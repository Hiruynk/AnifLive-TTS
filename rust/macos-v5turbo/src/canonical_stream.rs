//! Experimental completed-mel neural-vocoder streaming; not AR/CFM streaming.
use crate::{
    acoustic::AcousticEncoder,
    cfm_noise_field::CfmNoiseField,
    estimator::{Estimator, PreparedSchedule},
    rng::RequestRng,
    streaming::{StreamChunk, StreamConfig, StreamReference, WindowTrace},
    vocoder::Vocoder,
};
use mlx_rs::{
    ops::{concatenate, indexing::IndexOp, select, zeros},
    Array, Dtype,
};
use std::{
    cell::RefCell,
    error::Error,
    io,
    sync::atomic::{AtomicBool, Ordering},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}
fn precision_value(value: Array, precision: Dtype) -> Result<Array> {
    if value.dtype() == precision {
        Ok(value)
    } else {
        Ok(value.as_dtype(precision)?)
    }
}
const HOP: i32 = 480;
const CORE: i32 = 16;
const HALO: i32 = 12;
const BUCKET: i32 = 205;

pub struct CanonicalStream<'a> {
    acoustic: &'a AcousticEncoder,
    estimator: &'a Estimator,
    vocoder: &'a Vocoder,
    schedule: Option<&'a PreparedSchedule>,
    stage: Option<&'a RefCell<Option<crate::cfm_stage::CfmStage>>>,
    ge: Array,
    features: Array,
    mel: Array,
    phones: Array,
    reference_frames: i32,
    precision: Dtype,
    field: Option<CfmNoiseField>,
    capture_noise: bool,
    previous_tokens: Vec<i32>,
    done: bool,
    failed: bool,
    cancelled: bool,
    cursor: i32,
    cfm_windows: usize,
    vocoder_calls: usize,
    calls_at_first_emit: Option<usize>,
    degenerate_short: bool,
    traces: Vec<WindowTrace>,
}
impl<'a> CanonicalStream<'a> {
    pub fn new(
        acoustic: &'a AcousticEncoder,
        estimator: &'a Estimator,
        vocoder: &'a Vocoder,
        reference: StreamReference,
        phones: Array,
        config: StreamConfig,
        schedule: Option<&'a PreparedSchedule>,
        coordinate_v2: bool,
        rng: &mut RequestRng,
    ) -> Result<Self> {
        if (
            config.core_frames,
            config.future_frames,
            config.holdback_frames,
            config.fade_frames,
        ) != (200, 12, 50, 4)
        {
            return Err(invalid(
                "canonical completed-mel stream requires default input geometry",
            ));
        }
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
            return Err(invalid("canonical vocoder requires MLX_ENABLE_TF32=0"));
        }
        let f = reference.features.shape();
        let m = reference.mel.shape();
        let precision = reference.ge.dtype();
        if !matches!(precision, Dtype::Float32 | Dtype::Float16) {
            return Err(invalid(
                "canonical reference precision must be float32 or float16",
            ));
        }
        if precision == Dtype::Float16 && (schedule.is_some() || coordinate_v2) {
            return Err(invalid(
                "FP16 canonical stream requires ordinary local CFM and v1 noise",
            ));
        }
        if reference.ge.shape() != [1, 512, 1]
            || reference.ge.dtype() != precision
            || f.len() != 3
            || f[0] != 1
            || f[1] != 512
            || f[2] <= 0
            || m.len() != 3
            || m[0] != 1
            || m[1] != 100
            || m[2] <= 0
            || reference.features.dtype() != precision
            || reference.mel.dtype() != precision
            || phones.ndim() != 2
            || phones.shape()[0] != 1
            || phones.shape()[1] <= 0
            || !matches!(phones.dtype(), Dtype::Int32 | Dtype::Int64)
        {
            return Err(invalid(
                "invalid canonical stream reference or target phones",
            ));
        }
        let limit = f[2].min(m[2]);
        let r = limit.min(500);
        let field = if coordinate_v2 {
            Some(CfmNoiseField::new(rng)?)
        } else {
            None
        };
        Ok(Self {
            acoustic,
            estimator,
            vocoder,
            schedule,
            stage: None,
            ge: reference.ge,
            features: reference.features.index((.., .., limit - r..limit)),
            mel: reference.mel.index((.., .., limit - r..limit)),
            phones,
            reference_frames: r,
            precision,
            field,
            capture_noise: config.capture_noise,
            previous_tokens: Vec::new(),
            done: false,
            failed: false,
            cancelled: false,
            cursor: 0,
            cfm_windows: 0,
            vocoder_calls: 0,
            calls_at_first_emit: None,
            degenerate_short: false,
            traces: Vec::new(),
        })
    }
    pub fn attach_cfm_stage(
        &mut self,
        stage: &'a RefCell<Option<crate::cfm_stage::CfmStage>>,
    ) -> Result<()> {
        if self.schedule.is_some() || stage.borrow().is_none() {
            return Err(invalid(
                "isolated CFM requires an installed stage and no local schedule",
            ));
        }
        self.stage = Some(stage);
        Ok(())
    }
    pub fn is_finished(&self) -> bool {
        self.done && !self.failed && !self.cancelled
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
    pub fn frame_cursor(&self) -> i32 {
        self.cursor
    }
    pub fn traces(&self) -> &[WindowTrace] {
        &self.traces
    }
    pub fn cfm_windows(&self) -> usize {
        self.cfm_windows
    }
    pub fn vocoder_calls(&self) -> usize {
        self.vocoder_calls
    }
    pub fn calls_at_first_emit(&self) -> Option<usize> {
        self.calls_at_first_emit
    }
    pub fn degenerate_short(&self) -> bool {
        self.degenerate_short
    }
    pub fn coordinate_v2(&self) -> bool {
        self.field.is_some()
    }
    fn cancel(&mut self) {
        self.done = true;
        self.cancelled = true;
    }

    fn complete_mel(
        &mut self,
        tokens: &[i32],
        stop: &AtomicBool,
        rng: &mut RequestRng,
    ) -> Result<Option<Array>> {
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        let n = i32::try_from(tokens.len()).map_err(|_| invalid("too many semantic tokens"))?;
        let frames = n
            .checked_mul(4)
            .ok_or_else(|| invalid("target frame overflow"))?;
        if frames <= 0 {
            return Err(invalid("canonical stream requires nonempty target"));
        }
        let codes = Array::from_slice(tokens, &[1, 1, n]);
        let target = self.acoustic.forward(&codes, &self.phones, &self.ge)?;
        target.eval()?;
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        if target.shape() != [1, 512, frames] || target.dtype() != self.precision {
            return Err(invalid("invalid canonical acoustic features"));
        }
        let r = self.reference_frames;
        let width = (1000 - r).min(640);
        let rolling = frames > width;
        let mut prior: Option<(Array, Array)> = None;
        let mut pieces = Vec::new();
        for (index, offset) in (0..frames).step_by(width as usize).enumerate() {
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            let current_frames = (frames - offset).min(width);
            let current = if rolling {
                target.index((.., .., offset..offset + current_frames))
            } else {
                target.clone()
            };
            let (prompt_mel, features) = if let Some((prior_mel, prior_features)) = &prior {
                let tail = 32
                    .min(r)
                    .min(prior_mel.shape()[2])
                    .min(prior_features.shape()[2]);
                let prefix = r - tail;
                (
                    concatenate(
                        &[
                            self.mel.index((.., .., 0..prefix)),
                            prior_mel.index((.., .., prior_mel.shape()[2] - tail..)),
                        ],
                        2,
                    )?,
                    concatenate(
                        &[
                            self.features.index((.., .., 0..prefix)),
                            prior_features.index((.., .., prior_features.shape()[2] - tail..)),
                        ],
                        2,
                    )?,
                )
            } else {
                (self.mel.clone(), self.features.clone())
            };
            let mu = concatenate(&[features, current.clone()], 2)?.transpose_axes(&[0, 2, 1])?;
            let prompt = concatenate(
                &[
                    prompt_mel,
                    precision_value(zeros::<f32>(&[1, 100, current_frames])?, self.precision)?,
                ],
                2,
            )?;
            if mu.dtype() != self.precision || prompt.dtype() != self.precision {
                return Err(invalid("canonical CFM conditioning promoted precision"));
            }
            let len = r + current_frames;
            let mask =
                Array::from_slice(&(0..len).map(|i| i >= r).collect::<Vec<_>>(), &[1, 1, len]);
            let noise = if let Some(first) = &self.field {
                let next = if index == 0 {
                    if rng.cfm_draws() != 1 {
                        return Err(invalid("missing coordinate field zero"));
                    }
                    None
                } else {
                    if rng.cfm_draws() != index as u64 {
                        return Err(invalid("invalid coordinate field sequence"));
                    }
                    Some(CfmNoiseField::next(rng)?)
                };
                next.as_ref()
                    .unwrap_or(first)
                    .cumulative(r, current_frames)?
            } else {
                if rng.cfm_draws() != index as u64 {
                    return Err(invalid("invalid canonical v1 CFM draw sequence"));
                }
                rng.cfm_normal(&[1, 100, len])?
            };
            let noise = precision_value(noise, self.precision)?;
            noise.eval()?;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            if self.capture_noise {
                self.traces.push(WindowTrace {
                    start_frame: offset,
                    end_frame: offset + current_frames,
                    future_frames: 0,
                    reference_frames: r,
                    semantic_tokens: tokens.to_vec(),
                    noise: noise.clone(),
                });
            }
            let mut x = select(
                &mask,
                noise.multiply(precision_value(Array::from_f32(0.875), self.precision)?)?,
                precision_value(Array::from_f32(0.0), self.precision)?,
            )?;
            if x.dtype() != self.precision {
                return Err(invalid("canonical CFM initial state promoted precision"));
            }
            if let Some(stage) = self.stage {
                if stop.load(Ordering::Acquire) {
                    self.cancel();
                    return Ok(None);
                }
                x = stage
                    .borrow_mut()
                    .as_mut()
                    .ok_or_else(|| invalid("CFM stage unavailable"))?
                    .infer(&mu, &prompt, &noise, r)?;
                if stop.load(Ordering::Acquire) {
                    self.cancel();
                    return Ok(None);
                }
            } else {
                let context = self
                    .schedule
                    .map(|_| self.estimator.prepare_context(&mu))
                    .transpose()?;
                if stop.load(Ordering::Acquire) {
                    self.cancel();
                    return Ok(None);
                }
                let mut static_condition: Option<Array> = None;
                for step in 0..4 {
                    if stop.load(Ordering::Acquire) {
                        self.cancel();
                        return Ok(None);
                    }
                    let time = precision_value(
                        Array::from_slice(&[step as f32 * 0.25], &[1]),
                        self.precision,
                    )?;
                    let velocity =
                        if let (Some(context), Some(schedule)) = (&context, self.schedule) {
                            self.estimator
                                .forward_prepared(&x, &prompt, context, schedule, step, None)?
                        } else {
                            self.estimator.forward_cached(&x, &prompt, &mu, &time, &mut static_condition)?
                        };
                    x =
                        select(
                            &mask,
                            x.add(velocity.multiply(precision_value(
                                Array::from_f32(0.25),
                                self.precision,
                            )?)?)?,
                            precision_value(Array::from_f32(0.0), self.precision)?,
                        )?;
                    if x.dtype() != self.precision {
                        return Err(invalid("canonical CFM state promoted precision"));
                    }
                    // Match the complete path's lazy evaluation through the final mel.
                }
            }
            let generated = x.index((.., .., r..));
            generated.eval()?;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            prior = Some((generated.clone(), current));
            pieces.push(generated);
            self.cfm_windows += 1;
        }
        let mel = if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            concatenate(&pieces, 2)?
        };
        if mel.shape() != [1, 100, frames] || mel.dtype() != self.precision {
            return Err(invalid("canonical rolling CFM lost or repeated frames"));
        }
        mel.eval()?;
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        Ok(Some(mel))
    }

    fn samples(audio: Array) -> Result<(Vec<f32>, Vec<i16>)> {
        let values = precision_value(audio, Dtype::Float32)?
            .reshape(&[-1])?
            .contiguous()?
            .to_vec_exact::<f32>()?;
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("empty or nonfinite canonical vocoder tile"));
        }
        let pcm = values
            .iter()
            .map(|v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16)
            .collect();
        Ok((values, pcm))
    }
    fn audible(pcm: &[i16]) -> bool {
        let threshold = 10f64.powf(-45.0 / 20.0);
        pcm.chunks_exact(HOP as usize).any(|frame| {
            let mean = frame
                .iter()
                .map(|v| {
                    let x = f64::from(*v) / 32768.0;
                    x * x
                })
                .sum::<f64>()
                / HOP as f64;
            mean.sqrt() > threshold
        })
    }

    pub fn push<F>(
        &mut self,
        tokens: &[i32],
        finished: bool,
        stop: &AtomicBool,
        rng: &mut RequestRng,
        sink: &mut F,
    ) -> Result<()>
    where
        F: FnMut(StreamChunk) -> Result<()>,
    {
        if self.done {
            return Err(invalid("canonical stream is already terminal"));
        }
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(());
        }
        let result = self.push_inner(tokens, finished, stop, rng, sink);
        if result.is_err() {
            self.failed = true;
            self.done = true;
        }
        result
    }
    fn push_inner<F>(
        &mut self,
        tokens: &[i32],
        finished: bool,
        stop: &AtomicBool,
        rng: &mut RequestRng,
        sink: &mut F,
    ) -> Result<()>
    where
        F: FnMut(StreamChunk) -> Result<()>,
    {
        if tokens.iter().any(|id| !(0..1024).contains(id))
            || tokens.len() < self.previous_tokens.len()
            || !tokens.starts_with(&self.previous_tokens)
        {
            return Err(invalid("invalid or nonmonotonic canonical semantic prefix"));
        }
        self.previous_tokens = tokens.to_vec();
        if !finished {
            return Ok(());
        }
        if tokens.is_empty() {
            return Err(invalid("cannot finish empty canonical target"));
        }
        let Some(mel) = self.complete_mel(tokens, stop, rng)? else {
            return Ok(());
        };
        let frames = mel.shape()[2];
        let original_total_samples = usize::try_from(frames)
            .ok()
            .and_then(|n| n.checked_mul(HOP as usize))
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("invalid canonical output sample count"))?;
        let raw = mel
            .add(precision_value(Array::from_f32(1.0), self.precision)?)?
            .divide(precision_value(Array::from_f32(2.0), self.precision)?)?
            .multiply(precision_value(Array::from_f32(14.0), self.precision)?)?
            .subtract(precision_value(Array::from_f32(12.0), self.precision)?)?;
        if raw.dtype() != self.precision {
            return Err(invalid("canonical vocoder input promoted precision"));
        }
        let mut buffered_samples = Vec::new();
        let mut buffered_pcm = Vec::new();
        let mut sent = false;
        let mut cursor = 0;
        self.degenerate_short = frames < CORE + HALO;
        if self.degenerate_short {
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            // No verified partial tile exists for this case. Use the actual
            // full-length vocoder and report that it preceded the first PCM.
            let full = self.vocoder.forward(&raw)?.contiguous()?;
            self.vocoder_calls += 1;
            if full.dtype() != self.precision {
                return Err(invalid("canonical vocoder output promoted precision"));
            }
            full.eval()?;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            let (samples, pcm) = Self::samples(full)?;
            if samples.len() != frames as usize * HOP as usize {
                return Err(invalid(
                    "short canonical vocoder returned an incomplete waveform",
                ));
            }
            if !Self::audible(&pcm) {
                return Err(invalid("short canonical output has no audible 10 ms frame"));
            }
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            sink(StreamChunk {
                samples,
                pcm,
                frame_end: frames,
                before_eos: false,
                final_chunk: true,
                original_total_samples: Some(original_total_samples),
            })?;
            self.calls_at_first_emit = Some(self.vocoder_calls);
            self.cursor = frames;
            if stop.load(Ordering::Acquire) {
                self.cancel();
            } else {
                self.done = true;
            }
            return Ok(());
        }
        while cursor + CORE + HALO <= frames {
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            let end = cursor + CORE;
            let available = cursor + CORE + HALO;
            let start = (cursor - HALO).max(0);
            let window = raw.index((.., .., start..available));
            let window_frames = available - start;
            let padded = concatenate(
                &[
                    window,
                    precision_value(
                        zeros::<f32>(&[1, 100, BUCKET - window_frames])?,
                        self.precision,
                    )?,
                ],
                2,
            )?;
            let result = self.vocoder.forward(&padded)?;
            if result.dtype() != self.precision {
                return Err(invalid("canonical vocoder output promoted precision"));
            }
            self.vocoder_calls += 1;
            let crop = (cursor - start) * HOP;
            let audio = result
                .index((.., .., crop..crop + (end - cursor) * HOP))
                .contiguous()?;
            audio.eval()?;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            let (samples, pcm) = Self::samples(audio)?;
            if samples.len() != (end - cursor) as usize * HOP as usize {
                return Err(invalid("canonical vocoder returned an incomplete tile"));
            }
            buffered_samples.extend(samples);
            buffered_pcm.extend(pcm);
            cursor = end;
            if sent || Self::audible(&buffered_pcm) {
                if stop.load(Ordering::Acquire) {
                    self.cancel();
                    return Ok(());
                }
                let first = !sent;
                let chunk = StreamChunk {
                    samples: std::mem::take(&mut buffered_samples),
                    pcm: std::mem::take(&mut buffered_pcm),
                    frame_end: cursor,
                    before_eos: false,
                    final_chunk: cursor == frames,
                    original_total_samples: Some(original_total_samples),
                };
                sink(chunk)?;
                if first {
                    self.calls_at_first_emit = Some(self.vocoder_calls);
                }
                sent = true;
                self.cursor = cursor;
                if stop.load(Ordering::Acquire) {
                    self.cancel();
                    return Ok(());
                }
            }
        }
        if !sent {
            return Err(invalid(
                "no audible 10 ms frame in available partial vocoder tiles",
            ));
        }
        if cursor < frames {
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            // Tail-only full vocoder call is permitted only after a partial sink emission.
            let full = self.vocoder.forward(&raw)?;
            if full.dtype() != self.precision {
                return Err(invalid("canonical vocoder tail promoted precision"));
            }
            self.vocoder_calls += 1;
            let tail = full
                .index((.., .., cursor * HOP..frames * HOP))
                .contiguous()?;
            tail.eval()?;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            let (samples, pcm) = Self::samples(tail)?;
            if samples.len() != (frames - cursor) as usize * HOP as usize {
                return Err(invalid("canonical vocoder returned an incomplete tail"));
            }
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            sink(StreamChunk {
                samples,
                pcm,
                frame_end: frames,
                before_eos: false,
                final_chunk: true,
                original_total_samples: Some(original_total_samples),
            })?;
            self.cursor = frames;
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
        }
        if self.cursor != frames {
            return Err(invalid("canonical PCM accounting failed"));
        }
        self.done = true;
        Ok(())
    }
}
