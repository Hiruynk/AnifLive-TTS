//! Experimental v2-coordinate bootstrap stream. Not quality-qualified.
//!
//! One 32-token partial solve emits 16 frames and retains four overlap frames;
//! the independent full solve occurs only after EOS. The original reference
//! and four-step CFG0 model remain unchanged. This is a new noise policy.
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

pub struct BootstrapStream<'a> {
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
    field: CfmNoiseField,
    capture_noise: bool,
    previous_tokens: Vec<i32>,
    pending: Option<Array>,
    bootstrap_emitted: bool,
    done: bool,
    failed: bool,
    cancelled: bool,
    cursor: i32,
    solve_count: usize,
    eos_solve_count: usize,
    traces: Vec<WindowTrace>,
    trace_field_indices: Vec<u64>,
}

impl<'a> BootstrapStream<'a> {
    pub fn new(
        acoustic: &'a AcousticEncoder,
        estimator: &'a Estimator,
        vocoder: &'a Vocoder,
        reference: StreamReference,
        phones: Array,
        config: StreamConfig,
        schedule: Option<&'a PreparedSchedule>,
        rng: &mut RequestRng,
    ) -> Result<Self> {
        if (
            config.core_frames,
            config.future_frames,
            config.holdback_frames,
            config.fade_frames,
        ) != (20, 12, 50, 4)
        {
            return Err(invalid("v2 bootstrap requires 20/12/50/4 input geometry and emits 16 frames with a 4-frame fade"));
        }
        let f = reference.features.shape();
        let m = reference.mel.shape();
        if reference.ge.shape() != [1, 512, 1]
            || reference.ge.dtype() != Dtype::Float32
            || f.len() != 3
            || f[0] != 1
            || f[1] != 512
            || f[2] <= 0
            || m.len() != 3
            || m[0] != 1
            || m[1] != 100
            || m[2] <= 0
            || reference.features.dtype() != Dtype::Float32
            || reference.mel.dtype() != Dtype::Float32
            || phones.ndim() != 2
            || phones.shape()[0] != 1
            || phones.shape()[1] <= 0
            || !matches!(phones.dtype(), Dtype::Int32 | Dtype::Int64)
        {
            return Err(invalid("invalid v2 bootstrap reference or target phones"));
        }
        let limit = f[2].min(m[2]);
        let r = limit.min(500);
        if r <= 0 || r + 128 > 1000 {
            return Err(invalid(
                "reference cannot fit the required 32-token bootstrap",
            ));
        }
        let ar_before = rng.ar_draws();
        let field = CfmNoiseField::new(rng)?;
        if rng.ar_draws() != ar_before || rng.cfm_draws() != 1 {
            return Err(invalid(
                "v2 field changed AR draws or used an unexpected CFM draw",
            ));
        }
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
            field,
            capture_noise: config.capture_noise,
            previous_tokens: Vec::new(),
            pending: None,
            bootstrap_emitted: false,
            done: false,
            failed: false,
            cancelled: false,
            cursor: 0,
            solve_count: 0,
            eos_solve_count: 0,
            traces: Vec::new(),
            trace_field_indices: Vec::new(),
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
        self.done && !self.cancelled && !self.failed
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
    pub fn frame_cursor(&self) -> i32 {
        self.cursor
    }
    pub fn solve_count(&self) -> usize {
        self.solve_count
    }
    pub fn eos_solve_count(&self) -> usize {
        self.eos_solve_count
    }
    pub fn traces(&self) -> &[WindowTrace] {
        &self.traces
    }
    pub fn trace_field_indices(&self) -> &[u64] {
        &self.trace_field_indices
    }
    pub fn bootstrap_emitted(&self) -> bool {
        self.bootstrap_emitted
    }

    fn cancel(&mut self) {
        self.cancelled = true;
        self.done = true;
        self.pending = None;
    }

    /// None indicates cancellation; errors are terminal at the caller.
    fn solve(
        &mut self,
        tokens: &[i32],
        stop: &AtomicBool,
        rng: &mut RequestRng,
        eos: bool,
    ) -> Result<Option<Array>> {
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        let n = i32::try_from(tokens.len()).map_err(|_| invalid("too many semantic tokens"))?;
        let frames = n
            .checked_mul(4)
            .ok_or_else(|| invalid("target frame count overflow"))?;
        if n == 0 {
            return Err(invalid("v2 bootstrap requires a nonempty target"));
        }
        let waveform_samples = frames
            .checked_mul(480)
            .ok_or_else(|| invalid("target sample count exceeds supported integer range"))?;
        let codes = Array::from_slice(tokens, &[1, 1, n]);
        let target = self.acoustic.forward(&codes, &self.phones, &self.ge)?;
        target.eval()?;
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        if target.shape() != [1, 512, frames] || target.dtype() != Dtype::Float32 {
            return Err(invalid("unexpected acoustic features in v2 bootstrap"));
        }
        let r = self.reference_frames;
        let chunk_frames = (1000 - r).min(640);
        let rolling = frames > chunk_frames;
        let mut prior: Option<(Array, Array)> = None;
        let mut pieces = Vec::new();
        for (index, offset) in (0..frames).step_by(chunk_frames as usize).enumerate() {
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            let current_frames = (frames - offset).min(chunk_frames);
            let current = if rolling {
                target.index((.., .., offset..offset + current_frames))
            } else {
                target.clone()
            };
            // Match complete synthesis / official v5: retain the original
            // anchor and replace its last <=32 frames with the prior block.
            let (prompt_mel, features) = if let Some((previous_mel, previous_features)) = &prior {
                let tail = 32
                    .min(r)
                    .min(previous_mel.shape()[2])
                    .min(previous_features.shape()[2]);
                let prefix = r - tail;
                (
                    concatenate(
                        &[
                            self.mel.index((.., .., 0..prefix)),
                            previous_mel.index((.., .., previous_mel.shape()[2] - tail..)),
                        ],
                        2,
                    )?,
                    concatenate(
                        &[
                            self.features.index((.., .., 0..prefix)),
                            previous_features.index((
                                ..,
                                ..,
                                previous_features.shape()[2] - tail..,
                            )),
                        ],
                        2,
                    )?,
                )
            } else {
                (self.mel.clone(), self.features.clone())
            };
            if rng.cfm_draws() != if index == 0 { 1 } else { index as u64 } {
                return Err(invalid("unexpected v2 rolling CFM field draw index"));
            }
            let next_field = if index == 0 {
                None
            } else {
                Some(CfmNoiseField::next(rng)?)
            };
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            let field = next_field.as_ref().unwrap_or(&self.field);
            let noise = field.cumulative(r, current_frames)?;
            let mu = concatenate(&[features, current.clone()], 2)?.transpose_axes(&[0, 2, 1])?;
            let prompt = concatenate(&[prompt_mel, zeros::<f32>(&[1, 100, current_frames])?], 2)?;
            let len = r + current_frames;
            let mask =
                Array::from_slice(&(0..len).map(|i| i >= r).collect::<Vec<_>>(), &[1, 1, len]);
            if self.capture_noise {
                self.traces.push(WindowTrace {
                    start_frame: offset,
                    end_frame: offset + current_frames,
                    future_frames: 0,
                    reference_frames: r,
                    semantic_tokens: tokens.to_vec(),
                    noise: noise.clone(),
                });
                self.trace_field_indices.push(index as u64);
            }
            let mut x = select(
                &mask,
                noise.multiply(Array::from_f32(0.875))?,
                Array::from_f32(0.0),
            )?;
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
                for step in 0..4 {
                    if stop.load(Ordering::Acquire) {
                        self.cancel();
                        return Ok(None);
                    }
                    let time = Array::from_slice(&[step as f32 * 0.25], &[1]);
                    let velocity =
                        if let (Some(context), Some(schedule)) = (&context, self.schedule) {
                            self.estimator
                                .forward_prepared(&x, &prompt, context, schedule, step, None)?
                        } else {
                            self.estimator.forward(&x, &prompt, &mu, &time, None)?
                        };
                    x = select(
                        &mask,
                        x.add(velocity.multiply(Array::from_f32(0.25))?)?,
                        Array::from_f32(0.0),
                    )?;
                    x.eval()?;
                    if stop.load(Ordering::Acquire) {
                        self.cancel();
                        return Ok(None);
                    }
                }
            }
            let generated = x.index((.., .., r..));
            if rolling {
                generated.eval()?;
            }
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(None);
            }
            prior = Some((generated.clone(), current));
            pieces.push(generated);
            self.solve_count += 1;
            if eos {
                self.eos_solve_count += 1;
            }
        }
        let generated = if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            concatenate(&pieces, 2)?
        };
        if generated.shape() != [1, 100, frames] {
            return Err(invalid("v2 rolling CFM lost or repeated target frames"));
        }
        let denorm = generated
            .add(Array::from_f32(1.0))?
            .divide(Array::from_f32(2.0))?
            .multiply(Array::from_f32(14.0))?
            .subtract(Array::from_f32(12.0))?;
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        let waveform = self.vocoder.forward(&denorm)?;
        waveform.eval()?;
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(None);
        }
        if waveform.shape() != [1, 1, waveform_samples] || waveform.dtype() != Dtype::Float32 {
            return Err(invalid("invalid v2 bootstrap waveform"));
        }
        Ok(Some(waveform.index((0, 0, ..))))
    }

    fn chunk(audio: Array, end: i32, before_eos: bool, final_chunk: bool) -> Result<StreamChunk> {
        let samples = audio.reshape(&[-1])?.contiguous()?.to_vec_exact::<f32>()?;
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err(invalid("v2 bootstrap produced empty or non-finite audio"));
        }
        let pcm = samples
            .iter()
            .map(|v| (v.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        Ok(StreamChunk {
            samples,
            pcm,
            frame_end: end,
            before_eos,
            final_chunk,
            original_total_samples: None,
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
            return Err(invalid("v2 bootstrap stream is already terminal"));
        }
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(());
        }
        let result = self.push_inner(tokens, finished, stop, rng, sink);
        if result.is_err() {
            self.failed = true;
            self.done = true;
            self.pending = None;
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
        if rng.cfm_draws() != 1
            || tokens.iter().any(|id| !(0..1024).contains(id))
            || tokens.len() < self.previous_tokens.len()
            || !tokens.starts_with(&self.previous_tokens)
        {
            return Err(invalid(
                "invalid v2 RNG state or nonmonotonic semantic prefix",
            ));
        }
        let frames = tokens
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("target frame count overflow"))?;
        if frames > i32::MAX as usize {
            return Err(invalid(
                "v2 bootstrap target frame count exceeds supported integer range",
            ));
        }
        self.previous_tokens = tokens.to_vec();
        if !finished && !self.bootstrap_emitted && tokens.len() >= 32 {
            let Some(wave) = self.solve(&tokens[..32], stop, rng, false)? else {
                return Ok(());
            };
            let chunk = Self::chunk(wave.index(..16 * 480), 16, true, false)?;
            self.pending = Some(wave.index(16 * 480..20 * 480));
            if stop.load(Ordering::Acquire) {
                self.cancel();
                return Ok(());
            }
            sink(chunk)?;
            self.bootstrap_emitted = true;
            self.cursor = 16;
            if stop.load(Ordering::Acquire) {
                self.cancel();
            }
            return Ok(());
        }
        if !finished {
            return Ok(());
        }
        if tokens.is_empty() {
            return Err(invalid("cannot finish an empty v2 target"));
        }
        let Some(wave) = self.solve(tokens, stop, rng, true)? else {
            return Ok(());
        };
        let output = if let Some(pending) = self.pending.take() {
            let alpha: Vec<f32> = (0..4 * 480)
                .map(|i| i as f32 / (4 * 480 - 1) as f32)
                .collect();
            let alpha = Array::from_slice(&alpha, &[4 * 480]);
            let blend = pending
                .multiply(Array::from_f32(1.0).subtract(&alpha)?)?
                .add(wave.index(16 * 480..20 * 480).multiply(alpha)?)?;
            concatenate(&[blend, wave.index(20 * 480..)], 0)?
        } else {
            wave
        };
        let expected = frames * 480 - self.cursor as usize * 480;
        let chunk = Self::chunk(output, frames as i32, false, true)?;
        if chunk.pcm.len() != expected {
            return Err(invalid("v2 sample conservation failed"));
        }
        if stop.load(Ordering::Acquire) {
            self.cancel();
            return Ok(());
        }
        sink(chunk)?;
        self.cursor = frames as i32;
        if stop.load(Ordering::Acquire) {
            self.cancel();
        } else {
            self.done = true;
        }
        Ok(())
    }
}
