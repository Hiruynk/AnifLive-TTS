//! Fixed-window V5 semantic streaming, model computation in FP32.
//!
//! Mirrors official `stream_v5_windows` at speed 1, four CFM steps and CFG 0.
//! Input updates are cumulative semantic prefixes. PCM uses the official stream
//! convention: clamp to [-1,1], multiply by 32767, truncate toward zero.
use crate::{acoustic::AcousticEncoder, estimator::Estimator, rng::RequestRng, vocoder::Vocoder};
use mlx_rs::{
    ops::{concatenate, indexing::IndexOp, select, zeros},
    Array, Dtype,
};
use std::{
    error::Error,
    io,
    sync::atomic::{AtomicBool, Ordering},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
fn stopped(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}
fn require_f32(a: &Array, shape: &[i32], label: &str) -> Result<()> {
    if a.dtype() != Dtype::Float32 || a.shape() != shape {
        return Err(invalid(format!(
            "{label}: expected float32 {shape:?}, got {:?} {:?}",
            a.shape(),
            a.dtype()
        )));
    }
    Ok(())
}
/// Reference features [1,512,T], normalized mel [1,100,T'], style [1,512,1].
/// The constructor performs the same length alignment and last-500 trim as
/// official `v5_reference`; mel must already use the [-12,2] normalization.
pub struct StreamReference {
    pub ge: Array,
    pub features: Array,
    pub mel: Array,
}
/// Fixed-window sizes expressed at 100 frames/sec. `capture_noise` is optional
/// for numerical replay and retains one standard-normal tensor per CFM window.
#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub core_frames: i32,
    pub future_frames: i32,
    pub holdback_frames: i32,
    pub fade_frames: i32,
    pub capture_noise: bool,
}
impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            core_frames: 200,
            future_frames: 12,
            holdback_frames: 50,
            fade_frames: 4,
            capture_noise: false,
        }
    }
}
/// An emitted chunk. `frame_end` is the accepted acoustic-frame cursor, while
/// a retained fade means emitted samples can trail this cursor until flush.
pub struct StreamChunk {
    pub samples: Vec<f32>,
    pub pcm: Vec<i16>,
    pub frame_end: i32,
    pub before_eos: bool,
    pub final_chunk: bool,
    /// Total canonical waveform samples, known only after complete mel exists.
    pub original_total_samples: Option<usize>,
}
/// Diagnostic data for reproducing a CFM window with the same actual noise.
pub struct WindowTrace {
    pub start_frame: i32,
    pub end_frame: i32,
    pub future_frames: i32,
    pub reference_frames: i32,
    pub semantic_tokens: Vec<i32>,
    pub noise: Array,
}

pub struct V5StreamDecoder<'a> {
    acoustic: &'a AcousticEncoder,
    estimator: &'a Estimator,
    cfm_schedule: Option<&'a crate::estimator::PreparedSchedule>,
    cfm_stage: Option<&'a std::cell::RefCell<Option<crate::cfm_stage::CfmStage>>>,
    vocoder: &'a Vocoder,
    ge: Array,
    ref_features: Array,
    ref_mel: Array,
    target_phones: Array,
    config: StreamConfig,
    ref_frames: i32,
    fade_samples: i32,
    left_frames: i32,
    cursor: i32,
    history: Array,
    pending: Option<Array>,
    previous_tokens: Vec<i32>,
    done: bool,
    cancelled: bool,
    failed: bool,
    traces: Vec<WindowTrace>,
}
impl<'a> V5StreamDecoder<'a> {
    pub fn attach_cfm_stage(
        &mut self,
        stage: &'a std::cell::RefCell<Option<crate::cfm_stage::CfmStage>>,
    ) -> Result<()> {
        if self.cfm_schedule.is_some() || stage.borrow().is_none() {
            return Err(invalid(
                "isolated CFM requires an installed stage and no local schedule",
            ));
        }
        self.cfm_stage = Some(stage);
        Ok(())
    }
    pub fn new(
        acoustic: &'a AcousticEncoder,
        estimator: &'a Estimator,
        vocoder: &'a Vocoder,
        reference: StreamReference,
        target_phones: Array,
        config: StreamConfig,
    ) -> Result<Self> {
        Self::new_with_schedule(
            acoustic,
            estimator,
            vocoder,
            reference,
            target_phones,
            config,
            None,
        )
    }
    pub fn new_with_schedule(
        acoustic: &'a AcousticEncoder,
        estimator: &'a Estimator,
        vocoder: &'a Vocoder,
        reference: StreamReference,
        target_phones: Array,
        config: StreamConfig,
        cfm_schedule: Option<&'a crate::estimator::PreparedSchedule>,
    ) -> Result<Self> {
        if !(20..=500).contains(&config.core_frames)
            || !(0..=50).contains(&config.future_frames)
            || !(16..=200).contains(&config.holdback_frames)
            || !(0..=20).contains(&config.fade_frames)
        {
            return Err(invalid(
                "stream window settings outside official fixed-mode ranges",
            ));
        }
        require_f32(&reference.ge, &[1, 512, 1], "reference style")?;
        let f = reference.features.shape();
        let m = reference.mel.shape();
        if f.len() != 3
            || f[0] != 1
            || f[1] != 512
            || f[2] <= 0
            || reference.features.dtype() != Dtype::Float32
            || m.len() != 3
            || m[0] != 1
            || m[1] != 100
            || m[2] <= 0
            || reference.mel.dtype() != Dtype::Float32
        {
            return Err(invalid(
                "reference features/mel must be nonempty float32 [1,512,T] / [1,100,T]",
            ));
        }
        if target_phones.ndim() != 2
            || target_phones.shape()[0] != 1
            || target_phones.shape()[1] <= 0
            || !matches!(target_phones.dtype(), Dtype::Int32 | Dtype::Int64)
        {
            return Err(invalid("target phones must be nonempty integer [1,N]"));
        }
        let limit = f[2].min(m[2]);
        let count = limit.min(500);
        let start = limit - count;
        if count + config.core_frames + config.future_frames > 1000 {
            return Err(invalid(
                "reference plus streaming core and future exceeds 1000-frame context",
            ));
        }
        let fade_frames = config.fade_frames.min(config.core_frames - 1);
        let left_frames = 24.max(fade_frames);
        Ok(Self {
            acoustic,
            estimator,
            cfm_schedule,
            cfm_stage: None,
            vocoder,
            ge: reference.ge,
            ref_features: reference.features.index((.., .., start..limit)),
            ref_mel: reference.mel.index((.., .., start..limit)),
            target_phones,
            config,
            ref_frames: count,
            fade_samples: fade_frames * 480,
            left_frames,
            cursor: 0,
            history: zeros::<f32>(&[1, 100, 0])?,
            pending: None,
            previous_tokens: Vec::new(),
            done: false,
            cancelled: false,
            failed: false,
            traces: Vec::new(),
        })
    }
    pub fn is_finished(&self) -> bool {
        self.done && !self.cancelled && !self.failed
    }
    pub fn is_failed(&self) -> bool {
        self.failed
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

    /// Push a cumulative immutable token prefix. The sink receives each new
    /// chunk immediately before another window is computed. EOS flushes overlap
    /// once; cancellation is terminal and prevents further sink calls.
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
            return Err(invalid("stream is already finished or cancelled"));
        }
        if stopped(stop) {
            self.done = true;
            self.cancelled = true;
            self.pending = None;
            return Ok(());
        }
        match self.push_inner(tokens, finished, stop, rng, sink) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.done = true;
                self.failed = true;
                self.pending = None;
                Err(error)
            }
        }
    }
    fn cancel(&mut self) {
        self.done = true;
        self.cancelled = true;
        self.pending = None;
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
        if tokens.iter().any(|token| !(0..1024).contains(token)) {
            return Err(invalid(
                "target semantic token is outside the V5 codebook [0,1024)",
            ));
        }
        if tokens.len() < self.previous_tokens.len() || !tokens.starts_with(&self.previous_tokens) {
            return Err(invalid(
                "semantic stream is not a monotonic cumulative prefix",
            ));
        }
        self.previous_tokens = tokens.to_vec();
        let core = self.config.core_frames;
        let future = self.config.future_frames;
        let holdback = self.config.holdback_frames;
        let needed = (self.cursor + core + future + holdback + 3) / 4;
        if !finished && tokens.len() < needed as usize {
            return Ok(());
        }
        if tokens.is_empty() {
            return Err(invalid(
                "cannot finalize a stream with no target semantic tokens",
            ));
        }
        let n = i32::try_from(tokens.len()).map_err(|_| invalid("too many semantic tokens"))?;
        let codes = Array::from_slice(tokens, &[1, 1, n]);
        if stopped(stop) {
            return Ok(self.cancel());
        }
        let target = self
            .acoustic
            .forward(&codes, &self.target_phones, &self.ge)?;
        target.eval()?;
        if stopped(stop) {
            return Ok(self.cancel());
        }
        if target.ndim() != 3
            || target.shape()[0] != 1
            || target.shape()[1] != 512
            || target.dtype() != Dtype::Float32
        {
            return Err(invalid("acoustic encoder returned invalid target features"));
        }
        let total = target.shape()[2];
        if total < self.cursor {
            return Err(invalid("target acoustic prefix moved backwards"));
        }
        let safe_end = if finished {
            total
        } else {
            (total - future - holdback).max(0)
        };
        while self.cursor < total && (finished || self.cursor + core <= safe_end) {
            if stopped(stop) {
                return Ok(self.cancel());
            }
            let start = self.cursor;
            let end = (start + core).min(total);
            let (prompt, features) = if start == 0 {
                (self.ref_mel.clone(), self.ref_features.clone())
            } else {
                let tail = 32.min(self.ref_frames).min(self.history.shape()[2]);
                let prefix = self.ref_frames - tail;
                let hist = self
                    .history
                    .index((.., .., self.history.shape()[2] - tail..));
                let target_tail = target.index((.., .., start - tail..start));
                (
                    concatenate(&[self.ref_mel.index((.., .., 0..prefix)), hist], 2)?,
                    concatenate(
                        &[self.ref_features.index((.., .., 0..prefix)), target_tail],
                        2,
                    )?,
                )
            };
            let target_end = (end + future).min(total);
            let mu = concatenate(&[features, target.index((.., .., start..target_end))], 2)?
                .transpose_axes(&[0, 2, 1])?;
            let count = prompt.shape()[2];
            let frame_count = mu.shape()[1];
            let delta = frame_count - count;
            let prompt_full = concatenate(&[prompt, zeros::<f32>(&[1, 100, delta])?], 2)?;
            let mask = Array::from_slice(
                &(0..frame_count).map(|i| i >= count).collect::<Vec<_>>(),
                &[1, 1, frame_count],
            );
            let noise = rng.cfm_normal(&[1, 100, frame_count])?;
            if self.config.capture_noise {
                self.traces.push(WindowTrace {
                    start_frame: start,
                    end_frame: end,
                    future_frames: target_end - end,
                    reference_frames: count,
                    semantic_tokens: tokens.to_vec(),
                    noise: noise.clone(),
                });
            }
            let mut x = select(
                &mask,
                noise.multiply(Array::from_f32(0.875))?,
                Array::from_f32(0.0),
            )?;
            if let Some(connection) = self.cfm_stage {
                if stopped(stop) {
                    return Ok(self.cancel());
                }
                x = connection
                    .borrow_mut()
                    .as_mut()
                    .ok_or_else(|| invalid("CFM stage unavailable"))?
                    .infer(&mu, &prompt_full, &noise, count)?;
                if stopped(stop) {
                    return Ok(self.cancel());
                }
            } else {
                let prepared_context = self
                    .cfm_schedule
                    .map(|_| self.estimator.prepare_context(&mu))
                    .transpose()?;
                for step in 0..4 {
                    if stopped(stop) {
                        return Ok(self.cancel());
                    }
                    let t = Array::from_slice(&[step as f32 * 0.25], &[1]);
                    let velocity = if let (Some(context), Some(schedule)) =
                        (&prepared_context, self.cfm_schedule)
                    {
                        self.estimator.forward_prepared(
                            &x,
                            &prompt_full,
                            context,
                            schedule,
                            step,
                            None,
                        )?
                    } else {
                        self.estimator.forward(&x, &prompt_full, &mu, &t, None)?
                    };
                    x = select(
                        &mask,
                        x.add(velocity.multiply(Array::from_f32(0.25))?)?,
                        Array::from_f32(0.0),
                    )?;
                    x.eval()?;
                    if stopped(stop) {
                        return Ok(self.cancel());
                    }
                }
            }
            let generated = x.index((.., .., count..));
            let accepted = generated.index((.., .., 0..end - start));
            let history_count = self.history.shape()[2];
            let left = self
                .history
                .index((.., .., (history_count - self.left_frames).max(0)..));
            let left_count = left.shape()[2];
            let denorm = concatenate(&[left, generated], 2)?
                .add(Array::from_f32(1.0))?
                .divide(Array::from_f32(2.0))?
                .multiply(Array::from_f32(14.0))?
                .subtract(Array::from_f32(12.0))?;
            if stopped(stop) {
                return Ok(self.cancel());
            }
            let waveform = self.vocoder.forward(&denorm)?;
            waveform.eval()?;
            if stopped(stop) {
                return Ok(self.cancel());
            }
            if waveform.ndim() != 3
                || waveform.shape()[0] != 1
                || waveform.shape()[1] != 1
                || waveform.dtype() != Dtype::Float32
            {
                return Err(invalid("vocoder returned invalid waveform shape or dtype"));
            }
            let offset = left_count * 480;
            let expected = (end - start) * 480;
            if waveform.shape()[2] < offset + expected {
                return Err(invalid("vocoder returned incomplete acoustic window"));
            }
            let audio = waveform.index((0, 0, ..));
            let current = audio.index(offset..offset + expected);
            let prefix_audio = if let Some(pending) = self.pending.take() {
                if self.fade_samples <= 0 || offset < self.fade_samples {
                    return Err(invalid("invalid streaming fade state"));
                }
                let alpha: Vec<f32> = (0..self.fade_samples)
                    .map(|i| {
                        if self.fade_samples == 1 {
                            0.0
                        } else {
                            i as f32 / (self.fade_samples - 1) as f32
                        }
                    })
                    .collect();
                let alpha = Array::from_slice(&alpha, &[self.fade_samples]);
                let prev = audio.index(offset - self.fade_samples..offset);
                pending
                    .multiply(Array::from_f32(1.0).subtract(&alpha)?)?
                    .add(prev.multiply(alpha)?)?
            } else {
                zeros::<f32>(&[0])?
            };
            let emitted = if self.fade_samples > 0 && (!finished || end != total) {
                if current.shape()[0] < self.fade_samples {
                    return Err(invalid("streaming core shorter than overlap"));
                }
                let keep = current.shape()[0] - self.fade_samples;
                self.pending = Some(current.index(keep..));
                concatenate(&[prefix_audio, current.index(0..keep)], 0)?
            } else {
                self.pending = None;
                concatenate(&[prefix_audio, current], 0)?
            };
            self.history = concatenate(&[self.history.clone(), accepted], 2)?;
            let keep = self.left_frames.max(32);
            self.history = self
                .history
                .index((.., .., (self.history.shape()[2] - keep).max(0)..));
            self.cursor = end;
            if stopped(stop) {
                return Ok(self.cancel());
            }
            if let Some(chunk) = Self::chunk(emitted, end, !finished, finished && end == total)? {
                if stopped(stop) {
                    return Ok(self.cancel());
                }
                sink(chunk)?;
                if stopped(stop) {
                    return Ok(self.cancel());
                }
            }
        }
        if finished {
            if self.cursor != total {
                return Err(invalid("semantic streaming did not flush acoustic frames"));
            }
            if let Some(pending) = self.pending.take() {
                if stopped(stop) {
                    return Ok(self.cancel());
                }
                if let Some(chunk) = Self::chunk(pending, self.cursor, false, true)? {
                    if stopped(stop) {
                        return Ok(self.cancel());
                    }
                    sink(chunk)?;
                    if stopped(stop) {
                        return Ok(self.cancel());
                    }
                }
            }
            self.done = true;
        }
        Ok(())
    }
    fn chunk(
        audio: Array,
        end: i32,
        before_eos: bool,
        final_chunk: bool,
    ) -> Result<Option<StreamChunk>> {
        let flat = audio.reshape(&[-1])?.contiguous()?;
        let samples = flat.to_vec_exact::<f32>()?;
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite V5 streaming audio"));
        }
        if samples.is_empty() {
            return Ok(None);
        }
        let pcm = samples
            .iter()
            .map(|v| (v.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        Ok(Some(StreamChunk {
            samples,
            pcm,
            frame_end: end,
            before_eos,
            final_chunk,
            original_total_samples: None,
        }))
    }
}
