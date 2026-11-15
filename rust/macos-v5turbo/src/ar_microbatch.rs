//! Request-local, resumable AR generation with transactional speculative draws.
use crate::{
    ar::{ArModel, ArState},
    ar_decision::read_decision,
    rng::RequestRng,
    sampling,
};
use mlx_rs::{
    ops::{
        concatenate,
        indexing::{argmax_axis, IndexOp},
        select,
    },
    Array, Dtype,
};
use std::error::Error;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type DiagnosticDecode<'a> = dyn FnMut(&Array, &mut ArState) -> Result<Array> + 'a;

pub struct Generation {
    pub tokens: Vec<i32>,
    pub eos_sampled: i32,
    pub eos_greedy: i32,
    pub consumed_draws: u64,
    pub speculative_draws: u64,
}
/// A consumed draw, including EOS. Arrays can be saved by an optional caller
/// trace; a single-token raw-logit trace may require additional evaluation.
pub struct Record {
    pub step: usize,
    pub logits: Array,
    pub q: Array,
    pub sampled: i32,
    pub greedy: i32,
    pub vocab: i32,
}
/// Only consumed records are returned. `tokens` excludes the EOS record even
/// when greedy EOS terminates a draw whose sampled ID is not EOS.
pub struct Batch {
    pub records: Vec<Record>,
    pub tokens: Vec<i32>,
    pub finished: bool,
    pub speculative_draws: u64,
}
struct PendingDraw {
    logits: Array,
    q: Array,
}

fn gpu_adjust(logits: &Array, history: &Array, vocab: &Array) -> Result<Array> {
    let repeated = vocab.eq(history.expand_dims(1)?)?.any_axis(2, false)?;
    let negative = logits.lt(Array::from_f32(0.0))?;
    let scaled = select(
        &negative,
        logits.multiply(Array::from_f32(1.35))?,
        logits.divide(Array::from_f32(1.35))?,
    )?;
    // Half logits: Python-float penalty in F32 opmath, rounded back to Half
    // (torch.where(score<0, score*p, score/p) on a Half tensor).
    Ok(select(&repeated, &scaled, logits)?.as_dtype(logits.dtype())?)
}

/// Owns one request's AR state. The model stays on the calling GPU owner thread.
pub struct ArGenerator<'a> {
    model: &'a ArModel,
    state: Option<ArState>,
    logits: Option<Array>,
    history: Option<Array>,
    previous: Vec<i32>,
    vocab1024: Option<Array>,
    vocab1025: Option<Array>,
    batch_size: usize,
    sampling: sampling::SamplingParams,
    step: usize,
    target_count: usize,
    speculative: u64,
    rng_identity: Option<(u64, u64, u64)>,
    finished: bool,
    failed: bool,
    diagnostic_decode: Option<Box<DiagnosticDecode<'a>>>,
}
impl<'a> ArGenerator<'a> {
    pub fn new(
        model: &'a ArModel,
        phones: &Array,
        bert: &Array,
        prompt: &Array,
        batch_size: usize,
    ) -> Result<Self> {
        if batch_size != 1 && batch_size != 4 {
            return Err("AR batch size must be 1 or 4".into());
        }
        let (logits, state) = model.prefill(phones, bert, prompt)?;
        Self::from_prefill(model, prompt, logits, state, batch_size)
    }

    /// Reuse a prefill already performed by the owning runtime.
    pub(crate) fn from_prefill(
        model: &'a ArModel,
        prompt: &Array,
        logits: Array,
        state: ArState,
        batch_size: usize,
    ) -> Result<Self> {
        if batch_size != 1 && batch_size != 4 {
            return Err("AR batch size must be 1 or 4".into());
        }
        if prompt.ndim() != 2
            || prompt.shape()[0] != 1
            || prompt.shape()[1] <= 0
            || state.audio_len() != prompt.shape()[1]
            || logits.shape() != [1, 1025]
            || !matches!(logits.dtype(), Dtype::Float32 | Dtype::Float16)
        {
            return Err("invalid AR prefill state, prompt or logits".into());
        }
        let history = prompt.as_dtype(Dtype::Int32)?;
        let previous = if batch_size == 1 {
            history.contiguous()?.to_vec_exact::<i32>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            model,
            state: Some(state),
            logits: Some(logits),
            history: Some(history),
            previous,
            vocab1024: (batch_size == 4)
                .then(|| Array::from_slice(&(0..1024).collect::<Vec<i32>>(), &[1, 1024, 1])),
            vocab1025: (batch_size == 4)
                .then(|| Array::from_slice(&(0..1025).collect::<Vec<i32>>(), &[1, 1025, 1])),
            batch_size,
            sampling: sampling::SamplingParams::DEFAULT,
            step: 0,
            target_count: 0,
            speculative: 0,
            rng_identity: None,
            finished: false,
            failed: false,
            diagnostic_decode: None,
        })
    }
    /// Select a request's sampling policy before consuming any AR draws.
    pub(crate) fn set_sampling(&mut self, params: sampling::SamplingParams) -> Result<()> {
        if self.step != 0 {
            return Err("AR sampling must be selected before decoding".into());
        }
        params.validate()?;
        self.sampling = params;
        Ok(())
    }
    /// Standalone diagnostic only; install before any generated step. The
    /// callback receives the trusted GPU sampler output and owns state updates.
    pub fn set_diagnostic_decode_override(
        &mut self,
        callback: Box<DiagnosticDecode<'a>>,
    ) -> Result<()> {
        if self.batch_size != 4 || self.step != 0 || self.diagnostic_decode.is_some() {
            return Err("diagnostic decode override requires a fresh batch-four generator".into());
        }
        self.diagnostic_decode = Some(callback);
        Ok(())
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    pub fn consumed_draws(&self) -> usize {
        self.step
    }
    pub fn speculative_draws(&self) -> u64 {
        self.speculative
    }

    /// Emit at most min(batch_size, max_draws, remaining) consumed records.
    /// CFM draws may advance between calls; external AR draws may not.
    pub fn next_batch(&mut self, rng: &mut RequestRng, max_draws: usize) -> Result<Batch> {
        if self.finished {
            return Err("AR generator is already finished".into());
        }
        if self.failed {
            return Err("AR generator cannot resume after an error".into());
        }
        if max_draws == 0 {
            return Err("max_draws must be positive".into());
        }
        if self.step >= 1500 {
            self.failed = true;
            return Err("AR did not reach EOS within 1500 draws".into());
        }
        if let Some((ar_seed, cfm_seed, ar_count)) = self.rng_identity {
            if rng.ar_seed() != ar_seed || rng.cfm_seed() != cfm_seed || rng.ar_draws() != ar_count
            {
                return Err("AR RNG changed outside this generator".into());
            }
        }
        let result = if self.batch_size == 1 {
            self.next_one(rng)
        } else {
            self.next_many(rng, max_draws.min(4).min(1500 - self.step))
        };
        match result {
            Ok(batch) => {
                self.rng_identity = Some((rng.ar_seed(), rng.cfm_seed(), rng.ar_draws()));
                Ok(batch)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }
    fn next_one(&mut self, rng: &mut RequestRng) -> Result<Batch> {
        let logits = self.logits.as_ref().ok_or("missing AR logits")?.clone();
        if !matches!(logits.dtype(), Dtype::Float32 | Dtype::Float16) || logits.shape() != [1, 1025] {
            return Err("invalid AR raw logits".into());
        }
        let step_logits = if self.step < 11 {
            logits.index((.., 0..1024))
        } else {
            logits.clone()
        };
        let adjusted = sampling::apply_repetition_penalty(&step_logits, &self.previous, 1.35)?;
        let mut reserved = rng.clone();
        let q = reserved.ar_exponential(step_logits.shape())?.as_dtype(step_logits.dtype())?;
        let token = sampling::sample(&adjusted, &[], &q, self.sampling.top_k, self.sampling.top_p, self.sampling.temperature, 1.0)?
            .as_dtype(Dtype::Int32)?;
        let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
        let (id, greedy_id) = read_decision(&q, &token, &greedy)?;
        rng.commit_ar_prefix(&reserved, 1)?;
        let finished = id == 1024 || greedy_id == 1024;
        let record = Record {
            step: self.step,
            logits,
            q,
            sampled: id,
            greedy: greedy_id,
            vocab: step_logits.shape()[1],
        };
        self.step += 1;
        let tokens = if finished { Vec::new() } else { vec![id] };
        if finished {
            if self.target_count == 0 {
                return Err("AR generated no target tokens".into());
            }
            self.finished = true;
            self.state = None;
            self.logits = None;
            self.history = None;
        } else {
            self.previous.push(id);
            self.target_count += 1;
            self.logits = Some(self.model.decode_generated(
                &token,
                id,
                self.state.as_mut().ok_or("missing AR state")?,
            )?);
        }
        Ok(Batch {
            records: vec![record],
            tokens,
            finished,
            speculative_draws: 0,
        })
    }
    fn next_many(&mut self, rng: &mut RequestRng, count: usize) -> Result<Batch> {
        let start = self.step;
        let mut reserved = rng.clone();
        let mut roots = Vec::new();
        let mut packed = Vec::new();
        let mut pending = Vec::new();
        let mut logits = self.logits.take().ok_or("missing AR logits")?;
        let mut history = self.history.take().ok_or("missing AR history")?;
        for offset in 0..count {
            let step = start + offset;
            if !matches!(logits.dtype(), Dtype::Float32 | Dtype::Float16) || logits.shape() != [1, 1025] {
                return Err("invalid AR raw logits".into());
            }
            let step_logits = if step < 11 {
                logits.index((.., 0..1024))
            } else {
                logits.clone()
            };
            let vocab = if step < 11 {
                self.vocab1024.as_ref()
            } else {
                self.vocab1025.as_ref()
            }
            .ok_or("missing AR vocabulary")?;
            let adjusted = gpu_adjust(&step_logits, &history, vocab)?;
            let q = reserved.ar_exponential(step_logits.shape())?.as_dtype(step_logits.dtype())?;
            let token = sampling::sample(&adjusted, &[], &q, self.sampling.top_k, self.sampling.top_p, self.sampling.temperature, 1.0)?
                .as_dtype(Dtype::Int32)?;
            let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
            let valid = q
                .is_finite()?
                .logical_and(q.gt(Array::from_f32(0.0))?)?
                .all(false)?
                .as_dtype(Dtype::Int32)?
                .reshape(&[1])?;
            packed.push(concatenate(
                &[valid.clone(), token.reshape(&[1])?, greedy.reshape(&[1])?],
                0,
            )?);
            let state = self.state.as_mut().ok_or("missing AR state")?;
            let next = if let Some(callback) = &mut self.diagnostic_decode {
                callback(&token, state)?
            } else {
                self.model.diagnostic_decode_sampled_gpu(&token, state)?
            };
            history = concatenate(&[history, token.clone()], 1)?;
            roots.extend([
                logits.clone(),
                q.clone(),
                token.clone(),
                greedy.clone(),
                valid.clone(),
            ]);
            pending.push(PendingDraw { logits, q });
            logits = next;
        }
        let decisions = concatenate(&packed, 0)?;
        roots.push(decisions.clone());
        roots.push(logits.clone());
        roots.push(history.clone());
        roots.extend(
            self.state
                .as_ref()
                .ok_or("missing AR state")?
                .diagnostic_cache_arrays(),
        );
        mlx_rs::transforms::eval(roots.iter())?;
        let values = decisions.to_vec_exact::<i32>()?;
        let mut consumed = 0usize;
        let mut finished = false;
        let mut tokens = Vec::new();
        for i in 0..count {
            let (valid, id, greedy) = (values[3 * i], values[3 * i + 1], values[3 * i + 2]);
            let vocab = if start + i < 11 { 1024 } else { 1025 };
            if valid != 1 || !(0..vocab).contains(&id) || !(0..vocab).contains(&greedy) {
                return Err("consumed AR decision has invalid q or token".into());
            }
            consumed += 1;
            if id == 1024 || greedy == 1024 {
                finished = true;
                break;
            }
            tokens.push(id);
        }
        rng.commit_ar_prefix(&reserved, consumed as u64)?;
        let records = pending
            .into_iter()
            .take(consumed)
            .enumerate()
            .map(|(i, draw)| Record {
                step: start + i,
                logits: draw.logits,
                q: draw.q,
                sampled: values[3 * i + 1],
                greedy: values[3 * i + 2],
                vocab: if start + i < 11 { 1024 } else { 1025 },
            })
            .collect();
        let speculative = (count - consumed) as u64;
        self.step += consumed;
        self.target_count += tokens.len();
        self.speculative += speculative;
        if finished {
            if self.target_count == 0 {
                return Err("AR generated no target tokens".into());
            }
            self.finished = true;
            self.state = None;
            self.logits = None;
            self.history = None;
        } else {
            self.logits = Some(logits);
            self.history = Some(history);
        }
        Ok(Batch {
            records,
            tokens,
            finished,
            speculative_draws: speculative,
        })
    }
}

/// Existing complete-generation API, implemented via the resumable generator.
pub fn generate(
    model: &ArModel,
    phones: &Array,
    bert: &Array,
    prompt: &Array,
    rng: &mut RequestRng,
    batch_size: usize,
) -> Result<Generation> {
    let mut generator = ArGenerator::new(model, phones, bert, prompt, batch_size)?;
    let mut tokens = Vec::new();
    loop {
        let batch = generator.next_batch(rng, batch_size)?;
        tokens.extend(batch.tokens);
        if batch.finished {
            let last = batch.records.last().ok_or("missing EOS record")?;
            return Ok(Generation {
                tokens,
                eos_sampled: last.sampled,
                eos_greedy: last.greedy,
                consumed_draws: generator.consumed_draws() as u64,
                speculative_draws: generator.speculative_draws(),
            });
        }
    }
}
