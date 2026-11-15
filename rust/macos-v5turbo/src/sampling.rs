//! MLX implementation of the official GPT-SoVITS logits sampling policy.
//! Sampling noise is provided by the caller so cross-runtime comparisons use
//! the same exponential draws as PyTorch.
use mlx_rs::{
    ops::indexing::argmax_axis,
    ops::{argsort_axis, select, softmax_axis, sort_axis},
    Array, Dtype,
};
use std::{error::Error, io};

pub type BoxError = Box<dyn Error>;
type Result<T> = std::result::Result<T, BoxError>;

fn invalid(message: impl Into<String>) -> BoxError {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn validate_logits(logits: &Array) -> Result<i32> {
    if logits.ndim() != 2 || logits.shape()[0] != 1 || logits.shape()[1] <= 0 {
        return Err(invalid(format!(
            "logits must have nonempty shape [1,V], got {:?}",
            logits.shape()
        )));
    }
    if !logits.dtype().is_float() {
        return Err(invalid("logits must use a floating-point dtype"));
    }
    Ok(logits.shape()[1])
}

fn scalar(value: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::from_f32(value).as_dtype(dtype)?)
}

/// Apply the official sign-aware repetition penalty, leaving all other logits
/// unchanged. `previous` is host-side token history already known by the caller.
pub fn apply_repetition_penalty(
    logits: &Array,
    previous: &[i32],
    repetition_penalty: f32,
) -> Result<Array> {
    let vocabulary = validate_logits(logits)?;
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return Err(invalid("repetition penalty must be finite and positive"));
    }
    if previous.is_empty() || repetition_penalty == 1.0 {
        return Ok(logits.clone());
    }

    let mut repeated = vec![false; vocabulary as usize];
    for &token in previous {
        if token < 0 || token >= vocabulary {
            return Err(invalid(format!(
                "previous token {token} is outside vocabulary [0,{vocabulary})"
            )));
        }
        repeated[token as usize] = true;
    }

    let dtype = logits.dtype();
    let repeated = Array::from_slice(&repeated, &[1, vocabulary]);
    let negative = logits.lt(&scalar(0.0, dtype)?)?;
    let scaled_negative = logits.multiply(&scalar(repetition_penalty, dtype)?)?;
    let scaled_nonnegative = logits.divide(&scalar(repetition_penalty, dtype)?)?;
    let adjusted = select(&negative, &scaled_negative, &scaled_nonnegative)?;
    Ok(select(&repeated, &adjusted, logits)?)
}

/// Request-level AR sampling parameters (official GPT-SoVITS semantics).
/// Repetition penalty stays at the official 1.35. The default is the
/// qualified top_k=15, top_p=1, temperature=1 policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
}
impl SamplingParams {
    pub const DEFAULT: Self = Self { top_k: 15, top_p: 1.0, temperature: 1.0 };
    /// Same legal range as the Windows API: 1 <= top_k <= 50,
    /// 0 < top_p <= 1, temperature > 0 (all finite).
    pub fn validate(&self) -> Result<()> {
        if !(1..=50).contains(&self.top_k) {
            return Err(invalid("top_k must be between 1 and 50"));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(invalid("top_p must be finite and in (0,1]"));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(invalid("temperature must be finite and positive"));
        }
        Ok(())
    }
    pub fn is_default(&self) -> bool {
        *self == Self::DEFAULT
    }
}

fn apply_top_p(logits: &Array, top_p: f32) -> Result<Array> {
    if top_p >= 1.0 {
        return Ok(logits.clone());
    }
    let vocabulary = logits.shape()[1];
    let dtype = logits.dtype();
    let descending = logits.multiply(&scalar(-1.0, dtype)?)?;
    let sorted_indices = argsort_axis(&descending, -1)?;
    let sorted_logits = logits.take_along_axis(&sorted_indices, -1)?;
    let cumulative = softmax_axis(&sorted_logits, -1, None)?.cumsum(-1, None, None)?;
    let above_threshold = cumulative.gt(&scalar(top_p, dtype)?)?;

    // The official code clears sorted mask index zero, retaining the best
    // option even when its cumulative probability already exceeds top_p.
    let mut keep_first = vec![false; vocabulary as usize];
    keep_first[0] = true;
    let keep_first = Array::from_slice(&keep_first, &[1, vocabulary]);
    let sorted_remove = select(
        &keep_first,
        &Array::from_slice(&[false], &[1, 1]),
        &above_threshold,
    )?;
    let remove = Array::from_slice(&vec![false; vocabulary as usize], &[1, vocabulary])
        .put_along_axis(&sorted_indices, &sorted_remove, -1)?;
    let negative_infinity = scalar(f32::NEG_INFINITY, dtype)?;
    Ok(select(&remove, &negative_infinity, logits)?)
}

/// Apply official GPT-SoVITS logits-to-probabilities and categorical sampling.
///
/// `logits` and `q` are `[1,V]`; `q` is the caller-supplied exponential noise
/// tensor. The result is an `int32` token array with shape `[1,1]`.
pub fn sample(
    logits: &Array,
    previous: &[i32],
    q: &Array,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
) -> Result<Array> {
    let vocabulary = validate_logits(logits)?;
    if q.shape() != logits.shape() || q.dtype() != logits.dtype() {
        return Err(invalid(format!(
            "exponential noise must match logits shape and dtype ({:?}, {:?}), got ({:?}, {:?})",
            logits.shape(),
            logits.dtype(),
            q.shape(),
            q.dtype()
        )));
    }
    if top_k == 0 {
        return Err(invalid("top_k must be at least one"));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(invalid("top_p must be finite and in [0,1]"));
    }
    if !temperature.is_finite() {
        return Err(invalid("temperature must be finite"));
    }

    let mut filtered = apply_repetition_penalty(logits, previous, repetition_penalty)?;
    filtered = apply_top_p(&filtered, top_p)?;
    filtered = filtered.divide(&scalar(temperature.max(1e-5), filtered.dtype())?)?;

    let k = top_k.min(vocabulary as usize) as i32;
    let ascending = sort_axis(&filtered, -1)?;
    let pivot_index = Array::from_slice(&[vocabulary - k], &[1]);
    let pivot = ascending.take_axis(&pivot_index, -1)?;
    let below_pivot = filtered.lt(&pivot)?;
    let negative_infinity = scalar(f32::NEG_INFINITY, filtered.dtype())?;
    filtered = select(&below_pivot, &negative_infinity, &filtered)?;

    let probabilities = softmax_axis(&filtered, -1, None)?;
    let noisy_probabilities = probabilities.divide(q)?;
    Ok(argmax_axis(&noisy_probabilities, -1, true)?.as_dtype(Dtype::Int32)?)
}
