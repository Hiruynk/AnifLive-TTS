//! GPT-SoVITS v5 reference MelStyleEncoder, evaluation mode.
//! Implements the single, fully valid reference path of decode_encp.
use mlx_rs::{
    nn::{mish, sigmoid},
    ops::{conv1d, softmax_axis},
    Array, Dtype,
};
use std::{collections::HashMap, error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn source_mish(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() != Dtype::Float16 { return Ok(mish(x)?); }
    // Official MPS Half F.softplus evaluates log(1 + exp(x)) in Float32 (plain
    // log, not log1p) and rounds to Half: same-input style capture 100% exact.
    let f = x.as_dtype(Dtype::Float32)?;
    let soft = mlx_rs::ops::log(&mlx_rs::ops::exp(&f)?.add(Array::from_f32(1.0))?)?.as_dtype(Dtype::Float16)?;
    Ok(x.multiply(mlx_rs::ops::tanh(soft)?)?)
}

fn source_sigmoid(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() == mlx_rs::Dtype::Float16 {
        Ok(sigmoid(x.as_dtype(mlx_rs::Dtype::Float32)?)?.as_dtype(mlx_rs::Dtype::Float16)?)
    } else { Ok(sigmoid(x)?) }
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing reference encoder tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "reference encoder tensor {key}: expected {shape:?}, got {:?}",
            value.shape()
        )));
    }
    Ok(value.clone())
}
struct Linear {
    weight_t: Array,
    bias: Array,
}
impl Linear {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        input: i32,
        output: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight_t: get(weights, &format!("{prefix}.weight"), &[output, input])?
                .transpose_axes(&[1, 0])?,
            bias: get(weights, &format!("{prefix}.bias"), &[output])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == Dtype::Float16 {
            return Ok(mlx_rs::ops::addmm(&self.bias, x, &self.weight_t, None, None)?);
        }
        Ok(x.matmul(&self.weight_t)?.add(&self.bias)?)
    }
}
struct ConvGlu {
    weight: Array,
    bias: Array,
}
impl ConvGlu {
    fn load(weights: &HashMap<String, Array>, index: usize) -> Result<Self> {
        let prefix = format!("ref_enc.temporal.{index}.conv1.conv");
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), &[256, 128, 5])?
                .transpose_axes(&[0, 2, 1])?,
            bias: get(weights, &format!("{prefix}.bias"), &[256])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = if x.dtype() == Dtype::Float16 {
            conv1d(x.as_dtype(Dtype::Float32)?, self.weight.as_dtype(Dtype::Float32)?, 1, 2, 1, 1)?
                .add(self.bias.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)?
        } else { conv1d(x, &self.weight, 1, 2, 1, 1)?.add(&self.bias)? };
        let chunks = y.split_equal(2, 2)?;
        Ok(x.add(chunks[0].multiply(source_sigmoid(&chunks[1])?)?)?)
    }
}
/// Official V5 `MelStyleEncoder(704, style_vector_dim=512)`.
pub struct ReferenceEncoder {
    spectral1: Linear,
    spectral2: Linear,
    temporal: Vec<ConvGlu>,
    q: Linear,
    k: Linear,
    v: Linear,
    attention_out: Linear,
    fc: Linear,
}
impl ReferenceEncoder {
    /// Load name-preserving official `ref_enc.*` tensors. Dtype is preserved.
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        Ok(Self {
            spectral1: Linear::load(weights, "ref_enc.spectral.0.fc", 704, 128)?,
            spectral2: Linear::load(weights, "ref_enc.spectral.3.fc", 128, 128)?,
            temporal: (0..2)
                .map(|i| ConvGlu::load(weights, i))
                .collect::<Result<Vec<_>>>()?,
            q: Linear::load(weights, "ref_enc.slf_attn.w_qs", 128, 128)?,
            k: Linear::load(weights, "ref_enc.slf_attn.w_ks", 128, 128)?,
            v: Linear::load(weights, "ref_enc.slf_attn.w_vs", 128, 128)?,
            attention_out: Linear::load(weights, "ref_enc.slf_attn.fc", 128, 128)?,
            fc: Linear::load(weights, "ref_enc.fc.fc", 128, 512)?,
        })
    }
    /// Input is the first 704 bins of one reference spectrogram [1,704,T].
    /// Every reference frame is valid, as in official `decode_encp`; output is
    /// style conditioning `ge` [1,512,1]. Padded batches are not supported.
    pub fn forward(&self, reference: &Array) -> Result<Array> {
        if reference.ndim() != 3
            || reference.shape()[0] != 1
            || reference.shape()[1] != 704
            || reference.shape()[2] <= 0
        {
            return Err(invalid(format!(
                "expected reference spectrogram [1,704,T], got {:?}",
                reference.shape()
            )));
        }
        let frames = reference.shape()[2];
        let input = reference.transpose_axes(&[0, 2, 1])?;
        let mut x = source_mish(self.spectral1.forward(&input)?)?;
        x = source_mish(self.spectral2.forward(&x)?)?;
        for layer in &self.temporal {
            x = layer.forward(&x)?;
        }
        let heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[1, frames, 2, 64])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(self.q.forward(&x)?)?;
        let k = heads(self.k.forward(&x)?)?;
        let v = heads(self.v.forward(&x)?)?;
        // Official temperature is sqrt(d_model=128), not sqrt(head_dim=64).
        let raw = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
        let probability = if raw.dtype() == Dtype::Float16 {
            // np.power(128, 0.5) Python float divisor: Float32 opmath, then
            // Half; MPS softmax semantics as in source_half_attention.
            let scores = raw
                .as_dtype(Dtype::Float32)?
                .divide(Array::from_f32(128f64.sqrt() as f32))?
                .as_dtype(Dtype::Float16)?;
            crate::source_half_attention::softmax(&scores)?
        } else {
            softmax_axis(raw.divide(Array::from_f32(128f32.sqrt()))?, -1, false)?
        };
        let context = probability
            .matmul(&v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, frames, 128])?;
        x = x.add(self.attention_out.forward(&context)?)?;
        let output = self.fc.forward(&x)?;
        // The all-valid official mask path casts to f32, divides each element
        // by length, then sums over time and restores the original dtype.
        let pooled = output
            .as_dtype(Dtype::Float32)?
            .divide(Array::from_f32(frames as f32))?
            .sum_axes(&[1], false)?
            .as_dtype(output.dtype())?;
        Ok(pooled.expand_dims(2)?)
    }
}
