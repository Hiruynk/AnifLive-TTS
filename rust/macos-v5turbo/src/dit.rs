//! GPT-SoVITS v5 DiT transformer block, inference/evaluation mode.
//!
//! Operates on the original checkpoint tensor names; no weight quantization.
//! Implements the official AdaLN-Zero, partial rotary attention and feed-forward.
use std::{collections::HashMap, error::Error, io};

use mlx_rs::{
    fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn::{gelu_approximate, silu},
    ops::{concatenate, indexing::IndexOp, select},
    Array, Device, Dtype,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const DIM: i32 = 1024;
const HEADS: i32 = 16;
const HEAD_DIM: i32 = 64;

fn source_gelu_tanh(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() != Dtype::Float16 {
        return Ok(gelu_approximate(x)?);
    }
    let f = x.as_dtype(Dtype::Float32)?;
    let cube = f.multiply(&f)?.multiply(&f)?;
    let u = f
        .add(cube.multiply(Array::from_f32(0.044708251953125))?)?
        .multiply(Array::from_f32(0.7978515625))?;
    // MPSGraph Half tanh behaves as (e^2u-1)/(e^2u+1) in Float32 opmath;
    // same-input DiT FF capture 99.3-99.5% exact vs 93.5-95.8% for MLX tanh.
    // Clamp only avoids inf/inf; tanh is exactly 1.0 in Float32 beyond u=9.
    let e = mlx_rs::ops::exp(
        &mlx_rs::ops::minimum(&u, Array::from_f32(20.0))?.multiply(Array::from_f32(2.0))?,
    )?;
    // Division as multiplication by the Float32 reciprocal (same-input DiT
    // FF capture 99.88% exact vs 99.39% for a true division).
    Ok(e.subtract(Array::from_f32(1.0))?
        .multiply(mlx_rs::ops::reciprocal(&e.add(Array::from_f32(1.0))?)?)?
    .add(Array::from_f32(1.0))?
    .multiply(Array::from_f32(0.5))?
    .multiply(&f)?
    .as_dtype(Dtype::Float16)?)
}

fn source_silu(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() == Dtype::Float16 {
        Ok(silu(x.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)?)
    } else {
        Ok(silu(x)?)
    }
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing DiT tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "DiT tensor {key}: expected {shape:?}, got {:?}",
            value.shape()
        )));
    }
    Ok(value.clone())
}

// Optional diagnostic capture: explicitly requested by the caller. Evaluation
// boundaries may change fusion, so traced output must be compared separately.
fn trace(name: &str, value: &Array) -> Result<()> {
    if let Some(directory) = std::env::var_os("ANIFLIVE_DIT_TRACE_DIR") {
        let path = std::path::Path::new(&directory).join(format!("dit-stage-{name}.npy"));
        value.eval()?;
        mlx_rs::with_device(Device::cpu(), || value.save_numpy(path))?;
    }
    Ok(())
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
            if x.shape()[..x.ndim()-1].iter().product::<i32>() == 1 {
                // Half GEMV must add bias before the single half rounding.
                return Ok(mlx_rs::ops::addmm(self.bias.as_dtype(Dtype::Float32)?,
                    x.as_dtype(Dtype::Float32)?, self.weight_t.as_dtype(Dtype::Float32)?,
                    None, None)?.as_dtype(Dtype::Float16)?);
            }
            return Ok(mlx_rs::ops::addmm(
                &self.bias,
                x,
                &self.weight_t,
                None,
                None,
            )?);
        }
        Ok(x.matmul(&self.weight_t)?.add(&self.bias)?)
    }
}

// The official processor calls apply_rotary_pos_emb on projected [B,T,1024]
// BEFORE splitting heads. Thus only the first 64 projected channels rotate.
// x-transformers pairs adjacent channels: (a,b) -> (-b,a).
#[derive(Clone, Copy)]
enum Rotary<'a> {
    Angles(&'a Array),
    Cached(&'a Array, &'a Array),
}

fn rotate_projected(x: &Array, rope: Rotary<'_>, batch: i32, frames: i32) -> Result<Array> {
    let parts = x.split_at_indices(&[HEAD_DIM], 2)?;
    let paired = parts[0].reshape(&[batch, frames, HEAD_DIM / 2, 2])?;
    let pair = paired.split_equal(2, 3)?;
    let rotated = concatenate(&[pair[1].negative()?, pair[0].clone()], 3)?
        .reshape(&[batch, frames, HEAD_DIM])?;
    let value = match rope {
        Rotary::Angles(freqs) => parts[0]
            .multiply(freqs.cos()?)?
            .add(rotated.multiply(freqs.sin()?)?)?,
        Rotary::Cached(cos, sin) => parts[0].multiply(cos)?.add(rotated.multiply(sin)?)?,
    }
    .as_dtype(x.dtype())?;
    Ok(concatenate(&[value, parts[1].clone()], 2)?)
}

/// A single official v5 DiT block (1024 channels, 16 x 64 heads, FF x2).
enum PackedQkv {
    Horizontal(Array, Array),
    Batched(Array, Array),
}
pub struct DiTBlock {
    modulation: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    packed_qkv: Option<PackedQkv>,
}

impl DiTBlock {
    /// `prefix` is for example `cfm.estimator.transformer_blocks.0`.
    pub fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            modulation: Linear::load(weights, &format!("{prefix}.attn_norm.linear"), DIM, 6 * DIM)?,
            q: Linear::load(weights, &format!("{prefix}.attn.to_q"), DIM, DIM)?,
            k: Linear::load(weights, &format!("{prefix}.attn.to_k"), DIM, DIM)?,
            v: Linear::load(weights, &format!("{prefix}.attn.to_v"), DIM, DIM)?,
            out: Linear::load(weights, &format!("{prefix}.attn.to_out.0"), DIM, DIM)?,
            ff_in: Linear::load(weights, &format!("{prefix}.ff.ff.0.0"), DIM, 2 * DIM)?,
            ff_out: Linear::load(weights, &format!("{prefix}.ff.ff.2"), 2 * DIM, DIM)?,
            packed_qkv: None,
        })
    }

    /// `x` [B,T,1024], `time_embedding` [B,1024], optional boolean valid
    /// positions [B,T], and official raw rotary radians [1,T,64].
    /// V5 uses the default no-XPos rotary scale of 1. Dropout is disabled.
    pub fn forward(
        &self,
        x: &Array,
        time_embedding: &Array,
        mask: Option<&Array>,
        rope_freqs: &Array,
    ) -> Result<Array> {
        if self.packed_qkv.is_some() {
            return Err(invalid(
                "packed QKV supports only unmasked prepared forward",
            ));
        }
        if x.ndim() != 3 || x.shape()[0] <= 0 || x.shape()[1] <= 0 || x.shape()[2] != DIM {
            return Err(invalid(format!(
                "expected DiT input [B,T,{DIM}], got {:?}",
                x.shape()
            )));
        }
        let (batch, frames) = (x.shape()[0], x.shape()[1]);
        if time_embedding.shape() != [batch, DIM] {
            return Err(invalid(format!(
                "expected timestep embedding [{batch},{DIM}], got {:?}",
                time_embedding.shape()
            )));
        }
        if rope_freqs.shape() != [1, frames, HEAD_DIM] {
            return Err(invalid(format!(
                "expected rotary radians [1,{frames},{HEAD_DIM}], got {:?}",
                rope_freqs.shape()
            )));
        }
        if let Some(mask) = mask {
            if mask.shape() != [batch, frames] || mask.dtype() != Dtype::Bool {
                return Err(invalid(format!(
                    "expected boolean mask [{batch},{frames}], got {:?} {:?}",
                    mask.shape(),
                    mask.dtype()
                )));
            }
        }

        let conditioning = self.modulation.forward(&source_silu(time_embedding)?)?;
        self.forward_with_conditioning(x, &conditioning, mask, Rotary::Angles(rope_freqs))
    }

    pub(crate) fn prepare_modulation(&self, activated_time: &Array) -> Result<Array> {
        if activated_time.shape() != [1, DIM] || activated_time.dtype() != Dtype::Float32 {
            return Err(invalid("prepared DiT timestep must be float32 [1,1024]"));
        }
        self.modulation.forward(activated_time)
    }

    /// Read-only diagnostic extraction of the real pre-QKV AdaLN input.
    pub(crate) fn diagnostic_qkv_norm(&self, x: &Array, conditioning: &Array) -> Result<Array> {
        if x.ndim() != 3
            || x.shape()[0] != 1
            || x.shape()[2] != DIM
            || x.dtype() != Dtype::Float32
            || conditioning.shape() != [1, 6 * DIM]
            || conditioning.dtype() != Dtype::Float32
        {
            return Err(invalid("invalid diagnostic QKV normalization input"));
        }
        let chunks = conditioning.split_equal(6, 1)?;
        let shift = chunks[0].expand_dims(1)?;
        let scale = chunks[1].expand_dims(1)?;
        Ok(layer_norm(x, None, None, 1e-6)?
            .multiply(scale.add(Array::from_f32(1.0))?)?
            .add(shift)?)
    }

    /// Captured source projection tensors, in Q/K/V order, [K,N] + bias.
    pub(crate) fn diagnostic_qkv_weights(&self) -> Vec<Array> {
        vec![
            self.q.weight_t.clone(),
            self.k.weight_t.clone(),
            self.v.weight_t.clone(),
            self.q.bias.clone(),
            self.k.bias.clone(),
            self.v.bias.clone(),
        ]
    }

    pub(crate) fn prepare_packed_qkv(&self, batched: bool) -> Result<(Array, Array, Vec<Array>)> {
        let projections = self.diagnostic_qkv_weights();
        if projections.iter().any(|x| x.dtype() != Dtype::Float32) {
            return Err(invalid("packed QKV requires FP32 expanded weights"));
        }
        let source: Vec<Array> = projections[..3]
            .iter()
            .map(|x| x.transpose_axes(&[1, 0]))
            .collect::<std::result::Result<_, _>>()?;
        let packed = if batched {
            concatenate(
                &source
                    .iter()
                    .map(|x| x.expand_dims(0))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                0,
            )?
        } else {
            concatenate(&source, 0)?
        };
        let bias = if batched {
            concatenate(
                &projections[3..]
                    .iter()
                    .map(|x| x.expand_dims(0))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                0,
            )?
            .expand_dims(1)?
        } else {
            concatenate(&projections[3..], 0)?
        };
        let mut checks = Vec::new();
        for i in 0..3 {
            let start = (i * DIM as usize) as i32;
            let end = start + DIM;
            let w = if batched {
                packed.index((i as i32, .., ..))
            } else {
                packed.index((start..end, ..))
            }
            .contiguous()?
            .view_dtype(Dtype::Uint32)?;
            let original = source[i].contiguous()?.view_dtype(Dtype::Uint32)?;
            checks.push(w.eq(original)?.all(false)?);
            let b = if batched {
                bias.index((i as i32, 0, ..))
            } else {
                bias.index((start..end,))
            }
            .contiguous()?
            .view_dtype(Dtype::Uint32)?;
            let original_bias = projections[i + 3].contiguous()?.view_dtype(Dtype::Uint32)?;
            checks.push(b.eq(original_bias)?.all(false)?);
        }
        Ok((
            if batched {
                packed.transpose_axes(&[0, 2, 1])?
            } else {
                packed.transpose_axes(&[1, 0])?
            },
            bias,
            checks,
        ))
    }

    pub(crate) fn install_packed_qkv(&mut self, weight: Array, bias: Array, batched: bool) {
        self.packed_qkv = Some(if batched {
            PackedQkv::Batched(weight, bias)
        } else {
            PackedQkv::Horizontal(weight, bias)
        });
    }
    pub(crate) fn packed_qkv_enabled(&self) -> bool {
        self.packed_qkv.is_some()
    }
    pub(crate) fn batched_qkv_enabled(&self) -> bool {
        matches!(&self.packed_qkv, Some(PackedQkv::Batched(..)))
    }
    /// Inspect the logical V view independently; no contiguous conversion.
    pub(crate) fn diagnostic_batched_v(&self, norm: &Array) -> Result<Array> {
        match &self.packed_qkv {
            Some(PackedQkv::Batched(weight, bias)) => {
                if norm.ndim() != 3
                    || norm.shape()[0] != 1
                    || norm.shape()[2] != DIM
                    || norm.dtype() != Dtype::Float32
                {
                    return Err(invalid("invalid diagnostic batched V input"));
                }
                Ok(norm.matmul(weight)?.add(bias)?.split_equal(3, 0)?[2].clone())
            }
            _ => Err(invalid("batched QKV is not enabled")),
        }
    }

    /// Diagnostic-only prepared block with just the Q/K/V projection packed.
    /// V is made contiguous inside the graph before its normal reshape/SDPA.
    pub(crate) fn diagnostic_forward_packed_qkv(
        &self,
        x: &Array,
        conditioning: &Array,
        cos: &Array,
        sin: &Array,
        packed_weight: &Array,
        packed_bias: &Array,
    ) -> Result<Array> {
        self.diagnostic_forward_qkv_layout(
            x,
            conditioning,
            cos,
            sin,
            packed_weight,
            packed_bias,
            false,
        )
    }

    fn diagnostic_forward_qkv_layout(
        &self,
        x: &Array,
        conditioning: &Array,
        cos: &Array,
        sin: &Array,
        packed_weight: &Array,
        packed_bias: &Array,
        batched: bool,
    ) -> Result<Array> {
        if x.ndim() != 3
            || x.shape()[0] != 1
            || x.shape()[1] <= 0
            || x.shape()[2] != DIM
            || x.dtype() != Dtype::Float32
            || conditioning.shape() != [1, 6 * DIM]
            || conditioning.dtype() != Dtype::Float32
            || cos.shape() != [1, x.shape()[1], HEAD_DIM]
            || sin.shape() != cos.shape()
            || cos.dtype() != Dtype::Float32
            || sin.dtype() != Dtype::Float32
            || if batched {
                packed_weight.shape() != [3, DIM, DIM] || packed_bias.shape() != [3, 1, DIM]
            } else {
                packed_weight.shape() != [DIM, 3 * DIM] || packed_bias.shape() != [3 * DIM]
            }
            || packed_weight.dtype() != Dtype::Float32
            || packed_bias.dtype() != Dtype::Float32
        {
            return Err(invalid("invalid diagnostic packed QKV block input"));
        }
        let (batch, frames) = (x.shape()[0], x.shape()[1]);
        let chunks = conditioning.split_equal(6, 1)?;
        let shift_msa = chunks[0].expand_dims(1)?;
        let scale_msa = chunks[1].expand_dims(1)?;
        let gate_msa = chunks[2].expand_dims(1)?;
        let shift_mlp = chunks[3].expand_dims(1)?;
        let scale_mlp = chunks[4].expand_dims(1)?;
        let gate_mlp = chunks[5].expand_dims(1)?;
        let one = Array::from_f32(1.0).as_dtype(x.dtype())?;
        let norm = layer_norm(x, None, None, 1e-6)?
            .multiply(scale_msa.add(&one)?)?
            .add(&shift_msa)?;
        let packed = norm
            .matmul(packed_weight)?
            .add(packed_bias)?
            .split_equal(3, if batched { 0 } else { 2 })?;
        let rope = Rotary::Cached(cos, sin);
        let q = rotate_projected(&packed[0], rope, batch, frames)?
            .reshape(&[batch, frames, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = rotate_projected(&packed[1], rope, batch, frames)?
            .reshape(&[batch, frames, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = if batched {
            packed[2].clone()
        } else {
            packed[2].contiguous()?
        }
        .reshape(&[batch, frames, HEADS, HEAD_DIM])?
        .transpose_axes(&[0, 2, 1, 3])?;
        let attn = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / 8.0,
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, frames, DIM])?
        .as_dtype(q.dtype())?;
        let attention = self.out.forward(&attn)?;
        let residual = x.add(gate_msa.multiply(attention)?)?;
        let norm = layer_norm(&residual, None, None, 1e-6)?
            .multiply(scale_mlp.add(one)?)?
            .add(shift_mlp)?;
        let feed_forward = self
            .ff_out
            .forward(&source_gelu_tanh(self.ff_in.forward(&norm)?)?)?;
        Ok(residual.add(gate_mlp.multiply(feed_forward)?)?)
    }

    pub(crate) fn forward_prepared(
        &self,
        x: &Array,
        conditioning: &Array,
        mask: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        if x.ndim() != 3
            || x.shape()[0] != 1
            || x.shape()[1] <= 0
            || x.shape()[2] != DIM
            || x.dtype() != Dtype::Float32
            || conditioning.shape() != [1, 6 * DIM]
            || conditioning.dtype() != Dtype::Float32
            || cos.shape() != [1, x.shape()[1], HEAD_DIM]
            || sin.shape() != cos.shape()
            || cos.dtype() != Dtype::Float32
            || sin.dtype() != Dtype::Float32
        {
            return Err(invalid(
                "invalid FP32 prepared DiT input, conditioning or rotary values",
            ));
        }
        if let Some(packed) = &self.packed_qkv {
            if mask.is_some() {
                return Err(invalid(
                    "packed QKV supports only unmasked prepared forward",
                ));
            }
            return match packed {
                PackedQkv::Horizontal(weight, bias) => self.diagnostic_forward_qkv_layout(
                    x,
                    conditioning,
                    cos,
                    sin,
                    weight,
                    bias,
                    false,
                ),
                PackedQkv::Batched(weight, bias) => self.diagnostic_forward_qkv_layout(
                    x,
                    conditioning,
                    cos,
                    sin,
                    weight,
                    bias,
                    true,
                ),
            };
        }
        if let Some(m) = mask {
            if m.shape() != [1, x.shape()[1]] || m.dtype() != Dtype::Bool {
                return Err(invalid("invalid prepared DiT mask"));
            }
        }
        self.forward_with_conditioning(x, conditioning, mask, Rotary::Cached(cos, sin))
    }

    fn forward_with_conditioning(
        &self,
        x: &Array,
        conditioning: &Array,
        mask: Option<&Array>,
        rope: Rotary<'_>,
    ) -> Result<Array> {
        let (batch, frames) = (x.shape()[0], x.shape()[1]);
        trace("modulation", conditioning)?;
        let chunks = conditioning.split_equal(6, 1)?;
        let shift_msa = chunks[0].expand_dims(1)?;
        let scale_msa = chunks[1].expand_dims(1)?;
        let gate_msa = chunks[2].expand_dims(1)?;
        let shift_mlp = chunks[3].expand_dims(1)?;
        let scale_mlp = chunks[4].expand_dims(1)?;
        let gate_mlp = chunks[5].expand_dims(1)?;
        let one = Array::from_f32(1.0).as_dtype(x.dtype())?;
        let norm = layer_norm(x, None, None, 1e-6)?
            .multiply(scale_msa.add(&one)?)?
            .add(&shift_msa)?;

        trace("norm", &norm)?;
        let q = rotate_projected(&self.q.forward(&norm)?, rope, batch, frames)?
            .reshape(&[batch, frames, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = rotate_projected(&self.k.forward(&norm)?, rope, batch, frames)?
            .reshape(&[batch, frames, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&norm)?
            .reshape(&[batch, frames, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        trace("q", &q)?;
        trace("k", &k)?;
        trace("v", &v)?;
        let attn_mask = mask.map(|m| m.expand_dims_axes(&[1, 2])).transpose()?;
        let attention = if q.dtype() == Dtype::Float16 {
            crate::source_half_attention::attention(&q, &k, &v, attn_mask.as_ref())?
        } else {
            scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1.0 / 8.0,
                attn_mask.as_ref().map(ScaledDotProductAttentionMask::Array),
                None,
            )?
        };
        let attn = attention
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, frames, DIM])?
            .as_dtype(q.dtype())?;
        trace("sdpa", &attn)?;
        let mut attention = self.out.forward(&attn)?;
        if let Some(mask) = mask {
            attention = select(
                mask.expand_dims(2)?,
                &attention,
                Array::from_f32(0.0).as_dtype(x.dtype())?,
            )?;
        }
        trace("attention", &attention)?;
        let residual = x.add(gate_msa.multiply(attention)?)?;
        trace("residual", &residual)?;
        let norm = layer_norm(&residual, None, None, 1e-6)?
            .multiply(scale_mlp.add(one)?)?
            .add(shift_mlp)?;
        trace("ff_norm", &norm)?;
        let feed_forward = self
            .ff_out
            .forward(&source_gelu_tanh(self.ff_in.forward(&norm)?)?)?;
        trace("ff", &feed_forward)?;
        let output = residual.add(gate_mlp.multiply(feed_forward)?)?;
        trace("output", &output)?;
        Ok(output)
    }
}
