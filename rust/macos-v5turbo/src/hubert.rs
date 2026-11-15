//! CN-HuBERT base model, evaluation mode, single unpadded waveform.
//! Config: 7 valid convolutions, group-normalized first layer, 12 post-norm
//! Transformer layers, 12 heads, 768 hidden and 3072 feed-forward dimensions.
//! Waveform preparation belongs to the caller. Checkpoint names are unchanged.
use mlx_rs::{
    fast::{layer_norm, scaled_dot_product_attention},
    nn::gelu,
    ops::conv1d,
    Array, Device, Dtype,
};
use std::{collections::HashMap, error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing HuBERT tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "HuBERT tensor {key}: expected {shape:?}, got {:?}",
            value.shape()
        )));
    }
    Ok(value.clone())
}
// Opt-in debug snapshots force evaluation and may change graph fusion. Compare
// traced and untraced final output separately when interpreting these captures.
fn trace(name: &str, value: &Array) -> Result<()> {
    if let Some(dir) = std::env::var_os("ANIFLIVE_HUBERT_TRACE_DIR") {
        let path = std::path::Path::new(&dir).join(format!("hubert-stage-{name}.npy"));
        value.eval()?;
        mlx_rs::with_device(Device::cpu(), || value.save_numpy(path))?;
    }
    Ok(())
}
// Pinned official Torch MPS creates sqrt(1/2) in the half input dtype.
// Preserve that coefficient while using float opmath, then round the output.
fn pytorch_gelu(input: &Array) -> Result<Array> {
    if input.dtype() == Dtype::Float16 {
        let x = input.as_dtype(Dtype::Float32)?;
        Ok(mlx_rs::ops::erf(x.multiply(Array::from_f32(0.70703125))?)?
            .add(Array::from_f32(1.0))?
            .multiply(Array::from_f32(0.5))?
            .multiply(&x)?
            .as_dtype(Dtype::Float16)?)
    } else {
        Ok(gelu(input)?)
    }
}
struct Linear {
    weight_t: Array,
    bias: Array,
}
impl Linear {
    fn load(weights: &HashMap<String, Array>, key: &str, input: i32, output: i32) -> Result<Self> {
        Ok(Self {
            weight_t: get(weights, &format!("{key}.weight"), &[output, input])?
                .transpose_axes(&[1, 0])?,
            bias: get(weights, &format!("{key}.bias"), &[output])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == Dtype::Float16 {
            // Match source F.linear's single half output boundary (fused bias).
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
struct Norm {
    weight: Array,
    bias: Array,
}
impl Norm {
    fn load(weights: &HashMap<String, Array>, key: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{key}.weight"), &[dim])?,
            bias: get(weights, &format!("{key}.bias"), &[dim])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == Dtype::Float16 {
            return crate::source_half_attention::layer_norm(x, &self.weight, &self.bias, 1e-5);
        }
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-5)?)
    }
}
struct FeatureConv {
    weight: Array,
    stride: i32,
}
impl FeatureConv {
    fn load(
        weights: &HashMap<String, Array>,
        i: usize,
        input: i32,
        kernel: i32,
        stride: i32,
    ) -> Result<Self> {
        let key = format!("feature_extractor.conv_layers.{i}.conv.weight");
        Ok(Self {
            weight: get(weights, &key, &[512, input, kernel])?.transpose_axes(&[0, 2, 1])?,
            stride,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(conv1d(x, &self.weight, self.stride, 0, 1, 1)?)
    }
}
struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm1: Norm,
    ff1: Linear,
    ff2: Linear,
    norm2: Norm,
}
impl EncoderLayer {
    fn load(weights: &HashMap<String, Array>, i: usize) -> Result<Self> {
        let p = format!("encoder.layers.{i}");
        Ok(Self {
            q: Linear::load(weights, &format!("{p}.attention.q_proj"), 768, 768)?,
            k: Linear::load(weights, &format!("{p}.attention.k_proj"), 768, 768)?,
            v: Linear::load(weights, &format!("{p}.attention.v_proj"), 768, 768)?,
            out: Linear::load(weights, &format!("{p}.attention.out_proj"), 768, 768)?,
            norm1: Norm::load(weights, &format!("{p}.layer_norm"), 768)?,
            ff1: Linear::load(
                weights,
                &format!("{p}.feed_forward.intermediate_dense"),
                768,
                3072,
            )?,
            ff2: Linear::load(
                weights,
                &format!("{p}.feed_forward.output_dense"),
                3072,
                768,
            )?,
            norm2: Norm::load(weights, &format!("{p}.final_layer_norm"), 768)?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let frames = x.shape()[1];
        let heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[1, frames, 12, 64])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(self.q.forward(x)?)?;
        let k = heads(self.k.forward(x)?)?;
        let v = heads(self.v.forward(x)?)?;
        // Half: official MPS SDPA semantics (explicit Half QK, Half scale,
        // F32 exp*reciprocal softmax, Half PV); with F32 LayerNorm this lifts
        // teacher-forced encoder-layer parity from ~30% to 92-95% exact.
        let attention = if q.dtype() == Dtype::Float16 {
            crate::source_half_attention::attention(&q, &k, &v, None)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, 0.125, None, None)?
        };
        let a = attention
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, frames, 768])?;
        let x = self.norm1.forward(&x.add(self.out.forward(&a)?)?)?;
        self.norm2
            .forward(&x.add(self.ff2.forward(&pytorch_gelu(&self.ff1.forward(&x)?)?)?)?)
    }
}
/// Fixed Chinese HuBERT base architecture from the downloaded HF config.
pub struct Hubert {
    feature: Vec<FeatureConv>,
    group_weight: Array,
    group_bias: Array,
    projection_norm: Norm,
    projection: Linear,
    position_weight: Array,
    position_bias: Array,
    encoder_norm: Norm,
    layers: Vec<EncoderLayer>,
}
impl Hubert {
    /// Load original HF state-dict tensor keys; weights retain their values and dtype.
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        Self::load_impl(weights, None)
    }
    /// Diagnostic only: replace the effective positional convolution constant
    /// with a source-computed [768,48,128] tensor. The original parameter map
    /// remains unchanged; this does not alter any other neural component.
    pub fn load_with_precomputed_position(
        weights: &HashMap<String, Array>,
        position_weight: &Array,
    ) -> Result<Self> {
        Self::load_impl(weights, Some(position_weight))
    }
    fn load_impl(weights: &HashMap<String, Array>, precomputed: Option<&Array>) -> Result<Self> {
        let kernels = [10, 3, 3, 3, 3, 2, 2];
        let strides = [5, 2, 2, 2, 2, 2, 2];
        let feature = (0..7)
            .map(|i| {
                FeatureConv::load(
                    weights,
                    i,
                    if i == 0 { 1 } else { 512 },
                    kernels[i],
                    strides[i],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let v = get(
            weights,
            "encoder.pos_conv_embed.conv.weight_v",
            &[768, 48, 128],
        )?;
        let g = get(
            weights,
            "encoder.pos_conv_embed.conv.weight_g",
            &[1, 1, 128],
        )?;
        // PyTorch weight_norm(dim=2): norm over the other two axes, with no epsilon.
        let position_weight = if let Some(source_weight) = precomputed {
            if source_weight.shape() != [768, 48, 128] || source_weight.dtype() != v.dtype() {
                return Err(invalid(
                    "precomputed positional weight must match original shape and dtype",
                ));
            }
            source_weight.transpose_axes(&[0, 2, 1])?
        } else {
            if v.dtype() == Dtype::Float16 {
                crate::source_half_attention::weight_norm(&v, &g, &[0, 1])?.transpose_axes(&[0, 2, 1])?
            } else {
                let denom = v.square()?.sum_axes(&[0, 1], true)?.sqrt()?;
                v.multiply(g.divide(denom)?)?.transpose_axes(&[0, 2, 1])?
            }
        };
        Ok(Self {
            feature,
            group_weight: get(
                weights,
                "feature_extractor.conv_layers.0.layer_norm.weight",
                &[512],
            )?,
            group_bias: get(
                weights,
                "feature_extractor.conv_layers.0.layer_norm.bias",
                &[512],
            )?,
            projection_norm: Norm::load(weights, "feature_projection.layer_norm", 512)?,
            projection: Linear::load(weights, "feature_projection.projection", 512, 768)?,
            position_weight,
            position_bias: get(weights, "encoder.pos_conv_embed.conv.bias", &[768])?,
            encoder_norm: Norm::load(weights, "encoder.layer_norm", 768)?,
            layers: (0..12)
                .map(|i| EncoderLayer::load(weights, i))
                .collect::<Result<Vec<_>>>()?,
        })
    }
    /// Input is prepared mono 16 kHz samples [1,S], at least 400 samples;
    /// output is the HF last hidden state [1,T,768]. No waveform normalization,
    /// padding or attention mask is applied by this method.
    pub fn forward(&self, waveform: &Array) -> Result<Array> {
        if waveform.ndim() != 2 || waveform.shape()[0] != 1 || waveform.shape()[1] < 400 {
            return Err(invalid(format!(
                "expected waveform [1,S] with S>=400, got {:?}",
                waveform.shape()
            )));
        }
        let mut h = waveform.expand_dims(2)?;
        for (i, conv) in self.feature.iter().enumerate() {
            h = conv.forward(&h)?;
            if i == 0 {
                trace("conv0", &h)?;
                // GroupNorm(512,512) normalizes each channel across time.
                // PyTorch GroupNorm accumulates FP16 inputs in FP32, then
                // restores the input dtype after normalization and affine.
                let dtype = h.dtype();
                let normalized_input = h.as_dtype(mlx_rs::Dtype::Float32)?;
                let mean = normalized_input.mean_axes(&[1], true)?;
                let centered = normalized_input.subtract(mean)?;
                let variance = centered.square()?.mean_axes(&[1], true)?;
                h = centered
                    .multiply(variance.add(Array::from_f32(1e-5))?.rsqrt()?)?
                    .multiply(&self.group_weight)?
                    .add(&self.group_bias)?
                    .as_dtype(dtype)?;
                trace("groupnorm0", &h)?;
            }
            h = pytorch_gelu(&h)?;
            trace(&format!("feature-{i}"), &h)?;
        }
        h = self
            .projection
            .forward(&self.projection_norm.forward(&h)?)?;
        trace("projection", &h)?;
        let frames = h.shape()[1];
        let positional =
            conv1d(&h, &self.position_weight, 1, 64, 1, 16)?.add(&self.position_bias)?;
        // Kernel 128 with padding 64 adds a trailing frame; HF SamePad removes it.
        let positional = positional.split_at_indices(&[frames], 1)?[0].clone();
        let position = pytorch_gelu(&positional)?;
        trace("position", &position)?;
        h = self.encoder_norm.forward(&h.add(position)?)?;
        trace("encoder-norm", &h)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            trace(&format!("layer-{i:02}"), &h)?;
        }
        Ok(h)
    }
}
