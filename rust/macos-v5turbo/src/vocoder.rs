//! Official GPT-SoVITS v5 48 kHz Generator (ResBlock1 configuration).
//!
//! Weights retain their PyTorch state-dict names and layouts. Inputs and outputs
//! use PyTorch's [batch, channels, length]; MLX convolutions use NLC internally.
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io,
};

use mlx_rs::Dtype;
use mlx_rs::{
    nn::leaky_relu,
    ops::{conv1d, conv_transpose1d, tanh},
    Array,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing vocoder tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "vocoder tensor {key}: expected {shape:?}, got {:?}",
            value.shape()
        )));
    }
    Ok(value.clone())
}

// PyTorch's default weight_norm(dim=0) reduces every dimension except 0,
// including for ConvTranspose1d, whose first dimension is input channels.
fn weight(weights: &HashMap<String, Array>, prefix: &str, shape: &[i32]) -> Result<Array> {
    let plain = format!("{prefix}.weight");
    let g_key = format!("{prefix}.weight_g");
    let v_key = format!("{prefix}.weight_v");
    match (
        weights.contains_key(&plain),
        weights.contains_key(&g_key),
        weights.contains_key(&v_key),
    ) {
        (true, false, false) => get(weights, &plain, shape),
        (false, true, true) => {
            let v = get(weights, &v_key, shape)?;
            let g = get(weights, &g_key, &[shape[0], 1, 1])?;
            let norm = v.square()?.sum_axes(&[1, 2], true)?.sqrt()?;
            Ok(v.multiply(g.divide(norm)?)?)
        }
        _ => Err(invalid(format!(
            "{prefix}: expected either weight or the complete weight_g/weight_v pair"
        ))),
    }
}

struct Conv {
    weight: Array,
    bias: Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    transposed: bool,
}

impl Conv {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        input: i32,
        output: i32,
        kernel: i32,
        stride: i32,
        dilation: i32,
        transposed: bool,
    ) -> Result<Self> {
        let torch_shape = if transposed {
            [input, output, kernel]
        } else {
            [output, input, kernel]
        };
        let source_weight = weight(weights, prefix, &torch_shape)?;
        // [out,in,k] -> [out,k,in]; transposed [in,out,k] -> [out,k,in].
        let axes = if transposed { [1, 2, 0] } else { [0, 2, 1] };
        let padding = if transposed {
            (kernel - stride) / 2
        } else {
            (kernel * dilation - dilation) / 2
        };
        Ok(Self {
            weight: source_weight.transpose_axes(&axes)?,
            bias: get(weights, &format!("{prefix}.bias"), &[output])?,
            stride,
            padding,
            dilation,
            transposed,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array> {
        if input.dtype() == Dtype::Float16 && !self.transposed {
            // Half Conv1d: official MPS semantics are F32 accumulation with the
            // bias inside and one rounding to Half (same-input 99.7-99.9% exact
            // vs ~73% for MLX Half conv). Transposed upsampling convolutions
            // match MPS with the Half kernel below (99.9% vs 69% for F32).
            let x = input.as_dtype(Dtype::Float32)?;
            let w = self.weight.as_dtype(Dtype::Float32)?;
            let y = conv1d(&x, &w, self.stride, self.padding, self.dilation, 1)?;
            return Ok(y.add(self.bias.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)?);
        }
        let output = if self.transposed {
            conv_transpose1d(
                input,
                &self.weight,
                self.stride,
                self.padding,
                self.dilation,
                0,
                1,
            )?
        } else {
            conv1d(
                input,
                &self.weight,
                self.stride,
                self.padding,
                self.dilation,
                1,
            )?
        };
        Ok(output.add(&self.bias)?)
    }
}

struct ResBlock1 {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

impl ResBlock1 {
    fn load(
        weights: &HashMap<String, Array>,
        index: usize,
        channels: i32,
        kernel: i32,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        for (i, dilation) in [1, 3, 5].into_iter().enumerate() {
            convs1.push(Conv::load(
                weights,
                &format!("resblocks.{index}.convs1.{i}"),
                channels,
                channels,
                kernel,
                1,
                dilation,
                false,
            )?);
            convs2.push(Conv::load(
                weights,
                &format!("resblocks.{index}.convs2.{i}"),
                channels,
                channels,
                kernel,
                1,
                1,
                false,
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, input: &Array) -> Result<Array> {
        let mut x = input.clone();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let y = c1.forward(&leaky_relu(&x, 0.1)?)?;
            let y = c2.forward(&leaky_relu(&y, 0.1)?)?;
            x = x.add(y)?;
        }
        Ok(x)
    }
}

/// Fixed V5 Generator configuration from official `TTS.init_vocoder`.
pub struct Vocoder {
    pre: Conv,
    ups: Vec<Conv>,
    blocks: Vec<ResBlock1>,
    post: Conv,
}

impl Vocoder {
    /// Load exact official PyTorch state-dict keys. Unsupported shapes, missing
    /// weights, and ambiguous materialized/weight-normalized tensors are errors.
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        let mut prefixes = vec!["conv_pre".to_owned(), "conv_post".to_owned()];
        for i in 0..5 {
            prefixes.push(format!("ups.{i}"));
            for j in 0..3 {
                for k in 0..3 {
                    for conv in ["convs1", "convs2"] {
                        prefixes.push(format!("resblocks.{}.{}.{k}", i * 3 + j, conv));
                    }
                }
            }
        }
        let allowed: HashSet<String> = prefixes
            .iter()
            .flat_map(|prefix| {
                ["bias", "weight", "weight_g", "weight_v"]
                    .map(|suffix| format!("{prefix}.{suffix}"))
            })
            .collect();
        if let Some(key) = weights.keys().find(|key| !allowed.contains(*key)) {
            return Err(invalid(format!("unsupported vocoder tensor {key}")));
        }
        let pre = Conv::load(weights, "conv_pre", 100, 512, 7, 1, 1, false)?;
        let mut ups = Vec::with_capacity(5);
        let mut blocks = Vec::with_capacity(15);
        for (i, (stride, kernel)) in [(10, 20), (6, 12), (2, 4), (2, 4), (2, 4)]
            .into_iter()
            .enumerate()
        {
            let input = 512 >> i;
            let output = input / 2;
            ups.push(Conv::load(
                weights,
                &format!("ups.{i}"),
                input,
                output,
                kernel,
                stride,
                1,
                true,
            )?);
            for (j, block_kernel) in [3, 7, 11].into_iter().enumerate() {
                blocks.push(ResBlock1::load(weights, i * 3 + j, output, block_kernel)?);
            }
        }
        let post = Conv::load(weights, "conv_post", 16, 1, 7, 1, 1, false)?;
        Ok(Self {
            pre,
            ups,
            blocks,
            post,
        })
    }

    /// Input: denormalized log-mel [B,100,T]. Output: mono [B,1,480*T].
    pub fn forward(&self, input: &Array) -> Result<Array> {
        if input.ndim() != 3
            || input.shape()[0] <= 0
            || input.shape()[1] != 100
            || input.shape()[2] <= 0
        {
            return Err(invalid(format!(
                "expected vocoder input [B,100,T] with positive B and T, got {:?}",
                input.shape()
            )));
        }
        let mut x = self.pre.forward(&input.transpose_axes(&[0, 2, 1])?)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&leaky_relu(&x, 0.1)?)?;
            let a = self.blocks[i * 3].forward(&x)?;
            let b = self.blocks[i * 3 + 1].forward(&x)?;
            let c = self.blocks[i * 3 + 2].forward(&x)?;
            x = a
                .add(b)?
                .add(c)?
                .divide(Array::from_f32(3.0).as_dtype(x.dtype())?)?;
        }
        // PyTorch F.leaky_relu without an explicit slope defaults to 0.01 here.
        x = self.post.forward(&leaky_relu(&x, 0.01)?)?;
        Ok(tanh(x)?.transpose_axes(&[0, 2, 1])?)
    }
}
