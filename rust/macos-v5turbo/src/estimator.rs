//! Full GPT-SoVITS v5 DiT estimator, evaluation mode, CFG-positive path.
//! Inputs and output use CFM conventions; conditioning features are [B,T,512].
use std::{collections::HashMap, error::Error, io, sync::Arc};

use crate::dit::DiTBlock;
use mlx_rs::{
    fast::layer_norm,
    nn::{gelu, mish, silu},
    ops::{concatenate, conv1d, indexing::IndexOp, select},
    Array, Dtype,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Diagnostic only: tracing forces evaluation and must be compared against an
// untraced run. With the variable unset, no extra MLX operation is performed.
fn trace_estimator(name: &str, value: &Array) -> Result<()> {
    if let Some(dir) = std::env::var_os("ANIFLIVE_ESTIMATOR_TRACE_DIR") {
        value.eval()?;
        let cpu = mlx_rs::Stream::try_default_on_device(&mlx_rs::Device::cpu())?;
        mlx_rs::with_stream(&cpu, || {
            value.save_numpy(std::path::Path::new(&dir).join(format!("estimator-stage-{name}.npy")))
        })?;
    }
    Ok(())
}
fn source_mish(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() == Dtype::Float16 {
        // Official MPS Half nn.Mish follows the MPSGraph op order
        // exp -> +1 -> log -> tanh -> *x in Float32 opmath, with tanh
        // evaluated as (e^2y-1)/(e^2y+1). Same-input capture: 99.995% exact
        // versus 82% for MLX mish. y is clamped at 20 (tanh is exactly 1.0
        // in Float32 well before that) only to avoid inf/inf.
        let f = x.as_dtype(Dtype::Float32)?;
        let y = mlx_rs::ops::log(&mlx_rs::ops::exp(&f)?.add(Array::from_f32(1.0))?)?;
        let e = mlx_rs::ops::exp(
            &mlx_rs::ops::minimum(&y, Array::from_f32(20.0))?.multiply(Array::from_f32(2.0))?,
        )?;
        let t = e
            .subtract(Array::from_f32(1.0))?
            .divide(e.add(Array::from_f32(1.0))?)?;
        Ok(f.multiply(t)?.as_dtype(Dtype::Float16)?)
    } else {
        Ok(mish(x)?)
    }
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
        .ok_or_else(|| invalid(format!("missing estimator tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "estimator tensor {key}: expected {shape:?}, got {:?}",
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
    fn load(weights: &HashMap<String, Array>, key: &str, input: i32, output: i32) -> Result<Self> {
        Ok(Self {
            weight_t: get(weights, &format!("{key}.weight"), &[output, input])?
                .transpose_axes(&[1, 0])?,
            bias: get(weights, &format!("{key}.bias"), &[output])?,
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
struct Conv {
    weight: Array,
    bias: Array,
    pad: i32,
    groups: i32,
}
impl Conv {
    fn load(
        weights: &HashMap<String, Array>,
        key: &str,
        input: i32,
        output: i32,
        kernel: i32,
        groups: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight: get(
                weights,
                &format!("{key}.weight"),
                &[output, input / groups, kernel],
            )?
            .transpose_axes(&[0, 2, 1])?,
            bias: get(weights, &format!("{key}.bias"), &[output])?,
            pad: (kernel - 1) / 2,
            groups,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == Dtype::Float16 {
            return Ok(conv1d(
                x.as_dtype(Dtype::Float32)?,
                self.weight.as_dtype(Dtype::Float32)?,
                1,
                self.pad,
                1,
                self.groups,
            )?
            .add(self.bias.as_dtype(Dtype::Float32)?)?
            .as_dtype(Dtype::Float16)?);
        }
        Ok(conv1d(x, &self.weight, 1, self.pad, 1, self.groups)?.add(&self.bias)?)
    }
}
struct TextBlock {
    dw: Conv,
    norm_weight: Array,
    norm_bias: Array,
    up: Linear,
    down: Linear,
    gamma: Array,
    beta: Array,
}
impl TextBlock {
    fn load(weights: &HashMap<String, Array>, index: usize) -> Result<Self> {
        let p = format!("cfm.estimator.text_embed.text_blocks.{index}");
        Ok(Self {
            dw: Conv::load(weights, &format!("{p}.dwconv"), 512, 512, 7, 512)?,
            norm_weight: get(weights, &format!("{p}.norm.weight"), &[512])?,
            norm_bias: get(weights, &format!("{p}.norm.bias"), &[512])?,
            up: Linear::load(weights, &format!("{p}.pwconv1"), 512, 1024)?,
            down: Linear::load(weights, &format!("{p}.pwconv2"), 1024, 512)?,
            gamma: get(weights, &format!("{p}.grn.gamma"), &[1, 1, 1024])?,
            beta: get(weights, &format!("{p}.grn.beta"), &[1, 1, 1024])?,
        })
    }
    fn forward(&self, input: &Array) -> Result<Array> {
        let x = self.dw.forward(input)?;
        let x = if x.dtype() == Dtype::Float16 {
            crate::source_half_attention::layer_norm(&x, &self.norm_weight, &self.norm_bias, 1e-6)?
        } else {
            layer_norm(x, Some(&self.norm_weight), Some(&self.norm_bias), 1e-6)?
        };
        let up = self.up.forward(&x)?;
        // Torch MPS constructs the exact-GELU coefficient in input dtype.
        // The F16 coefficient is 0.70703125; round only the result to half.
        let x = if up.dtype() == Dtype::Float16 {
            let value = up.as_dtype(Dtype::Float32)?;
            mlx_rs::ops::erf(value.multiply(Array::from_f32(0.70703125))?)?
                .add(Array::from_f32(1.0))?
                .multiply(Array::from_f32(0.5))?
                .multiply(&value)?
                .as_dtype(Dtype::Float16)?
        } else {
            gelu(up)?
        };
        let gx = if x.dtype() == Dtype::Float16 {
            x.as_dtype(Dtype::Float32)?
                .square()?
                .sum_axes(&[1], true)?
                .sqrt()?
                .as_dtype(Dtype::Float16)?
        } else {
            x.square()?.sum_axes(&[1], true)?.sqrt()?
        };
        let mean = if gx.dtype() == Dtype::Float16 {
            gx.as_dtype(Dtype::Float32)?
                .mean_axes(&[-1], true)?
                .as_dtype(Dtype::Float16)?
        } else {
            gx.mean_axes(&[-1], true)?
        };
        let nx = gx.divide(mean.add(Array::from_f32(1e-6).as_dtype(gx.dtype())?)?)?;
        let x = self
            .gamma
            .multiply(x.multiply(nx)?)?
            .add(&self.beta)?
            .add(&x)?;
        Ok(input.add(self.down.forward(&x)?)?)
    }
}

/// Request-scoped FP32 text/rotary values, tied to their creating estimator.
pub struct PreparedContext {
    owner: Arc<()>,
    frames: i32,
    text: Array,
    cos: Array,
    sin: Array,
}
struct PreparedStep {
    block_modulation: Vec<Array>,
    final_modulation: Array,
}
/// Four fixed FP32 inference times, reusable with the creating estimator.
pub struct PreparedSchedule {
    owner: Arc<()>,
    steps: Vec<PreparedStep>,
}

/// Official 1024-dimensional, 22-block V5 DiT with four ConvNeXt text blocks.
pub struct Estimator {
    owner: Arc<()>,
    time1: Linear,
    time2: Linear,
    text_blocks: Vec<TextBlock>,
    input: Linear,
    pos1: Conv,
    pos2: Conv,
    blocks: Vec<DiTBlock>,
    final_mod: Linear,
    output: Linear,
    rotary_inv_freq: Array,
}
impl Estimator {
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        let p = "cfm.estimator";
        let text_blocks = (0..4)
            .map(|i| TextBlock::load(weights, i))
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..22)
            .map(|i| DiTBlock::load(weights, &format!("{p}.transformer_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owner: Arc::new(()),
            time1: Linear::load(weights, &format!("{p}.time_embed.time_mlp.0"), 256, 1024)?,
            time2: Linear::load(weights, &format!("{p}.time_embed.time_mlp.2"), 1024, 1024)?,
            text_blocks,
            input: Linear::load(weights, &format!("{p}.input_embed.proj"), 712, 1024)?,
            pos1: Conv::load(
                weights,
                &format!("{p}.input_embed.conv_pos_embed.conv1d.0"),
                1024,
                1024,
                31,
                16,
            )?,
            pos2: Conv::load(
                weights,
                &format!("{p}.input_embed.conv_pos_embed.conv1d.2"),
                1024,
                1024,
                31,
                16,
            )?,
            blocks,
            final_mod: Linear::load(weights, &format!("{p}.norm_out.linear"), 1024, 2048)?,
            output: Linear::load(weights, &format!("{p}.proj_out"), 1024, 100)?,
            rotary_inv_freq: get(weights, &format!("{p}.rotary_embed.inv_freq"), &[32])?,
        })
    }

    /// Explicit research opt-in. Pack once, verify every source bit before
    /// installing all 22 immutable blocks; source checkpoint remains intact.
    pub fn enable_packed_qkv(&mut self) -> Result<()> {
        self.enable_qkv_layout(false)
    }
    /// Explicit research opt-in to three batched source-layout projections.
    pub fn enable_batched_qkv(&mut self) -> Result<()> {
        self.enable_qkv_layout(true)
    }
    fn enable_qkv_layout(&mut self, batched: bool) -> Result<()> {
        let count = self.packed_qkv_blocks();
        if count == 22 && self.blocks.len() == 22 {
            if self.batched_qkv_enabled() == batched {
                return Ok(());
            }
            return Err(invalid("QKV projection layout is already installed"));
        }
        if count != 0 || self.blocks.len() != 22 || self.rotary_inv_freq.dtype() != Dtype::Float32 {
            return Err(invalid(
                "packed QKV requires a complete FP32 22-block estimator",
            ));
        }
        let candidates = self
            .blocks
            .iter()
            .map(|b| b.prepare_packed_qkv(batched))
            .collect::<Result<Vec<_>>>()?;
        mlx_rs::transforms::eval(
            candidates
                .iter()
                .flat_map(|(w, b, c)| std::iter::once(w).chain(std::iter::once(b)).chain(c.iter())),
        )?;
        for (index, (_, _, checks)) in candidates.iter().enumerate() {
            let mut valid = checks.len() == 6;
            for check in checks {
                valid &= check.try_item_exact::<bool>()?;
            }
            if !valid {
                return Err(invalid(format!(
                    "packed QKV changed source bits in block {index}"
                )));
            }
        }
        for (block, (weight, bias, _)) in self.blocks.iter_mut().zip(candidates) {
            block.install_packed_qkv(weight, bias, batched);
        }
        Ok(())
    }
    pub fn packed_qkv_blocks(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.packed_qkv_enabled())
            .count()
    }
    pub fn packed_qkv_enabled(&self) -> bool {
        self.blocks.len() == 22 && self.packed_qkv_blocks() == 22
    }
    pub fn batched_qkv_enabled(&self) -> bool {
        self.blocks.len() == 22 && self.blocks.iter().all(|b| b.batched_qkv_enabled())
    }
    /// Standalone logical V projection view, for layout verification only.
    pub fn diagnostic_block0_batched_v(&self, norm: &Array) -> Result<Array> {
        self.blocks
            .first()
            .ok_or_else(|| invalid("missing DiT block0"))?
            .diagnostic_batched_v(norm)
    }

    /// Precompute the fixed V5 times, without batching or changing any linear operation.
    pub fn prepare_schedule(&self) -> Result<PreparedSchedule> {
        if self.rotary_inv_freq.dtype() != Dtype::Float32 {
            return Err(invalid("prepared schedule supports FP32 estimators only"));
        }
        let time_freqs: Vec<f32> = (0..128)
            .map(|i| (-(10000f32.ln()) * i as f32 / 127.0).exp())
            .collect();
        let mut steps = Vec::with_capacity(4);
        for step in 0..4 {
            let time = Array::from_slice(&[step as f32 * 0.25], &[1]);
            let time_angles = time
                .as_dtype(Dtype::Float32)?
                .expand_dims(1)?
                .multiply(Array::from_f32(1000.0))?
                .multiply(Array::from_slice(&time_freqs, &[1, 128]))?;
            let time_hidden = concatenate(&[time_angles.sin()?, time_angles.cos()?], -1)?
                .as_dtype(time.dtype())?;
            let time_emb = self
                .time2
                .forward(&source_silu(self.time1.forward(&time_hidden)?)?)?;
            let activated = source_silu(&time_emb)?;
            let block_modulation = self
                .blocks
                .iter()
                .map(|block| block.prepare_modulation(&activated))
                .collect::<Result<Vec<_>>>()?;
            let final_modulation = self.final_mod.forward(&activated)?;
            if time_emb.dtype() != Dtype::Float32
                || final_modulation.dtype() != Dtype::Float32
                || block_modulation
                    .iter()
                    .any(|value| value.dtype() != Dtype::Float32)
            {
                return Err(invalid("prepared schedule requires FP32 estimator weights"));
            }
            steps.push(PreparedStep {
                block_modulation,
                final_modulation,
            });
        }
        // One evaluation for all outputs; no per-block synchronization.
        mlx_rs::transforms::eval(steps.iter().flat_map(|step| {
            step.block_modulation
                .iter()
                .chain(std::iter::once(&step.final_modulation))
        }))?;
        Ok(PreparedSchedule {
            owner: self.owner.clone(),
            steps,
        })
    }

    /// Read-only block-zero pre-QKV normalization for fixed scheduled t=0.
    pub fn diagnostic_block0_qkv_norm(
        &self,
        trunk: &Array,
        schedule: &PreparedSchedule,
    ) -> Result<Array> {
        if !Arc::ptr_eq(&self.owner, &schedule.owner)
            || schedule.steps.len() != 4
            || schedule.steps[0].block_modulation.len() != self.blocks.len()
            || self.blocks.is_empty()
        {
            return Err(invalid("invalid diagnostic QKV schedule"));
        }
        self.blocks[0].diagnostic_qkv_norm(trunk, &schedule.steps[0].block_modulation[0])
    }

    /// Read-only original block-zero Q/K/V projection weights and biases.
    pub fn diagnostic_block0_qkv_weights(&self) -> Result<Vec<Array>> {
        let block = self
            .blocks
            .first()
            .ok_or_else(|| invalid("missing DiT block0"))?;
        Ok(block.diagnostic_qkv_weights())
    }

    /// Read-only block-zero t0 comparisons using the actual prepared rotary
    /// arrays and the exact production forward_prepared control.
    pub fn diagnostic_block0_forward_prepared(
        &self,
        trunk: &Array,
        context: &PreparedContext,
        schedule: &PreparedSchedule,
    ) -> Result<Array> {
        if !Arc::ptr_eq(&self.owner, &context.owner)
            || !Arc::ptr_eq(&self.owner, &schedule.owner)
            || schedule.steps.len() != 4
            || schedule.steps[0].block_modulation.len() != self.blocks.len()
            || self.blocks.is_empty()
            || trunk.shape() != [1, context.frames, 1024]
        {
            return Err(invalid("invalid diagnostic prepared block0 inputs"));
        }
        self.blocks[0].forward_prepared(
            trunk,
            &schedule.steps[0].block_modulation[0],
            None,
            &context.cos,
            &context.sin,
        )
    }

    /// The paired diagnostic retains all block math except the packed QKV.
    pub fn diagnostic_block0_forward_packed_qkv(
        &self,
        trunk: &Array,
        context: &PreparedContext,
        schedule: &PreparedSchedule,
        packed_weight: &Array,
        packed_bias: &Array,
    ) -> Result<Array> {
        if !Arc::ptr_eq(&self.owner, &context.owner)
            || !Arc::ptr_eq(&self.owner, &schedule.owner)
            || schedule.steps.len() != 4
            || schedule.steps[0].block_modulation.len() != self.blocks.len()
            || self.blocks.is_empty()
            || trunk.shape() != [1, context.frames, 1024]
        {
            return Err(invalid("invalid diagnostic packed block0 inputs"));
        }
        self.blocks[0].diagnostic_forward_packed_qkv(
            trunk,
            &schedule.steps[0].block_modulation[0],
            &context.cos,
            &context.sin,
            packed_weight,
            packed_bias,
        )
    }

    /// Compute and materialize text embedding and rotary values once per request.
    pub fn prepare_context(&self, mu: &Array) -> Result<PreparedContext> {
        let context = self.prepare_context_unmaterialized(mu)?;
        mlx_rs::transforms::eval([&context.text, &context.cos, &context.sin])?;
        Ok(context)
    }

    /// Construct the same request context without evaluation for static-shape tracing.
    pub fn prepare_context_unmaterialized(&self, mu: &Array) -> Result<PreparedContext> {
        if mu.ndim() != 3
            || mu.shape()[0] != 1
            || mu.shape()[1] <= 0
            || mu.shape()[2] != 512
            || mu.dtype() != Dtype::Float32
            || self.rotary_inv_freq.dtype() != Dtype::Float32
        {
            return Err(invalid(
                "prepared context requires FP32 [1,T,512] and FP32 estimator",
            ));
        }
        let n = mu.shape()[1];
        let text_freqs: Vec<f32> = (0..256)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 512.0))
            .collect();
        let positions: Vec<f32> = (0..n).map(|i| i.min(4095) as f32).collect();
        let angles = Array::from_slice(&positions, &[n, 1])
            .multiply(Array::from_slice(&text_freqs, &[1, 256]))?;
        let pos = concatenate(&[angles.cos()?, angles.sin()?], -1)?.as_dtype(mu.dtype())?;
        let mut text = mu.add(pos)?;
        for block in &self.text_blocks {
            text = block.forward(&text)?;
        }
        let rope_positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let angle = Array::from_slice(&rope_positions, &[1, n, 1])
            .as_dtype(self.rotary_inv_freq.dtype())?
            .multiply(self.rotary_inv_freq.reshape(&[1, 1, 32])?)?;
        let rope = concatenate(&[angle.expand_dims(3)?, angle.expand_dims(3)?], 3)?
            .reshape(&[1, n, 64])?;
        let cos = rope.cos()?;
        let sin = rope.sin()?;
        if text.dtype() != Dtype::Float32
            || cos.dtype() != Dtype::Float32
            || sin.dtype() != Dtype::Float32
        {
            return Err(invalid("prepared context requires FP32 estimator values"));
        }
        Ok(PreparedContext {
            owner: self.owner.clone(),
            frames: n,
            text,
            cos,
            sin,
        })
    }

    /// Diagnostic-only unmasked input trunk for a selected contiguous suffix.
    /// Call with start=0 for the original full geometry, or include a 30-frame
    /// left halo when testing a suffix of its output. No solver path uses it.
    pub fn diagnostic_prepared_input_trunk(
        &self,
        x: &Array,
        prompt: &Array,
        context: &PreparedContext,
        start: i32,
    ) -> Result<Array> {
        if !Arc::ptr_eq(&self.owner, &context.owner) {
            return Err(invalid("prepared context belongs to another estimator"));
        }
        let n = context.frames;
        if x.dtype() != Dtype::Float32
            || x.shape() != [1, 100, n]
            || prompt.dtype() != Dtype::Float32
            || prompt.shape() != [1, 100, n]
            || context.text.dtype() != Dtype::Float32
            || context.text.shape() != [1, n, 512]
            || start < 0
            || start >= n
        {
            return Err(invalid("invalid diagnostic trunk inputs or suffix"));
        }
        let x_ntc = x.index((.., .., start..n)).transpose_axes(&[0, 2, 1])?;
        let p_ntc = prompt
            .index((.., .., start..n))
            .transpose_axes(&[0, 2, 1])?;
        let text = context.text.index((.., start..n, ..));
        let embedded = self
            .input
            .forward(&concatenate(&[x_ntc, p_ntc, text], 2)?)?;
        let pos_out = source_mish(
            self.pos2
                .forward(&source_mish(self.pos1.forward(&embedded)?)?)?,
        )?;
        Ok(embedded.add(pos_out)?)
    }

    /// Evaluate one fixed step using values prepared for this estimator and request.
    pub fn forward_prepared(
        &self,
        x: &Array,
        prompt: &Array,
        context: &PreparedContext,
        schedule: &PreparedSchedule,
        step: usize,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let packed = self.packed_qkv_blocks();
        if packed != 0 && (packed != 22 || mask.is_some()) {
            return Err(invalid(
                "packed QKV requires all 22 blocks and unmasked prepared forward",
            ));
        }
        if !Arc::ptr_eq(&self.owner, &context.owner) || !Arc::ptr_eq(&self.owner, &schedule.owner) {
            return Err(invalid("prepared values belong to another estimator"));
        }
        let n = context.frames;
        if x.dtype() != Dtype::Float32
            || x.shape() != [1, 100, n]
            || prompt.dtype() != Dtype::Float32
            || prompt.shape() != [1, 100, n]
            || context.text.dtype() != Dtype::Float32
            || context.text.shape() != [1, n, 512]
            || schedule.steps.len() != 4
            || step >= 4
        {
            return Err(invalid(
                "invalid prepared FP32 estimator inputs or schedule step",
            ));
        }
        if let Some(m) = mask {
            if m.dtype() != Dtype::Bool || m.shape() != [1, n] {
                return Err(invalid("invalid prepared estimator mask"));
            }
        }
        let prepared = &schedule.steps[step];
        if prepared.block_modulation.len() != self.blocks.len()
            || prepared.final_modulation.shape() != [1, 2048]
            || prepared.final_modulation.dtype() != Dtype::Float32
        {
            return Err(invalid("invalid prepared modulation schedule"));
        }
        let x_ntc = x.transpose_axes(&[0, 2, 1])?;
        let p_ntc = prompt.transpose_axes(&[0, 2, 1])?;
        let embedded = self
            .input
            .forward(&concatenate(&[x_ntc, p_ntc, context.text.clone()], 2)?)?;
        let mut pos_in = embedded.clone();
        if let Some(m) = mask {
            pos_in = select(
                m.expand_dims(2)?,
                &pos_in,
                Array::from_f32(0.0).as_dtype(embedded.dtype())?,
            )?;
        }
        let pos_out = source_mish(
            self.pos2
                .forward(&source_mish(self.pos1.forward(&pos_in)?)?)?,
        )?;
        let pos_out = if let Some(m) = mask {
            select(
                m.expand_dims(2)?,
                pos_out,
                Array::from_f32(0.0).as_dtype(embedded.dtype())?,
            )?
        } else {
            pos_out
        };
        let mut hidden = embedded.add(pos_out)?;
        for (block, modulation) in self.blocks.iter().zip(&prepared.block_modulation) {
            hidden =
                block.forward_prepared(&hidden, modulation, mask, &context.cos, &context.sin)?;
        }
        let final_cond = prepared.final_modulation.split_equal(2, 1)?;
        let norm = layer_norm(hidden, None, None, 1e-6)?
            .multiply(
                final_cond[0]
                    .expand_dims(1)?
                    .add(Array::from_f32(1.0).as_dtype(x.dtype())?)?,
            )?
            .add(final_cond[1].expand_dims(1)?)?;
        Ok(self.output.forward(&norm)?.transpose_axes(&[0, 2, 1])?)
    }

    /// Read-only diagnostic counterpart of the unmasked FP32 forward path.
    /// Returns the actual graph's pre-block input plus the normal velocity.
    /// This is deliberately separate from the production forward path.
    pub fn diagnostic_forward_with_trunk(
        &self,
        x: &Array,
        prompt: &Array,
        mu: &Array,
        time: &Array,
    ) -> Result<(Array, Array)> {
        if x.ndim() != 3
            || x.shape()[0] != 1
            || x.shape()[1] != 100
            || x.shape()[2] <= 0
            || x.dtype() != Dtype::Float32
            || prompt.shape() != x.shape()
            || prompt.dtype() != Dtype::Float32
            || mu.shape() != [1, x.shape()[2], 512]
            || mu.dtype() != Dtype::Float32
            || time.shape() != [1]
            || time.dtype() != Dtype::Float32
        {
            return Err(invalid(
                "diagnostic trunk requires unmasked FP32 B=1 estimator inputs",
            ));
        }
        let n = x.shape()[2];
        let time_freqs: Vec<f32> = (0..128)
            .map(|i| (-(10000f32.ln()) * i as f32 / 127.0).exp())
            .collect();
        let time_angles = time
            .as_dtype(Dtype::Float32)?
            .expand_dims(1)?
            .multiply(Array::from_f32(1000.0))?
            .multiply(Array::from_slice(&time_freqs, &[1, 128]))?;
        let time_hidden =
            concatenate(&[time_angles.sin()?, time_angles.cos()?], -1)?.as_dtype(time.dtype())?;
        let time_emb = self
            .time2
            .forward(&source_silu(self.time1.forward(&time_hidden)?)?)?;
        let text_freqs: Vec<f32> = (0..256)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 512.0))
            .collect();
        let positions: Vec<f32> = (0..n).map(|i| i.min(4095) as f32).collect();
        let angles = Array::from_slice(&positions, &[n, 1])
            .multiply(Array::from_slice(&text_freqs, &[1, 256]))?;
        let pos = concatenate(&[angles.cos()?, angles.sin()?], -1)?.as_dtype(mu.dtype())?;
        let mut text = mu.add(pos)?;
        for block in &self.text_blocks {
            text = block.forward(&text)?;
        }
        let embedded = self.input.forward(&concatenate(
            &[
                x.transpose_axes(&[0, 2, 1])?,
                prompt.transpose_axes(&[0, 2, 1])?,
                text,
            ],
            2,
        )?)?;
        let pos_out = source_mish(
            self.pos2
                .forward(&source_mish(self.pos1.forward(&embedded)?)?)?,
        )?;
        let trunk = embedded.add(pos_out)?;
        let rope_positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let angle = Array::from_slice(&rope_positions, &[1, n, 1])
            .as_dtype(self.rotary_inv_freq.dtype())?
            .multiply(self.rotary_inv_freq.reshape(&[1, 1, 32])?)?;
        let rope = concatenate(&[angle.expand_dims(3)?, angle.expand_dims(3)?], 3)?
            .reshape(&[1, n, 64])?;
        let mut hidden = trunk.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &time_emb, None, &rope)?;
        }
        let final_cond = self
            .final_mod
            .forward(&source_silu(&time_emb)?)?
            .split_equal(2, 1)?;
        let norm = layer_norm(hidden, None, None, 1e-6)?
            .multiply(
                final_cond[0]
                    .expand_dims(1)?
                    .add(Array::from_f32(1.0).as_dtype(x.dtype())?)?,
            )?
            .add(final_cond[1].expand_dims(1)?)?;
        let velocity = self.output.forward(&norm)?.transpose_axes(&[0, 2, 1])?;
        Ok((trunk, velocity))
    }

    /// Diagnostic only: run each stage on the *source* output of the previous
    /// stage, isolating per-stage local numerical error. `src` maps source
    /// stage names to captured tensors. Production paths never call this.
    pub fn diagnostic_teacher_stages(
        &self,
        src: &HashMap<String, Array>,
        mu: &Array,
        x: &Array,
        prompt: &Array,
    ) -> Result<Vec<(String, Array)>> {
        let s = |k: &str| src.get(k).cloned().ok_or_else(|| invalid(format!("missing source stage {k}")));
        let mut out = Vec::new();
        let time_hidden = s("time-hidden")?;
        out.push((
            "time-embedding".to_owned(),
            self.time2.forward(&source_silu(self.time1.forward(&time_hidden)?)?)?,
        ));
        let mut prev = mu.add(s("text-position")?)?;
        for (i, block) in self.text_blocks.iter().enumerate() {
            out.push((format!("text-block-{i:02}"), block.forward(&prev)?));
            prev = s(&format!("text-block-{i:02}"))?;
        }
        let p_ntc = prompt.transpose_axes(&[0, 2, 1])?;
        let condition = mlx_rs::ops::addmm(
            &self.input.bias,
            concatenate(&[p_ntc, prev], 2)?,
            self.input.weight_t.index((100.., ..)),
            None,
            None,
        )?;
        out.push(("static-condition".to_owned(), condition));
        let noised = x
            .transpose_axes(&[0, 2, 1])?
            .matmul(self.input.weight_t.index((..100, ..)))?;
        out.push(("embedded".to_owned(), noised.add(s("static-condition")?)?));
        let embedded = s("embedded")?;
        out.push((
            "position".to_owned(),
            source_mish(self.pos2.forward(&source_mish(self.pos1.forward(&embedded)?)?)?)?,
        ));
        out.push(("trunk".to_owned(), embedded.add(s("position")?)?));
        let n = x.shape()[2];
        let rope_positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let angle = Array::from_slice(&rope_positions, &[1, n, 1])
            .as_dtype(self.rotary_inv_freq.dtype())?
            .multiply(self.rotary_inv_freq.reshape(&[1, 1, 32])?)?;
        let rope = concatenate(&[angle.expand_dims(3)?, angle.expand_dims(3)?], 3)?
            .reshape(&[1, n, 64])?;
        let time_emb = s("time-embedding")?;
        let mut hidden = s("trunk")?;
        for (i, block) in self.blocks.iter().enumerate() {
            out.push((format!("block-{i:02}"), block.forward(&hidden, &time_emb, None, &rope)?));
            hidden = s(&format!("block-{i:02}"))?;
        }
        let final_cond = self
            .final_mod
            .forward(&source_silu(&time_emb)?)?
            .split_equal(2, 1)?;
        let norm = layer_norm(hidden, None, None, 1e-6)?
            .multiply(
                final_cond[0]
                    .expand_dims(1)?
                    .add(Array::from_f32(1.0).as_dtype(x.dtype())?)?,
            )?
            .add(final_cond[1].expand_dims(1)?)?;
        out.push(("head-norm".to_owned(), norm));
        out.push(("velocity".to_owned(), self.output.forward(&s("head-norm")?)?));
        Ok(out)
    }

    /// `x` and `prompt` [B,100,T], `mu` [B,T,512], `time` [B].
    /// `mask` is optional boolean valid positions [B,T]. When supplied, the
    /// position convolution follows official CFM static-cache mask behavior.
    pub fn forward(
        &self,
        x: &Array,
        prompt: &Array,
        mu: &Array,
        time: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        Ok(self.forward_impl(x, prompt, mu, time, mask, None)?.0)
    }

    /// FP16 four-step CFM: the Half static condition (text blocks + prompt/text
    /// projection) depends only on prompt and mu, exactly like the official
    /// static cache. The first call computes it; later steps reuse the same
    /// array, so velocities are bit-identical to recomputing it every step.
    pub fn forward_cached(
        &self,
        x: &Array,
        prompt: &Array,
        mu: &Array,
        time: &Array,
        condition: &mut Option<Array>,
    ) -> Result<Array> {
        if x.dtype() != Dtype::Float16 {
            return self.forward(x, prompt, mu, time, None);
        }
        let (velocity, computed) = self.forward_impl(x, prompt, mu, time, None, condition.as_ref())?;
        if condition.is_none() {
            *condition = computed;
        }
        Ok(velocity)
    }

    fn forward_impl(
        &self,
        x: &Array,
        prompt: &Array,
        mu: &Array,
        time: &Array,
        mask: Option<&Array>,
        precomputed: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        if self.packed_qkv_blocks() != 0 {
            return Err(invalid(
                "packed QKV supports only unmasked prepared forward",
            ));
        }
        if x.ndim() != 3 || x.shape()[0] <= 0 || x.shape()[1] != 100 || x.shape()[2] <= 0 {
            return Err(invalid(format!(
                "expected x [B,100,T], got {:?}",
                x.shape()
            )));
        }
        let (b, n) = (x.shape()[0], x.shape()[2]);
        if prompt.shape() != [b, 100, n] || mu.shape() != [b, n, 512] || time.shape() != [b] {
            return Err(invalid(format!(
                "incompatible estimator shapes prompt={:?}, mu={:?}, time={:?}",
                prompt.shape(),
                mu.shape(),
                time.shape()
            )));
        }
        if let Some(m) = mask {
            if m.dtype() != Dtype::Bool || m.shape() != [b, n] {
                return Err(invalid(format!(
                    "expected boolean mask [{b},{n}], got {:?} {:?}",
                    m.shape(),
                    m.dtype()
                )));
            }
        }
        // SinusPositionEmbedding(256): sin then cos, 1000*t, denominator 127.
        let time_freqs: Vec<f32> = (0..128)
            .map(|i| (-(10000f32.ln()) * i as f32 / 127.0).exp())
            .collect();
        let time_angles = time
            .as_dtype(Dtype::Float32)?
            .expand_dims(1)?
            .multiply(Array::from_f32(1000.0))?
            .multiply(Array::from_slice(&time_freqs, &[1, 128]))?;
        let time_hidden =
            concatenate(&[time_angles.sin()?, time_angles.cos()?], -1)?.as_dtype(time.dtype())?;
        trace_estimator("time-hidden", &time_hidden)?;
        let time_emb = self
            .time2
            .forward(&source_silu(self.time1.forward(&time_hidden)?)?)?;
        trace_estimator("time-embedding", &time_emb)?;
        // TextEmbedding's nonpersistent precomputed position table clamps at 4095.
        let text_freqs: Vec<f32> = (0..256)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 512.0))
            .collect();
        let positions: Vec<f32> = (0..n).map(|i| i.min(4095) as f32).collect();
        let angles = Array::from_slice(&positions, &[n, 1])
            .multiply(Array::from_slice(&text_freqs, &[1, 256]))?;
        let pos = concatenate(&[angles.cos()?, angles.sin()?], -1)?.as_dtype(mu.dtype())?;
        let x_ntc = x.transpose_axes(&[0, 2, 1])?;
        let p_ntc = prompt.transpose_axes(&[0, 2, 1])?;
        let mut computed_condition = None;
        let embedded = if x.dtype() == Dtype::Float16 {
            // Official v5 default static conditioning uses two half projections
            // and a half add, rather than one concatenated biased projection.
            let condition = if let Some(condition) = precomputed {
                condition.clone()
            } else {
                trace_estimator("text-position", &pos)?;
                let mut text = mu.add(&pos)?;
                for (index, block) in self.text_blocks.iter().enumerate() {
                    text = block.forward(&text)?;
                    if std::env::var_os("ANIFLIVE_ESTIMATOR_TRACE_DIR").is_some() {
                        trace_estimator(&format!("text-block-{index:02}"), &text)?;
                    }
                }
                let condition = mlx_rs::ops::addmm(
                    &self.input.bias,
                    concatenate(&[p_ntc, text], 2)?,
                    self.input.weight_t.index((100.., ..)),
                    None,
                    None,
                )?;
                computed_condition = Some(condition.clone());
                condition
            };
            trace_estimator("static-condition", &condition)?;
            let noised = x_ntc.matmul(self.input.weight_t.index((..100, ..)))?;
            noised.add(condition)?
        } else {
            trace_estimator("text-position", &pos)?;
            let mut text = mu.add(pos)?;
            for (index, block) in self.text_blocks.iter().enumerate() {
                text = block.forward(&text)?;
                if std::env::var_os("ANIFLIVE_ESTIMATOR_TRACE_DIR").is_some() {
                    trace_estimator(&format!("text-block-{index:02}"), &text)?;
                }
            }
            self.input
                .forward(&concatenate(&[x_ntc, p_ntc, text], 2)?)?
        };
        trace_estimator("embedded", &embedded)?;
        let mut pos_in = embedded.clone();
        if let Some(m) = mask {
            pos_in = select(
                m.expand_dims(2)?,
                &pos_in,
                Array::from_f32(0.0).as_dtype(embedded.dtype())?,
            )?;
        }
        let pos_out = source_mish(
            self.pos2
                .forward(&source_mish(self.pos1.forward(&pos_in)?)?)?,
        )?;
        let pos_out = if let Some(m) = mask {
            select(
                m.expand_dims(2)?,
                pos_out,
                Array::from_f32(0.0).as_dtype(embedded.dtype())?,
            )?
        } else {
            pos_out
        };
        trace_estimator("position", &pos_out)?;
        let mut hidden = embedded.add(pos_out)?;
        trace_estimator("trunk", &hidden)?;
        // RotaryEmbedding uses its checkpoint buffer and duplicates each angular
        // frequency for adjacent channel pairs before DiTBlock applies RoPE.
        let rope_positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let angle = Array::from_slice(&rope_positions, &[1, n, 1])
            .as_dtype(self.rotary_inv_freq.dtype())?
            .multiply(self.rotary_inv_freq.reshape(&[1, 1, 32])?)?;
        let rope = concatenate(&[angle.expand_dims(3)?, angle.expand_dims(3)?], 3)?
            .reshape(&[1, n, 64])?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &time_emb, mask, &rope)?;
            if std::env::var_os("ANIFLIVE_ESTIMATOR_TRACE_DIR").is_some() {
                trace_estimator(&format!("block-{index:02}"), &hidden)?;
            }
        }
        let final_cond = self
            .final_mod
            .forward(&source_silu(&time_emb)?)?
            .split_equal(2, 1)?;
        let norm = layer_norm(hidden, None, None, 1e-6)?
            .multiply(
                final_cond[0]
                    .expand_dims(1)?
                    .add(Array::from_f32(1.0).as_dtype(x.dtype())?)?,
            )?
            .add(final_cond[1].expand_dims(1)?)?;
        trace_estimator("head-norm", &norm)?;
        let velocity = self.output.forward(&norm)?;
        trace_estimator("velocity", &velocity)?;
        Ok((velocity.transpose_axes(&[0, 2, 1])?, computed_condition))
    }
}
