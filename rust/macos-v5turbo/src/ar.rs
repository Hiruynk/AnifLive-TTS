//! GPT-SoVITS autoregressive text-to-semantic transformer, eval mode.
//! Returns raw logits and explicit KV state; sampling and EOS policy are separate.
use crate::mixed_gemv::MixedGemv;
use mlx_rs::{
    fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn::relu,
    ops::concatenate,
    Array, Dtype,
};
use std::{cell::Cell, collections::HashMap, error::Error, ffi::c_int, io, rc::Rc, sync::Arc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing AR tensor {key}")))?;
    if value.shape() != shape {
        return Err(invalid(format!(
            "AR tensor {key}: expected {shape:?}, got {:?}",
            value.shape()
        )));
    }
    Ok(value.clone())
}
// Original linear matrices are stored transposed [K,N]. Recreate contiguous
// [N,K] FP16 storage only if every recovered FP32 bit is identical.
fn make_mirror(weight_t: &Array, name: String) -> Result<(String, Array, Array)> {
    if weight_t.ndim() == 2 && weight_t.dtype() == Dtype::Float16 {
        // FP16 model: the Half checkpoint matrix is itself the GEMV storage.
        let mirror = weight_t.transpose_axes(&[1, 0])?.contiguous()?;
        return Ok((name, mirror, Array::from_bool(true)));
    }
    if weight_t.ndim() != 2 || weight_t.dtype() != Dtype::Float32 {
        return Err(invalid(format!("mixed GEMV requires FP32 matrix: {name}")));
    }
    let original = weight_t.transpose_axes(&[1, 0])?.contiguous()?;
    let mirror = original.as_dtype(Dtype::Float16)?.contiguous()?;
    let restored = mirror.as_dtype(Dtype::Float32)?;
    let valid = original
        .view_dtype(Dtype::Uint32)?
        .eq(restored.view_dtype(Dtype::Uint32)?)?
        .all(false)?;
    Ok((name, mirror, valid))
}
#[derive(Clone)]
struct MixedProjection {
    kernel: Rc<MixedGemv>,
    weight: Array,
    fused_epilogue: bool,
}
#[derive(Clone)]
struct Linear {
    weight_t: Array,
    bias: Array,
    mixed: Option<MixedProjection>,
}
impl Linear {
    fn load(
        weights: &HashMap<String, Array>,
        weight_key: &str,
        bias_key: &str,
        input: i32,
        output: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight_t: get(weights, weight_key, &[output, input])?.transpose_axes(&[1, 0])?,
            bias: get(weights, bias_key, &[output])?,
            mixed: None,
        })
    }
    fn named(
        weights: &HashMap<String, Array>,
        prefix: &str,
        input: i32,
        output: i32,
    ) -> Result<Self> {
        Self::load(
            weights,
            &format!("{prefix}.weight"),
            &format!("{prefix}.bias"),
            input,
            output,
        )
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == Dtype::Float16 {
            if let Some(mixed) = &self.mixed {
                if mixed.fused_epilogue
                    && x.ndim() > 0
                    && x.size() == *x.shape().last().unwrap() as usize
                {
                    // Half single-token GEMV: F32 accumulation, bias added in
                    // F32 in-kernel, one rounding to Half (official MPS M=1).
                    return Ok(mixed.kernel.apply_with_bias(&mixed.weight, x, &self.bias, false)?);
                }
            }
            return Ok(mlx_rs::ops::addmm(
                &self.bias,
                x,
                &self.weight_t,
                None,
                None,
            )?);
        }
        let product = if let Some(mixed) = &self.mixed {
            if x.dtype() == Dtype::Float32
                && x.ndim() > 0
                && x.size() == *x.shape().last().unwrap() as usize
            {
                if mixed.fused_epilogue {
                    return Ok(mixed.kernel.apply_with_bias(
                        &mixed.weight,
                        x,
                        &self.bias,
                        false,
                    )?);
                }
                mixed.kernel.apply(&mixed.weight, x)?
            } else {
                x.matmul(&self.weight_t)?
            }
        } else {
            x.matmul(&self.weight_t)?
        };
        Ok(product.add(&self.bias)?)
    }

    fn forward_relu(&self, x: &Array) -> Result<Array> {
        if let Some(mixed) = &self.mixed {
            if mixed.fused_epilogue
                && x.dtype() == Dtype::Float16
                && x.ndim() > 0
                && x.size() == *x.shape().last().unwrap() as usize
            {
                return Ok(mixed.kernel.apply_with_bias(&mixed.weight, x, &self.bias, true)?);
            }
            if mixed.fused_epilogue
                && x.dtype() == Dtype::Float32
                && x.ndim() > 0
                && x.size() == *x.shape().last().unwrap() as usize
            {
                return Ok(mixed
                    .kernel
                    .apply_with_bias(&mixed.weight, x, &self.bias, true)?);
            }
        }
        Ok(relu(self.forward(x)?)?)
    }
}
#[derive(Clone)]
struct Norm {
    weight: Array,
    bias: Array,
    residual_fused: Option<Rc<crate::residual_layer_norm::ResidualLayerNorm>>,
}
impl Norm {
    fn load(weights: &HashMap<String, Array>, prefix: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), &[dim])?,
            bias: get(weights, &format!("{prefix}.bias"), &[dim])?,
            residual_fused: None,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == mlx_rs::Dtype::Float16 {
            return crate::source_half_attention::layer_norm(x, &self.weight, &self.bias, 1e-5);
        }
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-5)?)
    }
    fn forward_residual(&self, lhs: &Array, rhs: &Array) -> Result<Array> {
        if let Some(kernel) = &self.residual_fused {
            if lhs.dtype() == Dtype::Float16
                && lhs.shape() == [1, 1, 512]
                && rhs.dtype() == Dtype::Float16
                && rhs.shape() == [1, 1, 512]
            {
                // Half I/O kernel: Half residual add (official order), float
                // normalization, one rounding to Half.
                return Ok(kernel.forward(lhs, rhs, &self.weight, &self.bias)?);
            }
            if lhs.dtype() == Dtype::Float32
                && lhs.shape() == [1, 1, 512]
                && rhs.dtype() == Dtype::Float32
                && rhs.shape() == [1, 1, 512]
            {
                return Ok(kernel.forward(lhs, rhs, &self.weight, &self.bias)?);
            }
        }
        self.forward(&lhs.add(rhs)?)
    }
}
#[derive(Clone)]
struct Block {
    qkv: Linear,
    out: Linear,
    ff1: Linear,
    ff2: Linear,
    norm1: Norm,
    norm2: Norm,
}
#[derive(Clone)]
struct Cache {
    k: Array,
    v: Array,
}
impl Block {
    fn load(weights: &HashMap<String, Array>, index: usize, dim: i32) -> Result<Self> {
        let p = format!("model.h.layers.{index}");
        Ok(Self {
            qkv: Linear::load(
                weights,
                &format!("{p}.self_attn.in_proj_weight"),
                &format!("{p}.self_attn.in_proj_bias"),
                dim,
                dim * 3,
            )?,
            out: Linear::named(weights, &format!("{p}.self_attn.out_proj"), dim, dim)?,
            ff1: Linear::named(weights, &format!("{p}.linear1"), dim, dim * 4)?,
            ff2: Linear::named(weights, &format!("{p}.linear2"), dim * 4, dim)?,
            norm1: Norm::load(weights, &format!("{p}.norm1"), dim)?,
            norm2: Norm::load(weights, &format!("{p}.norm2"), dim)?,
        })
    }
    fn forward(
        &self,
        input: &Array,
        old: Option<&Cache>,
        heads: i32,
        mask: Option<&Array>,
    ) -> Result<(Array, Cache)> {
        let (length, dim) = (input.shape()[1], input.shape()[2]);
        let chunks = self.qkv.forward(input)?.split_equal(3, 2)?;
        // Keep cache in [B,T,D] just as the official T2SBlock does.
        let (k, v) = if let Some(cache) = old {
            (
                concatenate(&[&cache.k, &chunks[1]], 1)?,
                concatenate(&[&cache.v, &chunks[2]], 1)?,
            )
        } else {
            (chunks[1].clone(), chunks[2].clone())
        };
        let total = k.shape()[1];
        let head_dim = dim / heads;
        let q = chunks[0]
            .reshape(&[1, length, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let a = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (head_dim as f32).sqrt(),
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, length, dim])?;
        let h = self.norm1.forward_residual(input, &self.out.forward(&a)?)?;
        let output = self
            .norm2
            .forward_residual(&h, &self.ff2.forward(&self.ff1.forward_relu(&h)?)?)?;
        Ok((output, Cache { k, v }))
    }
}

// Standalone diagnostic owns ENABLED as its process default. No other thread
// may change compile mode or schedule model work while a region call runs.
unsafe extern "C" {
    fn mlx_set_compile_mode(mode: c_int) -> c_int;
}
type RegionGraph = dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception>;

pub struct Block0Regions {
    qkv: Box<RegionGraph>,
    post: Box<RegionGraph>,
    qkv_traces: Rc<Cell<usize>>,
    post_traces: Rc<Cell<usize>>,
    mode: c_int,
}

impl Block0Regions {
    pub fn trace_counts(&self) -> (usize, usize) {
        (self.qkv_traces.get(), self.post_traces.get())
    }

    fn call(graph: &mut RegionGraph, args: &[Array], mode: c_int) -> Result<Vec<Array>> {
        if unsafe { mlx_set_compile_mode(mode) } != 0 {
            if unsafe { mlx_set_compile_mode(3) } != 0 {
                return Err(invalid("failed to set region mode and restore ENABLED"));
            }
            return Err(invalid("failed to set region compile mode"));
        }
        let result = graph(args);
        if unsafe { mlx_set_compile_mode(3) } != 0 {
            return Err(invalid(format!(
                "failed to restore ENABLED after region call: {}",
                match result {
                    Ok(_) => "graph succeeded".to_owned(),
                    Err(e) => e.to_string(),
                }
            )));
        }
        Ok(result?)
    }

    /// Uses the same causal single-query attention as Block::forward; only
    /// the fixed-size projection regions are compiled. Inputs are the bundle
    /// produced by diagnostic_block0_inputs, with immutable captured weights.
    pub fn forward(&mut self, inputs: &[Array]) -> Result<Vec<Array>> {
        if inputs.len() != 15
            || inputs[0].shape() != [1, 1, 512]
            || inputs[1].ndim() != 3
            || inputs[1].shape()[0] != 1
            || inputs[1].shape()[1] <= 0
            || inputs[1].shape()[2] != 512
            || inputs[2].shape() != inputs[1].shape()
            || inputs[..3].iter().any(|x| x.dtype() != Dtype::Float32)
        {
            return Err(invalid("invalid block0 region inputs"));
        }
        let output = Self::call(&mut self.qkv, &[inputs[0].clone()], self.mode)?;
        if output.len() != 1
            || output[0].shape() != [1, 1, 1536]
            || output[0].dtype() != Dtype::Float32
        {
            return Err(invalid("invalid compiled qkv output"));
        }
        let chunks = output[0].split_equal(3, 2)?;
        let k = concatenate(&[&inputs[1], &chunks[1]], 1)?;
        let v = concatenate(&[&inputs[2], &chunks[2]], 1)?;
        let total = k.shape()[1];
        let q = chunks[0]
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let a = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let post = Self::call(&mut self.post, &[inputs[0].clone(), a], self.mode)?;
        if post.len() != 1 || post[0].shape() != [1, 1, 512] || post[0].dtype() != Dtype::Float32 {
            return Err(invalid("invalid compiled postattention output"));
        }
        Ok(vec![post[0].clone(), k, v])
    }
}
/// State owned by one sequence, including every layer's unprojected-head K/V.
#[derive(Clone)]
pub struct ArState {
    owner: Arc<()>,
    caches: Vec<Cache>,
    audio_len: i32,
}
impl ArState {
    pub fn audio_len(&self) -> i32 {
        self.audio_len
    }

    /// Diagnostic-only output roots for scheduling every K/V sibling.
    pub fn diagnostic_cache_arrays(&self) -> Vec<Array> {
        self.caches
            .iter()
            .flat_map(|cache| [cache.k.clone(), cache.v.clone()])
            .collect()
    }
}
/// Original checkpoint GPT; head count must come from checkpoint config.
#[derive(Clone)]
pub struct ArModel {
    owner: Arc<()>,
    dim: i32,
    heads: i32,
    phoneme_vocab: i32,
    semantic_vocab: i32,
    text_embedding: Array,
    audio_embedding: Array,
    text_alpha: Array,
    audio_alpha: Array,
    bert: Linear,
    blocks: Vec<Block>,
    predict_t: Array,
    mixed_predict: Option<MixedProjection>,
}
impl ArModel {
    /// Supports the official post-norm architecture. The caller supplies the
    /// head count recorded in checkpoint config; tensor shapes validate the rest.
    pub fn load(weights: &HashMap<String, Array>, heads: i32) -> Result<Self> {
        let text = weights
            .get("model.ar_text_embedding.word_embeddings.weight")
            .ok_or_else(|| invalid("missing AR text embedding"))?;
        let audio = weights
            .get("model.ar_audio_embedding.word_embeddings.weight")
            .ok_or_else(|| invalid("missing AR audio embedding"))?;
        if text.ndim() != 2 || audio.ndim() != 2 {
            return Err(invalid("AR embeddings must be rank two"));
        }
        let (phoneme_vocab, dim, semantic_vocab) =
            (text.shape()[0], text.shape()[1], audio.shape()[0]);
        if dim <= 0
            || dim % 2 != 0
            || heads <= 0
            || dim % heads != 0
            || phoneme_vocab <= 0
            || semantic_vocab <= 0
            || audio.shape()[1] != dim
        {
            return Err(invalid(
                "unsupported AR dimension, vocabulary, or head count",
            ));
        }
        let layer_count = weights
            .keys()
            .filter(|k| {
                k.starts_with("model.h.layers.") && k.ends_with(".self_attn.in_proj_weight")
            })
            .count();
        if layer_count == 0 {
            return Err(invalid("no AR transformer layers"));
        }
        let blocks = (0..layer_count)
            .map(|i| Block::load(weights, i, dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owner: Arc::new(()),
            dim,
            heads,
            phoneme_vocab,
            semantic_vocab,
            text_embedding: text.clone(),
            audio_embedding: audio.clone(),
            text_alpha: get(weights, "model.ar_text_position.alpha", &[1])?,
            audio_alpha: get(weights, "model.ar_audio_position.alpha", &[1])?,
            bert: Linear::named(weights, "model.bert_proj", 1024, dim)?,
            blocks,
            predict_t: get(
                weights,
                "model.ar_predict_layer.weight",
                &[semantic_vocab, dim],
            )?
            .transpose_axes(&[1, 0])?,
            mixed_predict: None,
        })
    }

    /// Opt in to exact FP16 weight storage with FP32 single-row computation.
    /// All coefficient mirrors are verified bit-for-bit before installation.
    pub fn enable_mixed_gemv(&mut self) -> Result<()> {
        if self.mixed_predict.is_some() {
            return Ok(());
        }
        self.enable_mixed_gemv_with_rows(4)
    }

    /// Select the verified row tile before installing lossless mirrors.
    pub fn enable_mixed_gemv_with_rows(&mut self, rows: i32) -> Result<()> {
        if rows != 1 && rows != 4 {
            return Err(invalid("mixed GEMV rows must be 1 or 4"));
        }
        if let Some(projection) = &self.mixed_predict {
            if projection.kernel.rows_per_simdgroup() != rows {
                return Err(invalid("mixed GEMV row tile is already installed"));
            }
        }
        if self.mixed_predict.is_some() {
            return Ok(());
        }
        if self.dim != 512
            || self.heads != 16
            || self.blocks.len() != 24
            || self.semantic_vocab != 1025
        {
            return Err(invalid(
                "mixed GEMV requires the 24-layer 512-wide FP32 AR model",
            ));
        }
        let kernel = Rc::new(MixedGemv::with_rows_per_simdgroup(rows)?);
        let mut mirrors = Vec::with_capacity(24 * 4 + 1);
        for (index, block) in self.blocks.iter().enumerate() {
            for (name, layer) in [
                ("qkv", &block.qkv),
                ("out", &block.out),
                ("ff1", &block.ff1),
                ("ff2", &block.ff2),
            ] {
                mirrors.push(make_mirror(
                    &layer.weight_t,
                    format!("layer {index} {name}"),
                )?);
            }
        }
        mirrors.push(make_mirror(&self.predict_t, "predictor".to_owned())?);
        // Evaluate every mirror and comparison in one synchronization before
        // any model field changes, so errors never partially enable the path.
        mlx_rs::transforms::eval(
            mirrors
                .iter()
                .flat_map(|(_, weight, valid)| [weight, valid]),
        )?;
        for (name, _, valid) in &mirrors {
            if !valid.item_exact::<bool>() {
                return Err(invalid(format!(
                    "mixed GEMV FP16 round-trip changed {name} coefficients"
                )));
            }
        }
        let mut mirrors = mirrors.into_iter();
        for block in &mut self.blocks {
            for layer in [
                &mut block.qkv,
                &mut block.out,
                &mut block.ff1,
                &mut block.ff2,
            ] {
                let (_, weight, _) = mirrors
                    .next()
                    .ok_or_else(|| invalid("missing mixed GEMV mirror"))?;
                layer.mixed = Some(MixedProjection {
                    kernel: kernel.clone(),
                    weight,
                    fused_epilogue: false,
                });
            }
        }
        let (_, weight, _) = mirrors
            .next()
            .ok_or_else(|| invalid("missing mixed GEMV predictor mirror"))?;
        self.mixed_predict = Some(MixedProjection {
            kernel,
            weight,
            fused_epilogue: false,
        });
        Ok(())
    }

    /// Research-only fused bias/ReLU launch reduction; model coefficients and
    /// F32 arithmetic are retained. Public/runtime defaults remain unchanged.
    pub fn enable_mixed_gemv_epilogue(&mut self) -> Result<()> {
        self.enable_mixed_gemv()?;
        for block in &mut self.blocks {
            for linear in [
                &mut block.qkv,
                &mut block.out,
                &mut block.ff1,
                &mut block.ff2,
            ] {
                linear
                    .mixed
                    .as_mut()
                    .ok_or_else(|| invalid("missing mixed GEMV projection"))?
                    .fused_epilogue = true;
            }
        }
        Ok(())
    }

    pub fn mixed_gemv_enabled(&self) -> bool {
        self.mixed_predict.is_some()
    }

    pub fn mixed_gemv_rows(&self) -> Option<i32> {
        self.mixed_predict
            .as_ref()
            .map(|projection| projection.kernel.rows_per_simdgroup())
    }

    pub fn mixed_gemv_epilogue_enabled(&self) -> bool {
        !self.blocks.is_empty()
            && self.blocks.iter().all(|block| {
                [&block.qkv, &block.out, &block.ff1, &block.ff2]
                    .iter()
                    .all(|linear| {
                        linear
                            .mixed
                            .as_ref()
                            .map(|mixed| mixed.fused_epilogue)
                            .unwrap_or(false)
                    })
            })
    }

    /// Research opt-in for single-query FP32 decode. Prefill and all other
    /// shapes keep the original add + MLX LayerNorm path.
    pub fn enable_residual_norm_fusion(&mut self) -> Result<()> {
        if !matches!(self.audio_embedding.dtype(), Dtype::Float32 | Dtype::Float16)
            || self.dim != 512
            || self.heads != 16
            || self.blocks.len() != 24
            || self.mixed_gemv_rows() != Some(1)
            || !self.mixed_gemv_epilogue_enabled()
        {
            return Err(invalid("residual norm fusion requires FP32/FP16 fused-TM1 AR"));
        }
        if self.residual_norm_fusion_enabled() {
            return Ok(());
        }
        if self
            .blocks
            .iter()
            .any(|b| b.norm1.residual_fused.is_some() || b.norm2.residual_fused.is_some())
        {
            return Err(invalid("inconsistent residual norm fusion configuration"));
        }
        let kernel = Rc::new(crate::residual_layer_norm::ResidualLayerNorm::new()?);
        for block in &mut self.blocks {
            block.norm1.residual_fused = Some(kernel.clone());
            block.norm2.residual_fused = Some(kernel.clone());
        }
        Ok(())
    }
    pub fn residual_norm_fusion_enabled(&self) -> bool {
        !self.blocks.is_empty()
            && self
                .blocks
                .iter()
                .all(|b| b.norm1.residual_fused.is_some() && b.norm2.residual_fused.is_some())
    }

    fn validate_ids(ids: &Array, label: &str, vocab: i32) -> Result<()> {
        if ids.dtype() != Dtype::Int32 && ids.dtype() != Dtype::Int64 {
            return Err(invalid(format!("{label} must be int32 or int64")));
        }
        let min = ids.min(false)?;
        let max = ids.max(false)?;
        let (lo, hi) = (min.try_item_cast::<i64>()?, max.try_item_cast::<i64>()?);
        if lo < 0 || hi >= i64::from(vocab) {
            return Err(invalid(format!("{label} IDs outside [0,{vocab})")));
        }
        Ok(())
    }
    fn position(&self, start: i32, length: i32, dtype: Dtype) -> Result<Array> {
        let freqs: Vec<f32> = (0..self.dim / 2)
            .map(|i| (-(10000f32.ln()) * (2 * i) as f32 / self.dim as f32).exp())
            .collect();
        let end = start
            .checked_add(length)
            .ok_or_else(|| invalid("AR position overflow"))?;
        let positions: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let a = Array::from_slice(&positions, &[length, 1])
            .multiply(Array::from_slice(&freqs, &[1, self.dim / 2]))?;
        let sin = a.sin()?.expand_dims(2)?;
        let cos = a.cos()?.expand_dims(2)?;
        Ok(concatenate(&[sin, cos], 2)?
            .reshape(&[1, length, self.dim])?
            .as_dtype(dtype)?)
    }
    fn logits(&self, hidden: &Array) -> Result<Array> {
        let last = hidden.split_at_indices(&[hidden.shape()[1] - 1], 1)?;
        let product = if let Some(mixed) = &self.mixed_predict {
            if last[1].dtype() == Dtype::Float32 && last[1].size() == self.dim as usize {
                mixed.kernel.apply(&mixed.weight, &last[1])?
            } else if last[1].dtype() == Dtype::Float16 && last[1].size() == self.dim as usize {
                mixed.kernel.apply(&mixed.weight, &last[1])?
            } else {
                last[1].matmul(&self.predict_t)?
            }
        } else {
            last[1].matmul(&self.predict_t)?
        };
        Ok(product.reshape(&[1, self.semantic_vocab])?)
    }
    /// `phones` [1,N], `bert` [1,1024,N], `prompt_semantic` [1,M].
    /// N and M must be positive. Raw logits include EOS; caller applies policy.
    pub fn prefill(
        &self,
        phones: &Array,
        bert: &Array,
        prompt_semantic: &Array,
    ) -> Result<(Array, ArState)> {
        if phones.ndim() != 2
            || phones.shape()[0] != 1
            || phones.shape()[1] <= 0
            || prompt_semantic.ndim() != 2
            || prompt_semantic.shape()[0] != 1
            || prompt_semantic.shape()[1] <= 0
        {
            return Err(invalid(
                "expected nonempty phone and prompt token arrays [1,N] and [1,M]",
            ));
        }
        let (n, m) = (phones.shape()[1], prompt_semantic.shape()[1]);
        if bert.shape() != [1, 1024, n] {
            return Err(invalid(format!(
                "expected bert [1,1024,{n}], got {:?}",
                bert.shape()
            )));
        }
        Self::validate_ids(phones, "phone", self.phoneme_vocab)?;
        Self::validate_ids(prompt_semantic, "semantic", self.semantic_vocab)?;
        let text = self
            .text_embedding
            .take_axis(phones, 0)?
            .add(self.bert.forward(&bert.transpose_axes(&[0, 2, 1])?)?)?;
        let text = text.add(
            self.position(0, n, text.dtype())?
                .multiply(&self.text_alpha)?,
        )?;
        let audio = self.audio_embedding.take_axis(prompt_semantic, 0)?;
        let audio = audio.add(
            self.position(0, m, audio.dtype())?
                .multiply(&self.audio_alpha)?,
        )?;
        let mut hidden = concatenate(&[text, audio], 1)?;
        let total = n
            .checked_add(m)
            .ok_or_else(|| invalid("AR sequence length overflow"))?;
        // MLX boolean attention mask uses true=allowed. Text attends all text;
        // audio attends all text and preceding/current audio, never future audio.
        let mut mask = Vec::with_capacity(total as usize * total as usize);
        for i in 0..total {
            for j in 0..total {
                mask.push(if i < n { j < n } else { j < n || j <= i });
            }
        }
        let mask = Array::from_slice(&mask, &[1, 1, total, total]);
        let mut caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (h, c) = block.forward(&hidden, None, self.heads, Some(&mask))?;
            hidden = h;
            caches.push(c);
        }
        Ok((
            self.logits(&hidden)?,
            ArState {
                owner: self.owner.clone(),
                caches,
                audio_len: m,
            },
        ))
    }
    fn validate_decode_input(&self, token: &Array, state: &ArState) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &state.owner) {
            return Err(invalid("AR state belongs to a different model"));
        }
        if token.shape() != [1, 1] {
            return Err(invalid(format!(
                "expected semantic token [1,1], got {:?}",
                token.shape()
            )));
        }
        if state.caches.len() != self.blocks.len() {
            return Err(invalid("AR cache does not match model layers"));
        }
        Ok(())
    }

    /// Consume one caller-supplied generated token [1,1], then return raw logits.
    /// Public inputs receive full array-based range validation.
    pub fn decode(&self, token: &Array, state: &mut ArState) -> Result<Array> {
        self.validate_decode_input(token, state)?;
        Self::validate_ids(token, "semantic", self.semantic_vocab)?;
        self.decode_inner(token, state)
    }

    /// Internal sampled-token path: `known_id` must be the value already read
    /// from this very token array by the engine. This avoids another GPU range
    /// reduction and preserves the public API's checks for external tokens.
    pub(crate) fn decode_generated(
        &self,
        token: &Array,
        known_id: i32,
        state: &mut ArState,
    ) -> Result<Array> {
        self.validate_decode_input(token, state)?;
        if token.dtype() != Dtype::Int32 && token.dtype() != Dtype::Int64 {
            return Err(invalid("semantic must be int32 or int64"));
        }
        if !(0..self.semantic_vocab).contains(&known_id) {
            return Err(invalid(format!(
                "semantic ID {known_id} outside [0,{})",
                self.semantic_vocab
            )));
        }
        self.decode_inner(token, state)
    }

    /// Diagnostic-only speculative decode from an internally generated GPU
    /// argmax token. Caller must ensure token provenance/range and must discard
    /// the state on failed validation or after an earlier EOS.
    pub fn diagnostic_decode_sampled_gpu(
        &self,
        token: &Array,
        state: &mut ArState,
    ) -> Result<Array> {
        self.validate_decode_input(token, state)?;
        if token.dtype() != Dtype::Int32 {
            return Err(invalid("diagnostic sampled token must be int32"));
        }
        self.decode_inner(token, state)
    }

    /// Diagnostic-only block 0 input/cache and ordered weight tensors for a
    /// single already-selected token; this does not modify decoder state.
    pub fn diagnostic_block0_inputs(&self, token: &Array, state: &ArState) -> Result<Vec<Array>> {
        self.validate_decode_input(token, state)?;
        Self::validate_ids(token, "semantic", self.semantic_vocab)?;
        if self.dim != 512 || self.heads != 16 || self.blocks.is_empty() {
            return Err(invalid(
                "diagnostic requires the 512-wide, 16-head AR model",
            ));
        }
        let audio = self.audio_embedding.take_axis(token, 0)?;
        let hidden = audio.add(
            self.position(state.audio_len, 1, audio.dtype())?
                .multiply(&self.audio_alpha)?,
        )?;
        let cache = &state.caches[0];
        let block = &self.blocks[0];
        Ok(vec![
            hidden,
            cache.k.clone(),
            cache.v.clone(),
            block.qkv.weight_t.clone(),
            block.qkv.bias.clone(),
            block.out.weight_t.clone(),
            block.out.bias.clone(),
            block.ff1.weight_t.clone(),
            block.ff1.bias.clone(),
            block.ff2.weight_t.clone(),
            block.ff2.bias.clone(),
            block.norm1.weight.clone(),
            block.norm1.bias.clone(),
            block.norm2.weight.clone(),
            block.norm2.bias.clone(),
        ])
    }

    /// Run the original block implementation on a diagnostic input bundle.
    pub fn diagnostic_block0_reference(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        if inputs.len() != 15 || self.blocks.is_empty() {
            return Err(invalid("invalid diagnostic block 0 input bundle"));
        }
        let cache = Cache {
            k: inputs[1].clone(),
            v: inputs[2].clone(),
        };
        let (hidden, next) = self.blocks[0].forward(&inputs[0], Some(&cache), self.heads, None)?;
        Ok(vec![hidden, next.k, next.v])
    }

    /// Diagnostic control: original block output/K/V plus its raw QKV.
    pub fn diagnostic_block0_qkv_cache_reference(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        if inputs.len() != 15 || self.blocks.is_empty() || self.heads != 16 || self.dim != 512 {
            return Err(invalid("invalid QKV reference diagnostic model/inputs"));
        }
        let block = &self.blocks[0];
        let qkv = block.qkv.forward(&inputs[0])?;
        let chunks = qkv.split_equal(3, 2)?;
        let k = concatenate(&[&inputs[1], &chunks[1]], 1)?;
        let v = concatenate(&[&inputs[2], &chunks[2]], 1)?;
        let total = k.shape()[1];
        let q = chunks[0]
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let attention = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let h = block
            .norm1
            .forward_residual(&inputs[0], &block.out.forward(&attention)?)?;
        let output = block
            .norm2
            .forward_residual(&h, &block.ff2.forward(&block.ff1.forward_relu(&h)?)?)?;
        Ok(vec![output, k, v, qkv])
    }

    /// Diagnostic candidate only; existing production Block::forward is unchanged.
    pub fn diagnostic_block0_qkv_cache_fused(
        &self,
        inputs: &[Array],
        fused: &crate::qkv_cache_fused::QkvCacheFused,
    ) -> Result<Vec<Array>> {
        if inputs.len() != 15
            || self.blocks.is_empty()
            || self.heads != 16
            || self.dim != 512
            || !self.residual_norm_fusion_enabled()
        {
            return Err(invalid("invalid fused QKV diagnostic model/inputs"));
        }
        let block = &self.blocks[0];
        let mixed = block
            .qkv
            .mixed
            .as_ref()
            .ok_or_else(|| invalid("missing mixed QKV"))?;
        if mixed.kernel.rows_per_simdgroup() != 1 || !mixed.fused_epilogue {
            return Err(invalid("fused QKV requires TM1 and bias epilogue"));
        }
        let parts = fused.forward(
            &mixed.weight,
            &inputs[0],
            &block.qkv.bias,
            &inputs[1],
            &inputs[2],
        )?;
        if parts.len() != 3 {
            return Err(invalid("fused QKV returned wrong output count"));
        }
        let chunks = parts[0].split_equal(3, 2)?;
        let k = &parts[1];
        let v = &parts[2];
        let total = k.shape()[1];
        let q = chunks[0]
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let attention = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let h = block
            .norm1
            .forward_residual(&inputs[0], &block.out.forward(&attention)?)?;
        let output = block
            .norm2
            .forward_residual(&h, &block.ff2.forward(&block.ff1.forward_relu(&h)?)?)?;
        Ok(vec![output, k.clone(), v.clone(), parts[0].clone()])
    }

    /// Diagnostic reconstruction exposing both real residual operands and
    /// affine parameters. Compare output/K/V against block0_reference first.
    pub fn diagnostic_block0_residual_operands(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        if inputs.len() != 15
            || self.blocks.is_empty()
            || inputs[0].shape() != [1, 1, 512]
            || inputs[1].ndim() != 3
            || inputs[1].shape()[0] != 1
            || inputs[1].shape()[2] != 512
            || inputs[2].shape() != inputs[1].shape()
            || self.heads != 16
        {
            return Err(invalid("invalid diagnostic residual operand inputs"));
        }
        let block = &self.blocks[0];
        let chunks = block.qkv.forward(&inputs[0])?.split_equal(3, 2)?;
        let k = concatenate(&[&inputs[1], &chunks[1]], 1)?;
        let v = concatenate(&[&inputs[2], &chunks[2]], 1)?;
        let total = k.shape()[1];
        let q = chunks[0]
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let attention = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let branch1 = block.out.forward(&attention)?;
        let h = block.norm1.forward(&inputs[0].add(&branch1)?)?;
        let branch2 = block.ff2.forward(&block.ff1.forward_relu(&h)?)?;
        let output = block.norm2.forward(&h.add(&branch2)?)?;
        Ok(vec![
            inputs[0].clone(),
            branch1,
            block.norm1.weight.clone(),
            block.norm1.bias.clone(),
            h,
            branch2,
            block.norm2.weight.clone(),
            block.norm2.bias.clone(),
            output,
            k,
            v,
        ])
    }

    /// Diagnostic-only whole block with fused residual norms. Production
    /// The default block path is the control; caller checks outputs against it.
    pub fn diagnostic_block0_fused_norm(
        &self,
        inputs: &[Array],
        norm: &crate::residual_layer_norm::ResidualLayerNorm,
    ) -> Result<Vec<Array>> {
        if inputs.len() != 15
            || self.blocks.is_empty()
            || inputs[0].shape() != [1, 1, 512]
            || inputs[1].ndim() != 3
            || inputs[1].shape()[0] != 1
            || inputs[1].shape()[2] != 512
            || inputs[2].shape() != inputs[1].shape()
            || self.heads != 16
        {
            return Err(invalid("invalid diagnostic fused block inputs"));
        }
        let block = &self.blocks[0];
        let chunks = block.qkv.forward(&inputs[0])?.split_equal(3, 2)?;
        let k = concatenate(&[&inputs[1], &chunks[1]], 1)?;
        let v = concatenate(&[&inputs[2], &chunks[2]], 1)?;
        let total = k.shape()[1];
        let q = chunks[0]
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let vh = v
            .reshape(&[1, total, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let attention = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let branch1 = block.out.forward(&attention)?;
        let h = norm.forward(&inputs[0], &branch1, &block.norm1.weight, &block.norm1.bias)?;
        let branch2 = block.ff2.forward(&block.ff1.forward_relu(&h)?)?;
        let output = norm.forward(&h, &branch2, &block.norm2.weight, &block.norm2.bias)?;
        Ok(vec![output, k, v])
    }

    /// Create fixed-shape block-zero regions in a standalone diagnostic. This
    /// object owns the process compile-mode policy and must stay on its GPU
    /// owner thread; external compile-mode mutation is unsupported.
    pub fn diagnostic_block0_regions(&self, no_fuse: bool) -> Result<Block0Regions> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0")
            || std::env::var_os("MLX_DISABLE_COMPILE").is_some()
            || self.mixed_gemv_rows() != Some(1)
            || !self.mixed_gemv_epilogue_enabled()
            || self.blocks.is_empty()
            || self.heads != 16
            || self.dim != 512
        {
            return Err(invalid(
                "block0 regions require TF32 off, compilation, and fused TM1 FP32 model",
            ));
        }
        if !matches!(
            mlx_rs::Stream::thread_local_or_default()
                .device()?
                .get_type()?,
            mlx_rs::DeviceType::Gpu
        ) {
            return Err(invalid("block0 regions require GPU owner thread"));
        }
        if unsafe { mlx_set_compile_mode(3) } != 0 {
            return Err(invalid(
                "failed to establish diagnostic compile mode ENABLED",
            ));
        }
        let block = &self.blocks[0];
        let qkv = block.qkv.clone();
        let out = block.out.clone();
        let ff1 = block.ff1.clone();
        let ff2 = block.ff2.clone();
        let norm1 = block.norm1.clone();
        let norm2 = block.norm2.clone();
        let qkv_traces = Rc::new(Cell::new(0));
        let qkv_count = qkv_traces.clone();
        let qkv_graph = mlx_rs::transforms::compile::compile(
            move |args: &[Array]| {
                qkv_count.set(qkv_count.get() + 1);
                qkv.forward(&args[0])
                    .map(|v| vec![v])
                    .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
            },
            Some(false),
        );
        let post_traces = Rc::new(Cell::new(0));
        let post_count = post_traces.clone();
        let post_graph = mlx_rs::transforms::compile::compile(
            move |args: &[Array]| {
                post_count.set(post_count.get() + 1);
                let result = (|| -> Result<Array> {
                    let h = norm1.forward(&args[0].add(out.forward(&args[1])?)?)?;
                    norm2.forward(&h.add(ff2.forward(&ff1.forward_relu(&h)?)?)?)
                })();
                result
                    .map(|v| vec![v])
                    .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
            },
            Some(false),
        );
        Ok(Block0Regions {
            qkv: Box::new(qkv_graph),
            post: Box::new(post_graph),
            qkv_traces,
            post_traces,
            mode: if no_fuse { 2 } else { 3 },
        })
    }

    /// Fixed-shape diagnostic inputs: token, its position, then 24 K/V pairs.
    /// The model's immutable weights are captured by the diagnostic closure.
    pub fn diagnostic_static_decode_inputs(
        &self,
        token: &Array,
        state: &ArState,
    ) -> Result<Vec<Array>> {
        self.validate_decode_input(token, state)?;
        Self::validate_ids(token, "semantic", self.semantic_vocab)?;
        self.static_decode_inputs(token, state)
    }

    /// Diagnostic-only GPU sampler output; caller must validate consumed IDs
    /// and discard state on an invalid decision or an earlier EOS.
    pub fn diagnostic_static_decode_sampled_gpu_inputs(
        &self,
        token: &Array,
        state: &ArState,
    ) -> Result<Vec<Array>> {
        self.validate_decode_input(token, state)?;
        if token.dtype() != Dtype::Int32 {
            return Err(invalid("diagnostic sampled token must be int32"));
        }
        // The diagnostic owner checks immutable model configuration once at
        // construction; avoid revalidating every projection on each token.
        self.static_decode_args(token, state)
    }

    fn static_decode_inputs(&self, token: &Array, state: &ArState) -> Result<Vec<Array>> {
        if self.dim != 512
            || self.heads != 16
            || self.blocks.len() != 24
            || !matches!(self.audio_embedding.dtype(), Dtype::Float32 | Dtype::Float16)
            || self.mixed_gemv_rows() != Some(1)
            || !self.mixed_gemv_epilogue_enabled()
        {
            return Err(invalid(
                "static decode diagnostic requires FP32/FP16 fused TM1 model",
            ));
        }
        self.static_decode_args(token, state)
    }

    fn static_decode_args(&self, token: &Array, state: &ArState) -> Result<Vec<Array>> {
        let mut args = vec![
            token.clone(),
            self.position(state.audio_len, 1, self.audio_embedding.dtype())?,
        ];
        for cache in &state.caches {
            args.extend([cache.k.clone(), cache.v.clone()]);
        }
        Ok(args)
    }

    /// Stateless diagnostic graph. Uses the same block and projection methods
    /// as decode_inner; callers validate state and install the returned K/V.
    pub fn diagnostic_static_decode_graph(&self, args: &[Array]) -> Result<Vec<Array>> {
        if args.len() != 50
            || args[0].shape() != [1, 1]
            || args[1].shape() != [1, 1, 512]
            || args[1].dtype() != self.audio_embedding.dtype()
        {
            return Err(invalid("invalid static diagnostic inputs"));
        }
        let audio = self.audio_embedding.take_axis(&args[0], 0)?;
        let mut hidden = audio.add(args[1].multiply(&self.audio_alpha)?)?;
        let mut caches = Vec::with_capacity(48);
        for (index, block) in self.blocks.iter().enumerate() {
            let cache = Cache {
                k: args[2 + index * 2].clone(),
                v: args[3 + index * 2].clone(),
            };
            let (next, updated) = block.forward(&hidden, Some(&cache), self.heads, None)?;
            hidden = next;
            caches.extend([updated.k, updated.v]);
        }
        let mut outputs = vec![self.logits(&hidden)?];
        outputs.extend(caches);
        Ok(outputs)
    }

    /// Diagnostic-only flattened stateless decode inputs: token, position,
    /// embedding, alpha, prediction weight, then K/V and weights per block.
    pub fn diagnostic_compiled_decode_inputs(
        &self,
        token: &Array,
        state: &ArState,
    ) -> Result<Vec<Array>> {
        self.validate_decode_input(token, state)?;
        Self::validate_ids(token, "semantic", self.semantic_vocab)?;
        self.compiled_decode_inputs(token, state)
    }

    /// The ID must have been read from this same generated token array.
    pub(crate) fn compiled_decode_inputs_generated(
        &self,
        token: &Array,
        known_id: i32,
        state: &ArState,
    ) -> Result<Vec<Array>> {
        self.validate_decode_input(token, state)?;
        if token.dtype() != Dtype::Int32 && token.dtype() != Dtype::Int64 {
            return Err(invalid("semantic must be int32 or int64"));
        }
        if !(0..self.semantic_vocab).contains(&known_id) {
            return Err(invalid(format!(
                "semantic ID {known_id} outside [0,{})",
                self.semantic_vocab
            )));
        }
        self.compiled_decode_inputs(token, state)
    }

    fn compiled_decode_inputs(&self, token: &Array, state: &ArState) -> Result<Vec<Array>> {
        if self.dim != 512
            || self.heads != 16
            || self.blocks.len() != 24
            || self.audio_embedding.dtype() != Dtype::Float32
        {
            return Err(invalid(
                "compiled decode requires the FP32 24-layer AR model",
            ));
        }
        let mut inputs = vec![
            token.clone(),
            self.position(state.audio_len, 1, Dtype::Float32)?,
            self.audio_embedding.clone(),
            self.audio_alpha.clone(),
            self.predict_t.clone(),
        ];
        for (block, cache) in self.blocks.iter().zip(&state.caches) {
            inputs.extend([
                cache.k.clone(),
                cache.v.clone(),
                block.qkv.weight_t.clone(),
                block.qkv.bias.clone(),
                block.out.weight_t.clone(),
                block.out.bias.clone(),
                block.ff1.weight_t.clone(),
                block.ff1.bias.clone(),
                block.ff2.weight_t.clone(),
                block.ff2.bias.clone(),
                block.norm1.weight.clone(),
                block.norm1.bias.clone(),
                block.norm2.weight.clone(),
                block.norm2.bias.clone(),
            ]);
        }
        Ok(inputs)
    }

    /// Install diagnostic graph outputs after validating all cache shapes.
    pub fn diagnostic_install_compiled_decode(
        &self,
        state: &mut ArState,
        outputs: &[Array],
    ) -> Result<()> {
        self.install_compiled_decode(state, outputs)
    }

    pub(crate) fn install_compiled_decode(
        &self,
        state: &mut ArState,
        outputs: &[Array],
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &state.owner) {
            return Err(invalid("AR state belongs to a different model"));
        }
        if state.caches.len() != self.blocks.len()
            || outputs.len() != 1 + 2 * self.blocks.len()
            || outputs[0].shape() != [1, self.semantic_vocab]
            || outputs[0].dtype() != self.audio_embedding.dtype()
        {
            return Err(invalid("invalid compiled diagnostic logits or cache count"));
        }
        let mut next = Vec::with_capacity(self.blocks.len());
        for (index, old) in state.caches.iter().enumerate() {
            if old.k.ndim() != 3
                || old.k.shape()[0] != 1
                || old.k.shape()[2] != self.dim
                || old.v.shape() != old.k.shape()
            {
                return Err(invalid("invalid existing diagnostic cache"));
            }
            let length = old.k.shape()[1]
                .checked_add(1)
                .ok_or_else(|| invalid("AR cache length overflow"))?;
            let k = &outputs[1 + 2 * index];
            let v = &outputs[2 + 2 * index];
            if k.dtype() != self.audio_embedding.dtype()
                || v.dtype() != self.audio_embedding.dtype()
                || k.shape() != [1, length, self.dim]
                || v.shape() != k.shape()
            {
                return Err(invalid("invalid compiled diagnostic cache output"));
            }
            next.push(Cache {
                k: k.clone(),
                v: v.clone(),
            });
        }
        state.audio_len = state
            .audio_len
            .checked_add(1)
            .ok_or_else(|| invalid("AR audio length overflow"))?;
        state.caches = next;
        Ok(())
    }

    fn decode_inner(&self, token: &Array, state: &mut ArState) -> Result<Array> {
        let audio = self.audio_embedding.take_axis(token, 0)?;
        let mut hidden = audio.add(
            self.position(state.audio_len, 1, audio.dtype())?
                .multiply(&self.audio_alpha)?,
        )?;
        let mut next = Vec::with_capacity(self.blocks.len());
        for (block, cache) in self.blocks.iter().zip(&state.caches) {
            let (h, c) = block.forward(&hidden, Some(cache), self.heads, None)?;
            hidden = h;
            next.push(c);
        }
        let logits = self.logits(&hidden)?;
        state.audio_len = state
            .audio_len
            .checked_add(1)
            .ok_or_else(|| invalid("AR audio length overflow"))?;
        state.caches = next;
        Ok(logits)
    }
}
