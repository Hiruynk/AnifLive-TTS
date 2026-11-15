//! Chinese RoBERTa WWM large (`BertForMaskedLM` encoder), FP32/FP16 inference.
//!
//! GPT-SoVITS uses HF hidden_states[-3]: with 24 layers plus embeddings this
//! is the output of encoder layer 21. Later layers and the MLM head do not
//! contribute to that selected state. Tokenization and phone alignment remain
//! caller responsibilities; this module returns the selected tokenizer states.
use mlx_rs::{
    fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn::gelu,
    ops::zeros,
    Array, Dtype,
};
use std::{collections::HashMap, error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
fn get(weights: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let value = weights
        .get(key)
        .ok_or_else(|| invalid(format!("missing BERT tensor {key}")))?;
    if value.shape() != shape || !matches!(value.dtype(), Dtype::Float32 | Dtype::Float16) {
        return Err(invalid(format!(
            "BERT tensor {key}: expected float32 or float16 {shape:?}, got {:?} {:?}",
            value.shape(),
            value.dtype()
        )));
    }
    Ok(value.clone())
}
fn check_ids(a: &Array, name: &str, limit: i64) -> Result<()> {
    if a.dtype() != Dtype::Int32 && a.dtype() != Dtype::Int64 {
        return Err(invalid(format!("{name} must be int32 or int64")));
    }
    let min = a.min(false)?.try_item_cast::<i64>()?;
    let max = a.max(false)?.try_item_cast::<i64>()?;
    if min < 0 || max >= limit {
        return Err(invalid(format!("{name} values must be in [0,{limit})")));
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
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), &[1024])?,
            bias: get(weights, &format!("{prefix}.bias"), &[1024])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        if x.dtype() == mlx_rs::Dtype::Float16 {
            return crate::source_half_attention::layer_norm(x, &self.weight, &self.bias, 1e-12);
        }
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-12)?)
    }
}
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    attention_norm: Norm,
    ff1: Linear,
    ff2: Linear,
    output_norm: Norm,
}
impl Layer {
    fn load(weights: &HashMap<String, Array>, i: usize) -> Result<Self> {
        let p = format!("bert.encoder.layer.{i}");
        Ok(Self {
            q: Linear::load(weights, &format!("{p}.attention.self.query"), 1024, 1024)?,
            k: Linear::load(weights, &format!("{p}.attention.self.key"), 1024, 1024)?,
            v: Linear::load(weights, &format!("{p}.attention.self.value"), 1024, 1024)?,
            out: Linear::load(weights, &format!("{p}.attention.output.dense"), 1024, 1024)?,
            attention_norm: Norm::load(weights, &format!("{p}.attention.output.LayerNorm"))?,
            ff1: Linear::load(weights, &format!("{p}.intermediate.dense"), 1024, 4096)?,
            ff2: Linear::load(weights, &format!("{p}.output.dense"), 4096, 1024)?,
            output_norm: Norm::load(weights, &format!("{p}.output.LayerNorm"))?,
        })
    }
    fn forward(&self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let n = x.shape()[1];
        let heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[1, n, 16, 64])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(self.q.forward(x)?)?;
        let k = heads(self.k.forward(x)?)?;
        let v = heads(self.v.forward(x)?)?;
        let attention = if q.dtype() == Dtype::Float16 {
            crate::source_half_attention::attention(&q, &k, &v, mask)?
        } else {
            scaled_dot_product_attention(
                &q,
                &k,
                &v,
                0.125,
                mask.map(ScaledDotProductAttentionMask::Array),
                None,
            )?
        };
        let a = attention
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, n, 1024])?;
        let hidden = self
            .attention_norm
            .forward(&x.add(self.out.forward(&a)?)?)?;
        let intermediate = self.ff1.forward(&hidden)?;
        let activated = if intermediate.dtype() == Dtype::Float16 {
            let value = intermediate.as_dtype(Dtype::Float32)?;
            mlx_rs::ops::erf(value.multiply(Array::from_f32(0.70703125))?)?
                .add(Array::from_f32(1.0))?
                .multiply(Array::from_f32(0.5))?
                .multiply(&value)?
                .as_dtype(Dtype::Float16)?
        } else {
            gelu(intermediate)?
        };
        self.output_norm
            .forward(&hidden.add(self.ff2.forward(&activated)?)?)
    }
}
/// Exact configuration of Chinese RoBERTa WWM large, selected layer only.
pub struct Bert {
    words: Array,
    positions: Array,
    types: Array,
    embedding_norm: Norm,
    layers: Vec<Layer>,
}
impl Bert {
    /// Expect name-preserving checkpoint tensors cast to the requested precision by the caller.
    /// Strictly validates the used layers and checks the source has 24 layers.
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        let precision = weights
            .get("bert.embeddings.word_embeddings.weight")
            .ok_or_else(|| invalid("missing BERT embeddings"))?
            .dtype();
        if !matches!(precision, Dtype::Float32 | Dtype::Float16)
            || weights.values().any(|v| v.dtype() != precision)
        {
            return Err(invalid("BERT tensors must have one floating precision"));
        }
        let count = weights
            .keys()
            .filter(|name| {
                name.starts_with("bert.encoder.layer.")
                    && name.ends_with(".attention.self.query.weight")
            })
            .count();
        if count != 24 {
            return Err(invalid(format!("expected 24 BERT layers, found {count}")));
        }
        // Verify the two later layers exist even though hidden_states[-3] does
        // not depend on their execution.
        for i in 22..24 {
            let _ = get(
                weights,
                &format!("bert.encoder.layer.{i}.attention.self.query.weight"),
                &[1024, 1024],
            )?;
        }
        Ok(Self {
            words: get(
                weights,
                "bert.embeddings.word_embeddings.weight",
                &[21128, 1024],
            )?,
            positions: get(
                weights,
                "bert.embeddings.position_embeddings.weight",
                &[512, 1024],
            )?,
            types: get(
                weights,
                "bert.embeddings.token_type_embeddings.weight",
                &[2, 1024],
            )?,
            embedding_norm: Norm::load(weights, "bert.embeddings.LayerNorm")?,
            layers: (0..22)
                .map(|i| Layer::load(weights, i))
                .collect::<Result<Vec<_>>>()?,
        })
    }
    /// Input arrays are [1,N], N<=512. Optional segment IDs default to zero;
    /// optional attention mask uses 1/true for keys to attend. Returns
    /// [1,N,1024], including special tokens. The caller strips [CLS]/[SEP]
    /// and expands per `word2ph` exactly as official `get_bert_feature` does.
    pub fn hidden_for_phones(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        if input_ids.ndim() != 2
            || input_ids.shape()[0] != 1
            || !(1..=512).contains(&input_ids.shape()[1])
        {
            return Err(invalid(format!(
                "BERT input IDs must be [1,N] with 1<=N<=512, got {:?}",
                input_ids.shape()
            )));
        }
        let n = input_ids.shape()[1];
        check_ids(input_ids, "BERT input IDs", 21128)?;
        let default_types;
        let type_ids = if let Some(ids) = token_type_ids {
            if ids.shape() != [1, n] {
                return Err(invalid("BERT token type IDs must match input shape"));
            }
            check_ids(ids, "BERT token type IDs", 2)?;
            ids
        } else {
            default_types = zeros::<i32>(&[1, n])?;
            &default_types
        };
        let mask = if let Some(mask) = attention_mask {
            if mask.shape() != [1, n] {
                return Err(invalid("BERT attention mask must match input shape"));
            }
            let ids = if mask.dtype() == Dtype::Bool {
                mask.as_dtype(Dtype::Int32)?
            } else {
                mask.clone()
            };
            check_ids(&ids, "BERT attention mask", 2)?;
            if ids.max(false)?.try_item_cast::<i64>()? == 0 {
                return Err(invalid("BERT attention mask cannot hide every token"));
            }
            Some(ids.as_dtype(Dtype::Bool)?.expand_dims_axes(&[1, 2])?)
        } else {
            None
        };
        let positions = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[1, n]);
        // HF BertEmbeddings adds token type before absolute position embedding.
        let word = self.words.take_axis(input_ids, 0)?;
        let segment = self.types.take_axis(type_ids, 0)?;
        let position = self.positions.take_axis(positions, 0)?;
        let mut hidden = self
            .embedding_norm
            .forward(&word.add(segment)?.add(position)?)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask.as_ref())?;
        }
        Ok(hidden)
    }
}
