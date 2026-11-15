//! Native FP32 G2PW graph, using the pinned ONNX model's original weights.
use mlx_rs::{
    fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn::{gelu, sigmoid},
    ops::{clip, indexing::argmax_axis},
    Array, Dtype,
};
use std::{collections::HashMap, error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}
fn get(w: &HashMap<String, Array>, key: &str, shape: &[i32]) -> Result<Array> {
    let a = w
        .get(key)
        .ok_or_else(|| invalid(format!("missing G2PW tensor {key}")))?;
    if a.dtype() != Dtype::Float32 || a.shape() != shape {
        return Err(invalid(format!(
            "G2PW tensor {key} expected float32 {shape:?}, got {:?} {:?}",
            a.dtype(),
            a.shape()
        )));
    }
    Ok(a.clone())
}
fn ids(a: &Array, name: &str, shape: &[i32], limit: i64) -> Result<()> {
    if a.shape() != shape || (a.dtype() != Dtype::Int32 && a.dtype() != Dtype::Int64) {
        return Err(invalid(format!("{name} must be integer {shape:?}")));
    }
    if a.min(false)?.try_item_cast::<i64>()? < 0 || a.max(false)?.try_item_cast::<i64>()? >= limit {
        return Err(invalid(format!("{name} outside [0,{limit})")));
    }
    Ok(())
}
const MATRIX_NAMES: [[&str; 6]; 12] = [
    ["1719", "1720", "1721", "1729", "1731", "1732"],
    ["1734", "1735", "1736", "1744", "1746", "1747"],
    ["1749", "1750", "1751", "1759", "1761", "1762"],
    ["1764", "1765", "1766", "1774", "1776", "1777"],
    ["1779", "1780", "1781", "1789", "1791", "1792"],
    ["1794", "1795", "1796", "1804", "1806", "1807"],
    ["1809", "1810", "1811", "1819", "1821", "1822"],
    ["1824", "1825", "1826", "1834", "1836", "1837"],
    ["1839", "1840", "1841", "1849", "1851", "1852"],
    ["1854", "1855", "1856", "1864", "1866", "1867"],
    ["1869", "1870", "1871", "1879", "1881", "1882"],
    ["1884", "1885", "1886", "1894", "1896", "1897"],
];
struct Linear {
    w: Array,
    b: Array,
}
impl Linear {
    fn encoder(w: &HashMap<String, Array>, name: &str, bias: &str, i: i32, o: i32) -> Result<Self> {
        Ok(Self {
            w: get(w, name, &[i, o])?,
            b: get(w, bias, &[o])?,
        })
    }
    fn head(w: &HashMap<String, Array>, prefix: &str, o: i32) -> Result<Self> {
        Ok(Self {
            w: get(w, &format!("{prefix}.weight"), &[o, 768])?.transpose_axes(&[1, 0])?,
            b: get(w, &format!("{prefix}.bias"), &[o])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(x.matmul(&self.w)?.add(&self.b)?)
    }
}
struct Norm {
    w: Array,
    b: Array,
}
impl Norm {
    fn load(w: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: get(w, &format!("{prefix}.weight"), &[768])?,
            b: get(w, &format!("{prefix}.bias"), &[768])?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(layer_norm(x, Some(&self.w), Some(&self.b), 1e-12)?)
    }
}
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm1: Norm,
    ff1: Linear,
    ff2: Linear,
    norm2: Norm,
}
impl Layer {
    fn load(w: &HashMap<String, Array>, index: usize) -> Result<Self> {
        let p = format!("bert.encoder.layer.{index}");
        let n = MATRIX_NAMES[index];
        let lin = |i: usize, suffix: &str, a: i32, b: i32| {
            Linear::encoder(w, n[i], &format!("{p}.{suffix}.bias"), a, b)
        };
        Ok(Self {
            q: lin(0, "attention.self.query", 768, 768)?,
            k: lin(1, "attention.self.key", 768, 768)?,
            v: lin(2, "attention.self.value", 768, 768)?,
            out: lin(3, "attention.output.dense", 768, 768)?,
            norm1: Norm::load(w, &format!("{p}.attention.output.LayerNorm"))?,
            ff1: lin(4, "intermediate.dense", 768, 3072)?,
            ff2: lin(5, "output.dense", 3072, 768)?,
            norm2: Norm::load(w, &format!("{p}.output.LayerNorm"))?,
        })
    }
    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        let (b, n) = (x.shape()[0], x.shape()[1]);
        let split = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, 12, 64])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = split(self.q.forward(x)?)?;
        let k = split(self.k.forward(x)?)?;
        let v = split(self.v.forward(x)?)?;
        let a = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            0.125,
            Some(ScaledDotProductAttentionMask::Array(mask)),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, n, 768])?;
        let x = self.norm1.forward(&x.add(self.out.forward(&a)?)?)?;
        self.norm2
            .forward(&x.add(self.ff2.forward(&gelu(self.ff1.forward(&x)?)?)?)?)
    }
}
/// Pinned GPT-SoVITS G2PW ONNX model; all values are unquantized FP32.
pub struct G2PW {
    words: Array,
    positions: Array,
    types: Array,
    embed_norm: Norm,
    layers: Vec<Layer>,
    classifier: Linear,
    pos_classifier: Linear,
    desc_bias: Array,
    char_desc: Array,
    second_desc: Array,
}
impl G2PW {
    pub fn load(w: &HashMap<String, Array>) -> Result<Self> {
        Ok(Self {
            words: get(w, "bert.embeddings.word_embeddings.weight", &[21128, 768])?,
            positions: get(w, "bert.embeddings.position_embeddings.weight", &[512, 768])?,
            types: get(w, "bert.embeddings.token_type_embeddings.weight", &[2, 768])?,
            embed_norm: Norm::load(w, "bert.embeddings.LayerNorm")?,
            layers: (0..12)
                .map(|i| Layer::load(w, i))
                .collect::<Result<Vec<_>>>()?,
            classifier: Linear::head(w, "classifier", 1305)?,
            pos_classifier: Linear::head(w, "pos_classifier", 11)?,
            desc_bias: get(w, "descriptor_bias.weight", &[1, 1305])?,
            char_desc: get(w, "char_descriptor.weight", &[3582, 1305])?,
            second_desc: get(w, "second_order_descriptor.weight", &[39402, 1305])?,
        })
    }
    /// Inputs follow ONNX names and shapes. Each batch row identifies one polyphonic query.
    pub fn forward(
        &self,
        input_ids: &Array,
        token_type_ids: &Array,
        attention_mask: &Array,
        phoneme_mask: &Array,
        char_ids: &Array,
        position_ids: &Array,
    ) -> Result<Array> {
        if input_ids.ndim() != 2
            || input_ids.shape()[0] < 1
            || !(1..=512).contains(&input_ids.shape()[1])
        {
            return Err(invalid(
                "input_ids must be [batch,sequence], with nonempty batch and sequence <=512",
            ));
        }
        let (b, n) = (input_ids.shape()[0], input_ids.shape()[1]);
        ids(input_ids, "input_ids", &[b, n], 21128)?;
        ids(token_type_ids, "token_type_ids", &[b, n], 2)?;
        ids(attention_mask, "attention_mask", &[b, n], 2)?;
        ids(char_ids, "char_ids", &[b], 3582)?;
        ids(position_ids, "position_ids", &[b], n as i64)?;
        if phoneme_mask.dtype() != Dtype::Float32 || phoneme_mask.shape() != [b, 1305] {
            return Err(invalid("phoneme_mask must be float32 [batch,1305]"));
        }
        let binary_mask = phoneme_mask
            .eq(Array::from_f32(0.0))?
            .logical_or(phoneme_mask.eq(Array::from_f32(1.0))?)?;
        if !binary_mask.all(false)?.item_exact::<bool>() {
            return Err(invalid(
                "phoneme_mask must contain only finite zeros or ones",
            ));
        }
        if phoneme_mask
            .sum_axis(-1, false)?
            .min(false)?
            .try_item_cast::<f32>()?
            == 0.0
        {
            return Err(invalid(
                "each phoneme_mask row must enable at least one phoneme",
            ));
        }
        let selected_attention = attention_mask.take_along_axis(position_ids.expand_dims(1)?, 1)?;
        if selected_attention.min(false)?.try_item_cast::<i64>()? != 1 {
            return Err(invalid(
                "each query position must be enabled in attention_mask",
            ));
        }
        // Preserve ONNX's additive attention mask instead of replacing it by a Boolean mask.
        let mask = Array::from_f32(1.0)
            .subtract(attention_mask.as_dtype(Dtype::Float32)?)?
            .multiply(Array::from_f32(-10000.0))?
            .expand_dims_axes(&[1, 2])?;
        let pos = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[1, n]);
        // The ONNX export adds position before token type.
        let mut x = self.embed_norm.forward(
            &self
                .words
                .take_axis(input_ids, 0)?
                .add(self.positions.take_axis(&pos, 0)?)?
                .add(self.types.take_axis(token_type_ids, 0)?)?,
        )?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        let offsets = Array::from_slice(&(0..b).map(|i| i * n).collect::<Vec<i32>>(), &[b]);
        let positions = position_ids.as_dtype(Dtype::Int32)?.add(&offsets)?;
        let hidden = x.reshape(&[b * n, 768])?.take_axis(&positions, 0)?;
        let pos = argmax_axis(self.pos_classifier.forward(&hidden)?, -1, false)?
            .as_dtype(Dtype::Int32)?;
        let index = char_ids
            .as_dtype(Dtype::Int32)?
            .multiply(Array::from_int(11))?
            .add(&pos)?;
        let descriptors = self
            .desc_bias
            .add(self.char_desc.take_axis(char_ids, 0)?)?
            .add(self.second_desc.take_axis(&index, 0)?)?;
        let weights = phoneme_mask.multiply(sigmoid(descriptors)?)?;
        let logits = self.classifier.forward(&hidden)?;
        let numerator = logits
            .subtract(logits.max_axis(-1, true)?)?
            .exp()?
            .multiply(weights)?;
        let probs = numerator.divide(numerator.sum_axis(-1, true)?)?;
        Ok(clip(probs, (1e-6f32, 0.999999f32))?)
    }
}
