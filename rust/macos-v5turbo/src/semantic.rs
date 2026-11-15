//! Official single-codebook reference semantic encoding after CN-HuBERT.
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{
    ops::{conv1d, indexing::argmax_axis},
    Array, Dtype,
};
use std::{collections::HashMap, error::Error, io};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn squared_norm(value: &Array, axis: i32, keep_dims: bool) -> Result<Array> {
    let squared = value.square()?;
    if value.dtype() == Dtype::Float16 {
        // Official Torch half sum uses float accumulators. Keep the preceding
        // square rounded to half and round the reduction result back to half.
        Ok(squared
            .as_dtype(Dtype::Float32)?
            .sum_axis(axis, keep_dims)?
            .as_dtype(Dtype::Float16)?)
    } else {
        Ok(squared.sum_axis(axis, keep_dims)?)
    }
}
pub struct SemanticEncoder {
    weight: Array,
    bias: Array,
    codebook: Array,
}
impl SemanticEncoder {
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        let get = |name: &str| {
            weights.get(name).cloned().ok_or_else(|| {
                Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing semantic tensor {name}"),
                )) as Box<dyn Error>
            })
        };
        let weight = get("ssl_proj.weight")?;
        let bias = get("ssl_proj.bias")?;
        let codebook = get("quantizer.vq.layers.0._codebook.embed")?;
        if weight.shape() != [768, 768, 2]
            || bias.shape() != [768]
            || codebook.shape() != [1024, 768]
        {
            return Err("Unsupported reference semantic model shapes".into());
        }
        Ok(Self {
            weight: weight.transpose_axes(&[0, 2, 1])?,
            bias,
            codebook,
        })
    }
    /// Hidden states NTC [1,T,768] -> projected NTC and semantic IDs [1,1,N].
    pub fn forward(&self, hidden: &Array) -> Result<(Array, Array)> {
        if hidden.ndim() != 3
            || hidden.shape()[0] != 1
            || hidden.shape()[1] < 2
            || hidden.shape()[2] != 768
        {
            return Err("Expected HuBERT hidden [1,T>=2,768]".into());
        }
        let projected = if hidden.dtype() == Dtype::Float16 {
            // Official MPS Half Conv1d(k=2, stride=2) behaves as one biased
            // GEMM over non-overlapping frame pairs (F32 accumulation, bias
            // inside, one Half rounding): 99.85% exact vs 71.5% for Half conv
            // followed by a separate Half bias add.
            let t = (hidden.shape()[1] - 2) / 2 + 1;
            let pairs = hidden.index((.., 0..2 * t, ..)).reshape(&[t, 2 * 768])?;
            let w = self.weight.reshape(&[768, 2 * 768])?;
            mlx_rs::ops::addmm(&self.bias, &pairs, w.transpose_axes(&[1, 0])?, None, None)?
                .reshape(&[1, t, 768])?
        } else {
            conv1d(hidden, &self.weight, 2, 0, 1, 1)?.add(&self.bias)?
        };
        let n = projected.shape()[1];
        let x = projected.reshape(&[n, 768])?;
        let distances = squared_norm(&x, 1, true)?
            .subtract(
                x.multiply(Array::from_f32(2.0).as_dtype(x.dtype())?)?
                    .matmul(self.codebook.transpose_axes(&[1, 0])?)?,
            )?
            .add(squared_norm(&self.codebook, 1, false)?)?
            .negative()?;
        let codes = argmax_axis(&distances, 1, false)?.reshape(&[1, 1, n])?;
        Ok((projected, codes))
    }
}
