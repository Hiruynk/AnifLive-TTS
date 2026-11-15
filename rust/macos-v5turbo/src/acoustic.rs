//! V5 decode_encp, single utterance, speed=1, all-valid semantic/text lengths.
use mlx_rs::{
    fast::layer_norm,
    nn::{leaky_relu, relu, sigmoid},
    ops::{conv1d, softmax_axis, tanh},
    Array,
};
use std::{collections::HashMap, error::Error, io};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn source_sigmoid(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    if x.dtype() == mlx_rs::Dtype::Float16 {
        Ok(sigmoid(x.as_dtype(mlx_rs::Dtype::Float32)?)?.as_dtype(mlx_rs::Dtype::Float16)?)
    } else { Ok(sigmoid(x)?) }
}

static TRACE_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
// Diagnostic only: with ANIFLIVE_ACOUSTIC_TRACE_DIR unset nothing is evaluated.
fn trace(name: &str, value: &Array) -> Result<()> {
    if let Some(dir) = std::env::var_os("ANIFLIVE_ACOUSTIC_TRACE_DIR") {
        let call = TRACE_CALL.load(std::sync::atomic::Ordering::SeqCst);
        value.eval()?;
        let cpu = mlx_rs::Stream::try_default_on_device(&mlx_rs::Device::cpu())?;
        mlx_rs::with_stream(&cpu, || {
            value.save_numpy(std::path::Path::new(&dir).join(format!("{call:04}-{name}.npy")))
        })?;
    }
    Ok(())
}
fn err(s: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, s.into()))
}
fn get(w: &HashMap<String, Array>, k: &str) -> Result<Array> {
    w.get(k)
        .cloned()
        .ok_or_else(|| err(format!("missing acoustic tensor {k}")))
}
struct Conv {
    w: Array,
    b: Array,
    pad: i32,
}
impl Conv {
    fn load(w: &HashMap<String, Array>, p: &str) -> Result<Self> {
        let raw = if w.contains_key(&format!("{p}.weight")) {
            get(w, &format!("{p}.weight"))?
        } else {
            let v = get(w, &format!("{p}.weight_v"))?;
            let g = get(w, &format!("{p}.weight_g"))?;
            if v.dtype() == mlx_rs::Dtype::Float16 {
                crate::source_half_attention::weight_norm(&v, &g, &[1, 2])?
            } else {
                v.multiply(g.divide(v.square()?.sum_axes(&[1, 2], true)?.sqrt()?)?)?
            }
        };
        if raw.ndim() != 3 {
            return Err(err(format!("invalid convolution {p}")));
        }
        Ok(Self {
            pad: (raw.shape()[2] - 1) / 2,
            w: raw.transpose_axes(&[0, 2, 1])?,
            b: get(w, &format!("{p}.bias"))?,
        })
    }
    fn f(&self, x: &Array) -> Result<Array> {
        if x.dtype() == mlx_rs::Dtype::Float16 {
            if let Some(y) = crate::mps_order::conv1d_same(x, &self.w, &self.b)? {
                return Ok(y);
            }
        }
        if x.dtype() == mlx_rs::Dtype::Float16 {
            return Ok(conv1d(x.as_dtype(mlx_rs::Dtype::Float32)?, self.w.as_dtype(mlx_rs::Dtype::Float32)?, 1, self.pad, 1, 1)?
                .add(self.b.as_dtype(mlx_rs::Dtype::Float32)?)?.as_dtype(mlx_rs::Dtype::Float16)?);
        }
        Ok(conv1d(x, &self.w, 1, self.pad, 1, 1)?.add(&self.b)?)
    }
}
struct Norm {
    g: Array,
    b: Array,
}
impl Norm {
    fn load(w: &HashMap<String, Array>, p: &str) -> Result<Self> {
        Ok(Self {
            g: get(w, &format!("{p}.gamma"))?,
            b: get(w, &format!("{p}.beta"))?,
        })
    }
    fn f(&self, x: &Array) -> Result<Array> {
        if x.dtype() == mlx_rs::Dtype::Float16 {
            return crate::source_half_attention::layer_norm(x, &self.g, &self.b, 1e-5);
        }
        Ok(layer_norm(x, Some(&self.g), Some(&self.b), 1e-5)?)
    }
}
struct Attention {
    q: Conv,
    k: Conv,
    v: Conv,
    o: Conv,
    heads: i32,
    rk: Option<Array>,
    rv: Option<Array>,
}
impl Attention {
    fn load(w: &HashMap<String, Array>, p: &str, heads: i32, relative: bool) -> Result<Self> {
        Ok(Self {
            q: Conv::load(w, &format!("{p}.conv_q"))?,
            k: Conv::load(w, &format!("{p}.conv_k"))?,
            v: Conv::load(w, &format!("{p}.conv_v"))?,
            o: Conv::load(w, &format!("{p}.conv_o"))?,
            heads,
            rk: if relative {
                Some(get(w, &format!("{p}.emb_rel_k"))?)
            } else {
                None
            },
            rv: if relative {
                Some(get(w, &format!("{p}.emb_rel_v"))?)
            } else {
                None
            },
        })
    }
    fn relative(table: &Array, t: i32) -> Result<Array> {
        let window = (table.shape()[1] - 1) / 2;
        let dim = table.shape()[2];
        let mut ids = Vec::new();
        let mut valid = Vec::new();
        for i in 0..t {
            for j in 0..t {
                let delta = j - i;
                ids.push((delta + window).clamp(0, 2 * window));
                valid.push(if delta.abs() <= window { 1f32 } else { 0f32 });
            }
        }
        let index = Array::from_slice(&ids, &[t * t]);
        let values = table
            .reshape(&[2 * window + 1, dim])?
            .take_axis(&index, 0)?
            .reshape(&[1, 1, t, t, dim])?;
        Ok(
            values
                .multiply(Array::from_slice(&valid, &[1, 1, t, t, 1]).as_dtype(table.dtype())?)?,
        )
    }
    /// Half attention following the official MPS operations: every product
    /// is an MPS-order GEMM (mps_order), including the relative-position
    /// logits (query x padded key embeddings) and values (relative weights x
    /// padded value embeddings) that official VITS attention runs as matmul.
    fn f_half(&self, x: &Array, c: &Array) -> Result<Array> {
        use crate::mps_order::gemm;
        use mlx_rs::Dtype::{Float16, Float32, Int32};
        let (b, t, s) = (x.shape()[0], x.shape()[1], c.shape()[1]);
        let dim = self.q.b.shape()[0] / self.heads;
        let bh = b * self.heads;
        let split = |z: Array, n: i32| -> Result<Array> {
            Ok(z.reshape(&[b, n, self.heads, dim])?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[bh, n, dim])?)
        };
        // Source divides by a Python float: Float32 opmath, one Half rounding.
        let q = split(self.q.f(x)?, t)?
            .as_dtype(Float32)?
            .divide(Array::from_f32((dim as f64).sqrt() as f32))?
            .as_dtype(Float16)?;
        let k = split(self.k.f(c)?, s)?;
        let v = split(self.v.f(c)?, s)?;
        let mut scores = gemm(&q, &k, 1.0, None)?;
        if let Some(rk) = &self.rk {
            if s != t {
                return Err(err("relative attention requires equal lengths"));
            }
            let w = (rk.shape()[1] - 1) / 2;
            let g = gemm(&q, &rk.reshape(&[1, 2 * w + 1, dim])?, 1.0, None)?;
            let mut ids = Vec::with_capacity((t * t) as usize);
            let mut valid = Vec::with_capacity((t * t) as usize);
            for i in 0..t {
                for j in 0..t {
                    let r = j - i + w;
                    ids.push(r.clamp(0, 2 * w));
                    valid.push(if (0..=2 * w).contains(&r) { 1f32 } else { 0f32 });
                }
            }
            let ids = mlx_rs::ops::broadcast_to(Array::from_slice(&ids, &[1, t, t]), &[bh, t, t])?;
            let local = g
                .take_along_axis(&ids, -1)?
                .multiply(Array::from_slice(&valid, &[1, t, t]).as_dtype(Float16)?)?;
            scores = scores.add(local)?;
        }
        let p = crate::source_half_attention::softmax(&scores)?;
        let mut out = gemm(&p, &v.transpose_axes(&[0, 2, 1])?, 1.0, None)?;
        if let Some(rv) = &self.rv {
            let w = (rv.shape()[1] - 1) / 2;
            let width = 2 * t - 1;
            // relative weights rw[i, r] = p[i, i + r - (t - 1)] (zero outside).
            let mut ids = Vec::with_capacity((t * width) as usize);
            let mut valid = Vec::with_capacity((t * width) as usize);
            for i in 0..t {
                for r in 0..width {
                    let j = i + r - (t - 1);
                    ids.push(j.clamp(0, t - 1));
                    valid.push(if (0..t).contains(&j) { 1f32 } else { 0f32 });
                }
            }
            let ids = mlx_rs::ops::broadcast_to(Array::from_slice(&ids, &[1, t, width]), &[bh, t, width])?;
            let rw = p
                .take_along_axis(&ids, -1)?
                .multiply(Array::from_slice(&valid, &[1, t, width]).as_dtype(Float16)?)?;
            // padded value embeddings: row r holds rv[r - (t - 1) + w] inside the window.
            let mut rows = Vec::with_capacity(width as usize);
            let mut keep = Vec::with_capacity(width as usize);
            for r in 0..width {
                let e = r - (t - 1) + w;
                rows.push(e.clamp(0, 2 * w));
                keep.push(if (0..=2 * w).contains(&e) { 1f32 } else { 0f32 });
            }
            let table = rv.reshape(&[2 * w + 1, dim])?;
            let padded = table
                .take_axis(&Array::from_slice(&rows, &[width]).as_dtype(Int32)?, 0)?
                .multiply(Array::from_slice(&keep, &[width, 1]).as_dtype(Float16)?)?;
            let rel = gemm(&rw, &padded.transpose_axes(&[1, 0])?.reshape(&[1, dim, width])?, 1.0, None)?;
            out = out.add(rel)?;
        }
        self.o.f(&out
            .reshape(&[b, self.heads, t, dim])?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, self.heads * dim])?)
    }
    fn f(&self, x: &Array, c: &Array) -> Result<Array> {
        if x.dtype() == mlx_rs::Dtype::Float16 {
            return self.f_half(x, c);
        }
        let (b, t, s) = (x.shape()[0], x.shape()[1], c.shape()[1]);
        let dim = self.q.b.shape()[0] / self.heads;
        let shape = |z: Array, n: i32| {
            z.reshape(&[b, n, self.heads, dim])?
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = shape(self.q.f(x)?, t)?;
        // Source divides by a Python float: Float32 opmath, one Half rounding.
        let q = if q.dtype() == mlx_rs::Dtype::Float16 {
            q.as_dtype(mlx_rs::Dtype::Float32)?
                .divide(Array::from_f32((dim as f64).sqrt() as f32))?
                .as_dtype(mlx_rs::Dtype::Float16)?
        } else {
            q.divide(Array::from_f32((dim as f32).sqrt()).as_dtype(x.dtype())?)?
        };
        let k = shape(self.k.f(c)?, s)?;
        let v = shape(self.v.f(c)?, s)?;
        let mut scores = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
        if let Some(rk) = &self.rk {
            if s != t {
                return Err(err("relative attention requires equal lengths"));
            }
            let rel = Self::relative(rk, t)?;
            let logits = if q.dtype() == mlx_rs::Dtype::Float16 {
                // Source half matmul accumulates unrounded products in float.
                q.as_dtype(mlx_rs::Dtype::Float32)?.expand_dims(3)?
                    .multiply(rel.as_dtype(mlx_rs::Dtype::Float32)?)?
                    .sum_axis(-1, false)?.as_dtype(mlx_rs::Dtype::Float16)?
            } else { q.expand_dims(3)?.multiply(rel)?.sum_axis(-1, false)? };
            scores = scores.add(logits)?;
        }
        let p = if scores.dtype() == mlx_rs::Dtype::Float16 {
            crate::source_half_attention::softmax(&scores)?
        } else {
            softmax_axis(&scores, -1, None)?
        };
        let mut out = p.matmul(&v)?;
        if let Some(rv) = &self.rv {
            let rel = Self::relative(rv, t)?;
            let values = if p.dtype() == mlx_rs::Dtype::Float16 {
                p.as_dtype(mlx_rs::Dtype::Float32)?.expand_dims(-1)?
                    .multiply(rel.as_dtype(mlx_rs::Dtype::Float32)?)?
                    .sum_axis(3, false)?.as_dtype(mlx_rs::Dtype::Float16)?
            } else { p.expand_dims(-1)?.multiply(rel)?.sum_axis(3, false)? };
            out = out.add(values)?;
        }
        self.o.f(&out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, self.heads * dim])?)
    }
}
struct Layer {
    a: Attention,
    n1: Norm,
    n2: Norm,
    f1: Conv,
    f2: Conv,
}
struct Encoder {
    layers: Vec<Layer>,
}
impl Encoder {
    fn load(w: &HashMap<String, Array>, p: &str, count: usize) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..count {
            layers.push(Layer {
                a: Attention::load(w, &format!("{p}.attn_layers.{i}"), 2, true)?,
                n1: Norm::load(w, &format!("{p}.norm_layers_1.{i}"))?,
                n2: Norm::load(w, &format!("{p}.norm_layers_2.{i}"))?,
                f1: Conv::load(w, &format!("{p}.ffn_layers.{i}.conv_1"))?,
                f2: Conv::load(w, &format!("{p}.ffn_layers.{i}.conv_2"))?,
            });
        }
        Ok(Self { layers })
    }
    fn f(&self, input: &Array, name: &str) -> Result<Array> {
        let mut x = input.clone();
        for (i, l) in self.layers.iter().enumerate() {
            let y = l.a.f(&x, &x)?;
            trace(&format!("{name}.attn_layers.{i}"), &y)?;
            x = l.n1.f(&x.add(&y)?)?;
            trace(&format!("{name}.norm_layers_1.{i}"), &x)?;
            let y = l.f2.f(&relu(l.f1.f(&x)?)?)?;
            trace(&format!("{name}.ffn_layers.{i}"), &y)?;
            x = l.n2.f(&x.add(&y)?)?;
            trace(&format!("{name}.norm_layers_2.{i}"), &x)?;
        }
        Ok(x)
    }
}
struct Wn {
    pre: Conv,
    proj: Conv,
    cond: Conv,
    ins: Vec<Conv>,
    skips: Vec<Conv>,
}
impl Wn {
    fn load(w: &HashMap<String, Array>) -> Result<Self> {
        let mut ins = Vec::new();
        let mut skips = Vec::new();
        for i in 0..8 {
            ins.push(Conv::load(w, &format!("wns1.enc.in_layers.{i}"))?);
            skips.push(Conv::load(w, &format!("wns1.enc.res_skip_layers.{i}"))?);
        }
        Ok(Self {
            pre: Conv::load(w, "wns1.pre")?,
            proj: Conv::load(w, "wns1.proj")?,
            cond: Conv::load(w, "wns1.enc.cond_layer")?,
            ins,
            skips,
        })
    }
    fn f(&self, input: &Array, ge: &Array) -> Result<Array> {
        let mut x = self.pre.f(input)?;
        let g = self.cond.f(ge)?;
        let gs = g.split_equal(8, -1)?;
        let mut out = mlx_rs::ops::zeros_like(&x)?;
        for i in 0..8 {
            let z = self.ins[i].f(&x)?.add(&gs[i])?;
            let z = z.split_equal(2, -1)?;
            let acts = tanh(&z[0])?.multiply(source_sigmoid(&z[1])?)?;
            let skip = self.skips[i].f(&acts)?;
            if i < 7 {
                let r = skip.split_equal(2, -1)?;
                x = x.add(&r[0])?;
                out = out.add(&r[1])?;
            } else {
                out = out.add(skip)?;
            }
        }
        self.proj.f(&out)
    }
}
pub struct AcousticEncoder {
    codebook: Array,
    embed: Array,
    ssl: Conv,
    ssl_enc: Encoder,
    text_enc: Encoder,
    post_enc: Encoder,
    cpre: Conv,
    tpre: Conv,
    cpost: Conv,
    cross: Attention,
    bridge: Conv,
    wn: Wn,
}
impl AcousticEncoder {
    pub fn load(w: &HashMap<String, Array>) -> Result<Self> {
        Ok(Self {
            codebook: get(w, "quantizer.vq.layers.0._codebook.embed")?,
            embed: get(w, "enc_p.text_embedding.weight")?,
            ssl: Conv::load(w, "enc_p.ssl_proj")?,
            ssl_enc: Encoder::load(w, "enc_p.encoder_ssl", 3)?,
            text_enc: Encoder::load(w, "enc_p.encoder_text", 6)?,
            post_enc: Encoder::load(w, "enc_p.encoder2", 3)?,
            cpre: Conv::load(w, "enc_p.mrte.c_pre")?,
            tpre: Conv::load(w, "enc_p.mrte.text_pre")?,
            cpost: Conv::load(w, "enc_p.mrte.c_post")?,
            cross: Attention::load(w, "enc_p.mrte.cross_attention", 4, false)?,
            bridge: Conv::load(w, "bridge.0")?,
            wn: Wn::load(w)?,
        })
    }
    /// codes [1,1,N], text [1,M], ge [1,512,1] -> features [1,512,4*N].
    pub fn forward(&self, codes: &Array, text: &Array, ge: &Array) -> Result<Array> {
        if codes.ndim() != 3
            || codes.shape()[0] != 1
            || codes.shape()[1] != 1
            || codes.shape()[2] < 1
            || text.ndim() != 2
            || text.shape()[0] != 1
            || text.shape()[1] < 1
            || ge.shape() != [1, 512, 1]
        {
            return Err(err(
                "decode_encp expects one nonempty utterance, speed 1 and ge [1,512,1]",
            ));
        }
        let n = codes.shape()[2];
        let c = codes.reshape(&[n])?;
        let t = text.reshape(&[text.shape()[1]])?;
        let quant = self.codebook.take_axis(&c, 0)?.reshape(&[1, n, 768])?;
        let quant = repeat2(&quant)?;
        let ssl = self.ssl.f(&quant)?;
        trace("enc_p.ssl_proj", &ssl)?;
        let y = self.ssl_enc.f(&ssl, "enc_p.encoder_ssl")?;
        let text = self
            .text_enc
            .f(&self.embed.take_axis(&t, 0)?.expand_dims(0)?, "enc_p.encoder_text")?;
        let y = self.cpre.f(&y)?;
        trace("enc_p.mrte.c_pre", &y)?;
        let text = self.tpre.f(&text)?;
        trace("enc_p.mrte.text_pre", &text)?;
        let ge = ge.transpose_axes(&[0, 2, 1])?;
        let cross = self.cross.f(&y, &text)?;
        trace("enc_p.mrte.cross_attention", &cross)?;
        let mixed = cross.add(&y)?.add(&ge)?;
        let post = self.cpost.f(&mixed)?;
        trace("enc_p.mrte", &post)?;
        let y = self.post_enc.f(&post, "enc_p.encoder2")?;
        let bridged = leaky_relu(self.bridge.f(&y)?, Some(0.01))?;
        trace("bridge", &bridged)?;
        let feature = repeat2(&bridged)?;
        let out = self.wn.f(&feature, &ge)?;
        trace("wns1", &out)?;
        TRACE_CALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(out.transpose_axes(&[0, 2, 1])?)
    }
}
fn repeat2(x: &Array) -> Result<Array> {
    let s = x.shape();
    Ok(
        mlx_rs::ops::broadcast_to(x.expand_dims(2)?, &[s[0], s[1], 2, s[2]])?.reshape(&[
            s[0],
            s[1] * 2,
            s[2],
        ])?,
    )
}
