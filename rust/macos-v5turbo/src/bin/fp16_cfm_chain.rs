//! FP16 causal CFM diagnostic: teacher-forced per-step velocity and full
//! four-step chains from captured official MPS conditioning versus native
//! conditioning; native vocoder on each result. Diagnostic only.
use aniflive_macos_v5turbo::{estimator::Estimator, vocoder::Vocoder};
use mlx_rs::{
    ops::{concatenate, indexing::IndexOp, select, zeros},
    Array, Device, Dtype, Stream,
};
use serde_json::json;
use std::{collections::HashMap, error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn load(path: &Path) -> Result<Array> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let v = mlx_rs::with_stream(&cpu, || Array::load_numpy(path))?;
    if v.dtype() != Dtype::Float16 {
        return Err(format!("{} is not F16", path.display()).into());
    }
    Ok(v)
}
fn weights(path: &Path) -> Result<HashMap<String, Array>> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let raw = mlx_rs::with_stream(&cpu, || Array::load_safetensors(path))?;
    let mut m = HashMap::new();
    for (k, v) in raw {
        m.insert(k, v.as_dtype(Dtype::Float16)?);
    }
    Ok(m)
}
fn h(v: f32) -> Result<Array> {
    Ok(Array::from_slice(&[v], &[1]).as_dtype(Dtype::Float16)?)
}

fn chain(
    est: &Estimator,
    voc: &Vocoder,
    mu: &Array,
    prompt: &Array,
    state0: &Array,
    mask: &Array,
    r: i32,
    out: &Path,
    name: &str,
) -> Result<()> {
    let mut x = state0.clone();
    for step in 0..4 {
        let v = est.forward(&x, prompt, mu, &h(step as f32 * 0.25)?, None)?;
        x = select(mask, x.add(v.multiply(h(0.25)?)?)?, h(0.0)?)?;
    }
    let mel = x.index((.., .., r..));
    mel.eval()?;
    mel.save_numpy(out.join(format!("{name}-mel.npy")))?;
    let raw = mel
        .add(h(1.0)?)?
        .divide(h(2.0)?)?
        .multiply(h(14.0)?)?
        .subtract(h(12.0)?)?;
    let wave = voc.forward(&raw)?.as_dtype(Dtype::Float32)?;
    wave.eval()?;
    aniflive_macos_v5turbo::audio::write_pcm16(
        wave.as_slice::<f32>(),
        &out.join(format!("{name}.wav")),
        &out.join(format!("{name}.pcm")),
    )?;
    Ok(())
}

fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        return Err("Usage: fp16_cfm_chain MODELS_DIR SOURCE_TRACE NATIVE_RUN FRESH_OUT".into());
    }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
        return Err("Explicit MLX_ENABLE_TF32=0 is required".into());
    }
    let models = Path::new(&a[1]);
    let cfm = Path::new(&a[2]).join("cfm");
    let native = Path::new(&a[3]);
    let out = Path::new(&a[4]);
    fs::create_dir(out)?;
    let est = Estimator::load(&weights(&models.join("s2Gv5turbo.safetensors"))?)?;
    let voc = Vocoder::load(&weights(&models.join("vocoder.safetensors"))?)?;
    let state0 = load(&cfm.join("step-0000-state.npy"))?;
    let t = state0.shape()[2];
    let prefix = load(&cfm.join("prompt.npy"))?;
    let r = prefix.shape()[2];
    let prompt = concatenate(
        &[prefix, zeros::<f32>(&[1, 100, t - r])?.as_dtype(Dtype::Float16)?],
        2,
    )?;
    let mask: Vec<bool> = (0..t).map(|i| i >= r).collect();
    let mask = Array::from_slice(&mask, &[1, 1, t]);
    let source_mu = load(&cfm.join("mu.npy"))?;
    // Teacher-forced: official state_k and official conditioning per step.
    for step in 0..4 {
        let s = load(&cfm.join(format!("step-{step:04}-state.npy")))?;
        let v = est.forward(&s, &prompt, &source_mu, &h(step as f32 * 0.25)?, None)?;
        let v = v.transpose_axes(&[0, 2, 1])?.contiguous()?;
        v.eval()?;
        v.save_numpy(out.join(format!("teacher-step-{step:04}-velocity.npy")))?;
    }
    chain(&est, &voc, &source_mu, &prompt, &state0, &mask, r, out, "chain-source-mu")?;
    let native_mu = concatenate(
        &[
            load(&native.join("reference-features.npy"))?.index((.., .., 0..r)),
            load(&native.join("target-features.npy"))?,
        ],
        2,
    )?
    .transpose_axes(&[0, 2, 1])?;
    if native_mu.shape() != source_mu.shape() {
        return Err(format!("native mu {:?} vs source {:?}", native_mu.shape(), source_mu.shape()).into());
    }
    native_mu.eval()?;
    native_mu.save_numpy(out.join("native-mu.npy"))?;
    chain(&est, &voc, &native_mu, &prompt, &state0, &mask, r, out, "chain-native-mu")?;
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&json!({"kind":"fp16_cfm_chain_diagnostic","qualified":false,
            "frames":t,"reference_frames":r,"production_math_changed":false}))?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    mlx_rs::with_device(Device::gpu(), run)
}
