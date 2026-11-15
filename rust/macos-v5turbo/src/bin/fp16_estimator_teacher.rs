//! Diagnostic: per-stage teacher-forced native FP16 estimator vs source stages.
use aniflive_macos_v5turbo::estimator::Estimator;
use mlx_rs::{ops::{concatenate, zeros}, Array, Device, Dtype, Stream};
use std::{collections::HashMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn load(p: &Path) -> Result<Array> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let v = mlx_rs::with_stream(&cpu, || Array::load_numpy(p))?;
    if v.dtype() != Dtype::Float16 { return Err(format!("{} not F16", p.display()).into()); }
    Ok(v)
}
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 { return Err("Usage: fp16_estimator_teacher S2.safetensors SOURCE_TRACE SOURCE_STAGES FRESH_OUT".into()); }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") { return Err("MLX_ENABLE_TF32=0 required".into()); }
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let raw = mlx_rs::with_stream(&cpu, || Array::load_safetensors(&a[1]))?;
    let mut w = HashMap::new();
    for (k, v) in raw { w.insert(k, v.as_dtype(Dtype::Float16)?); }
    let est = Estimator::load(&w)?;
    let cfm = Path::new(&a[2]).join("cfm");
    let stages = Path::new(&a[3]);
    let out = Path::new(&a[4]);
    fs::create_dir(out)?;
    let mut src = HashMap::new();
    for e in fs::read_dir(stages)? {
        let p = e?.path();
        if p.extension().map(|x| x == "npy").unwrap_or(false) {
            src.insert(p.file_stem().unwrap().to_string_lossy().into_owned(), load(&p)?);
        }
    }
    let x = load(&cfm.join("step-0000-state.npy"))?;
    let t = x.shape()[2];
    let prefix = load(&cfm.join("prompt.npy"))?;
    let r = prefix.shape()[2];
    let prompt = concatenate(&[prefix, zeros::<f32>(&[1, 100, t - r])?.as_dtype(Dtype::Float16)?], 2)?;
    let mu = load(&cfm.join("mu.npy"))?;
    for (name, v) in est.diagnostic_teacher_stages(&src, &mu, &x, &prompt)? {
        v.eval()?;
        v.save_numpy(out.join(format!("estimator-stage-{name}.npy")))?;
    }
    Ok(())
}
fn main() -> Result<()> { mlx_rs::with_device(Device::gpu(), run) }
