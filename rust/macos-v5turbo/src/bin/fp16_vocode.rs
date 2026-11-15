//! Diagnostic: native FP16 vocoder on given normalized CFM mel files [1,100,T] (F16 .npy).
use aniflive_macos_v5turbo::vocoder::Vocoder;
use mlx_rs::{Array, Device, Dtype, Stream};
use std::{collections::HashMap, error::Error, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn h(v: f32) -> Result<Array> { Ok(Array::from_slice(&[v], &[1]).as_dtype(Dtype::Float16)?) }
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { return Err("Usage: fp16_vocode VOCODER.safetensors MEL.npy...".into()); }
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let raw = mlx_rs::with_stream(&cpu, || Array::load_safetensors(&a[1]))?;
    let mut w = HashMap::new();
    for (k, v) in raw { w.insert(k, v.as_dtype(Dtype::Float16)?); }
    let voc = Vocoder::load(&w)?;
    for p in &a[2..] {
        let mel = mlx_rs::with_stream(&cpu, || Array::load_numpy(p))?;
        if mel.dtype() != Dtype::Float16 { return Err(format!("{p} not F16").into()); }
        let x = mel.add(h(1.0)?)?.divide(h(2.0)?)?.multiply(h(14.0)?)?.subtract(h(12.0)?)?;
        let wave = voc.forward(&x)?.as_dtype(Dtype::Float32)?;
        wave.eval()?;
        let stem = p.strip_suffix(".npy").ok_or("mel path must end with .npy")?;
        aniflive_macos_v5turbo::audio::write_pcm16(wave.as_slice::<f32>(), Path::new(&format!("{stem}.wav")), Path::new(&format!("{stem}.pcm")))?;
    }
    Ok(())
}
fn main() -> Result<()> { mlx_rs::with_device(Device::gpu(), run) }
