//! Same-input FP16 reference split diagnostic; no model or weight changes.
use aniflive_macos_v5turbo::{hubert::Hubert, semantic::SemanticEncoder};
use mlx_rs::{Array, Device, DeviceType, Dtype, Stream};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn sha(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = BufReader::new(fs::File::open(path)?);
    let mut bytes = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn load_weights(path: &Path) -> Result<HashMap<String, Array>> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let raw = mlx_rs::with_stream(&cpu, || Array::load_safetensors(path))?;
    let mut values = HashMap::new();
    for (name, value) in raw {
        values.insert(name, value.as_dtype(Dtype::Float16)?);
    }
    Ok(values)
}
fn load_array(path: &Path) -> Result<Array> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    Ok(mlx_rs::with_stream(&cpu, || Array::load_numpy(path))?)
}
fn require(value: &Array, shape: &[i32], dtype: Dtype, name: &str) -> Result<()> {
    if value.shape() != shape || value.dtype() != dtype {
        return Err(format!(
            "{name}: expected {shape:?} {dtype:?}, got {:?} {:?}",
            value.shape(),
            value.dtype()
        )
        .into());
    }
    Ok(())
}
fn run() -> Result<()> {
    let a: Vec<_> = std::env::args_os().collect();
    if a.len() != 6 && !(a.len() == 8 && a[6] == "--position-weight") {
        return Err("Usage: fp16_reference_split HUBERT.safetensors S2.safetensors MPS_WAVE.npy MPS_HIDDEN.npy FRESH_OUTPUT_DIR [--position-weight MPS_EFFECTIVE_WEIGHT.npy]".into());
    }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
        return Err("Explicit MLX_ENABLE_TF32=0 is required".into());
    }
    if !matches!(
        Stream::thread_local_or_default().device()?.get_type()?,
        DeviceType::Gpu
    ) {
        return Err("FP16 reference split requires MLX Metal GPU".into());
    }
    let hubert_path = PathBuf::from(&a[1]);
    let s2_path = PathBuf::from(&a[2]);
    let wave_path = PathBuf::from(&a[3]);
    let hidden_path = PathBuf::from(&a[4]);
    let out = PathBuf::from(&a[5]);
    fs::create_dir(&out)?;
    let hw = load_weights(&hubert_path)?;
    let sw = load_weights(&s2_path)?;
    let wave = load_array(&wave_path)?;
    let source_hidden = load_array(&hidden_path)?;
    require(&wave, &[1, 74080], Dtype::Float16, "MPS waveform")?;
    require(
        &source_hidden,
        &[1, 231, 768],
        Dtype::Float16,
        "MPS HuBERT hidden",
    )?;
    let v = hw
        .get("encoder.pos_conv_embed.conv.weight_v")
        .ok_or("missing position weight_v")?;
    let g = hw
        .get("encoder.pos_conv_embed.conv.weight_g")
        .ok_or("missing position weight_g")?;
    require(v, &[768, 48, 128], Dtype::Float16, "position weight_v")?;
    require(g, &[1, 1, 128], Dtype::Float16, "position weight_g")?;
    let position_path = if a.len() == 8 {
        Some(PathBuf::from(&a[7]))
    } else {
        None
    };
    let position_input = position_path.as_deref().map(load_array).transpose()?;
    if let Some(value) = &position_input {
        require(
            value,
            &[768, 48, 128],
            Dtype::Float16,
            "precomputed MPS position weight",
        )?;
    }
    let hubert = if let Some(value) = &position_input {
        Hubert::load_with_precomputed_position(&hw, value)?
    } else {
        Hubert::load(&hw)?
    };
    let semantic = SemanticEncoder::load(&sw)?;
    // Preserve the production lazy graph through final semantic codes before
    // materializing any intermediate. The second branch uses MPS hidden input.
    let native_hidden = hubert.forward(&wave)?;
    let (native_projection, native_codes) = semantic.forward(&native_hidden)?;
    native_codes.eval()?;
    let (source_projection, source_codes) = semantic.forward(&source_hidden)?;
    source_codes.eval()?;
    // The first expression exactly matches Hubert::load's current graph.
    // The second is an isolated FP32-opmath hypothesis, not an adopted fix.
    let denominator = v.square()?.sum_axes(&[0, 1], true)?.sqrt()?;
    let positional_half = v.multiply(g.divide(&denominator)?)?;
    let v32 = v.as_dtype(Dtype::Float32)?;
    let g32 = g.as_dtype(Dtype::Float32)?;
    let denom32 = v32.square()?.sum_axes(&[0, 1], true)?.sqrt()?;
    let positional_f32_opmath = v32
        .multiply(g32.divide(&denom32)?)?
        .as_dtype(Dtype::Float16)?;
    mlx_rs::transforms::eval([&positional_half, &positional_f32_opmath])?;
    require(
        &native_hidden,
        &[1, 231, 768],
        Dtype::Float16,
        "native hidden",
    )?;
    require(
        &native_projection,
        &[1, 115, 768],
        Dtype::Float16,
        "native projection",
    )?;
    require(
        &source_projection,
        &[1, 115, 768],
        Dtype::Float16,
        "source-hidden projection",
    )?;
    if native_codes.shape() != [1, 1, 115] || source_codes.shape() != [1, 1, 115] {
        return Err("Semantic code geometry differs".into());
    }
    require(
        &positional_half,
        &[768, 48, 128],
        Dtype::Float16,
        "position native",
    )?;
    require(
        &positional_f32_opmath,
        &[768, 48, 128],
        Dtype::Float16,
        "position F32 opmath",
    )?;
    let arrays = [
        ("native-hidden.npy", &native_hidden),
        ("native-projection.npy", &native_projection),
        ("native-codes.npy", &native_codes),
        ("source-hidden-projection.npy", &source_projection),
        ("source-hidden-codes.npy", &source_codes),
        ("positional-weight-half.npy", &positional_half),
        ("positional-weight-f32-opmath.npy", &positional_f32_opmath),
    ];
    let mut outputs = serde_json::Map::new();
    for (name, value) in arrays {
        let path = out.join(name);
        value.save_numpy(&path)?;
        outputs.insert(name.into(),json!({"sha256":sha(&path)?,"shape":value.shape(),"dtype":format!("{:?}",value.dtype())}));
    }
    let report = json!({"kind":"fp16_reference_same_input_split", "qualified":false,
        "neural_backend":"mlx-metal", "precision":"float16", "tf32":"0",
        "scope":"native full HuBERT→quantizer and captured MPS hidden→native quantizer; positional weight alternatives diagnostic only",
        "production_math_changed":false,"diagnostic_position_override":position_path.is_some(),
        "diagnostic_position_weight_sha256":position_path.as_deref().map(sha).transpose()?,"output_materialization":"native codes evaluated before any intermediate save; source codes evaluated separately before saves",
        "positional_layout":"source [768,48,128], before MLX convolution transpose",
        "sources":{"hubert":sha(&hubert_path)?,"s2":sha(&s2_path)?,"mps_waveform":sha(&wave_path)?,"mps_hidden":sha(&hidden_path)?},
        "outputs":outputs});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
fn main() -> Result<()> {
    mlx_rs::with_device(Device::gpu(), run)
}
