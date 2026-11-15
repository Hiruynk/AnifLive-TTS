//! Fixed synthetic same-input FP16 SDPA comparisons; no model modifications.
use mlx_rs::{
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{broadcast_to, select, softmax_axis},
    Array, Device, DeviceType, Dtype, Stream,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn sha(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = BufReader::new(fs::File::open(path)?);
    let mut b = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn load(dir: &Path, report: &Value, name: &str, shape: &[i32], dtype: Dtype) -> Result<Array> {
    let path = dir.join(format!("{name}.npy"));
    let digest = sha(&path)?;
    if report["arrays"][name]["sha256"].as_str() != Some(digest.as_str()) {
        return Err(format!("Source input hash mismatch: {name}").into());
    }
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let value = mlx_rs::with_stream(&cpu, || Array::load_numpy(&path))?;
    if value.shape() != shape || value.dtype() != dtype {
        return Err(format!(
            "Invalid input {name}: {:?} {:?}",
            value.shape(),
            value.dtype()
        )
        .into());
    }
    Ok(value)
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("Usage: fp16_sdpa_ops SOURCE_FIXTURE_DIR FRESH_OUTPUT_DIR".into());
    }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
        return Err("Explicit MLX_ENABLE_TF32=0 required".into());
    }
    if !matches!(
        Stream::thread_local_or_default().device()?.get_type()?,
        DeviceType::Gpu
    ) {
        return Err("Metal GPU required".into());
    }
    let source = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out)?;
    let report_path = source.join("report.json");
    let report: Value = serde_json::from_slice(&fs::read(&report_path)?)?;
    if report["passed"] != true
        || (report["kind"] != "synthetic_fp16_mps_sdpa_same_input_fixture" && report["kind"] != "actual_cfm_fp16_mps_sdpa_fixture")
        || report["scale"] != 0.125
    {
        return Err("Invalid source SDPA fixture metadata".into());
    }
    let q = load(&source, &report, "q", &[1, 16, 619, 64], Dtype::Float16)?;
    let k = load(&source, &report, "k", &[1, 16, 619, 64], Dtype::Float16)?;
    let v = load(&source, &report, "v", &[1, 16, 619, 64], Dtype::Float16)?;
    let base_mask = load(&source, &report, "mask", &[1, 1, 1, 619], Dtype::Bool)?;
    let mask = broadcast_to(&base_mask, &[1, 16, 619, 619])?;
    if !mask.all(false)?.try_item_exact::<bool>()? {
        return Err("Fixture mask must be all valid".into());
    }
    let fast = scaled_dot_product_attention(
        &q,
        &k,
        &v,
        0.125,
        Some(ScaledDotProductAttentionMask::Array(&mask)),
        None,
    )?;
    // Materialize the explicit half boundaries so lazy fusion cannot erase the
    // rounding being measured. precise=true selects FP32 softmax accumulation.
    let scores_half = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
    scores_half.eval()?;
    let scaled_half = scores_half.multiply(Array::from_f32(0.125).as_dtype(Dtype::Float16)?)?;
    scaled_half.eval()?;
    let masked_half = select(
        &mask,
        &scaled_half,
        Array::from_f32(f32::NEG_INFINITY).as_dtype(Dtype::Float16)?,
    )?;
    let probabilities_half = softmax_axis(&masked_half, -1, Some(true))?;
    probabilities_half.eval()?;
    let explicit_half = probabilities_half.matmul(&v)?;
    let q32 = q.as_dtype(Dtype::Float32)?;
    let k32 = k.as_dtype(Dtype::Float32)?;
    let v32 = v.as_dtype(Dtype::Float32)?;
    let scores32 = q32.matmul(k32.transpose_axes(&[0, 1, 3, 2])?)?;
    scores32.eval()?;
    let scaled32 = scores32.multiply(Array::from_f32(0.125))?;
    scaled32.eval()?;
    let masked32 = select(&mask, &scaled32, Array::from_f32(f32::NEG_INFINITY))?;
    let probabilities32 = softmax_axis(&masked32, -1, Some(true))?;
    probabilities32.eval()?;
    let explicit_f32 = probabilities32.matmul(&v32)?.as_dtype(Dtype::Float16)?;
    mlx_rs::transforms::eval([&fast, &explicit_half, &explicit_f32])?;
    let mut outputs = serde_json::Map::new();
    for (name, value) in [
        ("native-fast", &fast),
        ("native-explicit-half", &explicit_half),
        ("native-f32-opmath", &explicit_f32),
    ] {
        if value.shape() != [1, 16, 619, 64] || value.dtype() != Dtype::Float16 {
            return Err(format!("Invalid output {name}").into());
        }
        let path = out.join(format!("{name}.npy"));
        value.save_numpy(&path)?;
        outputs.insert(
            name.into(),
            json!({"path":path,"sha256":sha(&path)?,"shape":value.shape(),"dtype":"float16"}),
        );
    }
    let result = json!({"kind":"fp16_sdpa_same_input_operator_probe","qualified":false,"production_operator_changed":false,
        "source_fixture_report_sha256":sha(&report_path)?,"source_output_sha256":sha(&source.join("source-output.npy"))?,
        "scope":"synthetic Q/K/V; MPS source versus MLX fast, explicit half and F32 opmath; no model or quality claim",
        "backend":"mlx-metal","tf32":"0","shape":[1,16,619,64],"scale":0.125,
        "mask":"all-valid Boolean expanded to [1,16,619,619]","dropout":0,
        "explicit_half":"F16 QK matmul, F16 scale, precise softmax to F16, F16 PV matmul; materialized boundaries",
        "explicit_f32":"F32 QK, scale, precise softmax and PV, then F16 output; materialized boundaries",
        "kernel_dispatch":"not observed","outputs":outputs});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
fn main() -> Result<()> {
    mlx_rs::with_device(Device::gpu(), run)
}
