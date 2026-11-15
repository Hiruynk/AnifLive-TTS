//! Same-input FP16 HuBERT operator diagnostics; no production math changes.
use mlx_rs::{nn::gelu, ops::addmm, Array, Device, DeviceType, Dtype, Stream};
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
fn input(dir: &Path, manifest: &Value, name: &str, shape: &[i32]) -> Result<Array> {
    let path = dir.join(format!("{name}.npy"));
    if manifest["arrays"][name]["sha256"].as_str() != Some(sha(&path)?.as_str()) {
        return Err(format!("Source stage hash mismatch: {name}").into());
    }
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let value = mlx_rs::with_stream(&cpu, || Array::load_numpy(&path))?;
    if value.shape() != shape || value.dtype() != Dtype::Float16 {
        return Err(format!(
            "{name}: expected F16 {shape:?}, got {:?} {:?}",
            value.shape(),
            value.dtype()
        )
        .into());
    }
    Ok(value)
}
fn output(out: &Path, name: &str, value: &Array, shape: &[i32]) -> Result<Value> {
    if value.shape() != shape || value.dtype() != Dtype::Float16 {
        return Err(format!("{name} produced unexpected shape or precision").into());
    }
    let path = out.join(format!("{name}.npy"));
    value.save_numpy(&path)?;
    Ok(json!({"path":path,"sha256":sha(&path)?,"shape":value.shape(),"dtype":"float16"}))
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("Usage: fp16_hubert_ops SOURCE_STAGE_DIR FRESH_OUTPUT_DIR".into());
    }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
        return Err("Explicit MLX_ENABLE_TF32=0 is required".into());
    }
    if !matches!(
        Stream::thread_local_or_default().device()?.get_type()?,
        DeviceType::Gpu
    ) {
        return Err("FP16 HuBERT operator diagnostic requires MLX Metal GPU".into());
    }
    let source = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out)?;
    let manifest_path = source.join("report.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest["passed"] != true || manifest["kind"] != "official_cnhubert_mps_fp16_stage_capture"
    {
        return Err("Source stage manifest is not a successful official MPS FP16 capture".into());
    }
    let group = input(&source, &manifest, "groupnorm0", &[1, 14815, 512])?;
    let x = input(
        &source,
        &manifest,
        "projection-linear-input",
        &[1, 231, 512],
    )?;
    let weight = input(&source, &manifest, "projection-linear-weight", &[768, 512])?;
    let bias = input(&source, &manifest, "projection-linear-bias", &[768])?;
    // Both variants consume identical captured source values and preserve full
    // production geometry. Only final outputs are evaluated and saved.
    let gelu_direct = gelu(&group)?;
    let gelu_opmath = gelu(group.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)?;
    let gf = group.as_dtype(Dtype::Float32)?;
    // Pinned Torch MPS creates SQRT1_2 in the input dtype before graph fusion.
    let gelu_mps_half_constant = mlx_rs::ops::erf(gf.multiply(Array::from_f32(0.70703125))?)?
        .add(Array::from_f32(1.0))?.multiply(Array::from_f32(0.5))?
        .multiply(&gf)?.as_dtype(Dtype::Float16)?;
    let gelu_mps_f32_constant = mlx_rs::ops::erf(gf.multiply(Array::from_f32(std::f32::consts::FRAC_1_SQRT_2))?)?
        .add(Array::from_f32(1.0))?.multiply(Array::from_f32(0.5))?
        .multiply(&gf)?.as_dtype(Dtype::Float16)?;
    let weight_t = weight.transpose_axes(&[1, 0])?;
    let linear_separate = x.matmul(&weight_t)?.add(&bias)?;
    let linear_addmm = addmm(&bias, &x, &weight_t, None, None)?;
    mlx_rs::transforms::eval([&gelu_direct, &gelu_opmath, &linear_separate, &linear_addmm])?;
    mlx_rs::transforms::eval([&gelu_mps_half_constant, &gelu_mps_f32_constant])?;
    let mut outputs = serde_json::Map::new();
    for (name, value, shape) in [
        ("gelu-half-direct", &gelu_direct, &[1, 14815, 512][..]),
        ("gelu-mps-half-constant", &gelu_mps_half_constant, &[1, 14815, 512][..]),
        ("gelu-mps-f32-constant", &gelu_mps_f32_constant, &[1, 14815, 512][..]),
        ("gelu-f32-opmath-half", &gelu_opmath, &[1, 14815, 512][..]),
        (
            "projection-matmul-add",
            &linear_separate,
            &[1, 231, 768][..],
        ),
        ("projection-addmm", &linear_addmm, &[1, 231, 768][..]),
    ] {
        outputs.insert(name.into(), output(&out, name, value, shape)?);
    }
    let report = json!({"kind":"fp16_hubert_same_input_operator_probe","qualified":false,
        "source_stage_report_sha256":sha(&manifest_path)?,"source_stage_dir":source,
        "backend":"mlx-metal","precision":"float16","tf32":"0",
        "scope":"same captured MPS groupnorm output and projection input/weights; diagnostic only",
        "expected_source_outputs":{"gelu":"feature-0.npy","projection":"projection-linear-output.npy"},
        "linear_geometry":[1,231,512,768],"gelu_geometry":[1,14815,512],
        "materialization":"evaluate all four final outputs together before saving",
        "production_operator_changed":false,"outputs":outputs});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
fn main() -> Result<()> {
    mlx_rs::with_device(Device::gpu(), run)
}
