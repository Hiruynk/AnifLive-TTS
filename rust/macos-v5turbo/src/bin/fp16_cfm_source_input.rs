//! Same-source-input FP16 DiT step-zero diagnostic; no neural runtime changes.
use aniflive_macos_v5turbo::estimator::Estimator;
use mlx_rs::{
    ops::{concatenate, zeros},
    Array, Device, DeviceType, Dtype, Stream,
};
use serde_json::{json, Value};
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
fn input(trace: &Path, manifest: &Value, name: &str) -> Result<Array> {
    let key = format!("cfm/{name}");
    let path = trace.join(format!("{key}.npy"));
    if manifest["arrays"][&key]["sha256"].as_str() != Some(sha(&path)?.as_str())
        || manifest["arrays"][&key]["dtype"] != "torch.float16"
        || manifest["arrays"][&key]["device"] != "mps:0"
    {
        return Err(format!("Invalid captured source tensor {key}").into());
    }
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let value = mlx_rs::with_stream(&cpu, || Array::load_numpy(&path))?;
    if value.dtype() != Dtype::Float16 {
        return Err(format!("{key} is not F16").into());
    }
    Ok(value)
}
fn load_weights(path: &Path) -> Result<HashMap<String, Array>> {
    let cpu = Stream::try_default_on_device(&Device::cpu())?;
    let raw = mlx_rs::with_stream(&cpu, || Array::load_safetensors(path))?;
    let mut weights = HashMap::new();
    for (name, value) in raw {
        weights.insert(name, value.as_dtype(Dtype::Float16)?);
    }
    Ok(weights)
}
fn run() -> Result<()> {
    let a: Vec<_> = std::env::args_os().collect();
    if a.len() != 4 && !(a.len() == 5 && a[4] == "--all-valid-mask") {
        return Err("Usage: fp16_cfm_source_input S2.safetensors SOURCE_TRACE_DIR FRESH_OUTPUT_DIR [--all-valid-mask]".into());
    }
    if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
        return Err("Explicit MLX_ENABLE_TF32=0 is required".into());
    }
    if !matches!(
        Stream::thread_local_or_default().device()?.get_type()?,
        DeviceType::Gpu
    ) {
        return Err("FP16 CFM source-input probe requires Metal GPU".into());
    }
    let model_path = PathBuf::from(&a[1]);
    let trace = PathBuf::from(&a[2]);
    let out = PathBuf::from(&a[3]);
    fs::create_dir(&out)?;
    let manifest_path = trace.join("trace-manifest.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest["kind"] != "official_public_controlled_mps_trace"
        || !manifest["error"].is_null()
        || manifest["counts"]["cfm_noise"] != 1
        || manifest["counts"]["estimator"] != 4
    {
        return Err("Source trace is not a complete single-window four-step CFM capture".into());
    }
    let state = input(&trace, &manifest, "step-0000-state")?;
    let prompt_prefix = input(&trace, &manifest, "prompt")?;
    let mu = input(&trace, &manifest, "mu")?;
    let source_velocity = input(&trace, &manifest, "step-0000-velocity")?;
    if state.ndim() != 3
        || state.shape()[0] != 1
        || state.shape()[1] != 100
        || state.shape()[2] <= 0
    {
        return Err("Source state must be [1,100,T]".into());
    }
    let t = state.shape()[2];
    if mu.shape() != [1, t, 512]
        || source_velocity.shape() != [1, t, 100]
        || prompt_prefix.ndim() != 3
        || prompt_prefix.shape()[0] != 1
        || prompt_prefix.shape()[1] != 100
        || prompt_prefix.shape()[2] <= 0
        || prompt_prefix.shape()[2] >= t
    {
        return Err("Incompatible source CFM tensor geometries".into());
    }
    let reference = prompt_prefix.shape()[2];
    // Source CFM.inference creates a zero-filled prompt_x and copies this
    // recorded prompt prefix; no values or model weights are approximated.
    let prompt = concatenate(
        &[
            prompt_prefix,
            zeros::<f32>(&[1, 100, t - reference])?.as_dtype(Dtype::Float16)?,
        ],
        2,
    )?;
    let time = Array::from_slice(&[0.0_f32], &[1]).as_dtype(Dtype::Float16)?;
    let all_valid = if a.len() == 5 {
        Some(Array::from_slice(&vec![true; t as usize], &[1, t]))
    } else {
        None
    };
    let estimator = Estimator::load(&load_weights(&model_path)?)?;
    let velocity = estimator.forward(&state, &prompt, &mu, &time, all_valid.as_ref())?;
    if velocity.shape() != [1, 100, t] || velocity.dtype() != Dtype::Float16 {
        return Err("Native velocity has unexpected shape or dtype".into());
    }
    velocity.eval()?;
    let source_layout = velocity.transpose_axes(&[0, 2, 1])?.contiguous()?;
    source_layout.eval()?;
    let ncl = out.join("native-velocity-ncl.npy");
    let ntc = out.join("native-velocity-source-layout.npy");
    velocity.save_numpy(&ncl)?;
    source_layout.save_numpy(&ntc)?;
    let report = json!({"kind":"fp16_cfm_source_input_step0","qualified":false,
        "scope":"one current native Estimator.forward using captured source step-zero state, conditioning and time; no solver or vocoder",
        "backend":"mlx-metal","precision":"float16","tf32":"0","step":0,"time":0.0,"time_dtype":"float16",
        "prompt_construction":"captured prompt prefix followed by F16 zero target frames, as official CFM.inference",
        "mask_mode":if all_valid.is_some(){"all-valid Boolean mask"}else{"none, as current native production"},
        "source_mask_note":"official source constructs an all-valid mask from sequence length; optional mask isolates this difference",
        "frames":t,"reference_frames":reference,"model_sha256":sha(&model_path)?,"source_trace_manifest_sha256":sha(&manifest_path)?,
        "source_tensor_sha256":{"state":sha(&trace.join("cfm/step-0000-state.npy"))?,"prompt_prefix":sha(&trace.join("cfm/prompt.npy"))?,"mu":sha(&trace.join("cfm/mu.npy"))?,"source_velocity":sha(&trace.join("cfm/step-0000-velocity.npy"))?},
        "outputs":{"native_ncl":{"path":ncl,"sha256":sha(&ncl)?},"native_source_layout":{"path":ntc,"sha256":sha(&ntc)?}},
        "production_math_changed":false,"quality_claim":false});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
fn main() -> Result<()> {
    mlx_rs::with_device(Device::gpu(), run)
}
