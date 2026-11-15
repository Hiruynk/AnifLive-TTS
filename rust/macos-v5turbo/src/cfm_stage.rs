//! Isolated, synchronous CFM process. IPC files are a research prototype.
use crate::{
    cfm_compiled::{CompiledCfm, Metadata},
    estimator::Estimator,
    reference_cache::sha256_file,
};
use mlx_rs::{ops::select, thread_local_default_stream, Array, Device, Dtype, Stream};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(s: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, s.into()).into()
}

#[derive(Deserialize)]
#[serde(tag = "op", deny_unknown_fields)]
enum Request {
    #[serde(rename = "infer")]
    Infer {
        request_id: String,
        input_dir: PathBuf,
        output_dir: PathBuf,
        reference_frames: i32,
    },
    #[serde(rename = "shutdown")]
    Shutdown { request_id: String },
}
fn validate(mu: &Array, prompt: &Array, noise: &Array, r: i32) -> Result<i32> {
    let shape = mu.shape();
    if mu.dtype() != Dtype::Float32 || shape.len() != 3 || shape[0] != 1 || shape[2] != 512 {
        return Err(invalid("mu must be float32 [1,N,512]"));
    }
    let n = shape[1];
    if n > 1000
        || r <= 0
        || r >= n
        || prompt.dtype() != Dtype::Float32
        || noise.dtype() != Dtype::Float32
        || prompt.shape() != [1, 100, n]
        || noise.shape() != [1, 100, n]
    {
        return Err(invalid("invalid CFM shapes or reference_frames"));
    }
    for a in [mu, prompt, noise] {
        if !a.is_finite()?.all(false)?.item_exact::<bool>() {
            return Err(invalid("nonfinite CFM input"));
        }
    }
    Ok(n)
}
// ABI from mlx/c/compile.h; only the isolated compiled child owns this mode.
unsafe extern "C" {
    fn mlx_set_compile_mode(mode: std::ffi::c_int) -> std::ffi::c_int;
}
fn compiled_setting() -> Result<bool> {
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_COMPILED") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => Err(invalid("ANIFLIVE_TTS_NATIVE_CFM_COMPILED must be 0 or 1")),
    }
}
fn profiling_setting() -> Result<bool> {
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_PROFILE") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => Err(invalid("ANIFLIVE_TTS_NATIVE_CFM_PROFILE must be 0 or 1")),
    }
}
fn batched_qkv_setting() -> Result<bool> {
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        _ => Err(invalid("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV must be 0 or 1")),
    }
}
fn profile_elapsed(start: Option<Instant>) -> f64 {
    start
        .map(|value| value.elapsed().as_secs_f64())
        .unwrap_or(0.0)
}
fn validate_child_profile(value: &Value, compiled_enabled: bool) -> Result<()> {
    if !value.is_object()
        || value["compiled_cfm"] != compiled_enabled
        || value["inference_call_includes_device_eval"] != compiled_enabled
        || value["components_are_sequential_wall_intervals"] != true
    {
        return Err(invalid("invalid CFM child profile identity"));
    }
    for name in [
        "input_load_validation_seconds",
        "output_directory_seconds",
        "inference_call_seconds",
        "output_eval_validation_seconds",
        "output_save_seconds",
        "child_infer_total_seconds",
    ] {
        let Some(seconds) = value[name].as_f64() else {
            return Err(invalid(format!(
                "missing CFM child profile component {name}"
            )));
        };
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(invalid(format!(
                "invalid CFM child profile component {name}"
            )));
        }
    }
    Ok(())
}
fn load(path: &Path) -> Result<Array> {
    Ok(mlx_rs::with_device(Device::cpu(), || {
        Array::load_numpy(path)
    })?)
}
fn infer(
    est: &Estimator,
    compiled: Option<&mut CompiledCfm>,
    input: &Path,
    output: &Path,
    r: i32,
    profiling: bool,
) -> Result<(Option<Metadata>, Option<Value>)> {
    let total_started = profiling.then(Instant::now);
    let input_started = profiling.then(Instant::now);
    let mu = load(&input.join("mu.npy"))?;
    let prompt = load(&input.join("prompt.npy"))?;
    let noise = load(&input.join("noise.npy"))?;
    let n = validate(&mu, &prompt, &noise, r)?;
    let input_load_validation_seconds = profile_elapsed(input_started);
    let directory_started = profiling.then(Instant::now);
    if output.exists() {
        if !output.is_dir() || fs::read_dir(output)?.next().is_some() {
            return Err(invalid("output directory must be fresh or empty"));
        }
    } else {
        fs::create_dir(output)?;
    }
    let output_directory_seconds = profile_elapsed(directory_started);
    let inference_started = profiling.then(Instant::now);
    let compiled_path = compiled.is_some();
    let (x, metadata) = if let Some(compiled) = compiled {
        // The first request at each new geometry includes compilation; only one
        // geometry is retained. Failure propagates without an eager fallback.
        let (x, metadata) = compiled.infer(&mu, &prompt, &noise, r)?;
        (x, Some(metadata))
    } else {
        let mask = Array::from_slice(&(0..n).map(|i| i >= r).collect::<Vec<_>>(), &[1, 1, n]);
        let mut x = select(
            &mask,
            noise.multiply(Array::from_f32(0.875))?,
            Array::from_f32(0.0),
        )?;
        for step in 0..4 {
            let time = Array::from_slice(&[step as f32 * 0.25], &[1]);
            let v = est.forward(&x, &prompt, &mu, &time, None)?;
            if v.dtype() != Dtype::Float32 {
                return Err(invalid("CFM velocity is not float32"));
            }
            x = select(
                &mask,
                x.add(v.multiply(Array::from_f32(0.25))?)?,
                Array::from_f32(0.0),
            )?;
            if x.dtype() != Dtype::Float32 {
                return Err(invalid("CFM state is not float32"));
            }
        }
        (x, None)
    };
    let inference_call_seconds = profile_elapsed(inference_started);
    let output_check_started = profiling.then(Instant::now);
    x.eval()?;
    if x.dtype() != Dtype::Float32
        || x.shape() != [1, 100, n]
        || !x.is_finite()?.all(false)?.item_exact::<bool>()
    {
        return Err(invalid("invalid CFM output"));
    }
    let output_eval_validation_seconds = profile_elapsed(output_check_started);
    let save_started = profiling.then(Instant::now);
    x.save_numpy(output.join("x.npy"))?;
    let output_save_seconds = profile_elapsed(save_started);
    let profile = if profiling {
        Some(json!({
            "scope": "child CfmStage::infer request wall intervals; includes input NPY reads, validation, CFM call/eval and output NPY save; not GPU-only attribution",
            "compiled_cfm": compiled_path,
            "input_load_validation_seconds": input_load_validation_seconds,
            "output_directory_seconds": output_directory_seconds,
            "inference_call_seconds": inference_call_seconds,
            "inference_call_includes_device_eval": compiled_path,
            "output_eval_validation_seconds": output_eval_validation_seconds,
            "output_save_seconds": output_save_seconds,
            "child_infer_total_seconds": profile_elapsed(total_started),
            "components_are_sequential_wall_intervals": true
        }))
    } else {
        None
    };
    Ok((metadata, profile))
}
fn emit(v: &Value) -> Result<()> {
    let mut o = io::stdout().lock();
    serde_json::to_writer(&mut o, v)?;
    writeln!(o)?;
    o.flush()?;
    Ok(())
}
/// Run on the actual GPU default stream; all diagnostics go to stderr, protocol to stdout.
pub fn run(model_dir: &Path) -> Result<()> {
    let stream = Stream::try_default_on_device(&Device::gpu())?;
    mlx_rs::with_stream(&stream, || run_on_gpu(model_dir))
}
fn run_on_gpu(model_dir: &Path) -> Result<()> {
    let flag = std::env::var("MLX_ENABLE_TF32")?;
    if flag != "0" && flag != "1" {
        return Err(invalid("MLX_ENABLE_TF32 must be 0 or 1"));
    }
    let compiled_enabled = compiled_setting()?;
    let profiling = profiling_setting()?;
    let batched_qkv = batched_qkv_setting()?;
    if batched_qkv && !compiled_enabled {
        return Err(invalid("Batched CFM QKV requires compiled CFM"));
    }
    if compiled_enabled {
        if flag != "1" {
            return Err(invalid("compiled CFM requires MLX_ENABLE_TF32=1"));
        }
        if std::env::var_os("MLX_DISABLE_COMPILE").is_some() {
            return Err(invalid("compiled CFM requires compilation enabled"));
        }
        // This child process owns ENABLED=3 and never changes it thereafter.
        let status = unsafe { mlx_set_compile_mode(3) };
        if status != 0 {
            return Err(invalid(format!(
                "failed to enable MLX compilation: {status}"
            )));
        }
    }
    let gpu = Device::gpu();
    let stream = thread_local_default_stream()
        .ok_or_else(|| invalid("missing scoped GPU default stream"))?;
    if stream.device()? != gpu || stream != Stream::try_default_on_device(&gpu)? {
        return Err(invalid("CFM requires GPU default stream"));
    }
    // Force an operation on the selected stream before announcing readiness.
    Array::from_f32(1.0).add(Array::from_f32(1.0))?.eval()?;
    let model = model_dir.join("s2Gv5turbo.safetensors");
    let hash = sha256_file(&model)?;
    let all = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&model))?;
    let mut weights = HashMap::new();
    for (name, value) in all {
        if name.starts_with("cfm.estimator.") {
            weights.insert(name, value.as_dtype(Dtype::Float32)?);
        }
    }
    if weights.is_empty() {
        return Err(invalid("missing estimator weights"));
    }
    if compiled_enabled {
        mlx_rs::transforms::eval(weights.values())?;
    }
    let mut est = Estimator::load(&weights)?;
    if batched_qkv {
        est.enable_batched_qkv()?;
    }
    let packed_blocks = est.packed_qkv_blocks();
    if est.batched_qkv_enabled() != batched_qkv || packed_blocks != if batched_qkv { 22 } else { 0 } {
        return Err(invalid("CFM QKV layout was not configured as requested"));
    }
    let est = Rc::new(est);
    let mut compiled = if compiled_enabled {
        Some(CompiledCfm::new(est.clone())?)
    } else {
        None
    };
    emit(
        &json!({"event":"ready","backend":"mlx-metal","storage_precision":"float32","tf32":flag,"cfm_steps":4,"cfg":0,"model_sha256":hash,"pid":std::process::id(),"compiled_cfm":compiled_enabled,"cfm_profile_enabled":profiling,"cfm_batched_qkv":batched_qkv,"cfm_packed_qkv_blocks":packed_blocks}),
    )?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let req: Request = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                emit(&json!({"event":"error","request_id":null,"message":e.to_string()}))?;
                continue;
            }
        };
        match req {
            Request::Infer {
                request_id,
                input_dir,
                output_dir,
                reference_frames,
            } => match infer(
                &est,
                compiled.as_mut(),
                &input_dir,
                &output_dir,
                reference_frames,
                profiling,
            ) {
                Ok((metadata, profile)) => {
                    let mut event = json!({"event":"complete","request_id":request_id,"output_path":output_dir.join("x.npy"),"compiled_cfm":compiled_enabled,"cfm_batched_qkv":batched_qkv,"cfm_packed_qkv_blocks":packed_blocks,
                        "compiled_cfm_metadata":metadata.map(|m| json!({"frames":m.frames,"reference_frames":m.reference_frames,"rebuilds":m.rebuilds,"traces":m.traces,"total_traces":m.total_traces}))});
                    if let Some(profile) = profile {
                        event["profile"] = profile;
                    }
                    emit(&event)?;
                }
                Err(e) => {
                    emit(&json!({"event":"error","request_id":request_id,"message":e.to_string()}))?
                }
            },
            Request::Shutdown { request_id } => {
                emit(&json!({"event":"shutdown","request_id":request_id}))?;
                break;
            }
        }
    }
    Ok(())
}
static COUNTER: AtomicU64 = AtomicU64::new(0);
pub struct CfmStage {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    root: PathBuf,
    next: u64,
    healthy: bool,
    compiled_enabled: bool,
    batched_qkv: bool,
    profile_enabled: bool,
    last_profile: Option<Value>,
}
impl CfmStage {
    pub fn spawn(binary: &Path, model_dir: &Path, work_root: &Path, tf32: bool) -> Result<Self> {
        let compiled_enabled = compiled_setting()?;
        let profile_enabled = profiling_setting()?;
        let batched_qkv = batched_qkv_setting()?;
        if batched_qkv && (!compiled_enabled || !tf32) {
            return Err(invalid("Batched CFM QKV requires a compiled TF32-enabled child"));
        }
        if compiled_enabled && !tf32 {
            return Err(invalid("compiled CFM requires TF32"));
        }
        if compiled_enabled && std::env::var_os("MLX_DISABLE_COMPILE").is_some() {
            return Err(invalid("compiled CFM requires compilation enabled"));
        }
        let hash = sha256_file(model_dir.join("s2Gv5turbo.safetensors"))?;
        let root = work_root.join("cfm-stage");
        fs::create_dir_all(&root)?;
        let mut child = Command::new(binary)
            .arg("cfm-worker")
            .arg(model_dir)
            .env("MLX_ENABLE_TF32", if tf32 { "1" } else { "0" })
            .env(
                "ANIFLIVE_TTS_NATIVE_CFM_COMPILED",
                if compiled_enabled { "1" } else { "0" },
            )
            .env(
                "ANIFLIVE_TTS_NATIVE_CFM_PROFILE",
                if profile_enabled { "1" } else { "0" },
            )
            .env("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV", if batched_qkv { "1" } else { "0" })
            .env_remove("ANIFLIVE_COMPONENT_CPU_REFERENCE")
            .env_remove("CPU_REFERENCE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| invalid("missing child stdin"))?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| invalid("missing child stdout"))?,
        );
        let mut stage = Self {
            child,
            stdin,
            stdout,
            root,
            next: 0,
            healthy: true,
            compiled_enabled,
            batched_qkv,
            profile_enabled,
            last_profile: None,
        };
        let ready = stage.read()?;
        if ready["event"] != "ready"
            || ready["backend"] != "mlx-metal"
            || ready["storage_precision"] != "float32"
            || ready["tf32"] != if tf32 { "1" } else { "0" }
            || ready["cfm_steps"] != 4
            || ready["cfg"] != 0
            || ready["compiled_cfm"] != compiled_enabled
            || ready["model_sha256"] != hash
            || ready["pid"].as_u64() != Some(stage.child.id() as u64)
            || ready["cfm_profile_enabled"] != profile_enabled
            || ready["cfm_batched_qkv"] != batched_qkv
            || ready["cfm_packed_qkv_blocks"].as_u64() != Some(if batched_qkv { 22 } else { 0 })
        {
            return Err(invalid("invalid CFM stage ready event"));
        }
        Ok(stage)
    }
    fn read(&mut self) -> Result<Value> {
        let mut line = String::new();
        let result = (|| -> Result<Value> {
            if self.stdout.read_line(&mut line)? == 0 {
                return Err(invalid("CFM stage EOF"));
            }
            Ok(serde_json::from_str(&line)?)
        })();
        if result.is_err() {
            self.healthy = false;
        }
        result
    }
    fn send(&mut self, v: &Value) -> Result<Value> {
        if !self.healthy {
            return Err(invalid("CFM stage unavailable"));
        }
        if let Err(e) = serde_json::to_writer(&mut self.stdin, v)
            .map_err(io::Error::other)
            .and_then(|_| writeln!(self.stdin))
            .and_then(|_| self.stdin.flush())
        {
            self.healthy = false;
            return Err(invalid(format!("CFM stage request transport failed: {e}")));
        }
        self.read()
    }
    pub fn compiled_enabled(&self) -> bool {
        self.compiled_enabled
    }
    pub fn batched_qkv_enabled(&self) -> bool {
        self.batched_qkv
    }
    pub fn profile_enabled(&self) -> bool {
        self.profile_enabled
    }
    /// Timings from the last successful profiled inference; absent when profiling is off.
    pub fn last_profile(&self) -> Option<Value> {
        self.last_profile.clone()
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn healthy(&mut self) -> bool {
        if self.healthy {
            match self.child.try_wait() {
                Ok(None) => (),
                _ => self.healthy = false,
            }
        }
        self.healthy
    }
    pub fn infer(
        &mut self,
        mu: &Array,
        prompt: &Array,
        noise: &Array,
        reference_frames: i32,
    ) -> Result<Array> {
        self.last_profile = None;
        let overall_started = self.profile_enabled.then(Instant::now);
        let validation_started = self.profile_enabled.then(Instant::now);
        let n = validate(mu, prompt, noise, reference_frames)?;
        let input_validation_seconds = profile_elapsed(validation_started);
        let id = format!(
            "{}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
            self.next
        );
        self.next += 1;
        let dir = self.root.join(&id);
        let directory_started = self.profile_enabled.then(Instant::now);
        fs::create_dir(&dir)?;
        let request_directory_seconds = profile_elapsed(directory_started);
        let mut parent_input_save_seconds = 0.0;
        let mut parent_round_trip_seconds = 0.0;
        let mut parent_response_validation_seconds = 0.0;
        let mut parent_output_load_validation_seconds = 0.0;
        let mut child_profile: Option<Value> = None;
        let result = (|| -> Result<Array> {
            let input = dir.join("in");
            let output = dir.join("out");
            fs::create_dir(&input)?;
            let input_save_started = self.profile_enabled.then(Instant::now);
            mu.save_numpy(input.join("mu.npy"))?;
            prompt.save_numpy(input.join("prompt.npy"))?;
            noise.save_numpy(input.join("noise.npy"))?;
            parent_input_save_seconds = profile_elapsed(input_save_started);
            let round_trip_started = self.profile_enabled.then(Instant::now);
            let response = self.send(&json!({"op":"infer","request_id":id,"input_dir":input,"output_dir":output,"reference_frames":reference_frames}))?;
            parent_round_trip_seconds = profile_elapsed(round_trip_started);
            let response_validation_started = self.profile_enabled.then(Instant::now);
            if response["request_id"] != id {
                self.healthy = false;
                return Err(invalid("CFM response request ID mismatch"));
            }
            if response["event"] == "error" {
                if let Some(message) = response["message"].as_str() {
                    return Err(invalid(format!("CFM inference: {message}")));
                }
                self.healthy = false;
                return Err(invalid("invalid CFM error response"));
            }
            let expected = output.join("x.npy");
            if response["event"] != "complete"
                || response["output_path"].as_str().map(Path::new) != Some(expected.as_path())
            {
                self.healthy = false;
                return Err(invalid("invalid CFM completion"));
            }
            let metadata = &response["compiled_cfm_metadata"];
            let mode_valid = response["compiled_cfm"] == self.compiled_enabled;
            let qkv_mode_valid = response["cfm_batched_qkv"] == self.batched_qkv
                && response["cfm_packed_qkv_blocks"].as_u64() == Some(if self.batched_qkv { 22 } else { 0 });
            let metadata_valid = if self.compiled_enabled {
                metadata.is_object()
                    && metadata["frames"].as_i64() == Some(i64::from(n))
                    && metadata["reference_frames"].as_i64() == Some(i64::from(reference_frames))
                    && metadata["traces"].as_u64() == Some(1)
                    && metadata["rebuilds"].as_u64().is_some_and(|v| v > 0)
                    && match (
                        metadata["rebuilds"].as_u64(),
                        metadata["total_traces"].as_u64(),
                    ) {
                        (Some(rebuilds), Some(total)) => total >= rebuilds,
                        _ => false,
                    }
            } else {
                metadata.is_null()
            };
            if !mode_valid || !qkv_mode_valid || !metadata_valid {
                self.healthy = false;
                return Err(invalid("invalid CFM mode or compilation metadata"));
            }
            if self.profile_enabled {
                let Some(profile) = response.get("profile") else {
                    self.healthy = false;
                    return Err(invalid("CFM profile response is missing"));
                };
                if let Err(error) = validate_child_profile(profile, self.compiled_enabled) {
                    self.healthy = false;
                    return Err(error);
                }
                child_profile = Some(profile.clone());
            } else if response.get("profile").is_some() {
                self.healthy = false;
                return Err(invalid("CFM profile returned while profiling is disabled"));
            }
            parent_response_validation_seconds = profile_elapsed(response_validation_started);
            let output_validation_started = self.profile_enabled.then(Instant::now);
            let checked = (|| -> Result<Array> {
                let x = load(&expected)?;
                if x.dtype() != Dtype::Float32
                    || x.shape() != [1, 100, n]
                    || !x.is_finite()?.all(false)?.item_exact::<bool>()
                {
                    return Err(invalid("invalid CFM output"));
                }
                x.eval()?;
                Ok(x)
            })();
            if checked.is_err() {
                self.healthy = false;
            }
            let x = checked?;
            parent_output_load_validation_seconds = profile_elapsed(output_validation_started);
            Ok(x)
        })();
        let cleanup_started = self.profile_enabled.then(Instant::now);
        let cleanup = fs::remove_dir_all(&dir);
        let parent_workspace_cleanup_seconds = profile_elapsed(cleanup_started);
        match (result, cleanup) {
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(e.into()),
            (Ok(x), Ok(())) => {
                if self.profile_enabled {
                    let child_profile = child_profile
                        .ok_or_else(|| invalid("missing validated CFM child profile"))?;
                    self.last_profile = Some(json!({
                        "scope": "parent CfmStage::infer wall intervals; complete transport round trip includes child processing; not device-only attribution",
                        "parent": {
                            "overall_infer_wall_seconds": profile_elapsed(overall_started),
                            "input_validation_seconds": input_validation_seconds,
                            "request_directory_setup_seconds": request_directory_seconds,
                            "input_npy_save_seconds": parent_input_save_seconds,
                            "request_round_trip_seconds": parent_round_trip_seconds,
                            "response_validation_seconds": parent_response_validation_seconds,
                            "output_npy_load_validation_eval_seconds": parent_output_load_validation_seconds,
                            "workspace_cleanup_seconds": parent_workspace_cleanup_seconds
                        },
                        "child": child_profile,
                        "timing_relationship": {
                            "parent_request_round_trip_includes_child_infer_total": true,
                            "parent_request_round_trip_and_child_infer_overlap": true,
                            "parent_components_other_than_round_trip_are_sequential": true,
                            "component_seconds_are_not_additive_across_parent_and_child": true
                        }
                    }));
                }
                Ok(x)
            }
        }
    }
    pub fn shutdown(&mut self) -> Result<()> {
        if !self.healthy {
            return Err(invalid("CFM stage unavailable"));
        }
        let written = serde_json::to_writer(
            &mut self.stdin,
            &json!({"op":"shutdown","request_id":"shutdown"}),
        )
        .map_err(io::Error::other)
        .and_then(|_| writeln!(self.stdin))
        .and_then(|_| self.stdin.flush());
        self.healthy = false;
        written?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait()? {
                if !status.success() {
                    return Err(invalid(format!("CFM stage exited {status}")));
                }
                let response = self.read()?;
                if response["event"] != "shutdown" || response["request_id"] != "shutdown" {
                    return Err(invalid("invalid shutdown response"));
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(invalid("CFM shutdown timed out"));
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for CfmStage {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
