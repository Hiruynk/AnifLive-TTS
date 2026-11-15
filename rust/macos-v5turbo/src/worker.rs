//! Persistent local JSONL worker for the existing service adapter.
//!
//! `run` owns the model on its calling thread. The stdin thread only parses
//! control messages and toggles cancellation flags; it never accesses MLX.
use crate::{
    engine::Engine,
    frontend_models::FrontendModels,
    streaming::{StreamChunk, StreamConfig},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Output = Arc<Mutex<io::Stdout>>;
type SharedActive = Arc<Mutex<Option<ActiveRequest>>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    op: String,
    request_id: String,
    input_dir: Option<PathBuf>,
    audio: Option<PathBuf>,
    transcript: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    seed: Option<u64>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    core_frames: Option<i32>,
    future_frames: Option<i32>,
    holdback_frames: Option<i32>,
    fade_frames: Option<i32>,
    // Optional conditioning sources (expression policies). The main reference
    // supplies the AR semantic prompt; `style_*` supplies the global style
    // vector ge (reference spectrum) and `prompt_*` the CFM in-context acoustic
    // prompt (reference features and mel). Omitted groups follow the main one.
    style_input_dir: Option<PathBuf>,
    style_audio: Option<PathBuf>,
    style_transcript: Option<PathBuf>,
    prompt_input_dir: Option<PathBuf>,
    prompt_audio: Option<PathBuf>,
    prompt_transcript: Option<PathBuf>,
}
#[derive(Clone)]
struct RefSource {
    input: PathBuf,
    audio: PathBuf,
    transcript: PathBuf,
}
fn ref_source(
    input: Option<PathBuf>,
    audio: Option<PathBuf>,
    transcript: Option<PathBuf>,
    group: &str,
) -> std::result::Result<Option<RefSource>, String> {
    match (input, audio, transcript) {
        (None, None, None) => Ok(None),
        (input, audio, transcript) => Ok(Some(RefSource {
            input: path(input, &format!("{group}_input_dir"))?,
            audio: path(audio, &format!("{group}_audio"))?,
            transcript: path(transcript, &format!("{group}_transcript"))?,
        })),
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Synthesize,
    SynthesizeCached,
    Stream,
    PrepareReference,
    FrontendG2PW,
    FrontendBert,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Synthesize => "synthesize",
            Self::SynthesizeCached => "synthesize_cached",
            Self::Stream => "stream",
            Self::PrepareReference => "prepare_reference",
            Self::FrontendG2PW => "frontend_g2pw",
            Self::FrontendBert => "frontend_bert",
        }
    }
}
struct ActiveRequest {
    id: String,
    mode: Mode,
    stop: Arc<AtomicBool>,
}
struct Job {
    id: String,
    mode: Mode,
    input: PathBuf,
    audio: Option<PathBuf>,
    transcript: Option<PathBuf>,
    output: PathBuf,
    seed: u64,
    sampling: crate::sampling::SamplingParams,
    config: StreamConfig,
    stop: Arc<AtomicBool>,
    style: Option<RefSource>,
    prompt: Option<RefSource>,
}
enum Command {
    Job(Job),
    Shutdown(Option<String>),
    ReaderFailed(String),
}
enum Outcome {
    Complete(Value),
    Cancelled,
}

fn memory_snapshot() -> Result<Value> {
    Ok(json!({"active_bytes":mlx_rs::memory::active_memory()?,
              "peak_bytes":mlx_rs::memory::peak_memory()?,
              "cache_bytes":mlx_rs::memory::cache_memory()?,
              "scope":"MLX allocator only, not total process RSS"}))
}
fn lock_error() -> io::Error {
    io::Error::other("worker synchronization lock poisoned")
}
fn emit(output: &Output, value: &Value) -> Result<()> {
    let mut writer = output.lock().map_err(|_| lock_error())?;
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn error_event(id: Option<&str>, code: &str, message: &str) -> Value {
    json!({"event":"error", "request_id":id, "code":code, "message":message})
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
fn path(value: Option<PathBuf>, field: &str) -> std::result::Result<PathBuf, String> {
    match value {
        Some(value) if !value.as_os_str().is_empty() => Ok(value),
        _ => Err(format!("missing or empty {field}")),
    }
}
fn job(request: Request, mode: Mode) -> std::result::Result<Job, String> {
    if matches!(mode, Mode::FrontendG2PW | Mode::FrontendBert)
        && (request.audio.is_some()
            || request.transcript.is_some()
            || request.seed.is_some()
            || request.top_k.is_some()
            || request.top_p.is_some()
            || request.temperature.is_some()
            || request.core_frames.is_some()
            || request.future_frames.is_some()
            || request.holdback_frames.is_some()
            || request.fade_frames.is_some())
    {
        return Err("frontend operations accept only input_dir and output_dir".into());
    }
    if mode == Mode::PrepareReference
        && (request.seed.is_some()
            || request.top_k.is_some()
            || request.top_p.is_some()
            || request.temperature.is_some()
            || request.core_frames.is_some()
            || request.future_frames.is_some()
            || request.holdback_frames.is_some()
            || request.fade_frames.is_some())
    {
        return Err("reference preparation does not accept seed or stream geometry".into());
    }
    if matches!(mode, Mode::Synthesize | Mode::SynthesizeCached)
        && (request.core_frames.is_some()
            || request.future_frames.is_some()
            || request.holdback_frames.is_some()
            || request.fade_frames.is_some())
    {
        return Err("stream geometry is not supported for complete synthesis".into());
    }
    if mode == Mode::Synthesize && (request.audio.is_some() || request.transcript.is_some()) {
        return Err("complete synthesis uses prepared reference arrays; raw audio/transcript are not accepted in this mode".into());
    }
    let style = ref_source(request.style_input_dir, request.style_audio, request.style_transcript, "style")?;
    let prompt = ref_source(request.prompt_input_dir, request.prompt_audio, request.prompt_transcript, "prompt")?;
    if (style.is_some() || prompt.is_some()) && !matches!(mode, Mode::SynthesizeCached | Mode::Stream) {
        return Err("conditioning sources are accepted only by synthesize_cached and stream".into());
    }
    // Request-level AR sampling with the Windows-legal ranges; omitted
    // fields keep the default top_k=15, top_p=1, temperature=1.
    let default = crate::sampling::SamplingParams::DEFAULT;
    let sampling = crate::sampling::SamplingParams {
        top_k: request.top_k.unwrap_or(default.top_k),
        top_p: request.top_p.unwrap_or(default.top_p),
        temperature: request.temperature.unwrap_or(default.temperature),
    };
    sampling.validate().map_err(|error| error.to_string())?;
    let input = path(request.input_dir, "input_dir")?;
    let output = path(request.output_dir, "output_dir")?;
    let (audio, transcript) = if matches!(
        mode,
        Mode::Stream | Mode::PrepareReference | Mode::SynthesizeCached
    ) {
        (
            Some(path(request.audio, "audio")?),
            Some(path(request.transcript, "transcript")?),
        )
    } else {
        (request.audio, request.transcript)
    };
    let config = StreamConfig {
        core_frames: request.core_frames.unwrap_or(200),
        future_frames: request.future_frames.unwrap_or(12),
        holdback_frames: request.holdback_frames.unwrap_or(50),
        fade_frames: request.fade_frames.unwrap_or(4),
        capture_noise: false,
    };
    if mode == Mode::Stream
        && (!(20..=500).contains(&config.core_frames)
            || !(0..=50).contains(&config.future_frames)
            || !(16..=200).contains(&config.holdback_frames)
            || !(0..=20).contains(&config.fade_frames))
    {
        return Err("stream geometry outside supported fixed-mode ranges".into());
    }
    Ok(Job {
        id: request.request_id,
        mode,
        input,
        audio,
        transcript,
        output,
        seed: request.seed.unwrap_or(0),
        sampling,
        config,
        stop: Arc::new(AtomicBool::new(false)),
        style,
        prompt,
    })
}

/// Compose the request conditioning: semantic prompt from the main reference,
/// ge/spectrum from `style`, CFM features/mel from `prompt` (each optional).
/// Returns the reference and the cache keys actually used per component.
fn composed_reference(
    engine: &Engine,
    job: &Job,
    cache_root: &Path,
    main: crate::reference_cache::PreparedReference,
    main_key: &str,
) -> Result<(crate::reference_cache::PreparedReference, Value)> {
    let mut reference = main;
    let mut style_key = main_key.to_owned();
    let mut prompt_key = main_key.to_owned();
    if let Some(source) = &job.style {
        let (style, _, key) = engine.prepare_reference_cached(&source.input, &source.audio, &source.transcript, cache_root)?;
        reference.ge = style.ge;
        reference.spectrum = style.spectrum;
        style_key = key;
    }
    if let Some(source) = &job.prompt {
        let (prompt, _, key) = engine.prepare_reference_cached(&source.input, &source.audio, &source.transcript, cache_root)?;
        reference.features = prompt.features;
        reference.mel = prompt.mel;
        prompt_key = key;
    }
    reference.validate()?;
    Ok((reference, json!({"semantic_reference_key":main_key,"style_reference_key":style_key,
        "acoustic_prompt_reference_key":prompt_key})))
}

// Called only by the stdin thread; at most one model job can be admitted.
fn read_commands(output: &Output, active: &SharedActive, tx: &mpsc::Sender<Command>) -> Result<()> {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let request: Request = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                // Preserve correlation for a valid ID even when a different
                // field is malformed or an unsupported field was supplied.
                let id = serde_json::from_str::<Value>(&line)
                    .ok()
                    .and_then(|v| {
                        v.get("request_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .filter(|id| valid_id(id));
                emit(
                    output,
                    &error_event(id.as_deref(), "invalid_request", &error.to_string()),
                )?;
                continue;
            }
        };
        if !valid_id(&request.request_id) {
            emit(
                output,
                &error_event(
                    None,
                    "invalid_request_id",
                    "request_id must be 1-64 ASCII letters, digits, '.', '_' or '-'",
                ),
            )?;
            continue;
        }
        if matches!(request.op.as_str(), "cancel" | "shutdown")
            && (request.input_dir.is_some()
                || request.output_dir.is_some()
                || request.audio.is_some()
                || request.transcript.is_some()
                || request.seed.is_some()
                || request.top_k.is_some()
            || request.top_p.is_some()
            || request.temperature.is_some()
                || request.core_frames.is_some()
                || request.future_frames.is_some()
                || request.holdback_frames.is_some()
                || request.fade_frames.is_some())
        {
            emit(
                output,
                &error_event(
                    Some(&request.request_id),
                    "invalid_request",
                    "control operations accept only op and request_id",
                ),
            )?;
            continue;
        }
        match request.op.as_str() {
            "synthesize" | "synthesize_cached" | "stream" | "prepare_reference"
            | "frontend_g2pw" | "frontend_bert" => {
                let mode = match request.op.as_str() {
                    "synthesize_cached" => Mode::SynthesizeCached,
                    "stream" => Mode::Stream,
                    "prepare_reference" => Mode::PrepareReference,
                    "frontend_g2pw" => Mode::FrontendG2PW,
                    "frontend_bert" => Mode::FrontendBert,
                    _ => Mode::Synthesize,
                };
                let request_id = request.request_id.clone();
                let next = match job(request, mode) {
                    Ok(next) => next,
                    Err(message) => {
                        emit(
                            output,
                            &error_event(Some(&request_id), "invalid_request", &message),
                        )?;
                        continue;
                    }
                };
                let mut slot = active.lock().map_err(|_| lock_error())?;
                if slot.is_some() {
                    emit(
                        output,
                        &error_event(
                            Some(&next.id),
                            "busy",
                            "worker already has an active request",
                        ),
                    )?;
                    continue;
                }
                *slot = Some(ActiveRequest {
                    id: next.id.clone(),
                    mode,
                    stop: next.stop.clone(),
                });
                let mut accepted =
                    json!({"event":"accepted", "request_id":next.id, "mode":mode.name()});
                if matches!(
                    mode,
                    Mode::Synthesize | Mode::SynthesizeCached | Mode::Stream
                ) {
                    accepted["seed"] = json!(next.seed);
                }
                emit(output, &accepted)?;
                if tx.send(Command::Job(next)).is_err() {
                    *slot = None;
                    return Err("model owner thread is unavailable".into());
                }
            }
            "cancel" => {
                // Serialize the cancellation acknowledgement with PCM writes:
                // once acknowledged, a sink holding this lock cannot emit more.
                let slot = active.lock().map_err(|_| lock_error())?;
                match slot.as_ref() {
                    Some(current)
                        if current.id == request.request_id
                            && matches!(
                                current.mode,
                                Mode::Stream | Mode::Synthesize | Mode::SynthesizeCached
                            ) =>
                    {
                        current.stop.store(true, Ordering::Release);
                        emit(
                            output,
                            &json!({"event":"cancel_requested","request_id":request.request_id}),
                        )?;
                    }
                    Some(current) if current.id == request.request_id => {
                        emit(
                            output,
                            &json!({"event":"error", "request_id":request.request_id,
                            "operation":"cancel", "terminal":false, "code":"cancel_unsupported",
                            "message":match current.mode {
                                Mode::PrepareReference => "reference preparation does not support cancellation",
                                Mode::FrontendG2PW | Mode::FrontendBert => "frontend inference does not support cancellation",
                                _ => "complete synthesis does not support cancellation",
                            }}),
                        )?;
                    }
                    _ => emit(
                        output,
                        &error_event(
                            Some(&request.request_id),
                            "not_active",
                            "no matching active request",
                        ),
                    )?,
                }
            }
            "shutdown" => {
                let slot = active.lock().map_err(|_| lock_error())?;
                if let Some(current) = slot.as_ref() {
                    if matches!(
                        current.mode,
                        Mode::Stream | Mode::Synthesize | Mode::SynthesizeCached
                    ) {
                        current.stop.store(true, Ordering::Release);
                    }
                }
                emit(
                    output,
                    &json!({"event":"shutdown_requested","request_id":request.request_id}),
                )?;
                tx.send(Command::Shutdown(Some(request.request_id)))
                    .map_err(|_| "model owner thread is unavailable")?;
                return Ok(());
            }
            _ => emit(
                output,
                &error_event(
                    Some(&request.request_id),
                    "unknown_op",
                    "unknown worker operation",
                ),
            )?,
        }
    }
    // EOF is a shutdown request; active streams are cancelled, full synthesis
    // finishes because its engine path has no cancellation support.
    let slot = active.lock().map_err(|_| lock_error())?;
    if let Some(current) = slot.as_ref() {
        if current.mode == Mode::Stream {
            current.stop.store(true, Ordering::Release);
        }
    }
    tx.send(Command::Shutdown(None))
        .map_err(|_| "model owner thread is unavailable")?;
    Ok(())
}

fn process(
    engine: &Engine,
    frontend: &FrontendModels,
    cache_root: &Path,
    job: &Job,
    active: &SharedActive,
    output: &Output,
) -> Result<Outcome> {
    if job.mode == Mode::FrontendG2PW {
        return Ok(Outcome::Complete(frontend.g2pw(&job.input, &job.output)?));
    }
    if job.mode == Mode::FrontendBert {
        return Ok(Outcome::Complete(frontend.bert(&job.input, &job.output)?));
    }
    if job.mode == Mode::PrepareReference {
        std::fs::create_dir_all(&job.output)?;
        if std::fs::read_dir(&job.output)?.next().is_some() {
            return Err("Reference preparation output directory must be empty".into());
        }
        let (reference, hit, key) = engine.prepare_reference_cached(
            &job.input,
            job.audio.as_deref().ok_or("missing audio")?,
            job.transcript.as_deref().ok_or("missing transcript")?,
            cache_root,
        )?;
        reference.validate()?;
        let report_path = job.output.join("reference-preparation.json");
        let result = json!({"cache_key":key, "cache_hit":hit, "cache_root":cache_root,
            "reference_frames":reference.features.shape()[2],
            "reference_prompt_tokens":reference.prompt_codes.shape()[2],
            "precision":engine.precision_name(), "report_path":report_path});
        std::fs::write(&report_path, serde_json::to_vec_pretty(&result)?)?;
        return Ok(Outcome::Complete(result));
    }
    if job.mode == Mode::SynthesizeCached {
        let (reference, hit, key) = engine.prepare_reference_cached(
            &job.input,
            job.audio.as_deref().ok_or("missing audio")?,
            job.transcript.as_deref().ok_or("missing transcript")?,
            cache_root,
        )?;
        let (reference, conditioning) = composed_reference(engine, job, cache_root, reference, &key)?;
        engine.run_seeded_cached(&job.input, &job.output, job.seed, &reference, false)?;
        return Ok(Outcome::Complete(
            json!({"result_path":job.output.join("native-wave.wav"),"conditioning":conditioning,
            "sample_rate":48000,"format":"wav","cache_hit":hit,"cache_key":key,
            "reference_conditioning_reused":true,"stochastic_and_neural_capture":false,
            "top_k":job.sampling.top_k,"top_p":job.sampling.top_p,"temperature":job.sampling.temperature,"precision":engine.precision_name(),"cfm_noise_policy":if engine.coordinate_noise_v2_enabled() {crate::cfm_noise_field::POLICY} else {crate::rng::POLICY},"quality_qualified":false}),
        ));
    }
    if job.mode == Mode::Synthesize {
        engine.run_seeded(&job.input, &job.output, job.seed)?;
        return Ok(Outcome::Complete(
            json!({"result_path":job.output.join("native-wave.wav"),"sample_rate":48000,"format":"wav",
                "top_k":job.sampling.top_k,"top_p":job.sampling.top_p,"temperature":job.sampling.temperature,"precision":engine.precision_name(),"cfm_noise_policy":if engine.coordinate_noise_v2_enabled() {crate::cfm_noise_field::POLICY} else {crate::rng::POLICY},"quality_qualified":false}),
        ));
    }
    if job.stop.load(Ordering::Acquire) {
        return Ok(Outcome::Cancelled);
    }
    let (reference, hit, main_key) = engine.prepare_reference_cached(
        &job.input,
        job.audio.as_deref().ok_or("missing audio")?,
        job.transcript.as_deref().ok_or("missing transcript")?,
        cache_root,
    )?;
    let (reference, conditioning) = composed_reference(engine, job, cache_root, reference, &main_key)?;
    {
        let slot = active.lock().map_err(|_| lock_error())?;
        let current = slot.as_ref().ok_or("stream no longer owns worker")?;
        if current.id != job.id || current.mode != Mode::Stream {
            return Err("stream ownership changed unexpectedly".into());
        }
        if current.stop.load(Ordering::Acquire) {
            return Ok(Outcome::Cancelled);
        }
        emit(
            output,
            &json!({"event":"reference_ready","request_id":job.id,"cache_hit":hit}),
        )?;
    }
    let mut sink = |chunk: &StreamChunk| -> Result<()> {
        let slot = active.lock().map_err(|_| lock_error())?;
        let current = slot.as_ref().ok_or("stream no longer owns worker")?;
        if current.id != job.id || current.mode != Mode::Stream {
            return Err("stream ownership changed unexpectedly".into());
        }
        if current.stop.load(Ordering::Acquire) {
            // The decoder checks cancellation after this callback and does not
            // produce another window. Suppress this not-yet-delivered chunk.
            return Ok(());
        }
        let bytes: Vec<u8> = chunk
            .pcm
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        emit(
            output,
            &json!({"event":"pcm","request_id":job.id,"sample_rate":48000,
            "channels":1,"sample_format":"s16le","pcm_base64":BASE64.encode(bytes),
            "frame_end":chunk.frame_end,"before_eos":chunk.before_eos,"final_chunk":chunk.final_chunk,
            "original_total_samples":chunk.original_total_samples}),
        )?;
        Ok(())
    };
    engine.run_stream_seeded_with_sink(
        &reference,
        &job.input,
        &job.output,
        job.seed,
        job.config.clone(),
        &job.stop,
        false,
        &mut sink,
    )?;
    let report: Value =
        serde_json::from_slice(&std::fs::read(job.output.join("stream-report.json"))?)?;
    match (report["cancelled"].as_bool(), report["finished"].as_bool()) {
        (Some(true), _) => Ok(Outcome::Cancelled),
        (Some(false), Some(true)) => Ok(Outcome::Complete(
            json!({"result_path":job.output.join("stream.wav"),"conditioning":conditioning,"sample_rate":48000,"format":"wav",
                "top_k":job.sampling.top_k,"top_p":job.sampling.top_p,"temperature":job.sampling.temperature,"precision":engine.precision_name(),"cfm_noise_policy":report["noise_policy"].as_str().unwrap_or(crate::rng::POLICY),
                "quality_qualified":false}),
        )),
        _ => Err("stream ended without a truthful terminal state".into()),
    }
}

/// Run the worker on the calling thread. The caller selects the MLX/Metal
/// device before invoking this function. `Engine` never leaves this thread.
pub fn run(model_dir: &Path, cache_root: &Path) -> Result<()> {
    let compiled_cfm = match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_COMPILED").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        _ => return Err("ANIFLIVE_TTS_NATIVE_CFM_COMPILED must be 0 or 1".into()),
    };
    let precision = match std::env::var("ANIFLIVE_TTS_NATIVE_PRECISION").as_deref() {
        Ok("FP16") => mlx_rs::Dtype::Float16,
        Ok("FP32") | Err(std::env::VarError::NotPresent) => mlx_rs::Dtype::Float32,
        _ => return Err("ANIFLIVE_TTS_NATIVE_PRECISION must be FP32 or FP16".into()),
    };
    // Optional MLX buffer-cache ceiling. Freed GPU buffers above this size are
    // returned to the system instead of being retained (allocator policy only;
    // no numerical effect). Unset keeps MLX's default unbounded cache.
    let mlx_cache_limit_bytes = match std::env::var("ANIFLIVE_TTS_MLX_CACHE_LIMIT_MB") {
        Ok(value) => {
            let mb: usize = value
                .parse()
                .map_err(|_| "ANIFLIVE_TTS_MLX_CACHE_LIMIT_MB must be a nonnegative integer")?;
            let bytes = mb.checked_mul(1024 * 1024).ok_or("MLX cache limit overflows")?;
            mlx_rs::memory::set_cache_limit(bytes)?;
            Some(bytes)
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("ANIFLIVE_TTS_MLX_CACHE_LIMIT_MB must be valid UTF-8".into()),
    };
    let mut engine = Engine::load_with_precision(model_dir, precision)?;
    engine.set_cli_reports(false);
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_PREPARE").as_deref() {
        Ok("1") => engine.enable_prepared_cfm()?,
        Ok("0") | Err(_) => (),
        _ => return Err("ANIFLIVE_TTS_NATIVE_CFM_PREPARE must be 0 or 1".into()),
    }
    engine.configure_ar_from_env()?;
    // Worker requests select their own top-k; a missing field is always 15,
    // regardless of a CLI research setting inherited by this process.
    engine.set_sampling(crate::sampling::SamplingParams::DEFAULT)?;
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_STAGE").as_deref() {
        Ok("1") => engine.enable_cfm_stage(cache_root)?,
        Ok("0") | Err(_) => (),
        _ => return Err("ANIFLIVE_TTS_NATIVE_CFM_STAGE must be 0 or 1".into()),
    }
    if compiled_cfm && !engine.compiled_cfm_stage_enabled() {
        return Err("Compiled CFM requires an enabled isolated CFM stage".into());
    }
    engine.validate_batched_cfm_qkv_from_env()?;
    let frontend = FrontendModels::load(model_dir, precision)?;
    let output: Output = Arc::new(Mutex::new(io::stdout()));
    let active: SharedActive = Arc::new(Mutex::new(None));
    let mut ready = json!({"event":"ready","backend":"mlx-metal","precision":engine.precision_name(),
        "operational":true,"prototype":true,"qualified":false,"neural_fallback":false,
        "capabilities":{"synthesize":true,"synthesize_cached":true,"cfm_isolated_tf32":engine.cfm_stage_enabled(),"cfm_compiled":engine.compiled_cfm_stage_enabled(),"cfm_stage_pid":engine.cfm_stage_pid(),"cfm_prepared":engine.prepared_cfm_enabled(),"ar_combined":engine.combined_ar_enabled(),"ar_compiled":engine.compiled_ar_enabled(),"ar_mixed_gemv":engine.mixed_ar_gemv_enabled(),"ar_optimized":engine.optimized_ar_enabled(),"ar_batch_size":engine.ar_batch_size(),"ar_gemv_rows":engine.ar_gemv_rows(),"ar_fused_epilogue":engine.ar_fused_epilogue_enabled(),"single_window_synthesis":true,"rolling_complete_synthesis":true,"preprocessed_frontend_required":true,
            "reference_prepare":true,"reference_prepare_precision":engine.precision_name(),
            "frontend_g2pw":frontend.has_g2pw(),"frontend_bert":frontend.has_bert(),
            "stream":true,"stream_cancel":true,"synthesize_cancel":true,"cfm_steps":4,"cfg_rate":0.0,
            "fixed_sampling":{"top_k":15,"top_p":1.0,"temperature":1.0,"repetition_penalty":1.35},
            "sample_rate":48000,"stream_format":"pcm_s16le","default_stream_geometry":{"core_frames":200,"future_frames":12,"holdback_frames":50,"fade_frames":4},
            "stream_quality_qualified":false,
            "conditioning_components":["semantic","style","acoustic_prompt"]}});
    ready["memory"] = memory_snapshot()?;
    ready["capabilities"]["frontend_compute_precision"] =
        json!({"g2pw":"float32", "bert":engine.precision_name(), "bert_transport":"float32"});
    ready["capabilities"]["supported_top_k"] = json!([1, 15]);
    ready["capabilities"]["mlx_cache_limit_bytes"] = json!(mlx_cache_limit_bytes);
    ready["capabilities"]["canonical_vocoder_stream"] = json!({
        "enabled":engine.canonical_stream_enabled()?,
        "quality_qualified":false,
        "semantic_eos_before_pcm":true,
        "cfm_complete_before_pcm":true,
        "early_ar_audio":false,
        "vocoder_core_frames":16,"vocoder_halo_frames":12,"vocoder_bucket_frames":205,
        "original_total_samples_in_pcm":true
    });
    ready["capabilities"]["top_k_1_requires_research_opt_in"] = json!(false);
    ready["capabilities"]["sampling_parameters"] = json!({
        "top_k":{"minimum":1,"maximum":50},"top_p":{"exclusive_minimum":0.0,"maximum":1.0},
        "temperature":{"exclusive_minimum":0.0},"repetition_penalty":1.35,
        "default":{"top_k":15,"top_p":1.0,"temperature":1.0}});
    ready["capabilities"]["complete_cfm_noise_policy"] =
        json!(if engine.coordinate_noise_v2_enabled() {
            crate::cfm_noise_field::POLICY
        } else {
            crate::rng::POLICY
        });
    ready["capabilities"]["ar_static_decode"] = json!(engine.static_ar_enabled());
    ready["capabilities"]["cfm_batched_qkv"] = json!(engine.batched_cfm_qkv_enabled());
    ready["capabilities"]["cfm_packed_qkv_blocks"] = json!(if engine.batched_cfm_qkv_enabled() {
        22
    } else {
        0
    });
    ready["capabilities"]["ar_residual_norm_fused"] = json!(engine.residual_ar_norm_enabled());
    ready["capabilities"]["ar_static_cache"] = json!(engine.static_ar_cache_stats());
    emit(&output, &ready)?;
    let (tx, rx) = mpsc::channel();
    let reader_output = output.clone();
    let reader_active = active.clone();
    thread::Builder::new()
        .name("aniflive-tts-ipc-reader".into())
        .spawn(move || {
            if let Err(error) = read_commands(&reader_output, &reader_active, &tx) {
                let _ = emit(
                    &reader_output,
                    &error_event(None, "reader_failed", &error.to_string()),
                );
                if let Ok(slot) = reader_active.lock() {
                    if let Some(current) = slot.as_ref() {
                        if current.mode == Mode::Stream {
                            current.stop.store(true, Ordering::Release);
                        }
                    }
                }
                let _ = tx.send(Command::ReaderFailed(error.to_string()));
            }
        })?;
    while let Ok(command) = rx.recv() {
        match command {
            Command::ReaderFailed(message) => {
                return Err(format!("worker reader failed: {message}").into());
            }
            Command::Shutdown(request_id) => {
                engine.shutdown_cfm_stage()?;
                emit(
                    &output,
                    &json!({"event":"shutdown_complete","request_id":request_id}),
                )?;
                break;
            }
            Command::Job(job) => {
                mlx_rs::memory::reset_peak_memory()?;
                emit(&output, &json!({"event":"started","request_id":job.id}))?;
                let is_tts = matches!(
                    job.mode,
                    Mode::Synthesize | Mode::SynthesizeCached | Mode::Stream
                );
                let complete = matches!(job.mode, Mode::Synthesize | Mode::SynthesizeCached);
                if complete {
                    engine.set_complete_cancel(Some(job.stop.clone()));
                }
                let result = if is_tts {
                    engine.set_sampling(job.sampling).and_then(|()| {
                        process(&engine, &frontend, cache_root, &job, &active, &output)
                    })
                } else {
                    process(&engine, &frontend, cache_root, &job, &active, &output)
                };
                // No selected request policy leaks into the next request.
                engine.set_sampling(crate::sampling::SamplingParams::DEFAULT)?;
                engine.set_complete_cancel(None);
                // A cancelled complete synthesis is a cancellation, never a
                // partial result; its output directory is left to the caller.
                let result = match result {
                    Err(error) if complete && error.to_string() == crate::engine::COMPLETE_CANCELLED => {
                        Ok(Outcome::Cancelled)
                    }
                    Ok(Outcome::Complete(_)) if complete && job.stop.load(Ordering::Acquire) => {
                        Ok(Outcome::Cancelled)
                    }
                    other => other,
                };
                // Hold the admission lock while writing the terminal event so a
                // new request cannot be admitted ahead of it in the JSONL stream.
                let mut slot = active.lock().map_err(|_| lock_error())?;
                if slot.as_ref().map(|current| current.id.as_str()) != Some(job.id.as_str()) {
                    return Err("worker request ownership changed unexpectedly".into());
                }
                let mut terminal = match result {
                    Ok(Outcome::Complete(details)) => {
                        json!({"event":"complete","request_id":job.id,"result":details})
                    }
                    Ok(Outcome::Cancelled) => json!({"event":"cancelled","request_id":job.id}),
                    Err(error) => {
                        error_event(Some(&job.id), "inference_failed", &error.to_string())
                    }
                };
                let memory = memory_snapshot()?;
                if terminal["event"] == "complete" {
                    terminal["result"]["memory"] = memory;
                } else {
                    terminal["memory"] = memory;
                }
                emit(&output, &terminal)?;
                *slot = None;
                if !engine.cfm_stage_healthy() {
                    return Err("isolated native CFM stage became unavailable".into());
                }
            }
        }
    }
    engine.shutdown_cfm_stage()?;
    Ok(())
}
