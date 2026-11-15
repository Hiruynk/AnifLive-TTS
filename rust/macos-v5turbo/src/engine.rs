//! Controlled full-neural inference fixture. Text preparation and RNG draws are
//! explicit supplied inputs, not hidden neural fallbacks or HTTP measurements.
use crate::{
    acoustic::AcousticEncoder, ar::ArModel, estimator::Estimator, hubert::Hubert,
    reference_encoder::ReferenceEncoder, semantic::SemanticEncoder, vocoder::Vocoder,
};
use mlx_rs::{
    ops::{
        concatenate,
        indexing::{argmax_axis, IndexOp},
        select, zeros,
    },
    Array, Device, Dtype,
};
use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

enum ActiveStream<'a> {
    Fixed(crate::streaming::V5StreamDecoder<'a>),
    Bootstrap(crate::bootstrap_stream::BootstrapStream<'a>),
    Canonical(crate::canonical_stream::CanonicalStream<'a>),
}
impl ActiveStream<'_> {
    fn push<F>(
        &mut self,
        tokens: &[i32],
        finished: bool,
        stop: &std::sync::atomic::AtomicBool,
        rng: &mut crate::rng::RequestRng,
        sink: &mut F,
    ) -> Result<()>
    where
        F: FnMut(crate::streaming::StreamChunk) -> Result<()>,
    {
        match self {
            Self::Fixed(decoder) => decoder.push(tokens, finished, stop, rng, sink),
            Self::Bootstrap(decoder) => decoder.push(tokens, finished, stop, rng, sink),
            Self::Canonical(decoder) => decoder.push(tokens, finished, stop, rng, sink),
        }
    }
    fn is_finished(&self) -> bool {
        match self {
            Self::Fixed(d) => d.is_finished(),
            Self::Bootstrap(d) => d.is_finished(),
            Self::Canonical(d) => d.is_finished(),
        }
    }
    fn is_cancelled(&self) -> bool {
        match self {
            Self::Fixed(d) => d.is_cancelled(),
            Self::Bootstrap(d) => d.is_cancelled(),
            Self::Canonical(d) => d.is_cancelled(),
        }
    }
    fn frame_cursor(&self) -> i32 {
        match self {
            Self::Fixed(d) => d.frame_cursor(),
            Self::Bootstrap(d) => d.frame_cursor(),
            Self::Canonical(d) => d.frame_cursor(),
        }
    }
    fn traces(&self) -> &[crate::streaming::WindowTrace] {
        match self {
            Self::Fixed(d) => d.traces(),
            Self::Bootstrap(d) => d.traces(),
            Self::Canonical(d) => d.traces(),
        }
    }
}
fn load(path: impl AsRef<Path>) -> Result<Array> {
    Ok(mlx_rs::with_stream(
        &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
        || Array::load_numpy(path),
    )?)
}
// AR conditions on reference+target text. Target-only input may be numerically
// valid yet synthesize unrelated speech, so validate the actual bundle contract.
fn load_aligned_frontend(input: &Path) -> Result<(Array, Array, Array, Array)> {
    let phones = load(input.join("phones.npy"))?;
    let bert = load(input.join("bert.npy"))?;
    let reference = load(input.join("reference-phones.npy"))?;
    let target = load(input.join("target-phones.npy"))?;
    for array in [&phones, &reference, &target] {
        if array.ndim() != 2
            || array.shape()[0] != 1
            || array.shape()[1] <= 0
            || !matches!(array.dtype(), Dtype::Int32 | Dtype::Int64)
        {
            return Err("Frontend phones must be nonempty integer [1,N]".into());
        }
    }
    let total = reference.shape()[1] + target.shape()[1];
    if phones.shape() != [1, total]
        || bert.shape() != [1, 1024, total]
        || !matches!(bert.dtype(), Dtype::Float32 | Dtype::Float16)
    {
        return Err(
            "Frontend bundle must concatenate reference and target phones/BERT in order".into(),
        );
    }
    let ids = |a: &Array| {
        mlx_rs::with_stream(
            &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
            || a.as_dtype(Dtype::Int64)?.to_vec_exact::<i64>(),
        )
    };
    let mut expected = ids(&reference)?;
    expected.extend(ids(&target)?);
    if ids(&phones)? != expected {
        return Err(
            "Frontend combined phone IDs differ from reference+target concatenation".into(),
        );
    }
    Ok((phones, bert, reference, target))
}
fn weights(path: impl AsRef<Path>, precision: Dtype) -> Result<HashMap<String, Array>> {
    let raw = mlx_rs::with_stream(
        &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
        || Array::load_safetensors(path),
    )?;
    let mut map = HashMap::new();
    for (k, v) in raw {
        map.insert(k, v.as_dtype(precision)?);
    }
    Ok(map)
}
fn checked_dtype(value: &Array, expected: Dtype, stage: &str) -> Result<()> {
    if value.dtype() != expected {
        return Err(format!(
            "{stage} produced {:?}, expected {:?}",
            value.dtype(),
            expected
        )
        .into());
    }
    Ok(())
}
fn scalar(value: f32, precision: Dtype) -> Result<Array> {
    Ok(Array::from_f32(value).as_dtype(precision)?)
}
struct StageTimings {
    reference_ms: f64,
    autoregressive_ms: f64,
    acoustic_ms: f64,
    cfm_ms: f64,
    vocoder_ms: f64,
    pcm_write_ms: f64,
    total_ms: f64,
}
struct RunSummary {
    timings: StageTimings,
    tokens: usize,
    eos_sampled: i32,
    eos_argmax: i32,
    ar_speculative_draws: u64,
    noise_frames: i32,
    cfm_windows: Vec<serde_json::Value>,
    samples: usize,
}

/// Error text of a complete synthesis stopped by a worker cancel request.
pub const COMPLETE_CANCELLED: &str = "native complete synthesis cancelled";

pub struct Engine {
    precision: Dtype,
    complete_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    emit_cli_reports: bool,
    ar_combined: bool,
    ar_optimized: bool,
    sampling: crate::sampling::SamplingParams,
    cfm_coordinate_v2: bool,
    ar_compiled: std::cell::RefCell<Option<crate::ar_compiled::CompiledAr>>,
    ar_static: Option<Rc<std::cell::RefCell<crate::ar_static::StaticAr>>>,
    model_hashes: std::collections::BTreeMap<String, String>,
    vocoder_manifest: PathBuf,
    hubert: Hubert,
    semantic: SemanticEncoder,
    reference: ReferenceEncoder,
    ar: ArModel,
    acoustic: AcousticEncoder,
    estimator: Estimator,
    cfm_schedule: Option<crate::estimator::PreparedSchedule>,
    cfm_stage: std::cell::RefCell<Option<crate::cfm_stage::CfmStage>>,
    vocoder: Vocoder,
}
impl Engine {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_precision(dir, Dtype::Float32)
    }
    pub fn precision(&self) -> Dtype {
        self.precision
    }
    pub fn precision_name(&self) -> &'static str {
        if self.precision == Dtype::Float16 {
            "float16"
        } else {
            "float32"
        }
    }
    /// Disable CLI summaries when stdout is owned by the worker protocol.
    pub fn set_cli_reports(&mut self, enabled: bool) {
        self.emit_cli_reports = enabled;
    }
    /// FP16 uses the official is_half runtime cast; stored checkpoints are
    /// never modified. The official vocoder checkpoint itself is FP32.
    pub fn load_with_precision(dir: &Path, precision: Dtype) -> Result<Self> {
        if precision != Dtype::Float32 && precision != Dtype::Float16 {
            return Err(format!("Unsupported engine precision: {precision:?}").into());
        }
        let vocoder_manifest = dir.join("vocoder.manifest.json");
        if precision == Dtype::Float16 && !vocoder_manifest.is_file() {
            return Err(
                "FP16 requires the vocoder conversion manifest for source provenance".into(),
            );
        }
        let mut model_hashes = std::collections::BTreeMap::new();
        for name in [
            "s2Gv5turbo.safetensors",
            "s1v3.safetensors",
            "vocoder.safetensors",
            "chinese-hubert-base/hubert.safetensors",
        ] {
            model_hashes.insert(
                name.to_owned(),
                crate::reference_cache::sha256_file(&dir.join(name))?,
            );
        }
        let s2 = weights(dir.join("s2Gv5turbo.safetensors"), precision)?;
        Ok(Self {
            precision,
            emit_cli_reports: true,
            ar_combined: false,
            ar_optimized: false,
            sampling: crate::sampling::SamplingParams::DEFAULT,
            complete_cancel: None,
            cfm_coordinate_v2: false,
            ar_compiled: std::cell::RefCell::new(None),
            ar_static: None,
            model_hashes,
            vocoder_manifest,
            hubert: Hubert::load(&weights(
                dir.join("chinese-hubert-base/hubert.safetensors"),
                precision,
            )?)?,
            semantic: SemanticEncoder::load(&s2)?,
            reference: ReferenceEncoder::load(&s2)?,
            ar: ArModel::load(&weights(dir.join("s1v3.safetensors"), precision)?, 16)?,
            acoustic: AcousticEncoder::load(&s2)?,
            estimator: Estimator::load(&s2)?,
            cfm_schedule: None,
            cfm_stage: std::cell::RefCell::new(None),
            vocoder: Vocoder::load(&weights(dir.join("vocoder.safetensors"), precision)?)?,
        })
    }
    /// Opt-in fixed-time FP32 CFM preparation; startup work is explicit.
    pub fn enable_prepared_cfm(&mut self) -> Result<()> {
        if self.precision != Dtype::Float32 || self.cfm_stage.borrow().is_some() {
            return Err("Prepared local CFM requires FP32 and no isolated stage".into());
        }
        if self.cfm_schedule.is_none() {
            self.cfm_schedule = Some(self.estimator.prepare_schedule()?);
        }
        Ok(())
    }
    pub fn prepared_cfm_enabled(&self) -> bool {
        self.cfm_schedule.is_some()
    }

    pub fn enable_combined_ar(&mut self) -> Result<()> {
        if !matches!(self.precision, Dtype::Float32 | Dtype::Float16) {
            return Err("Combined AR decision requires FP32 or FP16".into());
        }
        self.ar_combined = true;
        Ok(())
    }
    pub fn combined_ar_enabled(&self) -> bool {
        self.ar_combined
    }

    /// Request-local quality experiment; the unchanged default is 15.
    pub fn set_sampling_top_k(&mut self, top_k: usize) -> Result<()> {
        if !matches!(top_k, 1 | 15) {
            return Err("Native AR top_k must be 1 or 15".into());
        }
        if top_k == 1
            && std::env::var("ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM").as_deref() != Ok("1")
        {
            return Err("Research top_k=1 requires unqualified research opt-in".into());
        }
        self.sampling = crate::sampling::SamplingParams { top_k, ..crate::sampling::SamplingParams::DEFAULT };
        Ok(())
    }
    /// Request-level sampling (worker): any Windows-legal parameter set.
    pub fn set_sampling(&mut self, params: crate::sampling::SamplingParams) -> Result<()> {
        params.validate()?;
        self.sampling = params;
        Ok(())
    }
    /// Cancellation flag polled by complete synthesis (AR batches, CFM
    /// windows, before the vocoder). A cancelled run writes no WAV.
    pub fn set_complete_cancel(
        &mut self,
        flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.complete_cancel = flag;
    }
    fn check_complete_cancel(&self) -> Result<()> {
        match &self.complete_cancel {
            Some(flag) if flag.load(std::sync::atomic::Ordering::Acquire) => {
                Err(COMPLETE_CANCELLED.into())
            }
            _ => Ok(()),
        }
    }
    pub fn sampling(&self) -> crate::sampling::SamplingParams {
        self.sampling
    }
    pub fn sampling_top_k(&self) -> usize {
        self.sampling.top_k
    }
    pub fn coordinate_noise_v2_enabled(&self) -> bool {
        self.cfm_coordinate_v2
    }
    /// Explicit completed-mel vocoder stream; never quality-qualified by this flag.
    pub fn canonical_stream_enabled(&self) -> Result<bool> {
        let enabled = match std::env::var("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(std::env::VarError::NotPresent) => false,
            _ => return Err("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM must be 0 or 1".into()),
        };
        if enabled {
            // Accepted product stream: no research opt-in (bootstrap v2 and
            // coordinate noise remain research-gated).
            if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
                return Err("Canonical stream requires MLX_ENABLE_TF32=0".into());
            }
            match std::env::var("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2").as_deref() {
                Ok("1") => {
                    return Err(
                        "Canonical and bootstrap stream modes are mutually exclusive".into(),
                    )
                }
                Ok("0") | Err(std::env::VarError::NotPresent) => (),
                _ => return Err("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2 must be 0 or 1".into()),
            }
        }
        Ok(enabled)
    }

    /// Apply the same explicit AR flag contract in CLI and persistent worker.
    pub fn configure_ar_from_env(&mut self) -> Result<()> {
        fn flag(name: &str) -> Result<bool> {
            match std::env::var(name) {
                Ok(value) if value == "0" => Ok(false),
                Ok(value) if value == "1" => Ok(true),
                Err(std::env::VarError::NotPresent) => Ok(false),
                _ => Err(format!("{name} must be 0 or 1").into()),
            }
        }
        let top_k = match std::env::var("ANIFLIVE_TTS_RESEARCH_AR_TOP_K").as_deref() {
            Ok("1") => 1,
            Ok("0") | Err(std::env::VarError::NotPresent) => 15,
            _ => return Err("ANIFLIVE_TTS_RESEARCH_AR_TOP_K must be 0 or 1".into()),
        };
        let coordinate = flag("ANIFLIVE_TTS_NATIVE_CFM_COORDINATE_V2")?;
        if coordinate
            && (self.precision != Dtype::Float32
                || std::env::var("ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM").as_deref() != Ok("1"))
        {
            return Err("Coordinate v2 noise requires FP32 and unqualified research opt-in".into());
        }
        self.set_sampling_top_k(top_k)?;
        self.cfm_coordinate_v2 = coordinate;
        let optimized = flag("ANIFLIVE_TTS_NATIVE_AR_OPTIMIZED")?;
        let combined = flag("ANIFLIVE_TTS_NATIVE_AR_COMBINED")?;
        let compiled = flag("ANIFLIVE_TTS_NATIVE_AR_COMPILED")?;
        let mixed = flag("ANIFLIVE_TTS_NATIVE_AR_MIXED_GEMV")?;
        let static_decode = flag("ANIFLIVE_TTS_NATIVE_AR_STATIC")?;
        let residual_norm = flag("ANIFLIVE_TTS_NATIVE_AR_RESIDUAL_NORM")?;
        if residual_norm && !static_decode {
            return Err("Residual norm fusion requires static optimized AR".into());
        }
        if static_decode
            && (!optimized || !combined || !mixed || compiled)
        {
            return Err("Static AR requires AR_OPTIMIZED=1, AR_COMBINED=1, AR_MIXED_GEMV=1 and AR_COMPILED=0".into());
        }
        if optimized && (!combined || !mixed || compiled) {
            return Err(
                "Optimized AR requires AR_COMBINED=1, AR_MIXED_GEMV=1 and AR_COMPILED=0".into(),
            );
        }
        if combined {
            self.enable_combined_ar()?;
        }
        if optimized {
            self.enable_optimized_ar()?;
        } else {
            if compiled {
                self.enable_compiled_ar()?;
            }
            if mixed {
                self.enable_mixed_ar_gemv()?;
            }
        }
        if residual_norm {
            if self.static_ar_enabled() && !self.ar.residual_norm_fusion_enabled() {
                return Err(
                    "Residual norm fusion must be configured before static AR graphs".into(),
                );
            }
            self.ar.enable_residual_norm_fusion()?;
        }
        if static_decode {
            self.enable_static_ar()?;
        }
        Ok(())
    }

    pub fn enable_optimized_ar(&mut self) -> Result<()> {
        if !self.ar_combined || self.compiled_ar_enabled() {
            return Err(
                "Optimized AR requires combined decision and compiled AR disabled".into(),
            );
        }
        self.ar.enable_mixed_gemv_with_rows(1)?;
        self.ar.enable_mixed_gemv_epilogue()?;
        self.ar_optimized = true;
        Ok(())
    }
    pub fn optimized_ar_enabled(&self) -> bool {
        self.ar_optimized
    }
    pub fn residual_ar_norm_enabled(&self) -> bool {
        self.ar.residual_norm_fusion_enabled()
    }
    /// Research opt-in; capacity is a provisional per-model graph bound.
    pub fn enable_static_ar(&mut self) -> Result<()> {
        if !self.ar_optimized
            || !self.ar_combined
            || self.compiled_ar_enabled()
            || self.ar.mixed_gemv_rows() != Some(1)
            || !self.ar.mixed_gemv_epilogue_enabled()
        {
            return Err(
                "Static AR requires optimized fused TM1 AR and compiled AR disabled".into(),
            );
        }
        if self.ar_static.is_none() {
            self.ar_static = Some(Rc::new(std::cell::RefCell::new(
                crate::ar_static::StaticAr::new(&self.ar, 1024)?,
            )));
        }
        Ok(())
    }
    pub fn static_ar_enabled(&self) -> bool {
        self.ar_static.is_some()
    }
    /// Cumulative per-worker LRU statistics; null when static AR is disabled.
    pub fn static_ar_cache_stats(&self) -> Option<serde_json::Value> {
        self.ar_static.as_ref().map(|cache| {
            let s = cache.borrow().stats();
            serde_json::json!({"entries":s.entries,"capacity":s.capacity,"hits":s.hits,
                "misses":s.misses,"evictions":s.evictions,"compiles":s.compiles})
        })
    }
    fn attach_static_ar<'a>(
        &self,
        generator: &mut crate::ar_microbatch::ArGenerator<'a>,
    ) -> Result<()> {
        if let Some(cache) = &self.ar_static {
            let cache = Rc::clone(cache);
            generator.set_diagnostic_decode_override(Box::new(move |token, state| {
                cache.borrow_mut().decode_sampled_gpu(token, state)
            }))?;
        }
        Ok(())
    }
    pub fn ar_batch_size(&self) -> usize {
        if self.ar_optimized {
            4
        } else {
            1
        }
    }
    pub fn ar_gemv_rows(&self) -> Option<i32> {
        self.ar.mixed_gemv_rows()
    }
    pub fn ar_fused_epilogue_enabled(&self) -> bool {
        self.ar.mixed_gemv_epilogue_enabled()
    }

    pub fn enable_cfm_stage(&mut self, work_root: &Path) -> Result<()> {
        if self.precision != Dtype::Float32 || self.cfm_schedule.is_some() {
            return Err(
                "Isolated CFM requires FP32 core and local CFM preparation disabled".into(),
            );
        }
        if self.cfm_stage.borrow().is_some() {
            return Err("CFM stage already enabled".into());
        }
        let models = self
            .vocoder_manifest
            .parent()
            .ok_or("missing model directory")?;
        let stage =
            crate::cfm_stage::CfmStage::spawn(&std::env::current_exe()?, models, work_root, true)?;
        *self.cfm_stage.borrow_mut() = Some(stage);
        Ok(())
    }
    pub fn cfm_stage_enabled(&self) -> bool {
        self.cfm_stage.borrow().is_some()
    }
    pub fn compiled_cfm_stage_enabled(&self) -> bool {
        self.cfm_stage
            .borrow()
            .as_ref()
            .map(|stage| stage.compiled_enabled())
            .unwrap_or(false)
    }
    pub fn batched_cfm_qkv_enabled(&self) -> bool {
        self.cfm_stage
            .borrow()
            .as_ref()
            .map(|stage| stage.batched_qkv_enabled())
            .unwrap_or(false)
    }
    pub fn validate_batched_cfm_qkv_from_env(&self) -> Result<()> {
        let requested = match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            _ => return Err("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV must be 0 or 1".into()),
        };
        if requested != self.batched_cfm_qkv_enabled() {
            return Err("Batched CFM QKV requires the compiled isolated CFM stage".into());
        }
        Ok(())
    }
    pub fn cfm_stage_pid(&self) -> Option<u32> {
        self.cfm_stage.borrow().as_ref().map(|s| s.pid())
    }
    pub fn cfm_stage_healthy(&self) -> bool {
        self.cfm_stage
            .borrow_mut()
            .as_mut()
            .map(|s| s.healthy())
            .unwrap_or(true)
    }
    pub fn shutdown_cfm_stage(&self) -> Result<()> {
        if let Some(mut stage) = self.cfm_stage.borrow_mut().take() {
            stage.shutdown()?;
        }
        Ok(())
    }

    pub fn enable_compiled_ar(&mut self) -> Result<()> {
        if self.ar.mixed_gemv_enabled() {
            return Err(
                "Compiled AR and custom mixed GEMV cannot be combined in this runtime".into(),
            );
        }
        if self.precision != Dtype::Float32 {
            return Err("Compiled AR requires FP32".into());
        }
        if self.ar_compiled.borrow().is_none() {
            *self.ar_compiled.borrow_mut() = Some(crate::ar_compiled::CompiledAr::initialize()?);
        }
        Ok(())
    }
    pub fn compiled_ar_enabled(&self) -> bool {
        self.ar_compiled.borrow().is_some()
    }
    pub fn ar_compile_trace_count(&self) -> Option<usize> {
        self.ar_compiled
            .borrow()
            .as_ref()
            .map(|decoder| decoder.trace_count())
    }
    fn decode_ar_generated(
        &self,
        token: &Array,
        id: i32,
        state: &mut crate::ar::ArState,
        fast: bool,
    ) -> Result<Array> {
        if let Some(decoder) = self.ar_compiled.borrow_mut().as_mut() {
            decoder.decode_generated(&self.ar, token, id, state)
        } else if fast {
            self.ar.decode_generated(token, id, state)
        } else {
            self.ar.decode(token, state)
        }
    }

    pub fn enable_mixed_ar_gemv(&mut self) -> Result<()> {
        if self.compiled_ar_enabled() {
            return Err("Mixed-storage GEMV requires compiled AR disabled".into());
        }
        self.ar.enable_mixed_gemv()
    }
    pub fn mixed_ar_gemv_enabled(&self) -> bool {
        self.ar.mixed_gemv_enabled()
    }

    /// Register immutable prepared reference inputs and persist actual native
    /// conditioning. Hashing is registration work, not an HTTP fast-path claim.
    pub fn prepare_reference_cached(
        &self,
        input: &Path,
        audio: &Path,
        transcript: &Path,
        root: &Path,
    ) -> Result<(crate::reference_cache::PreparedReference, bool, String)> {
        use crate::reference_cache::{sha256_file, PreparedReference, ReferenceIdentity};
        use std::collections::BTreeMap;
        let mut prepared_inputs_sha256 = BTreeMap::new();
        for name in [
            "waveform.npy",
            "refer.npy",
            "reference-mel.npy",
            "reference-phones.npy",
        ] {
            prepared_inputs_sha256.insert(name.to_owned(), sha256_file(&input.join(name))?);
        }
        let identity = ReferenceIdentity {
            schema: 1,
            reference_audio_sha256: sha256_file(audio)?,
            transcript_sha256: sha256_file(transcript)?,
            models_sha256: self.model_hashes.clone(),
            prepared_inputs_sha256,
            precision: self.precision_name().into(),
            preprocessing_version: if self.precision == Dtype::Float16 {
                "official-6d51ffce-mps-fp16-reference-v4-mps-reduction-order"
            } else {
                "official-6d51ffce-reference-arrays-v1"
            }.into(),
        };
        let key = crate::reference_cache::identity_key(&identity)?;
        if let Some(reference) = crate::reference_cache::load(root, &identity)? {
            return Ok((reference, true, key));
        }
        let waveform = load(input.join("waveform.npy"))?;
        let waveform = if self.precision == Dtype::Float16 {
            waveform.as_dtype(self.precision)?
        } else {
            waveform
        };
        let hidden = self.hubert.forward(&waveform)?;
        checked_dtype(&hidden, self.precision, "cached reference HuBERT hidden")?;
        let (projected, prompt_codes) = self.semantic.forward(&hidden)?;
        checked_dtype(
            &projected,
            self.precision,
            "cached reference semantic projection",
        )?;
        let spectrum = load(input.join("refer.npy"))?;
        let spectrum = if self.precision == Dtype::Float16 {
            spectrum.as_dtype(self.precision)?
        } else {
            spectrum
        };
        if spectrum.ndim() != 3 || spectrum.shape()[0] != 1 || spectrum.shape()[1] < 704 {
            return Err("Invalid reference spectrum".into());
        }
        let ge = self.reference.forward(&spectrum.index((.., 0..704, ..)))?;
        checked_dtype(&ge, self.precision, "cached reference style")?;
        let features = self.acoustic.forward(
            &prompt_codes,
            &load(input.join("reference-phones.npy"))?,
            &ge,
        )?;
        let mel = load(input.join("reference-mel.npy"))?
            .add(Array::from_f32(12.0))?
            .divide(Array::from_f32(14.0))?
            .multiply(Array::from_f32(2.0))?
            .subtract(Array::from_f32(1.0))?;
        let mel = if self.precision == Dtype::Float16 {
            mel.as_dtype(self.precision)?
        } else {
            mel
        };
        checked_dtype(&features, self.precision, "cached reference acoustic")?;
        checked_dtype(&mel, self.precision, "cached reference mel")?;
        let end = features.shape()[2].min(mel.shape()[2]);
        let begin = (end - 500).max(0);
        let reference = PreparedReference {
            prompt_codes,
            ge,
            spectrum,
            features: features.index((.., .., begin..end)),
            mel: mel.index((.., .., begin..end)),
        };
        reference.save(root, &identity)?;
        Ok((reference, false, key))
    }

    /// Real AR-incremental streaming from a registered conditioning cache.
    /// The callback is invoked per acoustic window before further AR decoding.
    pub fn run_stream_seeded(
        &self,
        reference: &crate::reference_cache::PreparedReference,
        input: &Path,
        out: &Path,
        seed: u64,
        config: crate::streaming::StreamConfig,
        stop: &std::sync::atomic::AtomicBool,
        cancel_after_first: bool,
    ) -> Result<()> {
        self.run_stream_seeded_with_sink(
            reference,
            input,
            out,
            seed,
            config,
            stop,
            cancel_after_first,
            &mut |_| Ok(()),
        )
    }
    pub fn run_stream_seeded_with_sink(
        &self,
        reference: &crate::reference_cache::PreparedReference,
        input: &Path,
        out: &Path,
        seed: u64,
        config: crate::streaming::StreamConfig,
        stop: &std::sync::atomic::AtomicBool,
        cancel_after_first: bool,
        external: &mut dyn FnMut(&crate::streaming::StreamChunk) -> Result<()>,
    ) -> Result<()> {
        use crate::streaming::{StreamReference, V5StreamDecoder};
        use std::{io::Write, sync::atomic::Ordering, time::Instant};
        let bootstrap = match std::env::var("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(std::env::VarError::NotPresent) => false,
            _ => return Err("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2 must be 0 or 1".into()),
        };
        if bootstrap && std::env::var("ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM").as_deref() != Ok("1")
        {
            return Err(
                "v2 bootstrap requires explicit unqualified research streaming opt-in".into(),
            );
        }
        let canonical = self.canonical_stream_enabled()?;
        if self.precision == Dtype::Float16 && (!canonical || bootstrap || self.cfm_stage_enabled())
        {
            return Err(
                "FP16 streaming requires canonical vocoder mode with local FP16 CFM".into(),
            );
        }
        if reference.ge.dtype() != self.precision {
            return Err("Reference precision differs from engine".into());
        }
        let (phones, bert, _reference_phones, target_phones) = load_aligned_frontend(input)?;
        let bert = if self.precision == Dtype::Float16 {
            bert.as_dtype(self.precision)?
        } else {
            bert
        };
        std::fs::create_dir_all(out)?;
        if std::fs::read_dir(out)?.next().is_some() {
            return Err("Stream output directory must be empty".into());
        }
        std::fs::create_dir(out.join("window-noise"))?;
        std::fs::write(
            out.join("stream-config.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"core_frames":config.core_frames,"future_frames":config.future_frames,"holdback_frames":config.holdback_frames,"fade_frames":config.fade_frames}),
            )?,
        )?;
        reference
            .prompt_codes
            .save_numpy(out.join("prompt-codes.npy"))?;
        let core_frames = config.core_frames;
        let start = Instant::now();
        let mut rng = crate::rng::RequestRng::new(seed);
        let stream_reference = StreamReference {
            ge: reference.ge.clone(),
            features: reference.features.clone(),
            mel: reference.mel.clone(),
        };
        let mut decoder = if bootstrap {
            let mut value = crate::bootstrap_stream::BootstrapStream::new(
                &self.acoustic,
                &self.estimator,
                &self.vocoder,
                stream_reference,
                target_phones,
                config,
                self.cfm_schedule.as_ref(),
                &mut rng,
            )?;
            if self.cfm_stage_enabled() {
                value.attach_cfm_stage(&self.cfm_stage)?;
            }
            ActiveStream::Bootstrap(value)
        } else if canonical {
            let mut value = crate::canonical_stream::CanonicalStream::new(
                &self.acoustic,
                &self.estimator,
                &self.vocoder,
                stream_reference,
                target_phones,
                config,
                self.cfm_schedule.as_ref(),
                self.cfm_coordinate_v2,
                &mut rng,
            )?;
            if self.cfm_stage_enabled() {
                value.attach_cfm_stage(&self.cfm_stage)?;
            }
            ActiveStream::Canonical(value)
        } else {
            let mut value = V5StreamDecoder::new_with_schedule(
                &self.acoustic,
                &self.estimator,
                &self.vocoder,
                stream_reference,
                target_phones,
                config,
                self.cfm_schedule.as_ref(),
            )?;
            if self.cfm_stage_enabled() {
                value.attach_cfm_stage(&self.cfm_stage)?;
            }
            ActiveStream::Fixed(value)
        };
        let prompt = reference
            .prompt_codes
            .reshape(&[1, reference.prompt_codes.shape()[2]])?
            .as_dtype(Dtype::Int32)?;
        prompt.eval()?;
        let mut previous = prompt.as_slice::<i32>().to_vec();
        let (logits, state) = self.ar.prefill(&phones, &bert, &prompt)?;
        logits.eval()?;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut legacy_logits = Some(logits);
        let mut legacy_state = Some(state);
        let mut generator = if self.ar_optimized {
            Some(crate::ar_microbatch::ArGenerator::from_prefill(
                &self.ar,
                &prompt,
                legacy_logits.take().ok_or("missing prefill logits")?,
                legacy_state.take().ok_or("missing prefill state")?,
                4,
            )?)
        } else {
            None
        };
        if let Some(generator) = generator.as_mut() {
            generator.set_sampling(self.sampling)?;
            self.attach_static_ar(generator)?;
        }
        let mut generated = Vec::new();
        let mut ended = false;
        let mut updates = Vec::new();
        let mut chunks = Vec::new();
        let mut samples = Vec::new();
        let mut pcm = Vec::new();
        let mut first_pcm_ms = None;
        let mut ar_eos_ms = None;
        let mut raw = std::fs::File::create(out.join("stream.pcm"))?;
        let mut ar_speculative_draws = 0u64;
        for step in 0..1500 {
            if stop.load(Ordering::Acquire) {
                break;
            }
            let (finished, pending_token) = if let Some(generator) = generator.as_mut() {
                // Do not commit across the next acoustic update boundary.
                let max_draws = (16 - generated.len() % 16).min(4);
                let batch = generator.next_batch(&mut rng, max_draws)?;
                ar_speculative_draws += batch.speculative_draws;
                if stop.load(Ordering::Acquire) {
                    break;
                }
                generated.extend(batch.tokens);
                (batch.finished, None)
            } else {
                let logits = legacy_logits.as_ref().ok_or("missing legacy AR logits")?;
                let logits_step = if step < 11 {
                    logits.index((.., 0..1024))
                } else {
                    logits.clone()
                };
                let adjusted =
                    crate::sampling::apply_repetition_penalty(&logits_step, &previous, 1.35)?;
                let q = rng.ar_exponential(logits_step.shape())?;
                let q = if self.precision == Dtype::Float16 {
                    q.as_dtype(logits_step.dtype())?
                } else {
                    q
                };
                if self.precision == Dtype::Float16 {
                    let minimum = q.min(false)?.try_item_cast::<f32>()?;
                    if !minimum.is_finite() || minimum <= 0.0 {
                        return Err("AR exponential draw invalid after FP16 conversion".into());
                    }
                }
                let token = crate::sampling::sample(
                    &adjusted,
                    &[],
                    &q,
                    self.sampling.top_k,
                    self.sampling.top_p,
                    self.sampling.temperature,
                    1.0,
                )?
                .as_dtype(Dtype::Int32)?;
                let (id, greedy_id) = if self.ar_combined {
                    let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
                    crate::ar_decision::read_decision(&q, &token, &greedy)?
                } else {
                    token.eval()?;
                    let id = token.item_exact::<i32>();
                    let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
                    greedy.eval()?;
                    (id, greedy.item_exact::<i32>())
                };
                let finished = id == 1024 || greedy_id == 1024;
                if !finished {
                    generated.push(id);
                    previous.push(id);
                }
                (finished, Some((token, id)))
            };
            if finished {
                ended = true;
                ar_eos_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            if finished || generated.len() % 16 == 0 {
                updates.push(serde_json::json!({"tokens":generated.len(),"finished":finished}));
                let token_count = generated.len();
                let mut sink = |chunk: crate::streaming::StreamChunk| -> Result<()> {
                    if stop.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    external(&chunk)?;
                    let index = chunks.len();
                    let before = pcm.len();
                    let bytes: Vec<u8> = chunk.pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
                    raw.write_all(&bytes)?;
                    raw.flush()?;
                    std::fs::write(out.join(format!("chunk-{index:04}.pcm")), &bytes)?;
                    let at = start.elapsed().as_secs_f64() * 1000.0;
                    if first_pcm_ms.is_none() {
                        first_pcm_ms = Some(at);
                    }
                    chunks.push(serde_json::json!({"index":index,"samples":chunk.pcm.len(),"sample_start":before,"sample_end":before+chunk.pcm.len(),"frame_end":chunk.frame_end,"before_eos":chunk.before_eos,"final_chunk":chunk.final_chunk,"ar_tokens_available":token_count,"emitted_ms":at}));
                    samples.extend_from_slice(&chunk.samples);
                    pcm.extend_from_slice(&chunk.pcm);
                    if cancel_after_first {
                        stop.store(true, Ordering::Release);
                    }
                    Ok(())
                };
                decoder.push(&generated, finished, stop, &mut rng, &mut sink)?;
            }
            if finished || stop.load(Ordering::Acquire) {
                break;
            }
            if let Some((token, id)) = pending_token {
                legacy_logits = Some(self.decode_ar_generated(
                    &token,
                    id,
                    legacy_state.as_mut().ok_or("missing legacy AR state")?,
                    self.ar_combined,
                )?);
            }
        }
        let cancelled = stop.load(Ordering::Acquire) || decoder.is_cancelled();
        if !cancelled && (!ended || !decoder.is_finished()) {
            return Err("Stream did not reach a complete EOS flush".into());
        }
        if !cancelled && pcm.len() != generated.len() * 4 * 480 {
            return Err("Streaming PCM sample accounting failed".into());
        }
        Array::from_slice(&generated, &[1, 1, generated.len() as i32])
            .save_numpy(out.join("target-codes.npy"))?;
        Array::from_slice(&samples, &[samples.len() as i32])
            .save_numpy(out.join("stream-wave.npy"))?;
        crate::audio::write_pcm16_values(&pcm, &out.join("stream.wav"))?;
        let mut windows = Vec::new();
        for (index, t) in decoder.traces().iter().enumerate() {
            t.noise
                .save_numpy(out.join(format!("window-noise/{index:04}.npy")))?;
            let mut window = serde_json::json!({"start_frame":t.start_frame,"end_frame":t.end_frame,"future_frames":t.future_frames,"reference_frames":t.reference_frames,"semantic_tokens":t.semantic_tokens});
            if let ActiveStream::Bootstrap(value) = &decoder {
                let draw_index = value
                    .trace_field_indices()
                    .get(index)
                    .ok_or("Bootstrap noise trace lacks a field draw index")?;
                window["cfm_field_draw_index"] = serde_json::json!(draw_index);
            } else if let ActiveStream::Canonical(_) = &decoder {
                window["cfm_field_draw_index"] = serde_json::json!(index);
            }
            windows.push(window);
        }
        std::fs::write(
            out.join("updates.json"),
            serde_json::to_vec_pretty(&updates)?,
        )?;
        std::fs::write(
            out.join("windows.json"),
            serde_json::to_vec_pretty(&windows)?,
        )?;
        std::fs::write(out.join("chunks.json"), serde_json::to_vec_pretty(&chunks)?)?;
        let mut report = serde_json::json!({"kind":"native_ar_incremental_stream","seed":seed,"precision":self.precision_name(),"sample_rate":48000,"cfm_steps":4,"cfg":0,"cancelled":cancelled,"finished":decoder.is_finished(),"semantic_tokens":generated.len(),"pcm_samples":pcm.len(),"expected_samples_at_eos":if ended{Some(generated.len()*4*480)}else{None},"first_pcm_before_ar_eos":chunks.first().map(|c|c["before_eos"].as_bool().unwrap_or(false)).unwrap_or(false),"chunk_count":chunks.len(),"window_count":(decoder.frame_cursor()+core_frames-1)/core_frames,"noise_trace_count":decoder.traces().len(),"ar_optimized":self.ar_optimized,"cfm_batched_qkv":self.batched_cfm_qkv_enabled(),"cfm_packed_qkv_blocks":if self.batched_cfm_qkv_enabled(){22}else{0},"ar_static_decode":self.static_ar_enabled(),"ar_residual_norm_fused":self.residual_ar_norm_enabled(),"ar_static_cache":self.static_ar_cache_stats(),"ar_batch_size":self.ar_batch_size(),"ar_gemv_rows":self.ar_gemv_rows(),"ar_fused_epilogue":self.ar_fused_epilogue_enabled(),"ar_discarded_speculative_draws":ar_speculative_draws,"ar_prefill_ms":prefill_ms,"first_pcm_ms_local_only":first_pcm_ms,"ar_eos_ms_local_only":ar_eos_ms,"complete_ms_local_only":start.elapsed().as_secs_f64()*1000.0,"http_benchmark":false});
        report["top_k"] = serde_json::json!(self.sampling.top_k);
        report["top_p"] = serde_json::json!(self.sampling.top_p);
        report["temperature"] = serde_json::json!(self.sampling.temperature);
        report["sampling_default"] = serde_json::json!(self.sampling.is_default());
        if let ActiveStream::Bootstrap(value) = &decoder {
            report["kind"] = serde_json::json!("native_experimental_bootstrap_v2_stream");
            report["algorithm"] = serde_json::json!("prefix32_first16_fade4_then_full_eos");
            report["noise_policy"] = serde_json::json!(crate::cfm_noise_field::POLICY);
            report["quality_qualified"] = serde_json::json!(false);
            report["cfm_draws"] = serde_json::json!(rng.cfm_draws());
            report["ar_draws"] = serde_json::json!(rng.ar_draws());
            report["window_count"] = serde_json::json!(value.solve_count());
            report["eos_cfm_windows"] = serde_json::json!(value.eos_solve_count());
            report["bootstrap_emitted"] = serde_json::json!(value.bootstrap_emitted());
        } else if let ActiveStream::Canonical(value) = &decoder {
            report["kind"] = serde_json::json!("native_experimental_canonical_vocoder_stream");
            report["algorithm"] =
                serde_json::json!("complete_ar_cfm_then_incremental_vocoder_16_12_205");
            report["noise_policy"] = serde_json::json!(if value.coordinate_v2() {
                crate::cfm_noise_field::POLICY
            } else {
                crate::rng::POLICY
            });
            report["quality_qualified"] = serde_json::json!(false);
            report["cfm_draws"] = serde_json::json!(rng.cfm_draws());
            report["ar_draws"] = serde_json::json!(rng.ar_draws());
            report["window_count"] = serde_json::json!(value.cfm_windows());
            report["vocoder_NN_calls_at_first_emit"] =
                serde_json::json!(value.calls_at_first_emit());
            report["vocoder_nn_calls_total"] = serde_json::json!(value.vocoder_calls());
            report["degenerate_short"] = serde_json::json!(value.degenerate_short());
            report["full_vocoder_before_first_emit"] = serde_json::json!(value
                .calls_at_first_emit()
                .map(|_| value.degenerate_short()));
            report["semantic_eos_before_first_pcm"] =
                serde_json::json!(value.calls_at_first_emit().map(|_| true));
            report["cfm_complete_before_first_pcm"] =
                serde_json::json!(value.calls_at_first_emit().map(|_| true));
            report["early_AR_audio"] = serde_json::json!(false);
            report["initial_pcm_buffer"] =
                serde_json::json!("preserved until an RMS frame above -45 dBFS; no trim");
            report["pcm_scaling"] =
                serde_json::json!("native complete-WAV signed PCM16 conversion");
        }
        std::fs::write(
            out.join("stream-report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        if self.emit_cli_reports {
            println!("{report}");
        }
        Ok(())
    }
    /// Controlled complete utterance, with captured per-window noise. No HTTP claim.
    pub fn run_fixture(&self, input: &Path, out: &Path) -> Result<()> {
        self.run_inner(input, out, None, false, None, true, None)
            .map(|_| ())
    }

    /// Native complete synthesis from explicitly prepared frontend
    /// inputs. The output directory must be empty; q and unscaled CFM noise are
    /// recorded for cross-runtime replay. No global random state is used.
    pub fn run_seeded(&self, input: &Path, out: &Path, seed: u64) -> Result<()> {
        self.run_seeded_inner(input, out, seed, false, None, true)
    }

    /// Experimental sampled-token validation optimization. All neural and RNG
    /// operations remain shared with the standard path; qualification is separate.
    pub fn run_seeded_fast_ar(&self, input: &Path, out: &Path, seed: u64) -> Result<()> {
        self.run_seeded_inner(input, out, seed, true, None, true)
    }

    /// Complete-mode reference-cache and capture-I/O candidate. Neural and RNG
    /// math remain shared with the captured baseline; caller prepares the same
    /// immutable reference identity with prepare_reference_cached first.
    pub fn run_seeded_cached(
        &self,
        input: &Path,
        out: &Path,
        seed: u64,
        reference: &crate::reference_cache::PreparedReference,
        capture: bool,
    ) -> Result<()> {
        reference.validate()?;
        if reference.ge.dtype() != self.precision {
            return Err("Reference precision differs from engine".into());
        }
        self.run_seeded_inner(input, out, seed, false, Some(reference), capture)
    }

    fn run_seeded_inner(
        &self,
        input: &Path,
        out: &Path,
        seed: u64,
        fast_ar: bool,
        reference: Option<&crate::reference_cache::PreparedReference>,
        capture: bool,
    ) -> Result<()> {
        std::fs::create_dir_all(out)?;
        if std::fs::canonicalize(input)? == std::fs::canonicalize(out)? {
            return Err("Input and output directories must be distinct".into());
        }
        if std::fs::read_dir(out)?.next().is_some() {
            return Err("Seeded synthesis output directory must be empty".into());
        }
        if capture {
            std::fs::create_dir(out.join("q"))?;
        }
        let mut rng = crate::rng::RequestRng::new(seed);
        // Create the fixed v2 field before the AR target is known. The v1 path
        // does not allocate a field or consume any CFM draw here.
        let coordinate_field = if self.cfm_coordinate_v2 {
            let field = crate::cfm_noise_field::CfmNoiseField::new(&mut rng)?;
            if capture {
                field
                    .field()
                    .save_numpy(out.join("cfm-coordinate-field.npy"))?;
            }
            Some(field)
        } else {
            None
        };
        let summary = self.run_inner(
            input,
            out,
            Some(&mut rng),
            fast_ar,
            reference,
            capture,
            coordinate_field.as_ref(),
        )?;
        let provenance = if self.precision == Dtype::Float16 {
            std::fs::copy(
                &self.vocoder_manifest,
                out.join("vocoder-source.manifest.json"),
            )?;
            "vocoder-source.manifest.json"
        } else {
            "not_recorded"
        };
        let control = format!(
            concat!("{{\n  \"kind\": \"seeded_native_single_window\",\n",
                    "  \"rng_policy\": \"{}\",\n  \"seed_u64\": \"{}\",\n",
                    "  \"dtype\": \"{}\",\n  \"vocoder_checkpoint_dtype\": \"float32\",\n",
                    "  \"vocoder_runtime_dtype\": \"{}\",\n  \"vocoder_runtime_cast\": \"{}\",\n",
                    "  \"vocoder_source_manifest\": \"{}\",\n  \"source_hash_runtime_revalidated\": false,\n",
                    "  \"noise_dtype_policy\": \"AR exponential and CFM normal drawn float32 then cast to runtime dtype\",\n",
                    "  \"ar_domain_hex\": \"{:016x}\",\n  \"cfm_domain_hex\": \"{:016x}\",\n",
                    "  \"ar_domain_seed_u64\": \"{}\",\n  \"cfm_domain_seed_u64\": \"{}\",\n",
                    "  \"key_derivation\": \"splitmix64(domain_seed XOR zero_based_draw_index)\",\n",
                    "  \"exponential_draw\": \"-log(max(MLX uniform [0,1), float32 minimum positive normal))\",\n",
                    "  \"ar_draws\": {},\n  \"cfm_draws\": {},\n  \"q_includes_eos_step\": true,\n",
                    "  \"semantic_target_length\": {},\n  \"eos_sampled\": {},\n  \"eos_argmax\": {},\n",
                    "  \"cfm_noise_shape\": [1, 100, {}],\n  \"cfm_noise_unscaled\": true,\n",
                    "  \"cfm_temperature\": 0.875,\n  \"cfm_steps\": 4,\n  \"cfg_rate\": 0,\n",
                    "  \"top_k\": 15,\n  \"top_p\": 1,\n  \"temperature\": 1,\n",
                    "  \"repetition_penalty\": 1.35,\n  \"sample_rate\": 48000,\n",
                    "  \"waveform_samples\": {}\n}}\n"),
            crate::rng::POLICY, seed,
            if self.precision == Dtype::Float16 { "float16" } else { "float32" },
            if self.precision == Dtype::Float16 { "float16" } else { "float32" },
            if self.precision == Dtype::Float16 { "official_is_half" } else { "none" },
            provenance,
            crate::rng::AR_DOMAIN, crate::rng::CFM_DOMAIN,
            rng.ar_seed(), rng.cfm_seed(), rng.ar_draws(), rng.cfm_draws(),
            summary.tokens, summary.eos_sampled, summary.eos_argmax,
            summary.noise_frames, summary.samples,
        );
        let mut control: serde_json::Value = serde_json::from_str(&control)?;
        control["top_k"] = serde_json::json!(self.sampling.top_k);
        control["top_p"] = serde_json::json!(self.sampling.top_p);
        control["temperature"] = serde_json::json!(self.sampling.temperature);
        control["sampling_default"] = serde_json::json!(self.sampling.is_default());
        control["cfm_noise_policy"] = serde_json::json!(if self.cfm_coordinate_v2 {
            crate::cfm_noise_field::POLICY
        } else {
            crate::rng::POLICY
        });
        if self.cfm_coordinate_v2 {
            control["coordinate_field_created_before_ar"] = serde_json::json!(true);
            control["historical_v1_noise_mapping_preserved"] = serde_json::json!(false);
            control["quality_qualified"] = serde_json::json!(false);
        }
        if summary.cfm_windows.len() > 1 {
            control["kind"] = serde_json::json!("seeded_native_rolling_complete");
            control["cfm_noise_shape"] = serde_json::Value::Null;
        }
        control["cfm_windows"] = serde_json::json!(summary.cfm_windows);
        control["complete_acoustic_windows"] = serde_json::json!(summary.cfm_windows.len());
        control["complete_vocoder_calls"] = serde_json::json!(1);
        control["true_incremental_stream"] = serde_json::json!(false);
        control["cfm_isolated_tf32"] = serde_json::json!(self.cfm_stage_enabled());
        control["cfm_compiled_stage"] = serde_json::json!(self.compiled_cfm_stage_enabled());
        control["cfm_batched_qkv"] = serde_json::json!(self.batched_cfm_qkv_enabled());
        control["cfm_packed_qkv_blocks"] = serde_json::json!(if self.batched_cfm_qkv_enabled() {
            22
        } else {
            0
        });
        control["cfm_static_prepared"] = serde_json::json!(self.cfm_schedule.is_some());
        control["reference_conditioning_reused"] = serde_json::json!(reference.is_some());
        control["stochastic_and_neural_capture"] = serde_json::json!(capture);
        control["ar_lossless_f16_storage_gemv"] = serde_json::json!(self.mixed_ar_gemv_enabled());
        control["ar_shapeless_compiled"] = serde_json::json!(self.compiled_ar_enabled());
        control["ar_compile_trace_count"] = serde_json::json!(self.ar_compile_trace_count());
        control["ar_combined_decision"] = serde_json::json!(self.ar_combined);
        control["ar_optimized"] = serde_json::json!(self.ar_optimized);
        control["ar_static_decode"] = serde_json::json!(self.static_ar_enabled());
        control["ar_residual_norm_fused"] = serde_json::json!(self.residual_ar_norm_enabled());
        control["ar_static_cache"] = serde_json::json!(self.static_ar_cache_stats());
        control["ar_batch_size"] = serde_json::json!(self.ar_batch_size());
        control["ar_gemv_rows"] = serde_json::json!(self.ar_gemv_rows());
        control["ar_fused_epilogue"] = serde_json::json!(self.ar_fused_epilogue_enabled());
        control["ar_discarded_speculative_draws"] = serde_json::json!(summary.ar_speculative_draws);
        control["ar_decode_validation"] = serde_json::json!(if self.ar_optimized {
            "gpu_sampled_token"
        } else if fast_ar || self.ar_combined {
            "known_sampled_token"
        } else {
            "full_array_validation"
        });
        control["stage_timings_ms"] = serde_json::json!({
            "reference_and_prepared_input_io": summary.timings.reference_ms,
            "ar_prefill_decode_sampling_and_q_io": summary.timings.autoregressive_ms,
            "acoustic_reference_and_target_with_output_io": summary.timings.acoustic_ms,
            "cfm_preparation_four_steps_and_mel_io": summary.timings.cfm_ms,
            "vocoder_denorm_through_wave_eval": summary.timings.vocoder_ms,
            "wave_save_and_pcm_write": summary.timings.pcm_write_ms,
            "run_inner_total": summary.timings.total_ms,
        });
        control["timing_scope"] = serde_json::json!(
            "elapsed wall time, includes existing diagnostic file I/O and synchronization; lazy evaluation may shift work across stages; not GPU compute time or HTTP latency; run_inner_total excludes model load and control/manifest writes"
        );
        std::fs::write(
            out.join("run-control.json"),
            serde_json::to_vec_pretty(&control)?,
        )?;
        Ok(())
    }

    fn run_inner(
        &self,
        input: &Path,
        out: &Path,
        mut rng: Option<&mut crate::rng::RequestRng>,
        fast_ar: bool,
        reference: Option<&crate::reference_cache::PreparedReference>,
        capture: bool,
        coordinate_field: Option<&crate::cfm_noise_field::CfmNoiseField>,
    ) -> Result<RunSummary> {
        if self.ar_optimized && rng.is_none() {
            return Err("Optimized AR requires request-local native RNG; captured-file replay is unsupported".into());
        }
        let sampling = if rng.is_some() {
            self.sampling
        } else {
            crate::sampling::SamplingParams::DEFAULT
        };
        let total_start = Instant::now();
        let reference_start = total_start;
        let (phones, bert, reference_phones, target_phones) = load_aligned_frontend(input)?;
        let bert = bert.as_dtype(self.precision)?;
        std::fs::create_dir_all(out)?;
        let (prompt_codes, ge) = if let Some(reference) = reference {
            (reference.prompt_codes.clone(), reference.ge.clone())
        } else {
            let waveform = load(input.join("waveform.npy"))?.as_dtype(self.precision)?;
            let hidden = self.hubert.forward(&waveform)?;
            checked_dtype(&hidden, self.precision, "HuBERT hidden state")?;
            let (projected, prompt_codes) = self.semantic.forward(&hidden)?;
            checked_dtype(&projected, self.precision, "semantic projection")?;
            let refer = load(input.join("refer.npy"))?.as_dtype(self.precision)?;
            if refer.ndim() != 3 || refer.shape()[0] != 1 || refer.shape()[1] < 704 {
                return Err("Invalid reference spectrum".into());
            }
            let ge = self.reference.forward(&refer.index((.., 0..704, ..)))?;
            (prompt_codes, ge)
        };
        prompt_codes.eval()?;
        if capture {
            prompt_codes.save_numpy(out.join("prompt-codes.npy"))?;
        }
        checked_dtype(&ge, self.precision, "reference style")?;
        ge.eval()?;
        if capture {
            ge.save_numpy(out.join("ge.npy"))?;
        }
        let reference_ms = reference_start.elapsed().as_secs_f64() * 1000.0;
        let ar_start = Instant::now();
        let prompt = prompt_codes
            .reshape(&[1, prompt_codes.shape()[2]])?
            .as_dtype(Dtype::Int32)?;
        let mut previous: Vec<i32> = prompt.as_slice::<i32>().to_vec();
        let (mut logits, mut state) = self.ar.prefill(&phones, &bert, &prompt)?;
        checked_dtype(&logits, self.precision, "AR prefill logits")?;
        let mut generated = Vec::new();
        let mut ended = false;
        let mut eos_sampled = 0;
        let mut eos_argmax = 0;
        let ar_trace = std::env::var_os("ANIFLIVE_AR_LOGITS_TRACE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                (std::env::var("ANIFLIVE_AR_LOGITS_TRACE").as_deref() == Ok("1"))
                    .then(|| out.join("ar-logits"))
            });
        let mut decisions = Vec::new();
        if let Some(path) = &ar_trace {
            std::fs::create_dir_all(path)?;
        }
        let mut ar_speculative_draws = 0u64;
        if self.ar_optimized {
            let mut generator = crate::ar_microbatch::ArGenerator::from_prefill(
                &self.ar, &prompt, logits, state, 4,
            )?;
            generator.set_sampling(sampling)?;
            self.attach_static_ar(&mut generator)?;
            let request_rng = rng
                .as_deref_mut()
                .ok_or("Optimized AR requires native RNG")?;
            loop {
                self.check_complete_cancel()?;
                let batch = generator.next_batch(request_rng, 4)?;
                ar_speculative_draws += batch.speculative_draws;
                for record in &batch.records {
                    if let Some(path) = &ar_trace {
                        record
                            .logits
                            .save_numpy(path.join(format!("{:04}.npy", record.step)))?;
                    }
                    if capture {
                        record
                            .q
                            .save_numpy(out.join(format!("q/{:04}.npy", record.step)))?;
                        decisions.push(serde_json::json!({"step":record.step,"sampled":record.sampled,"greedy":record.greedy,"vocab":record.vocab}));
                    }
                }
                generated.extend(batch.tokens);
                if batch.finished {
                    let last = batch
                        .records
                        .last()
                        .ok_or("Optimized AR missing EOS record")?;
                    eos_sampled = last.sampled;
                    eos_argmax = last.greedy;
                    ended = true;
                    break;
                }
            }
        } else {
            for step in 0..1500 {
                self.check_complete_cancel()?;
                if let Some(path) = &ar_trace {
                    logits.save_numpy(path.join(format!("{step:04}.npy")))?;
                }

                let logits_step = if step < 11 {
                    logits.index((.., 0..1024))
                } else {
                    logits.clone()
                };
                let adjusted =
                    crate::sampling::apply_repetition_penalty(&logits_step, &previous, 1.35)?;
                let q = if let Some(rng) = rng.as_deref_mut() {
                    let q = rng
                        .ar_exponential(logits_step.shape())?
                        .as_dtype(logits_step.dtype())?;
                    if !self.ar_combined {
                        let minimum = q.min(false)?.try_item_cast::<f32>()?;
                        if !minimum.is_finite() || minimum <= 0.0 {
                            return Err("AR exponential draw is not strictly positive after precision conversion".into());
                        }
                        q.eval()?;
                        if capture {
                            q.save_numpy(out.join(format!("q/{step:04}.npy")))?;
                        }
                    }
                    q
                } else {
                    load(input.join(format!("q/{step:04}.npy")))?.as_dtype(logits_step.dtype())?
                };
                let token =
                    crate::sampling::sample(&adjusted, &[], &q, sampling.top_k, sampling.top_p, sampling.temperature, 1.0)?
                        .as_dtype(Dtype::Int32)?;
                let (id, greedy_id) = if self.ar_combined {
                    let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
                    let decision = crate::ar_decision::read_decision(&q, &token, &greedy)?;
                    if capture && rng.is_some() {
                        q.save_numpy(out.join(format!("q/{step:04}.npy")))?;
                    }
                    decision
                } else {
                    token.eval()?;
                    let id = token.item_exact::<i32>();
                    let greedy = argmax_axis(&adjusted, -1, false)?.as_dtype(Dtype::Int32)?;
                    greedy.eval()?;
                    (id, greedy.item_exact::<i32>())
                };
                if capture {
                    decisions.push(serde_json::json!({"step":step,"sampled":id,"greedy":greedy_id,"vocab":q.shape()[1]}));
                }
                if id == 1024 || greedy_id == 1024 {
                    eos_sampled = id;
                    eos_argmax = greedy_id;
                    ended = true;
                    break;
                }
                generated.push(id);
                previous.push(id);
                logits =
                    self.decode_ar_generated(&token, id, &mut state, fast_ar || self.ar_combined)?;
                checked_dtype(&logits, self.precision, "AR decode logits")?;
            }
        }
        if !ended || generated.is_empty() {
            return Err("AR did not produce a complete nonempty EOS-terminated utterance".into());
        }
        if capture {
            std::fs::write(
                out.join("ar-decisions.json"),
                serde_json::to_vec_pretty(&decisions)?,
            )?;
        }
        let codes = Array::from_slice(&generated, &[1, 1, generated.len() as i32]);
        codes.eval()?;
        if capture {
            codes.save_numpy(out.join("target-codes.npy"))?;
        }
        let autoregressive_ms = ar_start.elapsed().as_secs_f64() * 1000.0;
        let acoustic_start = Instant::now();
        let ref_features = if let Some(reference) = reference {
            reference.features.clone()
        } else {
            self.acoustic
                .forward(&prompt_codes, &reference_phones, &ge)?
        };
        let target = self.acoustic.forward(&codes, &target_phones, &ge)?;
        checked_dtype(&ref_features, self.precision, "reference acoustic features")?;
        checked_dtype(&target, self.precision, "target acoustic features")?;
        ref_features.eval()?;
        if capture {
            ref_features.save_numpy(out.join("reference-features.npy"))?;
        }
        target.eval()?;
        if capture {
            target.save_numpy(out.join("target-features.npy"))?;
        }
        let acoustic_ms = acoustic_start.elapsed().as_secs_f64() * 1000.0;
        self.check_complete_cancel()?;
        let cfm_start = Instant::now();
        let mel = if let Some(reference) = reference {
            reference.mel.clone()
        } else {
            load(input.join("reference-mel.npy"))?
                .add(Array::from_f32(12.0))?
                .divide(Array::from_f32(14.0))?
                .multiply(Array::from_f32(2.0))?
                .subtract(Array::from_f32(1.0))?
                .as_dtype(self.precision)?
        };
        let end = mel.shape()[2].min(ref_features.shape()[2]);
        let start = (end - 500).max(0);
        let r = end - start;
        let t = target.shape()[2];
        if r <= 0 || t <= 0 {
            return Err("Complete synthesis requires nonempty reference and target frames".into());
        }
        let chunk_frames = (1000 - r).min(640);
        let original_mel = mel.index((.., .., start..end));
        let original_features = ref_features.index((.., .., start..end));
        let rolling = t > chunk_frames;
        if rolling && capture {
            std::fs::create_dir(out.join("cfm-noise"))?;
            std::fs::create_dir(out.join("cfm-mel"))?;
        }
        let mut prior: Option<(Array, Array)> = None;
        let mut pieces = Vec::new();
        let mut cfm_windows = Vec::new();
        for (index, offset) in (0..t).step_by(chunk_frames as usize).enumerate() {
            self.check_complete_cancel()?;
            let current_frames = (t - offset).min(chunk_frames);
            let current = if !rolling {
                target.clone()
            } else {
                target.index((.., .., offset..offset + current_frames))
            };
            // Official synthesize_v5_mel: keep the original reference anchor,
            // replacing only its final <=32 frames with the preceding output.
            let (mel, feature) = if let Some((previous_mel, previous_features)) = &prior {
                let tail = 32
                    .min(r)
                    .min(previous_mel.shape()[2])
                    .min(previous_features.shape()[2]);
                let prefix = r - tail;
                (
                    concatenate(
                        &[
                            original_mel.index((.., .., 0..prefix)),
                            previous_mel.index((.., .., previous_mel.shape()[2] - tail..)),
                        ],
                        2,
                    )?,
                    concatenate(
                        &[
                            original_features.index((.., .., 0..prefix)),
                            previous_features.index((
                                ..,
                                ..,
                                previous_features.shape()[2] - tail..,
                            )),
                        ],
                        2,
                    )?,
                )
            } else {
                (original_mel.clone(), original_features.clone())
            };
            let noise_file = if rolling {
                format!("cfm-noise/{index:04}.npy")
            } else {
                "cfm-noise.npy".to_owned()
            };
            let mu = concatenate(&[feature, current.clone()], 2)?.transpose_axes(&[0, 2, 1])?;
            checked_dtype(&mu, self.precision, "CFM conditioning")?;
            let prompt_mel = concatenate(
                &[
                    mel,
                    zeros::<f32>(&[1, 100, current_frames])?.as_dtype(self.precision)?,
                ],
                2,
            )?;
            checked_dtype(&prompt_mel, self.precision, "CFM prompt")?;
            let mask: Vec<bool> = (0..r + current_frames).map(|i| i >= r).collect();
            let mask = Array::from_slice(&mask, &[1, 1, r + current_frames]);
            let noise = if let Some(field) = coordinate_field {
                let next_field = if index == 0 {
                    if rng
                        .as_deref()
                        .ok_or("Coordinate synthesis requires request RNG")?
                        .cfm_draws()
                        != 1
                    {
                        return Err("Coordinate synthesis did not retain field draw zero".into());
                    }
                    None
                } else {
                    let request_rng = rng
                        .as_deref_mut()
                        .ok_or("Coordinate rolling synthesis requires request RNG")?;
                    if request_rng.cfm_draws() != index as u64 {
                        return Err(
                            "Coordinate rolling CFM draw index differs from block index".into()
                        );
                    }
                    Some(crate::cfm_noise_field::CfmNoiseField::next(request_rng)?)
                };
                let current_field = next_field.as_ref().unwrap_or(field);
                let noise = current_field.cumulative(r, current_frames)?;
                noise.eval()?;
                if capture {
                    noise.save_numpy(out.join(&noise_file))?;
                }
                noise
            } else if let Some(rng) = rng.as_deref_mut() {
                let noise = rng
                    .cfm_normal(&[1, 100, r + current_frames])?
                    .as_dtype(self.precision)?;
                noise.eval()?;
                if capture {
                    noise.save_numpy(out.join(&noise_file))?;
                }
                noise
            } else {
                load(input.join(&noise_file))?.as_dtype(self.precision)?
            };
            if noise.shape() != [1, 100, r + current_frames] {
                return Err(
                    "Captured CFM noise shape differs from actual semantic conditioning".into(),
                );
            }
            let mut x = select(
                &mask,
                noise.multiply(scalar(0.875, self.precision)?)?,
                scalar(0.0, self.precision)?,
            )?;
            checked_dtype(&x, self.precision, "CFM initial state")?;
            if self.cfm_stage_enabled() {
                let mut connection = self.cfm_stage.borrow_mut();
                x = connection.as_mut().ok_or("CFM stage unavailable")?.infer(
                    &mu,
                    &prompt_mel,
                    &noise,
                    r,
                )?;
            } else {
                let prepared_context = self
                    .cfm_schedule
                    .as_ref()
                    .map(|_| self.estimator.prepare_context(&mu))
                    .transpose()?;
                let mut static_condition: Option<Array> = None;
                for step in 0..4 {
                    let v = if let (Some(context), Some(schedule)) =
                        (&prepared_context, &self.cfm_schedule)
                    {
                        self.estimator.forward_prepared(
                            &x,
                            &prompt_mel,
                            context,
                            schedule,
                            step,
                            None,
                        )?
                    } else {
                        self.estimator.forward_cached(
                            &x,
                            &prompt_mel,
                            &mu,
                            &Array::from_slice(&[step as f32 * 0.25], &[1])
                                .as_dtype(self.precision)?,
                            &mut static_condition,
                        )?
                    };
                    checked_dtype(&v, self.precision, "CFM estimator velocity")?;
                    x = select(
                        &mask,
                        x.add(v.multiply(scalar(0.25, self.precision)?)?)?,
                        scalar(0.0, self.precision)?,
                    )?;
                    checked_dtype(&x, self.precision, "CFM updated state")?;
                }
            }
            let generated_mel = x.index((.., .., r..));
            generated_mel.eval()?;
            if capture && rolling {
                generated_mel.save_numpy(out.join(format!("cfm-mel/{index:04}.npy")))?;
            }
            let mut window = serde_json::json!({
                "index":index,"target_start_frame":offset,"target_frames":current_frames,
                "reference_frames":r,"noise_shape":[1,100,r+current_frames],
                "noise_file":if capture {Some(noise_file)} else {None},
                "cfm_steps":4,"cfg_rate":0,
            });
            if coordinate_field.is_some() {
                window["cfm_field_draw_index"] = serde_json::json!(index);
            }
            cfm_windows.push(window);
            prior = Some((generated_mel.clone(), current));
            pieces.push(generated_mel);
        }
        // Complete-WAV path only: all mel chunks are joined before one vocoder
        // call, exactly as the official V5 implementation. No streaming claim.
        let mel = if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            concatenate(&pieces, 2)?
        };
        if mel.shape() != [1, 100, t] {
            return Err("Rolling CFM lost or repeated target frames".into());
        }
        if capture {
            mel.save_numpy(out.join("mel.npy"))?;
        }
        let cfm_ms = cfm_start.elapsed().as_secs_f64() * 1000.0;
        self.check_complete_cancel()?;
        let vocoder_start = Instant::now();
        let raw = mel
            .add(scalar(1.0, self.precision)?)?
            .divide(scalar(2.0, self.precision)?)?
            .multiply(scalar(14.0, self.precision)?)?
            .subtract(scalar(12.0, self.precision)?)?;
        checked_dtype(&raw, self.precision, "vocoder input")?;
        let wave = self.vocoder.forward(&raw)?;
        checked_dtype(&wave, self.precision, "vocoder output")?;
        wave.eval()?;
        let vocoder_ms = vocoder_start.elapsed().as_secs_f64() * 1000.0;
        let pcm_start = Instant::now();
        if capture {
            wave.save_numpy(out.join("wave.npy"))?;
        }
        let wave_for_pcm = wave.as_dtype(Dtype::Float32)?;
        wave_for_pcm.eval()?;
        crate::audio::write_pcm16(
            wave_for_pcm.as_slice::<f32>(),
            &out.join("native-wave.wav"),
            &out.join("native-wave.pcm"),
        )?;
        let pcm_write_ms = pcm_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        if self.emit_cli_reports {
            let kind = if cfm_windows.len() == 1 {
                "full_neural_single_window"
            } else {
                "full_neural_rolling_complete"
            };
            println!("{{\"kind\":\"{}\",\"semantic_tokens\":{},\"samples\":{},\"acoustic_windows\":{},\"sample_rate\":48000,\"http_benchmark\":false}}",kind,generated.len(),wave.size(),cfm_windows.len());
        }
        Ok(RunSummary {
            timings: StageTimings {
                reference_ms,
                autoregressive_ms,
                acoustic_ms,
                cfm_ms,
                vocoder_ms,
                pcm_write_ms,
                total_ms,
            },
            tokens: generated.len(),
            eos_sampled,
            eos_argmax,
            ar_speculative_draws,
            noise_frames: r + t,
            cfm_windows,
            samples: wave.size(),
        })
    }
}
