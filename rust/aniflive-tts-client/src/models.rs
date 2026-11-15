use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub type Extra = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Zh,
    Yue,
    En,
    Ja,
    Ko,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuityPolicy {
    A,
    B,
    C,
    D,
    E,
    F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConditioningPolicy {
    FullSwitch,
    IdentityLock,
    SemanticStyle,
    AcousticStyle,
    SvOnly,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Expression {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<ConditioningPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpressionSegment {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub language: Language,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<ExpressionSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<Expression>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
}
impl SynthesisRequest {
    pub fn new(text: impl Into<String>, language: Language) -> Self {
        Self {
            language,
            text: Some(text.into()),
            segments: None,
            model: None,
            voice_profile: None,
            expression: None,
            generation: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub service: String,
    pub version: String,
    pub ready: bool,
    pub status: String,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub cuda: Option<Value>,
    #[serde(default)]
    pub tensorrt: Option<Value>,
    #[serde(default)]
    pub gpu: Option<Value>,
    #[serde(default)]
    pub engine_fingerprint: Option<String>,
    #[serde(default)]
    pub engine_count: Option<u64>,
    #[serde(default)]
    pub engines: Option<BTreeMap<String, bool>>,
    #[serde(default)]
    pub reference: Option<Extra>,
    #[serde(default)]
    pub expression: Option<Extra>,
    #[serde(default)]
    pub streaming: Option<Extra>,
    #[serde(default)]
    pub request_segment_char_limit: Option<u64>,
    #[serde(default)]
    pub startup_warmup: Option<Value>,
    #[serde(default)]
    pub active_requests: Option<u64>,
    #[serde(default)]
    pub gpu_resource_lease: Option<Extra>,
    #[serde(default)]
    pub warm_retention: Option<Value>,
    #[serde(default)]
    pub switching: Option<bool>,
    #[serde(default)]
    pub available_model_count: Option<u64>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingCapabilities {
    pub pcm16: bool,
    pub wav: bool,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionCapabilities {
    pub native: bool,
    pub controlled_profiles: bool,
    #[serde(default)]
    pub continuous_vector: bool,
    #[serde(default)]
    pub segmented: bool,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLimits {
    pub maximum_open: u64,
    pub ttl_seconds: f64,
    pub maximum_segments: u64,
    pub maximum_characters: u64,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCapabilities {
    pub committed_segments: bool,
    pub pcm_stream: bool,
    #[serde(default)]
    pub cancel: bool,
    #[serde(default)]
    pub flush: bool,
    #[serde(default)]
    pub context_mode: Option<String>,
    #[serde(default)]
    pub default_policy: Option<String>,
    #[serde(default)]
    pub qualification: Option<String>,
    #[serde(default)]
    pub neural_state_continuity: Option<bool>,
    #[serde(default)]
    pub acoustic_latent_continuity: Option<bool>,
    #[serde(default)]
    pub limits: Option<SessionLimits>,
    #[serde(default)]
    pub policies: Extra,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub service: String,
    pub version: String,
    pub languages: Vec<String>,
    pub streaming: StreamingCapabilities,
    pub expression: ExpressionCapabilities,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub speech_sessions: Option<SessionCapabilities>,
    #[serde(default)]
    pub voice_management: Option<VoiceManagementCapabilities>,
    #[serde(default)]
    pub model_family: Option<String>,
    #[serde(default)]
    pub precision: Option<String>,
    #[serde(default)]
    pub pytorch_fallback: Option<bool>,
    #[serde(default)]
    pub adaptive_punctuation_segments: Option<bool>,
    #[serde(default)]
    pub warm_retention_seconds: Option<f64>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse<T> {
    pub object: String,
    pub data: Vec<T>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub active: bool,
    pub voice_profiles: Vec<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub model_family: Option<String>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub reference_language: String,
    pub reference_configured: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub default_reference_id: Option<String>,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub reference_count: u64,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionProfileInfo {
    pub id: String,
    pub intensity_levels: Vec<f64>,
    pub languages: Vec<String>,
    pub reference_languages: Vec<String>,
    pub preferred_policies: Vec<String>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionCatalogInfo {
    pub object: String,
    pub enabled: bool,
    pub default: Option<String>,
    pub profiles: Vec<ExpressionProfileInfo>,
    pub policies: Vec<String>,
    #[serde(default)]
    pub preferred_policy: Option<String>,
    #[serde(default)]
    pub playback_policy: Option<Extra>,
    #[serde(default)]
    pub runtime_policy: Option<Extra>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationResult {
    pub code: i64,
    pub message: String,
    pub changed: bool,
    pub model: String,
    pub voice: String,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub pytorch_fallback: Option<bool>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationResult {
    pub cancelled: bool,
    pub model: String,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredFeature {
    PcmStreaming,
    Expressions,
    SpeechSessions,
    VoiceManagement,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub health: Health,
    pub capabilities: Capabilities,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    pub sample_rate: Option<u32>,
    pub model: Option<String>,
    pub service_version: Option<String>,
    pub backend: Option<String>,
    /// Protocol metadata only; Set-Cookie and arbitrary proxy headers are excluded.
    pub headers: BTreeMap<String, String>,
}
#[derive(Clone)]
pub struct SynthesisResult {
    pub audio: Vec<u8>,
    pub content_type: String,
    pub metadata: SynthesisMetadata,
}
impl std::fmt::Debug for SynthesisResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynthesisResult")
            .field("audio_bytes", &self.audio.len())
            .field("content_type", &self.content_type)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCreate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuity_policy: Option<ContinuityPolicy>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRequest {
    pub segment_id: String,
    pub text: String,
    pub language: Language,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paragraph_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_after_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<Expression>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    pub segment_id: String,
    pub sequence: u64,
    pub characters: u64,
    pub language: String,
    pub paragraph_id: Option<String>,
    pub pause_after_ms: u64,
    pub state: String,
    pub created_at: String,
    pub updated_at: String,
    pub error: Option<String>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub mode: String,
    pub policy: String,
    pub neural_state_continuity: bool,
    pub acoustic_latent_continuity: bool,
    pub committed_segments: u64,
    #[serde(default)]
    pub previous_language: Option<String>,
    #[serde(default)]
    pub previous_expression: Option<String>,
    #[serde(default)]
    pub paragraph_id: Option<String>,
    #[serde(default)]
    pub previous_pause_ms: Option<u64>,
    #[serde(default)]
    pub previous_output_samples: Option<u64>,
    #[serde(default)]
    pub total_output_samples: Option<u64>,
    #[serde(default)]
    pub hard_boundary: Option<bool>,
    #[serde(default)]
    pub retained_phonemes: Option<u64>,
    #[serde(default)]
    pub retained_semantic_tokens: Option<u64>,
    #[serde(default)]
    pub retained_neural_bytes: Option<u64>,
    #[serde(default)]
    pub last_plan: Option<Extra>,
    #[serde(flatten)]
    pub extra: Extra,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub object: String,
    pub state: String,
    pub model: String,
    pub voice_profile: String,
    pub sample_rate: u32,
    pub format: String,
    pub continuity_policy: String,
    pub context: SessionContext,
    pub audio_consumer_attached: bool,
    pub created_at: String,
    pub updated_at: String,
    pub segments: Vec<SegmentInfo>,
    #[serde(default)]
    pub active_audio_cancelled: Option<bool>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceManagementCapabilities {
    #[serde(default)]
    pub crud: bool,
    #[serde(default)]
    pub reference_upload: bool,
    #[serde(default)]
    pub reference_prepare: bool,
    #[serde(default)]
    pub persistent: bool,
    #[serde(default)]
    pub transcript_required: bool,
    #[serde(default)]
    pub reference_seconds: Option<Extra>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct VoiceCreate {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct VoiceUpdate {
    pub name: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ReferenceCreate {
    pub reference_language: Language,
    pub transcript: String,
    pub transcript_source: String,
    pub transcript_confirmed: bool,
    pub emotion_label: String,
    pub intensity: f64,
    pub is_default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_provider: Option<String>,
}
impl ReferenceCreate {
    pub fn new(reference_language: Language, transcript: impl Into<String>) -> Self {
        Self {
            reference_language,
            transcript: transcript.into(),
            transcript_source: "manual".into(),
            transcript_confirmed: false,
            emotion_label: "Neutral".into(),
            intensity: 0.5,
            is_default: false,
            id: None,
            emotion_id: None,
            transcript_provider: None,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ReferenceUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_language: Option<Language>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_confirmed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_default: Option<bool>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ReferenceInfo {
    pub id: String,
    pub voice_id: String,
    pub emotion_id: String,
    pub emotion_label: String,
    pub intensity: f64,
    pub reference_language: String,
    pub transcript: String,
    pub transcript_source: String,
    pub transcript_confirmed: bool,
    #[serde(default)]
    pub transcript_provider: Option<String>,
    pub is_default: bool,
    pub duration_seconds: f64,
    pub audio_sha256: String,
    pub revision: u64,
    pub conditioning_revision: u64,
    pub prepare_state: String,
    #[serde(default)]
    pub error_code: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(flatten)]
    pub extra: Extra,
}
impl std::fmt::Debug for ReferenceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceInfo")
            .field("id", &self.id)
            .field("voice_id", &self.voice_id)
            .field("prepare_state", &self.prepare_state)
            .field("revision", &self.revision)
            .finish_non_exhaustive()
    }
}
