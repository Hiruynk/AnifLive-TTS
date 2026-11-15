//! Strict, identity-keyed persistence for prepared reference tensors.
use mlx_rs::{Array, Device, Dtype};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub type BoxError = Box<dyn Error>;
type Result<T> = std::result::Result<T, BoxError>;

const IDENTITY_SCHEMA: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const TENSOR_FILE: &str = "prepared.safetensors";
const TENSOR_NAMES: [&str; 5] = ["prompt_codes", "ge", "spectrum", "features", "mel"];
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceIdentity {
    pub schema: u32,
    pub reference_audio_sha256: String,
    pub transcript_sha256: String,
    pub models_sha256: BTreeMap<String, String>,
    pub prepared_inputs_sha256: BTreeMap<String, String>,
    pub precision: String,
    pub preprocessing_version: String,
}

#[derive(Debug)]
pub struct PreparedReference {
    pub prompt_codes: Array,
    pub ge: Array,
    pub spectrum: Array,
    pub features: Array,
    pub mel: Array,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    schema: u32,
    identity: ReferenceIdentity,
    identity_key: String,
    tensor_file_sha256: String,
}

fn invalid(message: impl Into<String>) -> BoxError {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Hash a file incrementally, returning a lowercase SHA256 hex string.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let bytes = digest.finalize();
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(output)
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_identity(identity: &ReferenceIdentity) -> Result<()> {
    if identity.schema != IDENTITY_SCHEMA {
        return Err(invalid(format!(
            "unsupported reference identity schema {}",
            identity.schema
        )));
    }
    if !is_lower_sha256(&identity.reference_audio_sha256)
        || !is_lower_sha256(&identity.transcript_sha256)
    {
        return Err(invalid(
            "reference and transcript identities must be lowercase SHA256",
        ));
    }
    if identity.models_sha256.is_empty() || identity.prepared_inputs_sha256.is_empty() {
        return Err(invalid(
            "model and prepared-input SHA256 maps must not be empty",
        ));
    }
    if identity
        .models_sha256
        .values()
        .chain(identity.prepared_inputs_sha256.values())
        .any(|value| !is_lower_sha256(value))
    {
        return Err(invalid(
            "identity maps must contain lowercase SHA256 values",
        ));
    }
    if identity.precision.trim().is_empty() || identity.preprocessing_version.trim().is_empty() {
        return Err(invalid(
            "precision and preprocessing version must be nonempty",
        ));
    }
    Ok(())
}

/// Hash deterministic serde JSON for the reference identity and its sorted maps.
pub fn identity_key(identity: &ReferenceIdentity) -> Result<String> {
    validate_identity(identity)?;
    Ok(sha256_bytes(&serde_json::to_vec(identity)?))
}

impl PreparedReference {
    /// Validate tensor names, dtypes, and shape invariants before saving or use.
    pub fn validate(&self) -> Result<()> {
        if self.prompt_codes.ndim() != 3
            || self.prompt_codes.shape()[0] != 1
            || self.prompt_codes.shape()[1] != 1
            || self.prompt_codes.shape()[2] <= 0
        {
            return Err(invalid(format!(
                "prompt_codes must have shape [1,1,N] with N>0, got {:?}",
                self.prompt_codes.shape()
            )));
        }
        if !matches!(
            self.prompt_codes.dtype(),
            Dtype::Int8
                | Dtype::Int16
                | Dtype::Int32
                | Dtype::Int64
                | Dtype::Uint8
                | Dtype::Uint16
                | Dtype::Uint32
                | Dtype::Uint64
        ) {
            return Err(invalid("prompt_codes must have an integer dtype"));
        }
        let precision = self.ge.dtype();
        if !matches!(precision, Dtype::Float32 | Dtype::Float16) {
            return Err(invalid("reference tensors must be float32 or float16"));
        }
        if self.ge.shape() != [1, 512, 1] {
            return Err(invalid(format!(
                "ge must be [1,512,1], got {:?} {:?}",
                self.ge.shape(),
                self.ge.dtype()
            )));
        }
        if self.spectrum.ndim() != 3
            || self.spectrum.shape()[0] != 1
            || self.spectrum.shape()[1] < 704
            || self.spectrum.shape()[2] <= 0
            || self.spectrum.dtype() != precision
        {
            return Err(invalid(format!(
                "spectrum must match reference precision [1,>=704,T] with T>0, got {:?} {:?}",
                self.spectrum.shape(),
                self.spectrum.dtype()
            )));
        }
        if self.features.ndim() != 3
            || self.features.shape()[0] != 1
            || self.features.shape()[1] != 512
            || self.mel.ndim() != 3
            || self.mel.shape()[0] != 1
            || self.mel.shape()[1] != 100
            || self.features.shape()[2] <= 0
            || self.features.shape()[2] > 500
            || self.mel.shape()[2] != self.features.shape()[2]
            || self.features.dtype() != precision
            || self.mel.dtype() != precision
        {
            return Err(invalid(format!(
                "features and mel must match reference precision [1,512,R] and [1,100,R] with 1<=R<=500; got {:?} {:?} and {:?} {:?}",
                self.features.shape(),
                self.features.dtype(),
                self.mel.shape(),
                self.mel.dtype()
            )));
        }
        Ok(())
    }

    pub fn validate_precision(&self, identity: &ReferenceIdentity) -> Result<()> {
        let expected = match identity.precision.as_str() {
            "float32" => Dtype::Float32,
            "float16" => Dtype::Float16,
            _ => return Err(invalid("unsupported reference cache precision")),
        };
        if self.ge.dtype() != expected {
            return Err(invalid(
                "reference tensor dtype differs from cache identity",
            ));
        }
        Ok(())
    }

    /// Save this reference into a new identity-keyed cache directory.
    /// Existing entries are never overwritten.
    pub fn save(&self, root: impl AsRef<Path>, identity: &ReferenceIdentity) -> Result<PathBuf> {
        self.validate()?;
        self.validate_precision(identity)?;
        let key = identity_key(identity)?;
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let root_metadata = fs::symlink_metadata(root)?;
        if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
            return Err(invalid("cache root must be a regular directory"));
        }
        let final_dir = root.join(&key);
        if path_exists(&final_dir)? {
            return Err(invalid(format!(
                "reference cache already exists for identity {key}"
            )));
        }
        let temporary_dir = create_temporary_dir(root, &key)?;
        let write_result = self.write_cache(&temporary_dir, identity, &key);
        if let Err(error) = write_result {
            let _ = fs::remove_dir_all(&temporary_dir);
            return Err(error);
        }
        if path_exists(&final_dir)? {
            let _ = fs::remove_dir_all(&temporary_dir);
            return Err(invalid(format!(
                "reference cache already exists for identity {key}"
            )));
        }
        if let Err(error) = fs::rename(&temporary_dir, &final_dir) {
            let _ = fs::remove_dir_all(&temporary_dir);
            return Err(Box::new(error));
        }
        Ok(final_dir)
    }

    fn write_cache(
        &self,
        temporary_dir: &Path,
        identity: &ReferenceIdentity,
        key: &str,
    ) -> Result<()> {
        let tensor_path = temporary_dir.join(TENSOR_FILE);
        let tensors = BTreeMap::from([
            ("prompt_codes", self.prompt_codes.clone()),
            ("ge", self.ge.clone()),
            ("spectrum", self.spectrum.clone()),
            ("features", self.features.clone()),
            ("mel", self.mel.clone()),
        ]);
        mlx_rs::with_stream(
            &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
            || Array::save_safetensors(&tensors, None, &tensor_path),
        )?;
        let manifest = CacheManifest {
            schema: IDENTITY_SCHEMA,
            identity: identity.clone(),
            identity_key: key.to_owned(),
            tensor_file_sha256: sha256_file(&tensor_path)?,
        };
        let manifest_path = temporary_dir.join(MANIFEST_FILE);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(manifest_path)?;
        file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }
}

/// Load an identity-matched reference. A missing cache directory returns `Ok(None)`;
/// any existing but invalid entry returns an error.
pub fn load(
    root: impl AsRef<Path>,
    identity: &ReferenceIdentity,
) -> Result<Option<PreparedReference>> {
    let key = identity_key(identity)?;
    let root = root.as_ref();
    match fs::symlink_metadata(root) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Box::new(error)),
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(invalid("cache root must be a regular directory"));
        }
        Ok(_) => {}
    }
    let cache_dir = root.join(&key);
    match fs::symlink_metadata(&cache_dir) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Box::new(error)),
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(invalid("existing cache entry is not a regular directory"));
        }
        Ok(_) => {}
    }

    let manifest_path = cache_dir.join(MANIFEST_FILE);
    require_regular_file(&manifest_path, "cache manifest")?;
    let manifest: CacheManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema != IDENTITY_SCHEMA {
        return Err(invalid("unsupported reference cache manifest schema"));
    }
    if manifest.identity != *identity || manifest.identity_key != key {
        return Err(invalid(
            "cache manifest identity does not match requested identity",
        ));
    }
    if !is_lower_sha256(&manifest.tensor_file_sha256) {
        return Err(invalid("cache manifest contains an invalid tensor SHA256"));
    }

    let tensor_path = cache_dir.join(TENSOR_FILE);
    require_regular_file(&tensor_path, "cache tensor file")?;
    if sha256_file(&tensor_path)? != manifest.tensor_file_sha256 {
        return Err(invalid("reference cache tensor SHA256 mismatch"));
    }
    let mut tensors = mlx_rs::with_stream(
        &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
        || Array::load_safetensors(&tensor_path),
    )?;
    let expected: BTreeSet<&str> = TENSOR_NAMES.into_iter().collect();
    let actual: BTreeSet<&str> = tensors.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(invalid(format!(
            "reference cache tensor names mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    let prepared = PreparedReference {
        prompt_codes: tensors
            .remove("prompt_codes")
            .expect("validated tensor name"),
        ge: tensors.remove("ge").expect("validated tensor name"),
        spectrum: tensors.remove("spectrum").expect("validated tensor name"),
        features: tensors.remove("features").expect("validated tensor name"),
        mel: tensors.remove("mel").expect("validated tensor name"),
    };
    prepared.validate()?;
    prepared.validate_precision(identity)?;
    Ok(Some(prepared))
}

fn path_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(Box::new(error)),
    }
}

fn create_temporary_dir(root: &Path, key: &str) -> Result<PathBuf> {
    for _ in 0..128 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".{key}.tmp-{}-{counter}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Box::new(error)),
        }
    }
    Err(invalid(
        "could not allocate a unique temporary cache directory",
    ))
}

fn require_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!("{label} is not a regular file")));
    }
    Ok(())
}
