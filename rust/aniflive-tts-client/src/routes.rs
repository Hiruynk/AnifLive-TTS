use crate::ClientError;
use url::Url;

pub(crate) const HEALTH: &str = "health";
pub(crate) const CAPABILITIES: &str = "v1/capabilities";
pub(crate) const MODELS: &str = "v1/models";
pub(crate) const VOICES: &str = "v1/voices";
pub(crate) const EXPRESSIONS: &str = "v1/expressions";
pub(crate) const ACTIVATE: &str = "v1/models/activate";
pub(crate) const SPEECH: &str = "v1/audio/speech";
pub(crate) const CANCEL: &str = "v1/audio/cancel";

pub(crate) fn join(base: &Url, route: &str) -> Result<Url, ClientError> {
    base.join(route)
        .map_err(|_| ClientError::InvalidConfiguration)
}

pub(crate) fn session(base: &Url, id: &str, suffix: Option<&str>) -> Result<Url, ClientError> {
    if id.is_empty() || id == "." || id == ".." {
        return Err(ClientError::InvalidConfiguration);
    }
    let mut url = join(base, "v1/sessions")?;
    {
        let mut parts = url
            .path_segments_mut()
            .map_err(|_| ClientError::InvalidConfiguration)?;
        parts.push(id);
        if let Some(suffix) = suffix {
            parts.push(suffix);
        }
    }
    Ok(url)
}

pub(crate) fn voice(
    base: &Url,
    id: &str,
    reference: Option<&str>,
    suffix: Option<&str>,
) -> Result<Url, ClientError> {
    if id.is_empty()
        || id == "."
        || id == ".."
        || reference.is_some_and(|v| v.is_empty() || v == "." || v == "..")
    {
        return Err(ClientError::InvalidConfiguration);
    }
    let mut url = join(base, VOICES)?;
    {
        let mut parts = url
            .path_segments_mut()
            .map_err(|_| ClientError::InvalidConfiguration)?;
        parts.push(id);
        if reference.is_some() || suffix == Some("references") {
            parts.push("references");
        }
        if let Some(reference) = reference {
            parts.push(reference);
        }
        if let Some(suffix) = suffix
            && suffix != "references"
        {
            parts.push(suffix);
        }
    }
    Ok(url)
}
