use aniflive_tts_client as core;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

pub fn error(py: Python<'_>, error: core::ClientError) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let class = py
            .import("aniflive_tts.sdk.errors")?
            .getattr(error.kind())?;
        let kwargs = PyDict::new(py);
        let message = match &error {
            core::ClientError::Api(api) => {
                kwargs.set_item("status_code", api.status_code)?;
                kwargs.set_item(
                    "server_code",
                    value(py, &api.server_code.clone().unwrap_or(Value::Null))?,
                )?;
                kwargs.set_item("retry_after", api.retry_after.as_deref())?;
                kwargs.set_item("diagnostic", api.diagnostic.as_deref())?;
                api.message.clone()
            }
            _ => error.to_string(),
        };
        Ok(PyErr::from_value(class.call((message,), Some(&kwargs))?))
    };
    build().unwrap_or_else(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("native SDK error conversion failed")
    })
}
pub fn value(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(v) => PyBool::new(py, *v).to_owned().into_any().unbind(),
        Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                v.into_pyobject(py)?.into_any().unbind()
            } else if let Some(v) = v.as_u64() {
                v.into_pyobject(py)?.into_any().unbind()
            } else {
                v.as_f64().unwrap().into_pyobject(py)?.into_any().unbind()
            }
        }
        Value::String(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Array(v) => {
            let list = PyList::empty(py);
            for item in v {
                list.append(value(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(v) => {
            let dict = PyDict::new(py);
            for (k, v) in v {
                dict.set_item(k, value(py, v)?)?;
            }
            dict.into_any().unbind()
        }
    })
}
fn nested(parent: &str, field: &str) -> Option<&'static str> {
    match (parent, field) {
        ("Capabilities", "streaming") => Some("StreamingCapabilities"),
        ("Capabilities", "expression") => Some("ExpressionCapabilities"),
        ("Capabilities", "speech_sessions") => Some("SessionCapabilities"),
        ("SessionCapabilities", "limits") => Some("SessionLimits"),
        ("ServiceInfo", "health") => Some("Health"),
        ("ServiceInfo", "capabilities") => Some("Capabilities"),
        ("SessionInfo", "context") => Some("SessionContext"),
        _ => None,
    }
}
pub fn record(py: Python<'_>, kind: &str, v: Value) -> PyResult<Py<PyAny>> {
    let class_name = match kind {
        "Models" | "Voices" => "ListResponse",
        _ => kind,
    };
    let class = py.import("aniflive_tts.sdk.types")?.getattr(class_name)?;
    let fields = class.getattr("__dataclass_fields__")?;
    let fields = fields.cast::<PyDict>()?;
    let kwargs = PyDict::new(py);
    let extra = PyDict::new(py);
    let object = v
        .as_object()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected native record"))?;
    for (k, v) in object {
        if k == "extra" || !fields.contains(k)? {
            extra.set_item(k, value(py, v)?)?;
            continue;
        }
        let output = if v.is_null() {
            py.None()
        } else if kind == "SessionCapabilities" && k == "policies" {
            let policies = PyDict::new(py);
            let entries = v.as_object().ok_or_else(|| {
                error(
                    py,
                    core::ClientError::Protocol("invalid session policy metadata"),
                )
            })?;
            for (name, policy) in entries {
                if !policy.is_object() {
                    return Err(error(
                        py,
                        core::ClientError::Protocol("invalid session policy metadata"),
                    ));
                }
                policies.set_item(
                    name,
                    record(py, "ContinuityPolicyCapability", policy.clone())?,
                )?;
            }
            policies.into_any().unbind()
        } else if let Some(child) = nested(kind, k) {
            record(py, child, v.clone())?
        } else if let Some(child) = match (kind, k.as_str()) {
            ("Models", "data") => Some("ModelInfo"),
            ("Voices", "data") => Some("VoiceInfo"),
            ("ExpressionCatalogInfo", "profiles") => Some("ExpressionProfileInfo"),
            ("SessionInfo", "segments") => Some("SegmentInfo"),
            _ => None,
        } {
            let list = PyList::empty(py);
            for item in v.as_array().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("expected native record list")
            })? {
                list.append(record(py, child, item.clone())?)?;
            }
            list.into_any().unbind()
        } else {
            value(py, v)?
        };
        kwargs.set_item(k, output)?;
    }
    if fields.contains("extra")? {
        kwargs.set_item("extra", extra)?;
    }
    Ok(class.call((), Some(&kwargs))?.unbind())
}
pub fn serialized<T: serde::Serialize>(
    py: Python<'_>,
    kind: &str,
    result: T,
) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(result).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("native response conversion failed")
    })?;
    record(py, kind, v)
}
pub fn wav(py: Python<'_>, result: core::SynthesisResult) -> PyResult<Py<PyAny>> {
    let metadata = serialized(py, "SynthesisMetadata", result.metadata)?;
    Ok(py
        .import("aniflive_tts.sdk.types")?
        .getattr("SynthesisResult")?
        .call1((
            PyBytes::new(py, &result.audio),
            result.content_type,
            metadata,
        ))?
        .unbind())
}
pub fn chunk(py: Python<'_>, result: Option<Vec<u8>>) -> PyResult<Py<PyAny>> {
    Ok(match result {
        Some(bytes) => PyBytes::new(py, &bytes).into_any().unbind(),
        None => py.None(),
    })
}
fn fields(schema: &str) -> &'static [(&'static str, &'static str)] {
    match schema {
        "SynthesisRequest" => &[
            ("language", "str"),
            ("text", "str"),
            ("segments", "ExpressionSegment[]"),
            ("model", "str"),
            ("voice_profile", "str"),
            ("expression", "Expression"),
            ("generation", "Generation"),
        ],
        "ExpressionSegment" => &[("text", "str"), ("expression", "Expression")],
        "Expression" => &[
            ("enabled", "bool"),
            ("profile", "str"),
            ("intensity", "float"),
            ("policy", "str"),
        ],
        "Generation" => &[
            ("top_k", "u64"),
            ("top_p", "float"),
            ("temperature", "float"),
            ("seed", "i64"),
            ("noise_scale", "float"),
            ("speed", "float"),
        ],
        "SessionCreate" => &[
            ("model", "str"),
            ("voice_profile", "str"),
            ("continuity_policy", "str"),
        ],
        "SegmentRequest" => &[
            ("segment_id", "str"),
            ("text", "str"),
            ("language", "str"),
            ("paragraph_id", "str"),
            ("pause_after_ms", "u64"),
            ("expression", "Expression"),
            ("generation", "Generation"),
        ],
        _ => &[],
    }
}
fn attributes(object: &Bound<'_, PyAny>, schema: &str) -> PyResult<Value> {
    let mut map = Map::new();
    for &(name, kind) in fields(schema) {
        let attr = object.getattr(name)?;
        if attr.is_none() {
            continue;
        }
        let v = match kind {
            "str" => Value::String(attr.extract::<String>()?),
            "bool" => Value::Bool(attr.extract::<bool>()?),
            "u64" => Value::from(attr.extract::<u64>()?),
            "i64" => Value::from(attr.extract::<i64>()?),
            "float" => serde_json::Number::from_f64(attr.extract::<f64>()?)
                .map(Value::Number)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "generation and expression values must be finite",
                    )
                })?,
            "ExpressionSegment[]" => Value::Array(
                attr.try_iter()?
                    .map(|v| attributes(&v?, "ExpressionSegment"))
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            child => attributes(&attr, child)?,
        };
        map.insert(name.to_owned(), v);
    }
    Ok(Value::Object(map))
}
pub fn request<T: DeserializeOwned>(object: &Bound<'_, PyAny>, schema: &str) -> PyResult<T> {
    serde_json::from_value(attributes(object, schema)?)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid SDK request fields"))
}
