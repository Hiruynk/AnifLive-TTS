use crate::convert;
use aniflive_tts_client::ClientError;
use pyo3::prelude::*;
use std::future::Future;
use std::sync::{Arc, Mutex};
pub type Affinity = Arc<Mutex<Option<Py<PyAny>>>>;
pub fn affinity() -> Affinity {
    Arc::new(Mutex::new(None))
}
pub fn check_loop(py: Python<'_>, affinity: &Affinity) -> PyResult<()> {
    let current = py.import("asyncio")?.call_method0("get_running_loop")?;
    let mut saved = affinity.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(saved) = saved.as_ref() {
        let original = saved.bind(py).call0()?;
        if original.is_none() || !original.is(&current) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "async SDK handle belongs to a different event loop",
            ));
        }
    } else {
        // A callback-free weak reference avoids a hidden native-to-loop GC edge.
        // Keep the dead reference as an ownership tombstone; never rebind to another loop.
        *saved = Some(
            py.import("weakref")?
                .getattr("ref")?
                .call1((current,))?
                .unbind(),
        );
    }
    Ok(())
}
pub fn sync<T: Send, F, C>(py: Python<'_>, future: F, convert: C) -> PyResult<Py<PyAny>>
where
    F: Future<Output = Result<T, ClientError>> + Send,
    C: FnOnce(Python<'_>, T) -> PyResult<Py<PyAny>>,
{
    if py
        .import("asyncio")?
        .call_method0("get_running_loop")
        .is_ok()
        || tokio::runtime::Handle::try_current().is_ok()
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "use the async SDK inside a running event loop",
        ));
    }
    let result = py.detach(|| pyo3_async_runtimes::tokio::get_runtime().block_on(future));
    convert(py, result.map_err(|e| convert::error(py, e))?)
}
pub fn asynchronous<T: Send + 'static, F, C>(
    py: Python<'_>,
    affinity: &Affinity,
    future: F,
    convert: C,
) -> PyResult<Py<PyAny>>
where
    F: Future<Output = Result<T, ClientError>> + Send + 'static,
    C: FnOnce(Python<'_>, T) -> PyResult<Py<PyAny>> + Send + 'static,
{
    check_loop(py, affinity)?;
    Ok(pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = future.await;
        Python::attach(|py| convert(py, result.map_err(|e| convert::error(py, e))?))
    })?
    .unbind())
}
