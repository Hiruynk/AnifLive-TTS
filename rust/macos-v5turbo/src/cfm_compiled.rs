//! Fixed-geometry, single-entry compiled four-step CFM for an isolated GPU process.
//!
//! The owner must run this component on its MLX GPU thread in a process whose
//! compile mode is ENABLED (3). MLX exposes no mode getter; this component never
//! changes it. The caller must supply an estimator built from evaluated, immutable
//! F32 weights. TF32 must be explicitly enabled before MLX is initialized.
use crate::estimator::{Estimator, PreparedSchedule};
use mlx_rs::{error::Exception, ops::select, transforms::compile::compile, Array, Device, Dtype};
use std::{cell::Cell, error::Error, io, rc::Rc};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Graph = Box<dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception>>;
fn invalid(s: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, s.into()).into()
}
fn start(noise: &Array, mask: &Array) -> Result<Array> {
    Ok(select(
        mask,
        noise.multiply(Array::from_f32(0.875))?,
        Array::from_f32(0.0),
    )?)
}
fn update(x: &Array, v: &Array, mask: &Array) -> Result<Array> {
    if v.dtype() != Dtype::Float32 || v.shape() != x.shape() {
        return Err(invalid("invalid CFM velocity"));
    }
    Ok(select(
        mask,
        x.add(v.multiply(Array::from_f32(0.25))?)?,
        Array::from_f32(0.0),
    )?)
}
struct Entry {
    frames: i32,
    reference_frames: i32,
    traces: Rc<Cell<usize>>,
    graph: Graph,
}
/// Metadata for the graph used by one successful inference.
#[derive(Clone, Copy, Debug)]
pub struct Metadata {
    pub frames: i32,
    pub reference_frames: i32,
    pub rebuilds: u64,
    pub traces: usize,
    pub total_traces: u64,
}
/// One active geometry; old compiled graph is dropped before a replacement is created.
pub struct CompiledCfm {
    estimator: Rc<Estimator>,
    schedule: Rc<PreparedSchedule>,
    entry: Option<Entry>,
    rebuilds: u64,
    total_traces: Rc<Cell<u64>>,
}
impl CompiledCfm {
    /// Prepare the fixed four-step time schedule once, outside request inference.
    pub fn new(estimator: Rc<Estimator>) -> Result<Self> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("1") {
            return Err(invalid("compiled CFM requires MLX_ENABLE_TF32=1"));
        }
        if std::env::var_os("MLX_DISABLE_COMPILE").is_some() {
            return Err(invalid("compiled CFM requires compilation enabled"));
        }
        if std::env::var_os("ANIFLIVE_DIT_TRACE_DIR").is_some() {
            return Err(invalid("compiled CFM does not support DiT tracing"));
        }
        let stream = mlx_rs::thread_local_default_stream()
            .ok_or_else(|| invalid("compiled CFM requires a scoped GPU stream"))?;
        if stream.device()? != Device::gpu() {
            return Err(invalid("compiled CFM requires a GPU stream"));
        }
        let schedule = Rc::new(estimator.prepare_schedule()?);
        Ok(Self {
            estimator,
            schedule,
            entry: None,
            rebuilds: 0,
            total_traces: Rc::new(Cell::new(0)),
        })
    }
    /// Return full F32 state [1,100,N], with a zero reference prefix.
    pub fn infer(
        &mut self,
        mu: &Array,
        prompt: &Array,
        noise: &Array,
        reference_frames: i32,
    ) -> Result<(Array, Metadata)> {
        if mu.ndim() != 3
            || mu.shape()[0] != 1
            || mu.shape()[2] != 512
            || mu.dtype() != Dtype::Float32
        {
            return Err(invalid("mu must be F32 [1,N,512]"));
        }
        let n = mu.shape()[1];
        if n > 1000
            || reference_frames <= 0
            || reference_frames >= n
            || prompt.dtype() != Dtype::Float32
            || noise.dtype() != Dtype::Float32
            || prompt.shape() != [1, 100, n]
            || noise.shape() != [1, 100, n]
        {
            return Err(invalid("invalid F32 prompt/noise or reference_frames"));
        }
        if self.entry.as_ref().map(|e| (e.frames, e.reference_frames))
            != Some((n, reference_frames))
        {
            self.entry.take();
            let mask = Array::from_slice(
                &(0..n).map(|i| i >= reference_frames).collect::<Vec<_>>(),
                &[1, 1, n],
            );
            mask.eval()?;
            let estimator = self.estimator.clone();
            let schedule = self.schedule.clone();
            let traces = Rc::new(Cell::new(0usize));
            let current = traces.clone();
            let total = self.total_traces.clone();
            let graph = compile(
                move |args: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                    current.set(current.get() + 1);
                    total.set(total.get() + 1);
                    if args.len() != 3 {
                        return Err(Exception::custom("expected mu, prompt, noise"));
                    }
                    let inner = || -> Result<Array> {
                        let context = estimator.prepare_context_unmaterialized(&args[0])?;
                        let mut x = start(&args[2], &mask)?;
                        for step in 0..4 {
                            let v = estimator
                                .forward_prepared(&x, &args[1], &context, &schedule, step, None)?;
                            x = update(&x, &v, &mask)?;
                        }
                        Ok(x)
                    };
                    Ok(vec![inner().map_err(|e| Exception::custom(e.to_string()))?])
                },
                Some(false),
            );
            self.rebuilds += 1;
            self.entry = Some(Entry {
                frames: n,
                reference_frames,
                traces,
                graph: Box::new(graph),
            });
        }
        let entry = self
            .entry
            .as_mut()
            .ok_or_else(|| invalid("missing compiled CFM graph"))?;
        let outputs = (entry.graph)(&[mu.clone(), prompt.clone(), noise.clone()])?;
        if outputs.len() != 1 {
            return Err(invalid("compiled CFM returned wrong output count"));
        }
        let x = outputs
            .into_iter()
            .next()
            .ok_or_else(|| invalid("missing compiled CFM output"))?;
        if x.dtype() != Dtype::Float32 || x.shape() != [1, 100, n] {
            return Err(invalid("compiled CFM output shape/dtype mismatch"));
        }
        x.eval()?;
        Ok((
            x,
            Metadata {
                frames: n,
                reference_frames,
                rebuilds: self.rebuilds,
                traces: entry.traces.get(),
                total_traces: self.total_traces.get(),
            },
        ))
    }
}
