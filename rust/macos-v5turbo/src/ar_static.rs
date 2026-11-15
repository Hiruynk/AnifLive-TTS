//! Bounded owner-thread static AR graph candidate, enabled only by explicit opt-in.
//!
//! Each K/V length owns one fixed-shape compile closure. Dropping an entry
//! invokes mlx-rs CompiledState::drop's per-function cache erase; that Drop
//! suppresses native erase errors. No global compiler cache is cleared. Pending
//! arrays and GPU work can retain buffers after an entry is evicted.
use crate::ar::{ArModel, ArState};
use mlx_rs::{error::Exception, transforms::compile::compile, Array, DeviceType, Dtype, Stream};
use std::{
    cell::Cell,
    collections::HashMap,
    error::Error,
    ffi::c_int,
    rc::Rc,
    thread::{self, ThreadId},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Graph = dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception>;
unsafe extern "C" {
    fn mlx_set_compile_mode(mode: c_int) -> c_int;
}

struct Entry {
    graph: Box<Graph>,
    last_used: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct StaticArStats {
    pub entries: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub compiles: u64,
}
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    pub active: usize,
    pub cached: usize,
    pub peak: usize,
}

/// Owns the process compile-mode policy ENABLED except for scoped NO_FUSE
/// calls. Keep creation, use and destruction on the same Metal owner thread;
/// external compile-mode changes while alive are unsupported. Model weights
/// are private and immutable; constructor requires terminal fused-TM1 config,
/// whose existing mutators cannot later change rows or disable the epilogue.
pub struct StaticAr {
    model: ArModel,
    entries: HashMap<i32, Entry>,
    capacity: usize,
    stream: Stream,
    thread: ThreadId,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    compiles: Rc<Cell<u64>>,
}
impl StaticAr {
    pub fn new(model: &ArModel, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err("static AR capacity must be positive".into());
        }
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0")
            || std::env::var_os("MLX_DISABLE_COMPILE").is_some()
        {
            return Err("static AR requires MLX_ENABLE_TF32=0 and compilation enabled".into());
        }
        if model.mixed_gemv_rows() != Some(1) || !model.mixed_gemv_epilogue_enabled() {
            return Err("static AR requires FP32 model with fused TM1 GEMV".into());
        }
        let stream = Stream::thread_local_or_default();
        if !matches!(stream.device()?.get_type()?, DeviceType::Gpu) {
            return Err("static AR requires GPU owner".into());
        }
        if unsafe { mlx_set_compile_mode(3) } != 0 {
            return Err("cannot establish ENABLED compile mode".into());
        }
        Ok(Self {
            model: model.clone(),
            entries: HashMap::new(),
            capacity,
            stream,
            thread: thread::current().id(),
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            compiles: Rc::new(Cell::new(0)),
        })
    }
    pub fn stats(&self) -> StaticArStats {
        StaticArStats {
            entries: self.entries.len(),
            capacity: self.capacity,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            compiles: self.compiles.get(),
        }
    }
    pub fn memory_snapshot() -> Result<MemorySnapshot> {
        Ok(MemorySnapshot {
            active: mlx_rs::memory::active_memory()?,
            cached: mlx_rs::memory::cache_memory()?,
            peak: mlx_rs::memory::peak_memory()?,
        })
    }

    fn validate(inputs: &[Array]) -> Result<i32> {
        if inputs.len() != 50
            || inputs[0].shape() != [1, 1]
            || inputs[0].dtype() != Dtype::Int32
            || inputs[1].shape() != [1, 1, 512]
            || !matches!(inputs[1].dtype(), Dtype::Float32 | Dtype::Float16)
        {
            return Err("invalid static AR token or position".into());
        }
        let first = &inputs[2];
        if first.ndim() != 3
            || first.shape()[0] != 1
            || first.shape()[1] <= 0
            || first.shape()[2] != 512
        {
            return Err("invalid static AR cache geometry".into());
        }
        let length = first.shape()[1];
        if inputs[2..]
            .iter()
            .any(|x| x.dtype() != inputs[1].dtype() || x.shape() != [1, length, 512])
        {
            return Err("static AR caches have inconsistent shape or dtype".into());
        }
        Ok(length)
    }
    fn call(graph: &mut Graph, inputs: &[Array]) -> Result<Vec<Array>> {
        if unsafe { mlx_set_compile_mode(2) } != 0 {
            if unsafe { mlx_set_compile_mode(3) } != 0 {
                return Err("failed to set NO_FUSE and restore ENABLED".into());
            }
            return Err("failed to set NO_FUSE".into());
        }
        let result = graph(inputs);
        if unsafe { mlx_set_compile_mode(3) } != 0 {
            return Err(format!(
                "failed to restore ENABLED after graph call: {}",
                match result {
                    Ok(_) => "graph succeeded".into(),
                    Err(e) => e.to_string(),
                }
            )
            .into());
        }
        Ok(result?)
    }
    /// Token must be this request's GPU sampler argmax; caller validates
    /// consumed decisions and discards speculative state after EOS/errors.
    pub fn decode_sampled_gpu(&mut self, token: &Array, state: &mut ArState) -> Result<Array> {
        if thread::current().id() != self.thread || Stream::thread_local_or_default() != self.stream
        {
            return Err("static AR requires its original GPU owner thread and stream".into());
        }
        let inputs = self
            .model
            .diagnostic_static_decode_sampled_gpu_inputs(token, state)?;
        let length = Self::validate(&inputs)?;
        self.tick = self
            .tick
            .checked_add(1)
            .ok_or("static AR LRU counter overflow")?;
        let outputs = if let Some(entry) = self.entries.get_mut(&length) {
            self.hits += 1;
            let outputs = Self::call(&mut entry.graph, &inputs)?;
            entry.last_used = self.tick;
            outputs
        } else {
            self.misses += 1;
            let owned = self.model.clone();
            let count = self.compiles.clone();
            let mut graph = compile(
                move |args: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                    count.set(count.get() + 1);
                    owned
                        .diagnostic_static_decode_graph(args)
                        .map_err(|e| Exception::custom(e.to_string()))
                },
                Some(false),
            );
            let outputs = Self::call(&mut graph, &inputs)?;
            if self.entries.len() == self.capacity {
                let key = *self
                    .entries
                    .iter()
                    .min_by_key(|(_, v)| v.last_used)
                    .ok_or("missing LRU entry")?
                    .0;
                self.entries.remove(&key);
                self.evictions += 1;
            }
            self.entries.insert(
                length,
                Entry {
                    graph: Box::new(graph),
                    last_used: self.tick,
                },
            );
            outputs
        };
        self.model
            .diagnostic_install_compiled_decode(state, &outputs)?;
        Ok(outputs[0].clone())
    }
}
