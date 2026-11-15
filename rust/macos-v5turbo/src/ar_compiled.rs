//! Request-reusable shapeless FP32 AR decode on the worker owner thread.
//!
//! All tensors are explicit graph inputs. This component owns the worker's
//! process-global compile policy: ENABLED normally, NO_FUSE during graph calls.
//! The GPU owner serializes calls; external compile-mode mutation is unsupported.
use crate::ar::{ArModel, ArState};
use mlx_rs::{
    error::Exception,
    fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn::relu,
    ops::concatenate,
    transforms::compile::compile,
    Array, DeviceType, Dtype,
};
use std::{cell::Cell, error::Error, ffi::c_int, rc::Rc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type CompiledGraph = dyn FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception>;
// Exact mlx/c/compile.h enum values for this pinned runtime.
const COMPILE_NO_FUSE: c_int = 2;
const COMPILE_ENABLED: c_int = 3;
unsafe extern "C" {
    fn mlx_set_compile_mode(mode: c_int) -> c_int;
}

fn validate(args: &[Array]) -> Result<()> {
    if args.len() != 5 + 24 * 14
        || (args[0].dtype() != Dtype::Int32 && args[0].dtype() != Dtype::Int64)
        || args[0].shape() != [1, 1]
        || args[1].shape() != [1, 1, 512]
        || args[2].shape() != [1025, 512]
        || args[3].shape() != [1]
        || args[4].shape() != [512, 1025]
        || args[1..].iter().any(|a| a.dtype() != Dtype::Float32)
    {
        return Err("invalid diagnostic decode input count, dtype or fixed shapes".into());
    }
    let mut common_length = None;
    let shapes: [&[i32]; 12] = [
        &[512, 1536],
        &[1536],
        &[512, 512],
        &[512],
        &[512, 2048],
        &[2048],
        &[2048, 512],
        &[512],
        &[512],
        &[512],
        &[512],
        &[512],
    ];
    for layer in 0..24 {
        let i = 5 + 14 * layer;
        let k = &args[i];
        let v = &args[i + 1];
        if k.ndim() != 3
            || k.shape()[0] != 1
            || k.shape()[1] <= 0
            || k.shape()[2] != 512
            || v.shape() != k.shape()
            || args[i + 2..i + 14]
                .iter()
                .zip(shapes)
                .any(|(a, s)| a.shape() != s)
        {
            return Err("invalid diagnostic cache/weight shapes".into());
        }
        match common_length {
            None => common_length = Some(k.shape()[1]),
            Some(n) if n == k.shape()[1] => {}
            _ => return Err("inconsistent cache lengths".into()),
        }
    }
    Ok(())
}
#[doc(hidden)]
pub fn diagnostic_decode_graph(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    if args.len() != 5 + 24 * 14 {
        return Err(Exception::custom("invalid compiled decode input count"));
    }
    let audio = args[2].take_axis(&args[0], 0)?;
    let mut hidden = audio.add(args[1].multiply(&args[3])?)?;
    let mut caches = Vec::with_capacity(48);
    for layer in 0..24 {
        let i = 5 + 14 * layer;
        let qkv = hidden
            .matmul(&args[i + 2])?
            .add(&args[i + 3])?
            .reshape(&[1, 1, 3, 512])?;
        let q_part = qkv.take_axis(Array::from_int(0), 2)?;
        let k_part = qkv.take_axis(Array::from_int(1), 2)?;
        let v_part = qkv.take_axis(Array::from_int(2), 2)?;
        let k = concatenate(&[&args[i], &k_part], 1)?;
        let v = concatenate(&[&args[i + 1], &v_part], 1)?;
        let q = q_part
            .reshape(&[1, 1, 16, 32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kh = k.reshape(&[1, -1, 16, 32])?.transpose_axes(&[0, 2, 1, 3])?;
        let vh = v.reshape(&[1, -1, 16, 32])?.transpose_axes(&[0, 2, 1, 3])?;
        let a = scaled_dot_product_attention(
            &q,
            &kh,
            &vh,
            1.0 / (32f32).sqrt(),
            None::<ScaledDotProductAttentionMask<'_>>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, 1, 512])?;
        let projected = a.matmul(&args[i + 4])?.add(&args[i + 5])?;
        let h = layer_norm(
            hidden.add(projected)?,
            Some(&args[i + 10]),
            Some(&args[i + 11]),
            1e-5,
        )?;
        let ff = relu(h.matmul(&args[i + 6])?.add(&args[i + 7])?)?
            .matmul(&args[i + 8])?
            .add(&args[i + 9])?;
        hidden = layer_norm(h.add(ff)?, Some(&args[i + 12]), Some(&args[i + 13]), 1e-5)?;
        caches.push(k);
        caches.push(v);
    }
    let logits = hidden.matmul(&args[4])?.reshape(&[1, 1025])?;
    let mut outputs = Vec::with_capacity(49);
    outputs.push(logits);
    outputs.extend(caches);
    Ok(outputs)
}

/// Compiled graph and trace counter, deliberately bound to its owner thread.
pub struct CompiledAr {
    graph: Box<CompiledGraph>,
    traces: Rc<Cell<usize>>,
    owned_restore_mode: c_int,
}
impl CompiledAr {
    /// Establish the worker-owned ENABLED default and retain a shapeless graph.
    /// Call on the sole Metal owner during startup. External compile-mode
    /// changes while this component is alive are unsupported.
    pub fn initialize() -> Result<Self> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
            return Err("compiled AR requires MLX_ENABLE_TF32=0 before MLX initialization".into());
        }
        if std::env::var_os("MLX_DISABLE_COMPILE").is_some() {
            return Err("compiled AR requires MLX compilation enabled".into());
        }
        if !matches!(
            mlx_rs::Stream::thread_local_or_default()
                .device()?
                .get_type()?,
            DeviceType::Gpu
        ) {
            return Err("compiled AR requires the Metal GPU owner thread".into());
        }
        // The standalone worker explicitly owns this default; it is not a
        // guess about an unknown caller's previous process mode.
        if unsafe { mlx_set_compile_mode(COMPILE_ENABLED) } != 0 {
            return Err("failed to establish worker MLX compile mode ENABLED".into());
        }
        let traces = Rc::new(Cell::new(0));
        let count = traces.clone();
        let graph = compile(
            move |inputs: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
                count.set(count.get() + 1);
                diagnostic_decode_graph(inputs)
            },
            Some(true),
        );
        Ok(Self {
            graph: Box::new(graph),
            traces,
            owned_restore_mode: COMPILE_ENABLED,
        })
    }

    pub fn trace_count(&self) -> usize {
        self.traces.get()
    }

    /// Decode an internally sampled token. `known_id` must be the value read
    /// from this exact array by the caller; external tokens must use `ArModel::decode`.
    /// All output siblings are scheduled together using async_eval before the
    /// validated state is installed. This does not wait on the host. A later
    /// token/decision evaluation on this same owner thread synchronizes them;
    /// on failure that evaluation returns an error rather than falling back.
    pub fn decode_generated(
        &mut self,
        model: &ArModel,
        token: &Array,
        known_id: i32,
        state: &mut ArState,
    ) -> Result<Array> {
        let inputs = model.compiled_decode_inputs_generated(token, known_id, state)?;
        validate(&inputs)?;
        // Only the owner thread may execute model work during this short
        // process-global switch. Restore before scheduling any other work.
        if unsafe { mlx_set_compile_mode(COMPILE_NO_FUSE) } != 0 {
            if unsafe { mlx_set_compile_mode(self.owned_restore_mode) } != 0 {
                return Err(
                    "failed to set NO_FUSE and failed to restore worker compile mode".into(),
                );
            }
            return Err("failed to set MLX compile mode NO_FUSE".into());
        }
        let graph_result = (self.graph)(&inputs);
        if unsafe { mlx_set_compile_mode(self.owned_restore_mode) } != 0 {
            return Err(match graph_result {
                Ok(_) => "failed to restore worker MLX compile mode ENABLED".to_owned(),
                Err(error) => format!(
                    "failed to restore worker MLX compile mode ENABLED after graph error: {error}"
                ),
            }
            .into());
        }
        let outputs = graph_result?;
        mlx_rs::transforms::async_eval(outputs.iter())?;
        model.install_compiled_decode(state, &outputs)?;
        Ok(outputs[0].clone())
    }
}
