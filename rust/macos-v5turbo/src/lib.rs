//! Native v5 Turbo components under numerical qualification; not a full TTS runtime.
use mlx_rs::{error::Exception, Array};

/// Official v5 Euler flow: four evaluations, CFG=0, prompt prefix reset each step.
/// Noise is supplied explicitly to permit controlled cross-runtime RNG comparison.
/// `target_mask` is false on reference frames and true on target frames, broadcastable to noise.
/// The estimator must return [batch, mel, frames], with no CFG-negative evaluation.
pub fn cfm_four_step<F>(
    noise: &Array,
    target_mask: &Array,
    mut estimator: F,
) -> Result<Array, Exception>
where
    F: FnMut(&Array, f32) -> Result<Array, Exception>,
{
    let mut x = mlx_rs::ops::select(
        target_mask,
        noise.multiply(Array::from_f32(0.875))?,
        Array::from_f32(0.0),
    )?;
    for step in 0..4 {
        let velocity = estimator(&x, step as f32 * 0.25)?;
        x = mlx_rs::ops::select(
            target_mask,
            x.add(velocity.multiply(Array::from_f32(0.25))?)?,
            Array::from_f32(0.0),
        )?;
    }
    Ok(x)
}

pub mod vocoder;

pub mod dit;

pub mod estimator;

pub mod reference_encoder;

pub mod acoustic;

pub mod ar;

pub mod hubert;

pub mod semantic;

pub mod engine;
pub mod sampling;

pub mod audio;

pub mod rng;

pub mod reference_cache;
pub mod streaming;

pub mod bert;

pub mod worker;

pub mod g2pw;

pub mod frontend_models;

pub mod ar_decision;

pub mod cfm_stage;

pub mod ar_compiled;
pub mod ar_static;

pub mod mixed_gemv;
pub mod mps_order;
pub mod residual_layer_norm;
pub mod qkv_cache_fused;

pub mod ar_microbatch;

pub mod cfm_compiled;

pub mod bootstrap_stream;
pub mod canonical_stream;
pub mod cfm_noise_field;

pub mod source_half_attention;
