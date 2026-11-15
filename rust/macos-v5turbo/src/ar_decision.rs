//! One host read for an AR draw's validation and EOS decisions.
use mlx_rs::{ops::concatenate, Array, Dtype};
use std::{error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: &str) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message))
}

/// Validate every exponential draw and read the sampled and greedy token IDs.
/// q is floating [1,V], token is int32 [1,1], greedy is int32 [1]. The caller
/// must retain and pass the same `token` array to its internal decode path.
pub fn read_decision(q: &Array, token: &Array, greedy: &Array) -> Result<(i32, i32)> {
    if q.ndim() != 2 || q.shape()[0] != 1 || q.shape()[1] <= 0 || !q.dtype().is_float() {
        return Err(invalid("AR noise must be floating [1,V] with positive V"));
    }
    if token.dtype() != Dtype::Int32
        || token.shape() != [1, 1]
        || greedy.dtype() != Dtype::Int32
        || greedy.shape() != [1]
    {
        return Err(invalid(
            "sampled token must be int32 [1,1] and greedy token int32 [1]",
        ));
    }
    let zero = Array::from_f32(0.0).as_dtype(q.dtype())?;
    let valid = q
        .is_finite()?
        .logical_and(q.gt(zero)?)?
        .all(false)?
        .as_dtype(Dtype::Int32)?
        .reshape(&[1])?;
    let packed = concatenate(&[valid, token.reshape(&[1])?, greedy.clone()], 0)?;
    // Request all arrays so q and token remain materialized for capture and
    // decode. This is a single scheduling/wait boundary before the host read.
    mlx_rs::transforms::eval([q, token, greedy, &packed])?;
    let values = packed.to_vec_exact::<i32>()?;
    if values[0] != 1 {
        return Err(invalid(
            "AR exponential draw must contain only finite strictly positive values",
        ));
    }
    let (sampled, greedy_id) = (values[1], values[2]);
    if !(0..q.shape()[1]).contains(&sampled) || !(0..q.shape()[1]).contains(&greedy_id) {
        return Err(invalid("AR decision token is outside the noise vocabulary"));
    }
    Ok((sampled, greedy_id))
}
