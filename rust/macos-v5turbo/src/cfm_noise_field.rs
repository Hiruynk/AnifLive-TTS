//! Evaluation-only coordinate-stable CFM noise; not the default runtime policy.
use crate::rng::RequestRng;
use mlx_rs::{
    ops::{concatenate, indexing::IndexOp},
    thread_local_default_stream, Array, Device, Dtype,
};
use std::{error::Error, io};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const POLICY: &str = "splitmix64-domain-counter-v1-cfm-coordinate1000-metal-v2";
fn invalid(s: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, s).into()
}
/// Immutable one-draw field. No EOS, target length or future inputs are used at creation.
pub struct CfmNoiseField {
    field: Array,
}
impl CfmNoiseField {
    pub fn new(rng: &mut RequestRng) -> Result<Self> {
        if rng.cfm_draws() != 0 {
            return Err(invalid("coordinate field requires CFM draw zero"));
        }
        Self::draw(rng)
    }
    /// Draw a subsequent rolling-block field; draw zero remains reserved for `new`.
    pub fn next(rng: &mut RequestRng) -> Result<Self> {
        if rng.cfm_draws() == 0 {
            return Err(invalid(
                "subsequent coordinate field requires an existing draw",
            ));
        }
        Self::draw(rng)
    }
    fn draw(rng: &mut RequestRng) -> Result<Self> {
        let previous_cfm_draws = rng.cfm_draws();
        let expected_cfm_draws = previous_cfm_draws
            .checked_add(1)
            .ok_or_else(|| invalid("coordinate field draw counter overflow"))?;
        let stream = thread_local_default_stream()
            .ok_or_else(|| invalid("coordinate field requires a scoped Metal GPU stream"))?;
        if stream.device()? != Device::gpu() {
            return Err(invalid("coordinate field requires Metal GPU"));
        }
        let ar_draws = rng.ar_draws();
        let ar_seed = rng.ar_seed();
        let cfm_seed = rng.cfm_seed();
        let field = rng.cfm_normal(&[1, 100, 1000])?;
        if field.shape() != [1, 100, 1000] || field.dtype() != Dtype::Float32 {
            return Err(invalid("invalid coordinate field shape/dtype"));
        }
        field.eval()?;
        if !field.is_finite()?.all(false)?.item_exact::<bool>() {
            return Err(invalid("nonfinite coordinate field"));
        }
        if rng.cfm_draws() != expected_cfm_draws
            || rng.ar_draws() != ar_draws
            || rng.ar_seed() != ar_seed
            || rng.cfm_seed() != cfm_seed
        {
            return Err(invalid("unexpected RNG state after coordinate draw"));
        }
        Ok(Self { field })
    }
    pub fn field(&self) -> &Array {
        &self.field
    }
    /// Reference plus target frames [0,target_end), with end exclusive.
    pub fn cumulative(&self, r: i32, target_end: i32) -> Result<Array> {
        let total = r
            .checked_add(target_end)
            .ok_or_else(|| invalid("coordinate overflow"))?;
        if r <= 0 || target_end < 0 || total > 1000 {
            return Err(invalid("coordinate range outside field"));
        }
        Ok(self.field.index((.., .., ..total)).contiguous()?)
    }
    /// Reference plus target frames [start,end), with end exclusive.
    pub fn window(&self, r: i32, start: i32, end: i32) -> Result<Array> {
        let last = r
            .checked_add(end)
            .ok_or_else(|| invalid("coordinate overflow"))?;
        if r <= 0 || start < 0 || end <= start || last > 1000 {
            return Err(invalid("coordinate range outside field"));
        }
        Ok(concatenate(
            &[
                self.field.index((.., .., ..r)),
                self.field.index((.., .., r + start..last)),
            ],
            2,
        )?
        .contiguous()?)
    }
}
