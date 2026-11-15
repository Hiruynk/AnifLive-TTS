//! Request-local keyed randomness. No MLX global random state is used.
//!
//! Policy `splitmix64-domain-counter-v1`: domain_seed = splitmix64(seed XOR
//! domain); draw_seed = splitmix64(domain_seed XOR zero-based draw index).
//! Each draw_seed is passed to mlx::random::key, whose key has the seed's upper
//! and lower 32-bit words. AR and CFM use separate, fixed domain constants.
use mlx_rs::{ops::maximum, random, Array};
use std::{error::Error, io};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const POLICY: &str = "splitmix64-domain-counter-v1";
pub const AR_DOMAIN: u64 = 0x41525f5354524541;
pub const CFM_DOMAIN: u64 = 0x43464d5f5354524d;

/// SplitMix64 finalizer with its standard increment, evaluated with u64 wrapping.
pub fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[derive(Clone)]
pub struct RequestRng {
    ar_seed: u64,
    cfm_seed: u64,
    ar_draws: u64,
    cfm_draws: u64,
}
impl RequestRng {
    pub fn new(seed: u64) -> Self {
        Self {
            ar_seed: splitmix64(seed ^ AR_DOMAIN),
            cfm_seed: splitmix64(seed ^ CFM_DOMAIN),
            ar_draws: 0,
            cfm_draws: 0,
        }
    }
    pub fn ar_seed(&self) -> u64 {
        self.ar_seed
    }
    pub fn cfm_seed(&self) -> u64 {
        self.cfm_seed
    }
    pub fn ar_draws(&self) -> u64 {
        self.ar_draws
    }
    pub fn cfm_draws(&self) -> u64 {
        self.cfm_draws
    }

    /// Commit only the consumed AR prefix from a reservation cloned from this
    /// state. CFM and all seed domains must remain unchanged.
    pub fn commit_ar_prefix(&mut self, reserved: &Self, consumed: u64) -> Result<()> {
        if self.ar_seed != reserved.ar_seed
            || self.cfm_seed != reserved.cfm_seed
            || self.cfm_draws != reserved.cfm_draws
            || reserved.ar_draws < self.ar_draws
            || consumed > reserved.ar_draws - self.ar_draws
        {
            return Err(
                io::Error::new(io::ErrorKind::InvalidInput, "invalid AR RNG reservation").into(),
            );
        }
        self.ar_draws = self.ar_draws.checked_add(consumed).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "RNG draw counter overflow")
        })?;
        Ok(())
    }

    fn next(domain_seed: u64, counter: &mut u64) -> Result<Array> {
        let draw_seed = splitmix64(domain_seed ^ *counter);
        *counter = counter.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "RNG draw counter overflow")
        })?;
        Ok(random::key(draw_seed)?)
    }
    /// Exponential(1), generated as -log(U). MLX uniform is [0,1); clamp
    /// only its zero endpoint to the smallest positive normal float32 so q is
    /// finite and strictly positive, with no global PRNG state.
    pub fn ar_exponential(&mut self, shape: &[i32]) -> Result<Array> {
        let key = Self::next(self.ar_seed, &mut self.ar_draws)?;
        let uniform = random::uniform::<f32, f32>(0.0, 1.0, shape, &key)?;
        Ok(maximum(uniform, Array::from_f32(f32::MIN_POSITIVE))?
            .log()?
            .negative()?)
    }
    /// Standard normal values, prior to V5's 0.875 temperature scaling.
    pub fn cfm_normal(&mut self, shape: &[i32]) -> Result<Array> {
        let key = Self::next(self.cfm_seed, &mut self.cfm_draws)?;
        Ok(random::normal::<f32>(shape, None, None, &key)?)
    }
}
