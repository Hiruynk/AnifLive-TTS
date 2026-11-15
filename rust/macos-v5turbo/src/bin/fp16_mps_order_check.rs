//! Diagnostic: run mps_order kernels on .npy inputs (x, w, b) and save outputs.
use aniflive_macos_v5turbo::mps_order;
use mlx_rs::{Array, Device, Stream};
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    mlx_rs::with_device(Device::gpu(), || -> Result<(), Box<dyn Error>> {
        let a: Vec<String> = std::env::args().collect();
        let dir = Path::new(&a[1]);
        let cpu = Stream::try_default_on_device(&Device::cpu())?;
        let load = |n: &str| mlx_rs::with_stream(&cpu, || Array::load_numpy(dir.join(n)));
        let (x, w, b) = (load("x.npy")?, load("w.npy")?, load("b.npy")?);
        let y = mps_order::linear(&x, &w, Some(&b), false)?;
        y.eval()?;
        y.save_numpy(dir.join("y.npy"))?;
        Ok(())
    })
}
