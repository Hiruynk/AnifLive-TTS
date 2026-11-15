//! Diagnostic: repeat cached complete synthesis in one warm process and print
//! each run's stage timings (static AR graphs and allocator stay warm).
use aniflive_macos_v5turbo::engine::Engine;
use mlx_rs::{Device, Dtype};
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    mlx_rs::with_device(Device::gpu(), || -> Result<(), Box<dyn Error>> {
        let a: Vec<String> = std::env::args().collect();
        if a.len() != 9 {
            return Err("Usage: warm_profile MODELS INPUT AUDIO TRANSCRIPT CACHE OUT_ROOT SEED RUNS".into());
        }
        let precision = match std::env::var("ANIFLIVE_TTS_NATIVE_PRECISION").as_deref() {
            Ok("FP16") => Dtype::Float16,
            _ => Dtype::Float32,
        };
        let mut engine = Engine::load_with_precision(Path::new(&a[1]), precision)?;
        engine.configure_ar_from_env()?;
        let (reference, _, _) = engine.prepare_reference_cached(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]), Path::new(&a[5]))?;
        let runs: usize = a[8].parse()?;
        for i in 0..runs {
            let out = Path::new(&a[6]).join(format!("run-{i:02}"));
            std::fs::create_dir_all(&out)?;
            let start = std::time::Instant::now();
            engine.run_seeded_cached(Path::new(&a[2]), &out, a[7].parse()?, &reference, false)?;
            let wall = start.elapsed().as_secs_f64() * 1000.0;
            let control: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("run-control.json"))?)?;
            println!("{}", serde_json::json!({"run": i, "wall_ms": wall, "tokens": control["semantic_target_length"], "stages": control["stage_timings_ms"]}));
        }
        Ok(())
    })
}
