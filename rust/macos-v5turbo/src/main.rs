use aniflive_macos_v5turbo::cfm_four_step;
use mlx_rs::{Array, Device};
fn configure_native_candidates(
    engine: &mut aniflive_macos_v5turbo::engine::Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    let compiled_cfm = match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_COMPILED").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        _ => return Err("ANIFLIVE_TTS_NATIVE_CFM_COMPILED must be 0 or 1".into()),
    };
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_PREPARE").as_deref() {
        Ok("1") => engine.enable_prepared_cfm(),
        Ok("0") | Err(_) => Ok(()),
        _ => Err("ANIFLIVE_TTS_NATIVE_CFM_PREPARE must be 0 or 1".into()),
    }?;
    engine.configure_ar_from_env()?;
    match std::env::var("ANIFLIVE_TTS_NATIVE_CFM_STAGE").as_deref() {
        Ok("1") => {
            let root = std::env::var_os("ANIFLIVE_TTS_NATIVE_CACHE_DIR")
                .ok_or("CFM stage requires ANIFLIVE_TTS_NATIVE_CACHE_DIR")?;
            engine.enable_cfm_stage(std::path::Path::new(&root))
        }
        Ok("0") | Err(_) => Ok(()),
        _ => Err("ANIFLIVE_TTS_NATIVE_CFM_STAGE must be 0 or 1".into()),
    }?;
    if compiled_cfm && !engine.compiled_cfm_stage_enabled() {
        return Err("Compiled CFM requires an enabled isolated CFM stage".into());
    }
    engine.validate_batched_cfm_qkv_from_env()?;
    Ok(())
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("cfm-worker") {
        if args.len() != 3 {
            return Err("Usage: cfm-worker MODEL_DIR".into());
        }
        return aniflive_macos_v5turbo::cfm_stage::run(std::path::Path::new(&args[2]));
    }
    if args.get(1).map(String::as_str) == Some("worker") {
        if args.len() != 4 {
            return Err("Usage: worker MODEL_DIR CACHE_ROOT".into());
        }
        if !matches!(
            mlx_rs::Stream::thread_local_or_default()
                .device()?
                .get_type()?,
            mlx_rs::DeviceType::Gpu
        ) {
            return Err("Native worker requires Metal GPU; CPU reference mode is forbidden".into());
        }
        return aniflive_macos_v5turbo::worker::run(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
        );
    }
    if args.get(1).map(String::as_str) == Some("g2pw") {
        if args.len() != 4 {
            return Err("Usage: g2pw WEIGHTS.safetensors INPUT_DIR".into());
        }
        if !matches!(
            mlx_rs::Stream::thread_local_or_default()
                .device()?
                .get_type()?,
            mlx_rs::DeviceType::Gpu
        ) {
            return Err("G2PW native qualification requires the Metal GPU".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let model = aniflive_macos_v5turbo::g2pw::G2PW::load(&raw)?;
        let dir = std::path::Path::new(&args[3]);
        let output = dir.join("native-probabilities.npy");
        if output.exists() {
            return Err("G2PW qualification output already exists".into());
        }
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let ids = load("input_ids.npy")?;
        let types = load("token_type_ids.npy")?;
        let mask = load("attention_mask.npy")?;
        let phonemes = load("phoneme_mask.npy")?;
        let chars = load("char_ids.npy")?;
        let positions = load("position_ids.npy")?;
        let probabilities = model.forward(&ids, &types, &mask, &phonemes, &chars, &positions)?;
        probabilities.eval()?;
        probabilities.save_numpy(output)?;
        println!("{{\"kind\":\"g2pw_component_only\",\"backend\":\"mlx-metal\",\"dtype\":\"float32\",\"quantized\":false}}");
        return Ok(());
    }
    if matches!(args.get(1).map(String::as_str), Some("bert" | "bert-half")) {
        if args.len() != 4 {
            return Err("Usage: bert WEIGHTS.safetensors INPUT_DIR".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        let precision = if args[1] == "bert-half" { mlx_rs::Dtype::Float16 } else { mlx_rs::Dtype::Float32 };
        for (key, value) in raw {
            weights.insert(key, value.as_dtype(precision)?);
        }
        let model = aniflive_macos_v5turbo::bert::Bert::load(&weights)?;
        let dir = std::path::Path::new(&args[3]);
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let ids = load("input_ids.npy")?;
        let types = load("token_type_ids.npy")?;
        let mask = load("attention_mask.npy")?;
        let hidden = model.hidden_for_phones(&ids, Some(&types), Some(&mask))?;
        hidden.eval()?;
        hidden.save_numpy(dir.join("native-hidden.npy"))?;
        println!(
            "{{\"kind\":\"bert_selected_hidden_component_only\",\"selected_hidden_layer\":22}}"
        );
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("cache-key") {
        if args.len() != 3 {
            return Err("Usage: cache-key CACHE_MANIFEST.json".into());
        }
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        let identity: aniflive_macos_v5turbo::reference_cache::ReferenceIdentity =
            serde_json::from_value(manifest["identity"].clone())?;
        println!(
            "{}",
            aniflive_macos_v5turbo::reference_cache::identity_key(&identity)?
        );
        return Ok(());
    }
    if matches!(
        args.get(1).map(String::as_str),
        Some("cache" | "stream" | "stream-cancel")
    ) {
        if args.len() != 9 {
            return Err("Usage: cache|stream|stream-cancel MODEL_DIR PREPARED_INPUT_DIR AUDIO TRANSCRIPT CACHE_ROOT OUTPUT_DIR SEED".into());
        }
        let mut model =
            aniflive_macos_v5turbo::engine::Engine::load_with_precision(std::path::Path::new(&args[2]), cli_precision()?)?;
        configure_native_candidates(&mut model)?;
        let (reference, hit, key) = model.prepare_reference_cached(
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            std::path::Path::new(&args[5]),
            std::path::Path::new(&args[6]),
        )?;
        println!(
            "{{\"reference_cache_hit\":{},\"reference_cache_key\":\"{}\"}}",
            hit, key
        );
        if args[1] == "cache" {
            return Ok(());
        }
        let stop = std::sync::atomic::AtomicBool::new(false);
        let config = aniflive_macos_v5turbo::streaming::StreamConfig {
            core_frames: if std::env::var("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM").as_deref() == Ok("1") {
                200
            } else {
                20
            },
            capture_noise: true,
            ..Default::default()
        };
        return model.run_stream_seeded(
            &reference,
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[7]),
            args[8].parse()?,
            config,
            &stop,
            args[1] == "stream-cancel",
        );
    }
    if matches!(
        args.get(1).map(String::as_str),
        Some("seeded-cached" | "seeded-cached-capture")
    ) {
        if args.len() != 9 {
            return Err("Usage: seeded-cached[-capture] MODEL_DIR INPUT_DIR AUDIO TRANSCRIPT CACHE_ROOT OUTPUT_DIR SEED".into());
        }
        let mut model =
            aniflive_macos_v5turbo::engine::Engine::load_with_precision(std::path::Path::new(&args[2]), cli_precision()?)?;
        configure_native_candidates(&mut model)?;
        let (reference, hit, key) = model.prepare_reference_cached(
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            std::path::Path::new(&args[5]),
            std::path::Path::new(&args[6]),
        )?;
        println!(
            "{{\"reference_cache_hit\":{},\"reference_cache_key\":\"{}\"}}",
            hit, key
        );
        return model.run_seeded_cached(
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[7]),
            args[8].parse()?,
            &reference,
            args[1] == "seeded-cached-capture",
        );
    }
    if args.get(1).map(String::as_str) == Some("rng-check") {
        use aniflive_macos_v5turbo::rng::RequestRng;
        let equal = |a: &Array, b: &Array| -> Result<bool, Box<dyn std::error::Error>> {
            a.eval()?;
            b.eval()?;
            Ok(a.shape() == b.shape()
                && a.as_slice::<f32>()
                    .iter()
                    .zip(b.as_slice::<f32>())
                    .all(|(x, y)| x.to_bits() == y.to_bits()))
        };
        let mut a = RequestRng::new(20261112);
        let mut b = RequestRng::new(20261112);
        if !equal(
            &a.ar_exponential(&[1, 1025])?,
            &b.ar_exponential(&[1, 1025])?,
        )? {
            return Err("AR seed replay mismatch".into());
        }
        a.cfm_normal(&[1, 100, 7])?.eval()?;
        a.cfm_normal(&[1, 100, 13])?.eval()?;
        b.cfm_normal(&[1, 100, 29])?.eval()?;
        if !equal(
            &a.ar_exponential(&[1, 1024])?,
            &b.ar_exponential(&[1, 1024])?,
        )? {
            return Err("CFM draw sequence perturbed AR".into());
        }
        let mut c = RequestRng::new(20261112);
        let mut d = RequestRng::new(20261112);
        c.ar_exponential(&[1, 1024])?.eval()?;
        c.ar_exponential(&[1, 1025])?.eval()?;
        if !equal(&c.cfm_normal(&[1, 100, 32])?, &d.cfm_normal(&[1, 100, 32])?)? {
            return Err("AR draw sequence perturbed CFM".into());
        }
        let mut other = RequestRng::new(20261113);
        let mut original = RequestRng::new(20261112);
        if equal(
            &other.ar_exponential(&[1, 1025])?,
            &original.ar_exponential(&[1, 1025])?,
        )? {
            return Err("Different seed produced identical checked noise".into());
        }
        println!("{{\"ar_replay_exact\":true,\"ar_independent_of_cfm_shape_and_count\":true,\"cfm_independent_of_ar_draws\":true,\"different_seed_changes_noise\":true}}");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("sampler") {
        if args.len() != 4 {
            return Err("Usage: sampler FIXTURE_DIR TOP_P".into());
        }
        let dir = std::path::Path::new(&args[2]);
        let p: f32 = args[3].parse()?;
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let previous = load("previous.npy")?;
        let previous = previous.as_slice::<i32>();
        let token = aniflive_macos_v5turbo::sampling::sample(
            &load("logits.npy")?,
            previous,
            &load("q.npy")?,
            15,
            p,
            0.8,
            1.35,
        )?;
        token.eval()?;
        token.save_numpy(dir.join("native-token.npy"))?;
        return Ok(());
    }
    if matches!(
        args.get(1).map(String::as_str),
        Some("seeded" | "seeded-half" | "seeded-fast-ar")
    ) {
        if args.len() != 6 {
            return Err("Usage: seeded MODEL_DIR PREPARED_INPUT_DIR OUTPUT_DIR SEED".into());
        }
        let dtype = if args[1] == "seeded-half" {
            mlx_rs::Dtype::Float16
        } else {
            mlx_rs::Dtype::Float32
        };
        let mut model = aniflive_macos_v5turbo::engine::Engine::load_with_precision(
            std::path::Path::new(&args[2]),
            dtype,
        )?;
        configure_native_candidates(&mut model)?;
        if args[1] == "seeded-fast-ar" {
            return model.run_seeded_fast_ar(
                std::path::Path::new(&args[3]),
                std::path::Path::new(&args[4]),
                args[5].parse()?,
            );
        }
        return model.run_seeded(
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            args[5].parse()?,
        );
    }
    if args.get(1).map(String::as_str) == Some("speech") {
        if args.len() != 5 {
            return Err("Usage: speech MODEL_DIR INPUT_FIXTURE_DIR OUTPUT_DIR".into());
        }
        let mut model =
            aniflive_macos_v5turbo::engine::Engine::load_with_precision(std::path::Path::new(&args[2]), cli_precision()?)?;
        configure_native_candidates(&mut model)?;
        return model.run_fixture(
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
        );
    }
    if matches!(
        args.get(1).map(String::as_str),
        Some("hubert" | "hubert-half")
    ) {
        if args.len() != 5 {
            return Err("Usage: hubert HUBERT.safetensors V5.safetensors FIXTURE_DIR".into());
        }
        let dtype = if args[1] == "hubert-half" {
            mlx_rs::Dtype::Float16
        } else {
            mlx_rs::Dtype::Float32
        };
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            weights.insert(key, tensor.as_dtype(dtype)?);
        }
        let model = aniflive_macos_v5turbo::hubert::Hubert::load(&weights)?;
        let dir = std::path::Path::new(&args[4]);
        let wave = mlx_rs::with_device(Device::cpu(), || {
            Array::load_numpy(dir.join("waveform.npy"))
        })?;
        let hidden = model.forward(&wave.as_dtype(dtype)?)?;
        hidden.eval()?;
        hidden.save_numpy(dir.join("native-hubert.npy"))?;
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[3]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            if key.starts_with("ssl_proj.") || key.starts_with("quantizer.") {
                weights.insert(key, tensor.as_dtype(dtype)?);
            }
        }
        let encoder = aniflive_macos_v5turbo::semantic::SemanticEncoder::load(&weights)?;
        let (projected, codes) = encoder.forward(&hidden)?;
        projected
            .transpose_axes(&[0, 2, 1])?
            .save_numpy(dir.join("native-projected.npy"))?;
        codes.save_numpy(dir.join("native-codes.npy"))?;
        println!("{{\"kind\":\"hubert_reference_semantic_component_only\"}}");
        return Ok(());
    }
    if matches!(args.get(1).map(String::as_str), Some("ar" | "ar-half")) {
        if args.len() != 4 {
            return Err("Usage: ar WEIGHTS.safetensors FIXTURE_DIR".into());
        }
        let dtype = if args[1] == "ar-half" {
            mlx_rs::Dtype::Float16
        } else {
            mlx_rs::Dtype::Float32
        };
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            weights.insert(key, tensor.as_dtype(dtype)?);
        }
        let model = aniflive_macos_v5turbo::ar::ArModel::load(&weights, 16)?;
        let dir = std::path::Path::new(&args[3]);
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let (prefill, mut state) = model.prefill(
            &load("phones.npy")?,
            &load("bert.npy")?.as_dtype(dtype)?,
            &load("prompt.npy")?,
        )?;
        prefill.eval()?;
        prefill.save_numpy(dir.join("native-prefill.npy"))?;
        let logits = model.decode(&load("token.npy")?, &mut state)?;
        logits.eval()?;
        logits.save_numpy(dir.join("native-decode.npy"))?;
        println!("{{\"kind\":\"ar_prefill_decode_component_only\"}}");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("acoustic") {
        if args.len() != 4 {
            return Err("Usage: acoustic WEIGHTS.safetensors FIXTURE_DIR".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            if !key.starts_with("cfm.") {
                weights.insert(key, tensor.as_dtype(mlx_rs::Dtype::Float32)?);
            }
        }
        let model = aniflive_macos_v5turbo::acoustic::AcousticEncoder::load(&weights)?;
        let dir = std::path::Path::new(&args[3]);
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let features = model.forward(
            &load("codes.npy")?,
            &load("text.npy")?,
            &load("torch-ge.npy")?,
        )?;
        features.eval()?;
        features.save_numpy(dir.join("native-acoustic.npy"))?;
        println!("{{\"kind\":\"decode_encp_component_only\"}}");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("reference") {
        if args.len() != 4 {
            return Err("Usage: reference WEIGHTS.safetensors FIXTURE_DIR".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            if key.starts_with("ref_enc.") {
                weights.insert(key, tensor.as_dtype(mlx_rs::Dtype::Float32)?);
            }
        }
        let model = aniflive_macos_v5turbo::reference_encoder::ReferenceEncoder::load(&weights)?;
        let dir = std::path::Path::new(&args[3]);
        let reference = mlx_rs::with_device(Device::cpu(), || {
            Array::load_numpy(dir.join("reference.npy"))
        })?;
        let ge = model.forward(&reference)?;
        ge.eval()?;
        ge.save_numpy(dir.join("native-ge.npy"))?;
        println!("{{\"kind\":\"reference_style_component_only\"}}");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("estimator") {
        if args.len() != 4 {
            return Err("Usage: estimator WEIGHTS.safetensors FIXTURE_DIR".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            if key.starts_with("cfm.estimator.") {
                weights.insert(key, tensor.as_dtype(mlx_rs::Dtype::Float32)?);
            }
        }
        let model = aniflive_macos_v5turbo::estimator::Estimator::load(&weights)?;
        let dir = std::path::Path::new(&args[3]);
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let x = load("x.npy")?;
        let prompt = load("prompt.npy")?;
        let mu = load("mu.npy")?;
        let time = load("time.npy")?;
        let mask = load("mask.npy")?;
        let output = model.forward(&x, &prompt, &mu, &time, Some(&mask))?;
        output.eval()?;
        output.save_numpy(dir.join("native-estimator.npy"))?;
        let target_mask = load("target-mask.npy")?;
        let mut flow = x;
        for step in 0..4 {
            let velocity = model.forward(
                &flow,
                &prompt,
                &mu,
                &Array::from_slice(&[step as f32 * 0.25], &[1]),
                Some(&mask),
            )?;
            flow = mlx_rs::ops::select(
                &target_mask,
                flow.add(velocity.multiply(Array::from_f32(0.25))?)?,
                Array::from_f32(0.0),
            )?;
        }
        flow.eval()?;
        flow.save_numpy(dir.join("native-cfm.npy"))?;
        println!("{{\"kind\":\"22block_estimator_and_cfm_component_only\",\"steps\":4,\"cfg\":0,\"dtype\":\"float32\"}}");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("dit") {
        if args.len() != 4 {
            return Err("Usage: dit WEIGHTS.safetensors FIXTURE_DIR".into());
        }
        let raw = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let mut weights = std::collections::HashMap::new();
        for (key, tensor) in raw {
            if key.starts_with("cfm.estimator.transformer_blocks.0.") {
                weights.insert(key, tensor.as_dtype(mlx_rs::Dtype::Float32)?);
            }
        }
        let model = aniflive_macos_v5turbo::dit::DiTBlock::load(
            &weights,
            "cfm.estimator.transformer_blocks.0",
        )?;
        let dir = std::path::Path::new(&args[3]);
        let load =
            |name: &str| mlx_rs::with_device(Device::cpu(), || Array::load_numpy(dir.join(name)));
        let x = load("x.npy")?;
        let time = load("time.npy")?;
        let mask = load("mask.npy")?;
        let rope = load("rope.npy")?;
        let output = model.forward(&x, &time, Some(&mask), &rope)?;
        output.eval()?;
        output.save_numpy(dir.join("native-dit.npy"))?;
        println!("{{\"kind\":\"dit_block_component_only\",\"dtype\":\"float32\"}}");
        return Ok(());
    }
    if matches!(args.get(1).map(String::as_str), Some("vocoder" | "vocoder-half")) {
        if args.len() != 5 {
            return Err("Usage: vocoder WEIGHTS.safetensors MEL.npy OUTPUT.npy".into());
        }
        let weights = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(&args[2]))?;
        let weights = if args[1] == "vocoder-half" {
            weights.into_iter().map(|(key, value)| Ok((key, value.as_dtype(mlx_rs::Dtype::Float16)?)))
                .collect::<Result<std::collections::HashMap<_, _>, mlx_rs::error::Exception>>()?
        } else { weights };
        let model = aniflive_macos_v5turbo::vocoder::Vocoder::load(&weights)?;
        let mel = mlx_rs::with_device(Device::cpu(), || Array::load_numpy(&args[3]))?;
        let start = std::time::Instant::now();
        let output = model.forward(&mel)?;
        output.eval()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        output.save_numpy(&args[4])?;
        println!(
            "{{\"kind\":\"component_only\",\"vocoder_wall_ms\":{},\"samples\":{}}}",
            elapsed,
            output.size()
        );
        return Ok(());
    }
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = a.matmul(&a)?;
    b.eval()?;
    if b.as_slice::<f32>() != [7.0, 10.0, 15.0, 22.0] {
        return Err("GPU matmul mismatch".into());
    }
    let noise = Array::from_slice(&[1.0f32, 2.0, -1.0, 0.5], &[1, 1, 4]);
    let mask = Array::from_slice(&[false, false, true, true], &[1, 1, 4]);
    let out = cfm_four_step(&noise, &mask, |x, t| x.add(Array::from_f32(t)))?;
    out.eval()?;
    let values = out.as_slice::<f32>();
    if values != [0.0, 0.0, -1.69482421875, 1.509521484375] {
        return Err("CFM fixture mismatch".into());
    }
    println!("{{\"kind\":\"component_probe_only\",\"device\":\"gpu\",\"cfm_steps\":4,\"cfg\":0,\"flow\":{:?}}}", values);
    if let Some(path) = std::env::args().nth(1) {
        let weights = mlx_rs::with_device(Device::cpu(), || Array::load_safetensors(path))?;
        for tensor in weights.values() {
            tensor.eval()?;
        }
        println!(
            "{{\"loaded_tensors\":{},\"speech_inference_verified\":false}}",
            weights.len()
        );
    }
    Ok(())
}

/// CLI precision follows the worker contract: ANIFLIVE_TTS_NATIVE_PRECISION
/// FP16 selects Half; unset or FP32 keeps the original Float32 engine.
fn cli_precision() -> Result<mlx_rs::Dtype, Box<dyn std::error::Error>> {
    match std::env::var("ANIFLIVE_TTS_NATIVE_PRECISION").as_deref() {
        Ok("FP16") => Ok(mlx_rs::Dtype::Float16),
        Ok("FP32") | Err(std::env::VarError::NotPresent) => Ok(mlx_rs::Dtype::Float32),
        _ => Err("ANIFLIVE_TTS_NATIVE_PRECISION must be FP32 or FP16".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = if std::env::var_os("ANIFLIVE_COMPONENT_CPU_REFERENCE").is_some() {
        Device::cpu()
    } else {
        Device::gpu()
    };
    mlx_rs::with_device(device, run)
}
