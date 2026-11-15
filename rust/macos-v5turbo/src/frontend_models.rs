//! Optional persistent neural frontend components on the worker's model thread.
use crate::{bert::Bert, g2pw::G2PW};
use mlx_rs::{Array, Device, Dtype};
use serde_json::{json, Value};
use std::{collections::HashMap, error::Error, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct FrontendModels {
    g2pw: Option<G2PW>,
    bert: Option<Bert>,
    bert_precision: Dtype,
}

fn weights(path: &Path, precision: Dtype) -> Result<HashMap<String, Array>> {
    let values = mlx_rs::with_stream(
        &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
        || Array::load_safetensors(path),
    )?;
    let mut weights = HashMap::new();
    for (name, array) in values {
        weights.insert(name, array.as_dtype(precision)?);
    }
    Ok(weights)
}

fn input(dir: &Path, name: &str) -> Result<Array> {
    Ok(mlx_rs::with_stream(
        &mlx_rs::Stream::try_default_on_device(&Device::cpu())?,
        || Array::load_numpy(dir.join(name)),
    )?)
}

fn empty_output(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    if std::fs::read_dir(dir)?.next().is_some() {
        return Err("frontend output directory must be empty".into());
    }
    Ok(())
}

impl FrontendModels {
    /// Missing components are unavailable; existing invalid weights fail startup.
    pub fn load(model_dir: &Path, precision: Dtype) -> Result<Self> {
        if !matches!(precision, Dtype::Float32 | Dtype::Float16) {
            return Err("Invalid frontend BERT precision".into());
        }
        let g2pw_path = model_dir.join("g2pw/g2pw.safetensors");
        let bert_path = model_dir.join("chinese-roberta-wwm-ext-large/bert.safetensors");
        let g2pw = if g2pw_path.try_exists()? {
            Some(G2PW::load(&weights(&g2pw_path, Dtype::Float32)?)?)
        } else {
            None
        };
        let bert = if bert_path.try_exists()? {
            Some(Bert::load(&weights(&bert_path, precision)?)?)
        } else {
            None
        };
        Ok(Self {
            g2pw,
            bert,
            bert_precision: precision,
        })
    }

    pub fn has_g2pw(&self) -> bool {
        self.g2pw.is_some()
    }

    pub fn has_bert(&self) -> bool {
        self.bert.is_some()
    }

    pub fn g2pw(&self, source: &Path, output: &Path) -> Result<Value> {
        let model = self
            .g2pw
            .as_ref()
            .ok_or("frontend_g2pw unavailable: model weights are not installed")?;
        empty_output(output)?;
        let ids = input(source, "input_ids.npy")?;
        let types = input(source, "token_type_ids.npy")?;
        let attention = input(source, "attention_mask.npy")?;
        let phonemes = input(source, "phoneme_mask.npy")?;
        let chars = input(source, "char_ids.npy")?;
        let positions = input(source, "position_ids.npy")?;
        let probabilities =
            model.forward(&ids, &types, &attention, &phonemes, &chars, &positions)?;
        probabilities.eval()?;
        let path = output.join("probabilities.npy");
        probabilities.save_numpy(&path)?;
        Ok(
            json!({"backend":"mlx-metal", "precision":"float32", "shape":probabilities.shape(), "output_path":path}),
        )
    }

    pub fn bert(&self, source: &Path, output: &Path) -> Result<Value> {
        let model = self
            .bert
            .as_ref()
            .ok_or("frontend_bert unavailable: model weights are not installed")?;
        empty_output(output)?;
        let ids = input(source, "input_ids.npy")?;
        let types = input(source, "token_type_ids.npy")?;
        let attention = input(source, "attention_mask.npy")?;
        let hidden = model.hidden_for_phones(&ids, Some(&types), Some(&attention))?;
        if hidden.dtype() != self.bert_precision {
            return Err("BERT output dtype differs from runtime precision".into());
        }
        let transported = if self.bert_precision == Dtype::Float16 {
            hidden.as_dtype(Dtype::Float32)?
        } else {
            hidden
        };
        transported.eval()?;
        let path = output.join("hidden.npy");
        transported.save_numpy(&path)?;
        Ok(
            json!({"backend":"mlx-metal", "precision":if self.bert_precision == Dtype::Float16 {"float16"} else {"float32"}, "transport_dtype":"float32", "shape":transported.shape(), "output_path":path}),
        )
    }
}
