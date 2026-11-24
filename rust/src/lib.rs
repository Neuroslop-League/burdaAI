//! Checked C07.1 weights. Tensor views borrow one immutable backing buffer.
mod json;
pub mod tokenizer;

use anyhow::{ensure, Context, Result};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

const FILES: [&str; 4] = [
    "config.json",
    "model.safetensors",
    "tokenizer-manifest.json",
    "tokenizer.json",
];

#[derive(Clone, Copy, PartialEq)]
pub enum Profile {
    Reference,
    Fixture,
}

pub struct Bundle {
    /// Present for reference bundles; the synthetic fixture uses WordLevel.
    pub tokenizer: Option<tokenizer::TextTokenizer>,
    pub config: Value,
    weights: Vec<u8>,
}

fn integer(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        v[key]
            .as_u64()
            .with_context(|| format!("invalid integer {key}"))?,
    )
    .context("integer too large")
}

fn path_size(dir: &Path, name: &str) -> Result<u64> {
    let meta = fs::symlink_metadata(dir.join(name)).with_context(|| name.to_owned())?;
    ensure!(
        meta.file_type().is_file(),
        "{name}: expected a regular file"
    );
    Ok(meta.len())
}

fn read_file(dir: &Path, files: &Value, name: &str, limit: u64) -> Result<Vec<u8>> {
    let entry = &files[name];
    let size = entry["bytes"].as_u64().context("invalid file length")?;
    ensure!(
        size <= limit && size == path_size(dir, name)?,
        "{name}: size mismatch"
    );
    let data = fs::read(dir.join(name)).with_context(|| name.to_owned())?;
    ensure!(
        data.len() as u64 == size
            && entry["sha256"].as_str() == Some(&format!("{:x}", Sha256::digest(&data))),
        "{name}: digest mismatch"
    );
    Ok(data)
}

fn layout(c: &Value, profile: Profile) -> Result<BTreeMap<String, Vec<usize>>> {
    let (layers, h, heads, kv_heads, f) = match profile {
        Profile::Fixture => (1, 8, 4, 2, 12),
        Profile::Reference => (20, 2048, 16, 4, 5632),
    };
    for (key, val) in [
        ("architecture_version", 1),
        ("num_layers", layers),
        ("hidden_size", h),
        ("num_attention_heads", heads),
        ("num_key_value_heads", kv_heads),
        ("intermediate_size", f),
        ("max_sequence_length", 4096),
    ] {
        ensure!(integer(c, key)? == val, "config.json: unsupported {key}");
    }
    for (key, val) in [
        ("rope_theta", 10000.0),
        ("rms_norm_eps", 1e-5),
        ("initializer_std", 0.02),
    ] {
        ensure!(c[key].as_f64() == Some(val), "config.json: invalid {key}");
    }
    let vocab = integer(c, "vocab_size")?;
    ensure!(
        vocab > 0 && (profile != Profile::Fixture || vocab == 8),
        "config.json: invalid vocabulary size"
    );
    let mut shapes = BTreeMap::from([
        ("embedding.weight".into(), vec![vocab, h]),
        ("norm.weight".into(), vec![h]),
    ]);
    let kv = kv_heads * (h / heads);
    for i in 0..layers {
        for (name, shape) in [
            ("attn_norm", vec![h]),
            ("ffn_norm", vec![h]),
            ("attn.q", vec![h, h]),
            ("attn.k", vec![kv, h]),
            ("attn.v", vec![kv, h]),
            ("attn.o", vec![h, h]),
            ("gate", vec![f, h]),
            ("up", vec![f, h]),
            ("down", vec![h, f]),
        ] {
            shapes.insert(format!("blocks.{i}.{name}.weight"), shape);
        }
    }
    Ok(shapes)
}

impl Bundle {
    pub fn load(dir: &Path, profile: Profile) -> Result<Self> {
        ensure!(
            path_size(dir, "manifest.json")? <= 1 << 20,
            "manifest.json too large"
        );
        let manifest =
            json::parse(&fs::read(dir.join("manifest.json"))?).context("manifest.json")?;
        ensure!(
            integer(&manifest, "schema_version")? == 1,
            "unsupported schema_version"
        );
        let expected = if profile == Profile::Fixture {
            "loader-fixture"
        } else {
            "reference-f32"
        };
        ensure!(manifest["profile"] == expected, "unsupported profile");
        let files = &manifest["files"];
        ensure!(files.as_object().is_some_and(|m| m.len()==FILES.len() && FILES.iter().all(|k| m.contains_key(*k))), "invalid file inventory");
        let config =
            json::parse(&read_file(dir, files, "config.json", 1 << 20)?).context("config.json")?;
        let shapes = layout(&config, profile)?;
        let tm = json::parse(&read_file(dir, files, "tokenizer-manifest.json", 1 << 20)?)
            .context("tokenizer-manifest.json")?;
        let token_bytes = read_file(dir, files, "tokenizer.json", 1 << 28)?;
        let tok = json::parse(&token_bytes).context("tokenizer.json")?;
        let vocab_size = integer(&config, "vocab_size")?;
        let digest = files["tokenizer.json"]["sha256"]
            .as_str()
            .context("tokenizer digest")?;
        let (prefix, kind, roles) = if profile == Profile::Fixture {
            (
                "fixture-wordlevel-sha256-",
                "WordLevel",
                ["<pad>", "<bos>", "<eos>"],
            )
        } else {
            (
                "burda-bpe-v1-sha256-",
                "BPE",
                ["<|pad|>", "<|bos|>", "<|eos|>"],
            )
        };
        ensure!(
            tm["file"] == "tokenizer.json"
                && tm["sha256"] == digest
                && tm["tokenizer_id"] == format!("{prefix}{digest}")
                && tm["tokenizer_id"] == manifest["tokenizer_id"]
                && tm["tokenizer_id"] == config["tokenizer_id"],
            "tokenizer-manifest.json: identity mismatch"
        );
        ensure!(
            integer(&tm, "vocab_size")? == vocab_size && tok["model"]["type"] == kind,
            "tokenizer.json: incompatible model"
        );
        let vocab = tok["model"]["vocab"]
            .as_object()
            .context("tokenizer vocabulary")?;
        ensure!(vocab.len() == vocab_size, "tokenizer vocabulary length");
        let mut seen = vec![false; vocab_size];
        for id in vocab.values() {
            let i = usize::try_from(id.as_u64().context("invalid vocabulary ID")?)?;
            ensure!(
                i < vocab_size && !seen[i],
                "invalid or repeated vocabulary ID"
            );
            seen[i] = true;
        }
        for (i, (role, token)) in ["PAD", "BOS", "EOS"].into_iter().zip(roles).enumerate() {
            ensure!(
                integer(&tm["special_tokens"], role)? == i
                    && vocab.get(token).and_then(Value::as_u64) == Some(i as u64),
                "tokenizer.json: invalid {role}"
            );
        }
        for item in tok["added_tokens"].as_array().context("added_tokens")? {
            let token = item["content"].as_str().context("added token content")?;
            ensure!(
                vocab.get(token) == Some(&item["id"]),
                "added token outside vocabulary"
            );
        }

        let tokenizer = match profile {
            Profile::Reference => Some(tokenizer::TextTokenizer::from_artifact(&token_bytes, &tm)?),
            Profile::Fixture => None,
        };
        let data_size = shapes
            .values()
            .try_fold(0u64, |total, s| {
                let bytes = s.iter().try_fold(4u64, |n, &d| n.checked_mul(d as u64))?;
                total.checked_add(bytes)
            })
            .context("tensor size overflow")?;
        let max = data_size
            .checked_add(100_000_008)
            .context("tensor file size overflow")?;
        let weights = read_file(dir, files, "model.safetensors", max)?;
        let header_len = u64::from_le_bytes(
            weights
                .get(..8)
                .context("short safetensors prefix")?
                .try_into()?,
        );
        ensure!(
            header_len <= 100_000_000
                && data_size.checked_add(header_len + 8) == Some(weights.len() as u64),
            "model.safetensors: inconsistent header or data length"
        );
        json::parse(&weights[8..8 + header_len as usize]).context("model.safetensors header")?;
        let tensors = SafeTensors::deserialize(&weights).context("model.safetensors")?;
        for (name, shape) in &shapes {
            let t = tensors
                .tensor(name)
                .with_context(|| format!("model.safetensors: {name}"))?;
            ensure!(
                t.dtype() == Dtype::F32 && t.shape() == shape,
                "model.safetensors: invalid type or axes for {name}"
            );
        }
        ensure!(
            tensors.len() == shapes.len(),
            "model.safetensors: unexpected tensor"
        );
        Ok(Self {
            config,
            weights,
            tokenizer,
        })
    }

    pub fn tensors(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.weights)?)
    }
}

pub mod cpu;
