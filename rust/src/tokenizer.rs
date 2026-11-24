//! C02 text encoding for Rust inference. The caller inserts document boundaries.
use anyhow::{anyhow, ensure, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub struct TextTokenizer {
    inner: Tokenizer,
    pub tokenizer_id: String,
}

impl TextTokenizer {
    pub fn load_c02(directory: &Path) -> Result<Self> {
        let manifest = super::json::parse(&fs::read(directory.join("manifest.json"))?)?;
        let bytes = fs::read(directory.join("tokenizer.json"))?;
        Self::from_artifact(&bytes, &manifest)
    }

    pub(crate) fn from_artifact(bytes: &[u8], manifest: &Value) -> Result<Self> {
        let digest = format!("{:x}", Sha256::digest(bytes));
        let tokenizer_id = format!("burda-bpe-v1-sha256-{digest}");
        ensure!(
            manifest["file"] == "tokenizer.json"
                && manifest["sha256"] == digest
                && manifest["tokenizer_id"] == tokenizer_id,
            "C02 artifact identity mismatch"
        );
        let mut inner = Tokenizer::from_bytes(bytes).map_err(|e| anyhow!("{e}"))?;
        let size = super::integer(manifest, "vocab_size")?;
        let mut ids: Vec<_> = inner.get_vocab(true).into_values().collect();
        ids.sort_unstable();
        ensure!(
            ids.len() == size && ids.iter().enumerate().all(|(i, &id)| i == id as usize),
            "C02 vocabulary mismatch"
        );
        for (i, (role, literal)) in [("PAD", "<|pad|>"), ("BOS", "<|bos|>"), ("EOS", "<|eos|>")]
            .into_iter()
            .enumerate()
        {
            ensure!(
                super::integer(&manifest["special_tokens"], role)? == i
                    && inner.token_to_id(literal) == Some(i as u32),
                "invalid C02 {role}"
            );
        }
        inner.set_encode_special_tokens(true);
        Ok(Self {
            inner,
            tokenizer_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("{e}"))
            .context("encode text")?;
        Ok(tokens.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("{e}"))
            .context("decode IDs")
    }
}
