//! Single-sequence FP32 reference forward. ndarray supplies matrix multiplication.
use crate::Bundle;
use anyhow::{ensure, Context, Result};
use ndarray::{s, Array2, ArrayView2};
use std::collections::BTreeMap;

pub struct CpuDecoder<'a> {
    weights: BTreeMap<String, &'a [f32]>,
    vocab: usize,
}

impl<'a> CpuDecoder<'a> {
    pub fn new(bundle: &'a Bundle) -> Result<Self> {
        ensure!(
            bundle.tokenizer.is_some(),
            "CPU forward requires reference-f32"
        );
        ensure!(cfg!(target_endian = "little"), "little-endian CPU required");
        let mut weights = BTreeMap::new();
        for (key, view) in bundle.tensors()?.tensors() {
            let values = bytemuck::try_cast_slice(view.data())
                .map_err(|_| anyhow::anyhow!("unaligned F32 tensor {key}"))?;
            weights.insert(key, values);
        }
        Ok(Self {
            weights,
            vocab: bundle.config["vocab_size"].as_u64().context("vocabulary")? as usize,
        })
    }

    fn linear(&self, x: &Array2<f32>, key: &str) -> Array2<f32> {
        let w = self.weights[key];
        let matrix = ArrayView2::from_shape((w.len() / x.ncols(), x.ncols()), w).unwrap();
        x.dot(&matrix.t())
    }

    fn norm(&self, x: &Array2<f32>, key: &str) -> Array2<f32> {
        let mut y = x.clone();
        for mut row in y.rows_mut() {
            let scale = (row.dot(&row) / row.len() as f32 + 1e-5).sqrt().recip();
            for (z, g) in row.iter_mut().zip(self.weights[key]) {
                *z *= scale * g;
            }
        }
        y
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Array2<f32>> {
        ensure!(
            (1..=4096).contains(&ids.len()),
            "expected 1..4096 input IDs"
        );
        ensure!(
            ids.iter().all(|&i| (i as usize) < self.vocab),
            "ID outside vocabulary"
        );
        let t = ids.len();
        let e = ArrayView2::from_shape((self.vocab, 2048), self.weights["embedding.weight"])?;
        let mut x = Array2::from_shape_fn((t, 2048), |(p, h)| e[(ids[p] as usize, h)]);
        let angles = Array2::from_shape_fn((t, 64), |(p, i)| {
            (p as f32 * 10000f32.powf(-(2.0 * i as f32) / 128.0)).sin_cos()
        });
        for layer in 0..20 {
            let prefix = format!("blocks.{layer}.");
            let key = |s: &str| format!("{prefix}{s}.weight");
            let z = self.norm(&x, &key("attn_norm"));
            let mut q = self.linear(&z, &key("attn.q"));
            let mut k = self.linear(&z, &key("attn.k"));
            let v = self.linear(&z, &key("attn.v"));
            for tensor in [&mut q, &mut k] {
                for (p, mut row) in tensor.rows_mut().into_iter().enumerate() {
                    for (i, pair) in row.as_slice_mut().unwrap().chunks_exact_mut(2).enumerate() {
                        let (sin, cos) = angles[(p, i % 64)];
                        let (a, b) = (pair[0], pair[1]);
                        pair[0] = a * cos - b * sin;
                        pair[1] = a * sin + b * cos;
                    }
                }
            }
            let mut attention = Array2::zeros((t, 2048));
            for p in 0..t {
                for h in 0..16 {
                    let kv = (h / 4) * 128;
                    let keys = k.slice(s![..=p, kv..kv + 128]);
                    let query = q.slice(s![p, h * 128..(h + 1) * 128]);
                    let mut scores = keys.dot(&query) / 128f32.sqrt();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    scores.mapv_inplace(|n| (n - max).exp());
                    scores /= scores.sum();
                    let values = v.slice(s![..=p, kv..kv + 128]);
                    attention
                        .slice_mut(s![p, h * 128..(h + 1) * 128])
                        .assign(&scores.dot(&values));
                }
            }
            x += &self.linear(&attention, &key("attn.o"));
            let z = self.norm(&x, &key("ffn_norm"));
            let mut gate = self.linear(&z, &key("gate"));
            gate.mapv_inplace(|a| a / (1.0 + (-a).exp()));
            gate *= &self.linear(&z, &key("up"));
            x += &self.linear(&gate, &key("down"));
        }
        Ok(self.linear(&self.norm(&x, "norm.weight"), "embedding.weight"))
    }
}

pub fn generate(
    mut ids: Vec<u32>,
    max_new: usize,
    mut next: impl FnMut(&[u32]) -> Result<u32>,
) -> Result<Vec<u32>> {
    ensure!(
        (1..=4096).contains(&ids.len()),
        "prompt including BOS exceeds context or is empty"
    );
    let start = ids.len();
    for _ in 0..max_new.min(4096 - start) {
        let id = next(&ids)?;
        if id == 2 {
            break;
        }
        ids.push(id);
    }
    Ok(ids.split_off(start))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stops_at_eos_and_window() {
        let mut calls = 0;
        let out = generate(vec![1], 8, |_| {
            calls += 1;
            Ok(if calls == 1 { 7 } else { 2 })
        })
        .unwrap();
        assert_eq!((out, calls), (vec![7], 2));
        assert_eq!(generate(vec![1; 4095], 8, |_| Ok(7)).unwrap(), [7]);
        assert!(generate(vec![1; 4096], 8, |_| panic!()).unwrap().is_empty());
        assert!(generate(vec![1; 4097], 1, |_| panic!()).is_err());
    }
}
