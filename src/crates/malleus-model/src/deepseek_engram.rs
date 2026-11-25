//! The Engram n-gram memory of DeepSeek-V4.1 (issue #83).
//!
//! Two layers read a hash table: the last `max_ngram - 1` tokens before each position are hashed into
//! `(order, head)` buckets of the layer's table, the rows are added into the residual streams through a gate.
//!
//! The addressing is fixed for a model and does not depend on the prompt: a compressed token map (tokens that
//! normalise alike share an id), one prime per (layer, order, head), one multiplier per (layer, lookback), and an
//! offset per bucket. `scripts/deepseek_engram_state.py` computes them once with the reference's own code, so this
//! module only has to reproduce the arithmetic. The rows themselves stay in the companion file and are read one at
//! a time (`read_rows`): the two tables are 183 GB and are never a matrix the forward multiplies.

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use serde_json::Value;

/// Reads rows `rows[i]` of a `RowI4` table as f32: `k / 2` bytes of nibbles per row (LSB first, stored as
/// `value + 8`, so the levels are `nibble - 8`) and one f16 scale per row. A table is 98.31B rows, so it is never materialised: every row is a
/// read of `k / 2 + 2` bytes from the file, which is what makes it a lookup and not a matmul.
pub fn read_rows(src: &dyn TensorSource, name: &str, rows: &[u64], k: usize) -> Result<Vec<f32>> {
    let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        info.dtype == DType::RowI4 && info.shape.len() == 2,
        "{name}: expected a RowI4 [rows, k] table, not {:?} {:?}",
        info.dtype,
        info.shape
    );
    let (n, width) = (info.shape[0], info.shape[1]);
    ensure!(width == k, "{name}: rows are {width} wide, the graph reads {k}");
    let packed = k.div_ceil(2);
    let scales = format!("{name}.scales");
    let mut out = vec![0f32; rows.len() * k];
    let mut codes = vec![0u8; packed];
    let mut scale = [0u8; 2];
    for (i, &r) in rows.iter().enumerate() {
        ensure!((r as usize) < n, "{name}: row {r} outside the table's {n}");
        src.read(name, r as usize * packed, &mut codes)?;
        src.read(&scales, r as usize * 2, &mut scale)?;
        let s = half::f16::from_le_bytes(scale).to_f32();
        malleus_gguf::quant::dequant_rows_i4(&codes, s, &mut out[i * k..(i + 1) * k]);
    }
    Ok(out)
}

/// The hash state of one checkpoint, read from the file's metadata.
pub struct EngramState {
    /// 0-based layers that carry a table, in the table order.
    pub layers: Vec<usize>,
    pub num_embeddings: Vec<usize>,
    pub max_ngram: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pad: u32,
    /// `[layer][order][head]` bucket modulus.
    primes: Vec<Vec<Vec<u64>>>,
    /// `[layer][order * heads + head]` bucket offset.
    offsets: Vec<Vec<u64>>,
    /// `[layer][lookback]`.
    multipliers: Vec<Vec<i64>>,
    /// One compressed id per token of the tokenizer's vocabulary.
    token_map: Vec<i32>,
}

impl EngramState {
    /// `state` is the JSON `scripts/deepseek_engram_state.py` writes, `token_map` the file it writes beside it.
    pub fn from_json(state: &str, token_map: &[u8]) -> Result<Self> {
        let v: Value = serde_json::from_str(state).context("engram state")?;
        let list = |k: &str| -> Vec<u64> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default()
        };
        let nested = |k: &str| -> Vec<Vec<Vec<u64>>> {
            v[k].as_array()
                .map(|a| {
                    a.iter()
                        .map(|per_layer| {
                            per_layer
                                .as_array()
                                .map(|b| {
                                    b.iter()
                                        .map(|o| {
                                            o.as_array()
                                                .map(|c| c.iter().filter_map(|x| x.as_u64()).collect())
                                                .unwrap_or_default()
                                        })
                                        .collect()
                                })
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let tokens: Vec<i32> = token_map
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let e = EngramState {
            layers: list("layers").iter().map(|&l| l as usize).collect(),
            num_embeddings: list("num_embeddings").iter().map(|&l| l as usize).collect(),
            max_ngram: v["max_ngram"].as_u64().context("max_ngram")? as usize,
            heads: v["n_heads"].as_u64().context("n_heads")? as usize,
            head_dim: v["head_dim"].as_u64().context("head_dim")? as usize,
            pad: v["pad"].as_u64().context("pad")? as u32,
            primes: nested("primes"),
            offsets: v["offsets"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|l| {
                            l.as_array()
                                .map(|o| o.iter().filter_map(|x| x.as_u64()).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            multipliers: v["multipliers"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|l| {
                            l.as_array()
                                .map(|m| m.iter().filter_map(|x| x.as_i64()).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            token_map: tokens,
        };
        e.validate()?;
        Ok(e)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.layers.len() == self.num_embeddings.len() && !self.layers.is_empty(),
            "engram: {} layers for {} tables",
            self.layers.len(),
            self.num_embeddings.len()
        );
        anyhow::ensure!(
            self.primes.len() == self.layers.len() && self.offsets.len() == self.layers.len(),
            "engram: primes and offsets must cover every layer"
        );
        for (i, layer) in self.primes.iter().enumerate() {
            anyhow::ensure!(
                layer.len() == self.max_ngram - 1 && layer.iter().all(|o| o.len() == self.heads),
                "engram: layer {} wants {} orders x {} heads",
                i,
                self.max_ngram - 1,
                self.heads
            );
            anyhow::ensure!(
                self.offsets[i].len() == (self.max_ngram - 1) * self.heads,
                "engram: layer {} offsets",
                i
            );
            anyhow::ensure!(
                self.multipliers[i].len() == self.max_ngram,
                "engram: layer {} multipliers",
                i
            );
        }
        anyhow::ensure!(
            (self.pad as usize) < self.token_map.len(),
            "engram: the pad token is outside the vocabulary"
        );
        Ok(())
    }

    /// Hash ids of every position of `ids`: `[t, (max_ngram - 1) * heads]`, one row per position, for the
    /// Engram table `layer` (the position in `layers`, not the layer index).
    pub fn hash_ids(&self, ids: &[u32], layer: usize) -> Vec<u64> {
        let compressed: Vec<i64> = ids
            .iter()
            .map(|&id| self.token_map.get(id as usize).copied().unwrap_or(-1) as i64)
            .collect();
        self.hash_compressed(&compressed, layer)
    }

    /// The same, with a token mask: a position whose entry is false — every position of an image span — is hashed
    /// as a dead token, so no n-gram reaches into it and none spans one. `text` is the release's `token_mask`,
    /// which is `~image_mask`; it must cover `ids`.
    pub fn hash_ids_masked(&self, ids: &[u32], layer: usize, text: &[bool]) -> Vec<u64> {
        assert_eq!(text.len(), ids.len(), "engram: the mask must cover the sequence");
        let mut compressed: Vec<i64> = ids
            .iter()
            .map(|&id| self.token_map.get(id as usize).copied().unwrap_or(-1) as i64)
            .collect();
        for (c, &keep) in compressed.iter_mut().zip(text) {
            if !keep {
                *c = -1;
            }
        }
        self.hash_compressed(&compressed, layer)
    }

    /// The addressing itself, on the compressed ids (`-1` is a dead position: the pad fills its n-gram slots and
    /// the lookback stops there).
    fn hash_compressed(&self, compressed: &[i64], layer: usize) -> Vec<u64> {
        let t = compressed.len();
        let order = self.max_ngram - 1;
        let cols = order * self.heads;
        let pad = self.token_map[self.pad as usize] as i64;
        let mut out = vec![0u64; t * cols];
        for i in 0..t {
            // The n-gram is hashed one lookback at a time: after step k the running value is the (k+1)-gram's
            // hash, and it lands in the (k+1)-th order's buckets. A dead position — the start of the sequence, a
            // token without a compressed id, or a position inside an image span — blocks that n-gram and every
            // longer one behind it, which is what the release's cumulative mask does.
            let mut rolling = 0i64;
            let mut blocked = false;
            for shift in 0..self.max_ngram {
                let src = if shift > i { -1 } else { compressed[i - shift] };
                blocked |= src == -1;
                let token = if blocked { pad } else { src };
                let product = token * self.multipliers[layer][shift];
                if shift == 0 {
                    rolling = product;
                } else {
                    rolling ^= product;
                    let bucket = self.primes[layer][shift - 1].iter().enumerate();
                    for (head, &prime) in bucket {
                        let at = (shift - 1) * self.heads + head;
                        out[i * cols + at] = (rolling.rem_euclid(prime as i64)) as u64 + self.offsets[layer][at];
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash ids against `engram.py` on a real prompt, plain and with an image span masked out: the addressing
    /// has to agree exactly, integer for integer, and a masked position must take no part in an n-gram and stop the
    /// lookback that reaches into it.
    ///
    /// `DEEPSEEK_ENGRAM=/golden/deepseek-engram DEEPSEEK_ENGRAM_HASH=/golden/deepseek-engram-hash`; the masked dump
    /// is the `-masked` sibling `scripts/deepseek_engram_state.py` writes.
    #[test]
    #[ignore = "reads the state the python script wrote"]
    fn the_hash_matches_the_reference() {
        let base = std::env::var("DEEPSEEK_ENGRAM").expect("DEEPSEEK_ENGRAM");
        let hash = std::env::var("DEEPSEEK_ENGRAM_HASH").expect("DEEPSEEK_ENGRAM_HASH");
        let state = std::fs::read_to_string(format!("{base}-state.json")).unwrap();
        let map = std::fs::read(format!("{base}-token-map.i32")).unwrap();
        let e = EngramState::from_json(&state, &map).unwrap();
        for (what, path) in [("plain", hash.clone()), ("masked", format!("{hash}-masked"))] {
            let (t, layers, cols) = compare_hash_dump(&e, &path);
            eprintln!("{what}: {t} positions x {layers} layers x {cols} buckets agree with the reference");
        }
    }

    /// Compares one dump against `hash_ids` (or `hash_ids_masked` when the dump carries a mask) and returns its
    /// `(positions, layers, buckets)`.
    fn compare_hash_dump(e: &EngramState, path: &str) -> (usize, usize, usize) {
        let manifest: Value = serde_json::from_str(&std::fs::read_to_string(format!("{path}.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mask: Vec<bool> = manifest["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        // The reference dumps `[positions, layers, buckets]`.
        let shape: Vec<usize> = manifest["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want: Vec<i64> = std::fs::read(format!("{path}.i64"))
            .unwrap()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        let (t, layers, cols) = (shape[0], shape[1], shape[2]);
        assert_eq!(t, ids.len(), "{path}: positions");
        assert_eq!(mask.len(), t, "{path}: the mask must cover the prompt");
        assert_eq!((layers, cols), (e.layers.len(), (e.max_ngram - 1) * e.heads));
        let plain = mask.iter().all(|&keep| keep);
        for l in 0..layers {
            let got = if plain {
                e.hash_ids(&ids, l)
            } else {
                e.hash_ids_masked(&ids, l, &mask)
            };
            let mut wrong = 0;
            for i in 0..t {
                for c in 0..cols {
                    let w = want[(i * layers + l) * cols + c] as u64;
                    if got[i * cols + c] != w {
                        wrong += 1;
                        if wrong < 4 {
                            eprintln!("layer {l} pos {i} col {c}: got {} want {w}", got[i * cols + c]);
                        }
                    }
                }
            }
            assert_eq!(wrong, 0, "{path}: layer {l} has {wrong} of {} ids wrong", t * cols);
        }
        (t, layers, cols)
    }
}
