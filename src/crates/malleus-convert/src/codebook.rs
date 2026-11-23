//! A learned codebook for 2-bit weights (issue #83).
//!
//! The 4-level scalar grid MLS2 uses is near-optimal *for a scalar*: Lloyd-Max on a Gaussian at 2 bits leaves
//! 0.1175 of the variance, which is what the 0.94 cosine on a real expert measures. A vector quantizer over
//! blocks of 8 weights does better at the same rate, because it can spend the code on the shape of the block
//! rather than on each coordinate. This module learns that codebook and encodes with it.
//!
//! The layout is the one the GGUF builds use: an 8-bit index per block of 8 weights, plus a scale per group of
//! the caller's choosing. At 8 weights per index and a scale per 32, the file is 1.5 bits per weight.

use rayon::prelude::*;

/// Weights per codebook index.
pub const BLOCK: usize = 8;

/// A codebook: `k` entries of `BLOCK` signed values.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub entries: Vec<[f32; BLOCK]>,
}

impl Codebook {
    /// Learns `k` entries by Lloyd's algorithm on blocks normalised by their group's RMS: the codebook then
    /// carries the *shape* of a block and the group scale carries its magnitude, which is what lets one scale
    /// serve 32 weights without the codebook memorising magnitudes.
    pub fn learn(rows: &[f32], group: usize, k: usize, iters: usize) -> Codebook {
        let mut normalised = Vec::with_capacity(rows.len());
        for g in rows.chunks(group) {
            let rms = (g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / g.len() as f64).sqrt();
            let rms = if rms > 0.0 { rms } else { 1.0 };
            normalised.extend(g.iter().map(|x| (*x as f64 / rms) as f32));
        }
        Codebook::learn_normalised(&normalised, k, iters)
    }

    /// Lloyd's algorithm on blocks that are already normalised.
    pub fn learn_normalised(rows: &[f32], k: usize, iters: usize) -> Codebook {
        assert!(k <= 256, "an index is one byte");
        let blocks = rows.len() / BLOCK;
        assert!(blocks >= k, "fewer blocks than entries");
        // Seed the centroids by farthest-point: the first block, then the block farthest from those already
        // chosen. Uniform spreading collapses when two seeds land on the same shape.
        let mut entries: Vec<[f32; BLOCK]> = Vec::with_capacity(k);
        let mut first = [0f32; BLOCK];
        first.copy_from_slice(&rows[..BLOCK]);
        entries.push(first);
        while entries.len() < k {
            let mut best = 0usize;
            let mut best_d = f64::NEG_INFINITY;
            for b in 0..blocks {
                let block = &rows[b * BLOCK..b * BLOCK + BLOCK];
                let d = entries
                    .iter()
                    .map(|e| {
                        e.iter()
                            .zip(block)
                            .map(|(x, y)| ((*x - *y) as f64).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                if d > best_d {
                    best_d = d;
                    best = b;
                }
            }
            let mut e = [0f32; BLOCK];
            e.copy_from_slice(&rows[best * BLOCK..best * BLOCK + BLOCK]);
            entries.push(e);
        }
        let mut sums = vec![[0f64; BLOCK]; k];
        let mut counts = vec![0usize; k];
        for _ in 0..iters {
            sums.iter_mut().for_each(|s| *s = [0.0; BLOCK]);
            counts.iter_mut().for_each(|c| *c = 0);
            for b in 0..blocks {
                let block = &rows[b * BLOCK..b * BLOCK + BLOCK];
                let (mut best, mut best_d) = (0usize, f64::INFINITY);
                for (i, e) in entries.iter().enumerate() {
                    let d: f64 = e.iter().zip(block).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                counts[best] += 1;
                for j in 0..BLOCK {
                    sums[best][j] += block[j] as f64;
                }
            }
            for i in 0..k {
                if counts[i] > 0 {
                    for j in 0..BLOCK {
                        entries[i][j] = (sums[i][j] / counts[i] as f64) as f32;
                    }
                }
            }
        }
        Codebook { entries }
    }

    /// The index of the entry nearest to a block.
    pub fn encode(&self, block: &[f32]) -> u8 {
        let mut best = 0u8;
        let mut best_d = f64::INFINITY;
        for (i, e) in self.entries.iter().enumerate() {
            let d: f64 = e.iter().zip(block).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        best
    }

    pub fn decode(&self, index: u8) -> &[f32; BLOCK] {
        &self.entries[index as usize]
    }
}

/// Encodes a row: one index per block of 8 and one f32 scale per group. The scale is fitted to the entries the
/// row actually chose, by least squares, twice: the first pass picks entries at the group's RMS, the second
/// re-picks them at the fitted scale.
pub fn encode_row(row: &[f32], codebook: &Codebook, group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut indices = Vec::with_capacity(row.len() / BLOCK);
    let mut scales = Vec::with_capacity(row.len() / group);
    for g in row.chunks(group) {
        let rms = (g.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / g.len() as f64).sqrt() as f32;
        let mut scale = if rms > 0.0 { rms } else { 1.0 };
        let mut chosen: Vec<usize> = Vec::with_capacity(g.len() / BLOCK);
        for _ in 0..2 {
            chosen.clear();
            for b in g.chunks(BLOCK) {
                let mut best = (f32::INFINITY, 0usize);
                for (i, e) in codebook.entries.iter().enumerate() {
                    let mut err = 0f32;
                    for (x, c) in b.iter().zip(e.iter()) {
                        let d = x / scale - c;
                        err += d * d;
                    }
                    if err < best.0 {
                        best = (err, i);
                    }
                }
                chosen.push(best.1);
            }
            let mut num = 0f64;
            let mut den = 0f64;
            for (b, &i) in g.chunks(BLOCK).zip(&chosen) {
                for (x, c) in b.iter().zip(codebook.entries[i].iter()) {
                    num += (*x as f64) * (*c as f64);
                    den += (*c as f64).powi(2);
                }
            }
            if den > 0.0 {
                scale = (num / den) as f32;
            }
        }
        scales.push(scale);
        indices.extend(chosen.iter().map(|&i| i as u8));
    }
    (indices, scales)
}

/// Decodes a row back to f32.
pub fn decode_row(indices: &[u8], scales: &[f32], codebook: &Codebook, group: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * BLOCK);
    for (gi, &s) in scales.iter().enumerate() {
        for &i in &indices[gi * group / BLOCK..(gi + 1) * group / BLOCK] {
            for c in codebook.entries[i as usize].iter() {
                out.push(c * s);
            }
        }
    }
    out
}

/// A row's cosine against the original, which is what the round-trip gate reports.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Quantizes a whole matrix with the codebook and returns the mean cosine over its rows.
pub fn round_trip(rows: &[f32], n: usize, k: usize, codebook: &Codebook, group: usize) -> f64 {
    let coss: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|r| {
            let row = &rows[r * k..(r + 1) * k];
            let (idx, sc) = encode_row(row, codebook, group);
            let back = decode_row(&idx, &sc, codebook, group);
            cosine(row, &back)
        })
        .collect();
    coss.iter().sum::<f64>() / coss.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_codebook_learns_the_blocks_it_is_given() {
        // 64 blocks of 8, all of them one of two shapes.
        let mut rows = Vec::new();
        for i in 0..64 {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            for j in 0..BLOCK {
                rows.push(s * (j as f32 + 1.0) * 0.1);
            }
        }
        let cb = Codebook::learn(&rows, 32, 2, 8);
        // Two groups of 16: two indices each, one scale each.
        let back = decode_row(&[0, 1, 0, 1], &[1.0; 2], &cb, 16);
        let want = &rows[0..32];
        assert!(cosine(&back, want) > 0.999, "{}", cosine(&back, want));
    }

    #[test]
    fn the_index_is_one_byte() {
        let rows: Vec<f32> = (0..256 * BLOCK).map(|i| (i as f32 * 0.01).sin()).collect();
        let cb = Codebook::learn(&rows, 32, 256, 2);
        assert_eq!(cb.entries.len(), 256);
        let (idx, _) = encode_row(&rows[0..128], &cb, 32);
        assert_eq!(idx.len(), 16);
    }
}

#[cfg(test)]
mod experiment {
    use super::*;
    use crate::deepseek::fp4_rows;
    use malleus_gguf::safetensors::SafeTensors;
    use malleus_gguf::TensorSource;

    /// Measures what a 2-bit quantizer costs on a real expert, so the codebook is chosen on numbers and not on
    /// theory. Run with the checkpoint mounted:
    ///
    ///   DEEPSEEK_CKPT=/models/deepseekv41-flash cargo test -p malleus-convert experiment -- --ignored --nocapture
    #[test]
    #[ignore = "reads the real checkpoint"]
    fn what_a_two_bit_quantizer_costs() {
        let root = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let st = SafeTensors::open_dir(std::path::Path::new(&root)).unwrap();
        let name = "layers.0.ffn.experts.0.w1.weight";
        let info = st.info(name).unwrap();
        let (n, k) = (info.shape[0], info.shape[1] * 2);
        let mut q = vec![0u8; info.size];
        st.read(name, 0, &mut q).unwrap();
        let mut s = vec![0u8; n * (k / 32)];
        st.read(&format!("{}.scale", name.trim_end_matches(".weight")), 0, &mut s)
            .unwrap();
        let mut w = vec![0f32; n * k];
        fp4_rows(&q, &s, n, k, &mut w).unwrap();
        println!("{name}: {n}x{k}, {} values", w.len());

        // Rotate every row the way the engine does, so the quantizer sees what the kernel will see.
        let signs = malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
        let mut rot = w.clone();
        for r in 0..n {
            malleus_gguf::quant::rotate_row(&mut rot[r * k..(r + 1) * k], &signs);
        }

        // The scalar grid the engine ships, at the group it ships and at finer ones.
        for group in [128usize, 32] {
            let mut coss = 0f64;
            for r in 0..n {
                let row = &rot[r * k..(r + 1) * k];
                let mut back = Vec::with_capacity(k);
                for g in row.chunks(group) {
                    let mut blk = [0f32; 128];
                    let src = &g[..group.min(128)];
                    blk[..src.len()].copy_from_slice(src);
                    let (codes, sc) =
                        malleus_gguf::quant::quantize_group_int2(&blk[..128], malleus_gguf::quant::Int2Grid::E2m1);
                    let mut out = [0f32; 128];
                    malleus_gguf::quant::dequant_int2(&codes, sc, malleus_gguf::quant::Int2Grid::E2m1, &mut out);
                    back.extend_from_slice(&out[..src.len()]);
                }
                coss += cosine(row, &back);
            }
            let bits = 2.0 + 16.0 / group as f64;
            println!(
                "scalar 4-level, scale per {group:3}: cosine {:.5}, {bits:.3} bits/weight",
                coss / n as f64
            );
        }

        // A learned codebook: 256 entries over 8 weights, one scale per group.
        let cb = Codebook::learn(&rot[..64 * k], 32, 256, 4);
        for group in [128usize, 32, 16, 8] {
            let cos = round_trip(&rot, n, k, &cb, group);
            let bits = 1.0 + 16.0 / group as f64;
            println!("codebook 256x8, scale per {group:3}: cosine {cos:.5}, {bits:.3} bits/weight");
        }
    }
}
