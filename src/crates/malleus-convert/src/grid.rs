//! A learned 2-bit grid (issue #83).
//!
//! MLS2 ships four fixed levels, `-1.5, -0.5, 0.5, 1.5`, taken from the e2m1 table's shape. That grid is a
//! compromise: it was chosen for the format, not for the weights. A rotated row is very close to Gaussian, and
//! the best four levels for a Gaussian are not those four numbers.
//!
//! This module learns the four levels per tensor by Lloyd's algorithm on the rotated weights, keeping the
//! storage exactly as it is: 2-bit codes, one fp16 scale per group of 128. The only change is that the levels
//! travel with the tensor instead of being a constant in the kernel.
//!
//! It is the scalar half of the GSQ idea: learn the grid, keep the format. Vector quantization is a different
//! trade and is measured separately.

use rayon::prelude::*;

/// Bits per code.
pub const CODE_BITS: usize = 2;
/// Codes per byte, LSB first.
pub const CODES_PER_BYTE: usize = 4;
/// The grid MLS2 ships, for comparison and as the starting point.
pub const E2M1: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

/// Four levels, in code order. The format stores them as f32 next to the tensor's scales.
pub type Levels = [f32; 4];

/// Learns four levels for a matrix, minimising the squared error of the values divided by their group scale.
///
/// `rows` are the rotated weights; `group` is the scale's group width. The scale is recomputed each iteration
/// the way the encoder computes it, so the levels are learned for the representation that will be stored.
pub fn learn(rows: &[f32], group: usize, iters: usize) -> Levels {
    let mut levels = E2M1;
    for _ in 0..iters {
        // Every group's scale first, so the values the levels see are already normalised.
        let scaled: Vec<f32> = rows
            .par_chunks(group)
            .flat_map_iter(|g| {
                let s = best_scale(g, &levels);
                g.iter().map(move |x| x / s)
            })
            .collect();
        // Assign each value to its nearest level, then move the level to the mean of its own values.
        let mut sum = [0f64; 4];
        let mut count = [0usize; 4];
        for &v in &scaled {
            let code = nearest(v, &levels);
            sum[code] += v as f64;
            count[code] += 1;
        }
        for i in 0..4 {
            if count[i] > 0 {
                levels[i] = (sum[i] / count[i] as f64) as f32;
            }
        }
        // Keep the grid in code order: the format reads level `i` for code `i`.
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    levels
}

/// The code of a value already divided by its scale.
pub fn nearest(v: f32, levels: &Levels) -> usize {
    let mut best = 0usize;
    let mut err = (v - levels[0]).abs();
    for (i, &level) in levels.iter().enumerate().skip(1) {
        let e = (v - level).abs();
        if e < err {
            best = i;
            err = e;
        }
    }
    best
}

/// The scale that minimises the squared error of a group at these levels: the encoder's own search, a coarse
/// sweep from the group's largest magnitude downwards.
pub fn best_scale(g: &[f32], levels: &Levels) -> f32 {
    let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return 1.0;
    }
    let mut best = (f32::INFINITY, amax);
    for step in 0..=80 {
        let s = amax / (1.0 + step as f32 * 0.025);
        let mut err = 0f32;
        for &v in g {
            let q = levels[nearest(v / s, levels)] * s;
            err += (v - q) * (v - q);
        }
        if err < best.0 {
            best = (err, s);
        }
    }
    best.1
}

/// Encodes a row: codes packed four per byte and one f32 scale per group.
pub fn encode_row(row: &[f32], levels: &Levels, group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(row.len() / CODES_PER_BYTE);
    let mut scales = Vec::with_capacity(row.len() / group);
    for g in row.chunks(group) {
        let s = best_scale(g, levels);
        scales.push(s);
        for &v in g {
            codes.push(nearest(v / s, levels) as u8);
        }
    }
    let mut packed = vec![0u8; codes.len().div_ceil(CODES_PER_BYTE)];
    for (i, &c) in codes.iter().enumerate() {
        packed[i / CODES_PER_BYTE] |= c << ((i % CODES_PER_BYTE) * CODE_BITS);
    }
    (packed, scales)
}

/// Decodes a row back to f32.
pub fn decode_row(packed: &[u8], scales: &[f32], levels: &Levels, group: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(scales.len() * group);
    for (gi, &s) in scales.iter().enumerate() {
        for i in 0..group {
            let at = gi * group + i;
            let code = ((packed[at / CODES_PER_BYTE] >> ((at % CODES_PER_BYTE) * CODE_BITS)) & 3) as usize;
            out.push(levels[code] * s);
        }
    }
    out
}

/// The mean cosine between the rows and their reconstruction, which is what the round-trip gate reports.
pub fn mean_cosine(rows: &[f32], k: usize, levels: &Levels, group: usize) -> f64 {
    let n = rows.len() / k;
    (0..n)
        .into_par_iter()
        .map(|r| {
            let row = &rows[r * k..(r + 1) * k];
            let (packed, scales) = encode_row(row, levels, group);
            let got = decode_row(&packed, &scales, levels, group);
            let mut dot = 0f64;
            let (mut a, mut b) = (0f64, 0f64);
            for (x, y) in row.iter().zip(&got) {
                dot += (*x as f64) * (*y as f64);
                a += (*x as f64).powi(2);
                b += (*y as f64).powi(2);
            }
            dot / (a.sqrt() * b.sqrt())
        })
        .sum::<f64>()
        / n as f64
}

#[cfg(test)]
mod experiment {
    use super::*;
    use crate::deepseek::fp4_rows;
    use malleus_gguf::TensorSource;
    use malleus_gguf::safetensors::SafeTensors;

    /// The same tensor the codebook experiment reads, so the two numbers are comparable.
    #[test]
    #[ignore = "reads the real checkpoint"]
    fn a_learned_grid_on_a_real_expert() {
        let root = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let st = SafeTensors::open_dir(std::path::Path::new(&root)).unwrap();
        let name = "layers.0.ffn.experts.0.w1.weight";
        let info = st.info(name).unwrap();
        let n = info.shape[0];
        let k = info.shape[1] * 2;
        let mut q = vec![0u8; info.size];
        st.read(name, 0, &mut q).unwrap();
        let mut s = vec![
            0u8;
            st.info(&format!("{}.scale", name.trim_end_matches(".weight")))
                .unwrap()
                .size
        ];
        st.read(&format!("{}.scale", name.trim_end_matches(".weight")), 0, &mut s)
            .unwrap();
        let mut w = vec![0f32; n * k];
        fp4_rows(&q, &s, n, k, &mut w).unwrap();

        // The engine rotates every row before quantizing, so the grid is learned on what the kernel sees.
        let signs = malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
        for r in 0..n {
            malleus_gguf::quant::rotate_row(&mut w[r * k..(r + 1) * k], &signs);
        }

        let fixed = mean_cosine(&w, k, &E2M1, 128);
        println!("fixed e2m1 grid: cosine {fixed:.5} at 2.125 bits/weight");
        for iters in [4usize, 8] {
            let levels = learn(&w, 128, iters);
            let got = mean_cosine(&w, k, &levels, 128);
            println!("learned grid ({iters} iters): cosine {got:.5}, levels {levels:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gaussian_grid_beats_the_fixed_one() {
        // A deterministic pseudo-Gaussian sample: a sum of uniforms is close enough and needs no dependency.
        let mut rows = Vec::with_capacity(64 * 128);
        let mut x = 12345u64;
        for _ in 0..64 * 128 {
            let mut acc = 0f32;
            for _ in 0..12 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                acc += ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            }
            rows.push(acc);
        }
        let learned = learn(&rows, 128, 8);
        let fixed = mean_cosine(&rows, 128, &E2M1, 128);
        let got = mean_cosine(&rows, 128, &learned, 128);
        println!("fixed {fixed:.5} learned {got:.5} levels {learned:?}");
        assert!(got > fixed, "the learned grid must beat the fixed one");
    }

    #[test]
    fn the_codes_are_packed_four_per_byte() {
        let row = [0.0f32, 1.0, -1.0, 2.0];
        let levels = [0.0, 1.0, -1.0, 2.0];
        let (packed, scales) = encode_row(&row, &levels, 4);
        assert_eq!(packed.len(), 1);
        assert_eq!(scales.len(), 1);
        let got = decode_row(&packed, &scales, &levels, 4);
        assert_eq!(got, row);
    }
}
