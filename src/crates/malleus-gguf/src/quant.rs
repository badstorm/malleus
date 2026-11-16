//! MLS4 / MLS8 weight formats (see PLAN.md §2.1). Both operate on rows rotated by a randomized
//! Hadamard transform on 128-wide blocks, with fixed signs derived from `HADAMARD_SEED`.

pub const BLOCK: usize = 128;
pub const HADAMARD_SEED: u64 = 0x4D4C_5334_0001;
pub const LAYOUT_VERSION: u32 = 1;

/// Bytes of 3-bit codes of one group: 8 codes per 3 bytes, LSB-first bit order.
pub const INT3_GROUP_BYTES: usize = BLOCK * 3 / 8;

/// MLS3 codes are the 8 levels -4..3 stored as `value + 4`.
pub const INT3_OFFSET: i32 = 4;

pub fn hadamard_signs(seed: u64) -> [i8; BLOCK] {
    let mut s = seed;
    let mut out = [1i8; BLOCK];
    for o in out.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        *o = if z & 1 == 1 { -1 } else { 1 };
    }
    out
}

/// In-place randomized Hadamard on one 128-element block: x <- H (signs * x) / sqrt(128).
pub fn rotate_block(x: &mut [f32], signs: &[i8; BLOCK]) {
    debug_assert_eq!(x.len(), BLOCK);
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s as f32;
    }
    let mut h = 1;
    while h < BLOCK {
        for i in (0..BLOCK).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let n = 1.0 / (BLOCK as f32).sqrt();
    for v in x.iter_mut() {
        *v *= n;
    }
}

pub fn rotate_row(row: &mut [f32], signs: &[i8; BLOCK]) {
    for b in row.as_chunks_mut::<BLOCK>().0 {
        rotate_block(b, signs);
    }
}

/// Inverse of `rotate_block`: y <- signs * (H y / sqrt(128)). H is symmetric and H H = 128 I, and the signs
/// square to one, so this is exactly what undoes the rotation — used to read an MLS4 tensor back as plain fp32
/// when a kernel needs the matrix itself rather than its dot products (MLA absorbs `kv_b_proj` into the query,
/// which is a product against the transpose).
pub fn unrotate_block(x: &mut [f32], signs: &[i8; BLOCK]) {
    debug_assert_eq!(x.len(), BLOCK);
    let mut h = 1;
    while h < BLOCK {
        for i in (0..BLOCK).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let n = 1.0 / (BLOCK as f32).sqrt();
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s as f32 * n;
    }
}

pub fn unrotate_row(row: &mut [f32], signs: &[i8; BLOCK]) {
    for b in row.as_chunks_mut::<BLOCK>().0 {
        unrotate_block(b, signs);
    }
}

/// Quantizes one rotated group of 128 to int4 with an error-minimizing scale. Returns (nibbles[64], scale).
pub fn quantize_group_int4(g: &[f32]) -> ([u8; BLOCK / 2], f32) {
    let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut best = (f32::INFINITY, 0f32);
    if amax > 0.0 {
        for step in 0..=40 {
            let s = amax / (7.0 + step as f32 * 0.025);
            let mut err = 0f32;
            for &v in g {
                let q = (v / s).round().clamp(-8.0, 7.0);
                let d = v - q * s;
                err += d * d;
            }
            if err < best.0 {
                best = (err, s);
            }
        }
    }
    let s = half::f16::from_f32(best.1).to_f32();
    let mut out = [0u8; BLOCK / 2];
    for (i, o) in out.iter_mut().enumerate() {
        let q = |v: f32| {
            if s > 0.0 {
                (v / s).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            }
        };
        *o = (q(g[2 * i]) as u8 & 0xF) | ((q(g[2 * i + 1]) as u8 & 0xF) << 4);
    }
    (out, s)
}

/// Quantizes one rotated group of 128 to 3-bit codes with an error-minimizing scale.
///
/// Levels are -4..3 (all eight codes used), packed LSB-first 8 per 3 bytes: code `i` lives at bit `3i`, so a
/// kernel can read the 24 bits of eight codes with one 4-byte aligned load at `3i & !3` shifted by `(3i & 3) * 8`
/// and never read past the group. Returns (48 bytes, scale).
pub fn quantize_group_int3(g: &[f32]) -> ([u8; INT3_GROUP_BYTES], f32) {
    let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut best = (f32::INFINITY, 0f32);
    if amax > 0.0 {
        // The positive end is 3, but the negative one reaches -4: search scales from amax / 3 up to amax / 6.
        for step in 0..=60 {
            let s = amax / (3.0 + step as f32 * 0.05);
            let mut err = 0f32;
            for &v in g {
                let q = (v / s).round().clamp(-4.0, 3.0);
                let d = v - q * s;
                err += d * d;
            }
            if err < best.0 {
                best = (err, s);
            }
        }
    }
    let s = half::f16::from_f32(best.1).to_f32();
    let mut out = [0u8; INT3_GROUP_BYTES];
    for (i, &v) in g.iter().enumerate() {
        let code = if s > 0.0 {
            ((v / s).round().clamp(-4.0, 3.0) as i32 + INT3_OFFSET) as u32
        } else {
            INT3_OFFSET as u32
        };
        let (b, sh) = (i * 3 / 8, i * 3 % 8);
        out[b] |= (code << sh) as u8;
        if sh > 5 {
            out[b + 1] |= (code >> (8 - sh)) as u8;
        }
    }
    (out, s)
}

/// Inverse of `quantize_group_int3`: 48 bytes and a scale into 128 values. The output is an array, not a slice,
/// so a caller cannot hand over a longer row and have the bit index run off the end of `bytes`.
pub fn dequant_int3(bytes: &[u8], s: f32, out: &mut [f32; BLOCK]) {
    debug_assert!(bytes.len() >= INT3_GROUP_BYTES);
    for (i, o) in out.iter_mut().enumerate() {
        let (b, sh) = (i * 3 / 8, i * 3 % 8);
        let mut c = (bytes[b] >> sh) as u32;
        if sh > 5 {
            c |= (bytes[b + 1] as u32) << (8 - sh);
        }
        *o = ((c & 7) as i32 - INT3_OFFSET) as f32 * s;
    }
}

/// Quantizes a full rotated row to int8 with a per-row scale. Returns the scale.
pub fn quantize_row_int8(row: &[f32], out: &mut [i8]) -> f32 {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    for (o, &v) in out.iter_mut().zip(row) {
        *o = if s > 0.0 {
            (v / s).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
    }
    s
}

pub fn dequant_int4(nib: &[u8], s: f32, out: &mut [f32]) {
    for (i, &b) in nib.iter().enumerate() {
        out[2 * i] = (((b << 4) as i8) >> 4) as f32 * s;
        out[2 * i + 1] = ((b as i8) >> 4) as f32 * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_match_calibration_script() {
        let mask = hadamard_signs(HADAMARD_SEED)
            .iter()
            .enumerate()
            .fold(0u128, |m, (i, &s)| if s < 0 { m | 1 << i } else { m });
        assert_eq!(mask, 0x98913798d17e8d905c16e345695a0124);
    }

    #[test]
    fn unrotate_undoes_rotate() {
        let signs = hadamard_signs(HADAMARD_SEED);
        let x: Vec<f32> = (0..BLOCK).map(|i| ((i * 53 % 97) as f32 - 48.0) / 11.0).collect();
        let mut y = x.clone();
        rotate_row(&mut y, &signs);
        unrotate_row(&mut y, &signs);
        let worst = x.iter().zip(&y).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-5, "round trip off by {worst}");
    }

    #[test]
    fn rotation_is_orthonormal_and_quant_roundtrips() {
        let signs = hadamard_signs(HADAMARD_SEED);
        assert!(signs.iter().any(|&s| s < 0) && signs.iter().any(|&s| s > 0));
        let w: Vec<f32> = (0..BLOCK).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let x: Vec<f32> = (0..BLOCK).map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0).collect();
        let dot: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let (mut wr, mut xr) = (w.clone(), x.clone());
        rotate_block(&mut wr, &signs);
        rotate_block(&mut xr, &signs);
        let rdot: f32 = wr.iter().zip(&xr).map(|(a, b)| a * b).sum();
        assert!((dot - rdot).abs() < 1e-3, "{dot} vs {rdot}");
        let (nib, s) = quantize_group_int4(&wr);
        let mut dq = vec![0f32; BLOCK];
        dequant_int4(&nib, s, &mut dq);
        let err: f32 = wr.iter().zip(&dq).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let norm: f32 = wr.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!(err / norm < 0.12, "relative error {}", err / norm);
    }

    /// Every one of the 128 bit positions is exercised: code i sits at bit 3i, so the shift cycles through
    /// 0, 3, 6, 1, 4, 7, 2, 5 and half of the codes straddle two bytes.
    #[test]
    fn int3_packing_roundtrips_every_level() {
        for offset in 0..8 {
            let g: Vec<f32> = (0..BLOCK).map(|i| ((i + offset) % 8) as f32 - 4.0).collect();
            let (bytes, s) = quantize_group_int3(&g);
            assert_eq!(bytes.len(), INT3_GROUP_BYTES);
            let mut dq = [0f32; BLOCK];
            dequant_int3(&bytes, s, &mut dq);
            for (i, (&a, &b)) in g.iter().zip(&dq).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6 * s.abs().max(1.0),
                    "code {i}: {a} vs {b} (scale {s})"
                );
            }
        }
    }

    #[test]
    fn int3_is_exact_on_representable_values() {
        // Scale 0.25 makes -4..3 exactly representable; the search must land on it and the roundtrip be exact.
        let g: Vec<f32> = (0..BLOCK).map(|i| ((i % 8) as f32 - 4.0) * 0.25).collect();
        let (bytes, s) = quantize_group_int3(&g);
        assert!((s - 0.25).abs() < 1e-3, "scale {s}");
        let mut dq = [0f32; BLOCK];
        dequant_int3(&bytes, s, &mut dq);
        for (a, b) in g.iter().zip(&dq) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// 3 bits must cost less memory than 4 and more error, but not catastrophically more: that trade-off is why
    /// MLS3 exists for the expert `gate`/`up` of Ling-3.0-flash.
    #[test]
    fn int3_error_stays_close_to_int4() {
        let signs = hadamard_signs(HADAMARD_SEED);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let (mut e3, mut e4) = (0f64, 0f64);
        for _ in 0..64 {
            let mut g: Vec<f32> = (0..BLOCK).map(|_| rnd()).collect();
            rotate_row(&mut g, &signs);
            let norm = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let (n3, s3) = quantize_group_int3(&g);
            let mut d3 = [0f32; BLOCK];
            dequant_int3(&n3, s3, &mut d3);
            let (n4, s4) = quantize_group_int4(&g);
            let mut d4 = [0f32; BLOCK];
            dequant_int4(&n4, s4, &mut d4);
            e3 += g
                .iter()
                .zip(&d3)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm;
            e4 += g
                .iter()
                .zip(&d4)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / norm;
        }
        let (e3, e4) = (e3 / 64.0, e4 / 64.0);
        println!("relative L2 error: MLS3 {e3:.4}  MLS4 {e4:.4}  ratio {:.2}", e3 / e4);
        assert!(e4 < e3, "int4 {e4} should beat int3 {e3}");
        assert!(e3 < 2.2 * e4, "int3 {e3} should not be far worse than int4 {e4}");
        assert!(e3 < 0.30, "int3 relative error {e3} too large");
    }
}
