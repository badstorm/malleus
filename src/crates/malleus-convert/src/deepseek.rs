//! Reading a DeepSeek-V4.1 checkpoint's weights (issue #83).
//!
//! The checkpoint stores two formats, and both are row-blocked: one scale per row and group of 32 columns.
//!
//! - the dense and attention weights are **e4m3** bytes with an **e8m0** scale per 32 columns;
//! - the routed experts are **e2m1** pairs in an `I8` tensor, with an e8m0 scale per 32 columns.
//!
//! An e8m0 scale is an exponent and nothing else: `2^(b - 127)`, so a block's scale is a power of two and the
//! mantissa carries the whole value. This module decodes both, row by row, into the f32 the quantizer eats.

use anyhow::{Result, bail};

/// The group of columns a scale covers, and the group of rows an fp8 block covers.
pub const BLOCK: usize = 32;
/// The fp8 weights are blocked 32x32; the fp4 experts are blocked one row by 32 columns.
pub const FP8_BLOCK: usize = 32;

/// An e8m0 scale byte as a number: `2^(b - 127)`. `0` is the smallest exponent the format has, `255` is NaN.
pub fn e8m0(b: u8) -> f32 {
    if b == 255 {
        return f32::NAN;
    }
    if b == 0 {
        // 2^-127 is a subnormal f32: exponent 0, the top mantissa bit set.
        return f32::from_bits(0x0040_0000);
    }
    // 2^(b - 127) as an exponent-only float: the byte *is* the biased exponent.
    f32::from_bits((b as u32) << 23)
}

/// An e4m3 byte as a number: sign, 4-bit exponent biased by 7, 3-bit mantissa. `0x7F` and `0xFF` are NaN.
pub fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) as u32, ((b >> 3) & 0x0F) as u32, (b & 0x07) as u32);
    let mag = if e == 0x0F && m == 0x07 {
        f32::NAN
    } else if e == 0 {
        // Subnormal: mantissa/8 scaled by 2^-6.
        (m as f32) / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -mag } else { mag }
}

/// An e2m1 nibble as a number: sign, 2-bit exponent, 1-bit mantissa. The eight magnitudes are fixed.
pub fn e2m1(n: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(n & 0x07) as usize];
    if n & 0x08 != 0 { -v } else { v }
}

/// Dequantizes a block-scaled fp8 tensor: `q` is `[n, k]` e4m3 bytes, `s` is `[n / 32, k / 32]` e8m0 scales,
/// one per 32x32 block. The scale tensor is small enough to keep in memory while the weight streams.
pub fn fp8_rows(q: &[u8], s: &[u8], n: usize, k: usize, out: &mut [f32]) -> Result<()> {
    check(n, k, q.len(), 1, "fp8")?;
    let groups = k / BLOCK;
    let rows = n.div_ceil(FP8_BLOCK);
    if s.len() < rows * groups {
        bail!("fp8: {} scale bytes for a [{n}, {k}] tensor", s.len());
    }
    for r in 0..n {
        let (qr, sr, or) = (r * k, (r / FP8_BLOCK) * groups, r * k);
        for g in 0..groups {
            let scale = e8m0(s[sr + g]);
            for c in 0..BLOCK {
                out[or + g * BLOCK + c] = e4m3(q[qr + g * BLOCK + c]) * scale;
            }
        }
    }
    Ok(())
}

/// Dequantizes a row-blocked fp4 tensor: `q` is `[n, k / 2]` bytes of e2m1 pairs, `s` is `[n, k / 32]` e8m0
/// scales. The low nibble of a byte is the first column, the high nibble the second.
pub fn fp4_rows(q: &[u8], s: &[u8], n: usize, k: usize, out: &mut [f32]) -> Result<()> {
    check(n, k, q.len(), 2, "fp4")?;
    if s.len() < n * (k / BLOCK) {
        bail!("fp4: {} scale bytes for a [{n}, {k}] tensor", s.len());
    }
    let groups = k / BLOCK;
    let half = k / 2;
    for r in 0..n {
        let (qr, sr, or) = (r * half, r * groups, r * k);
        for g in 0..groups {
            let scale = e8m0(s[sr + g]);
            for c in 0..BLOCK {
                let byte = q[qr + g * BLOCK / 2 + c / 2];
                let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out[or + g * BLOCK + c] = e2m1(nib) * scale;
            }
        }
    }
    Ok(())
}

fn check(n: usize, k: usize, q: usize, per: usize, what: &str) -> Result<()> {
    if !k.is_multiple_of(BLOCK) {
        bail!("{what}: K {k} is not a multiple of {BLOCK}");
    }
    if q < n * k / per {
        bail!("{what}: {q} bytes for a [{n}, {k}] tensor");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_byte_is_an_exponent() {
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(128), 2.0);
        assert_eq!(e8m0(126), 0.5);
        assert_eq!(e8m0(0), 2f32.powi(-127));
        assert!(e8m0(255).is_nan());
    }

    #[test]
    fn the_fp8_byte_decodes_like_the_format_says() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x80), -0.0);
        // 0x3F = 1.875 * 2^0, the largest of the low exponent range.
        assert_eq!(e4m3(0x3F), 1.875);
        assert_eq!(e4m3(0x40), 2.0);
        assert_eq!(e4m3(0x7E), 448.0);
        assert!(e4m3(0x7F).is_nan());
        // Subnormal: 0x01 = 1/8 * 2^-6.
        assert_eq!(e4m3(0x01), 0.125 / 64.0);
    }

    #[test]
    fn the_fp4_nibble_decodes_like_the_format_says() {
        let mags: Vec<f32> = (0..8).map(e2m1).collect();
        assert_eq!(mags, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        assert_eq!(e2m1(0x0F), -6.0);
    }

    #[test]
    fn a_block_is_dequantized_by_its_own_scale() {
        // 64 rows of 32 columns: one 32x32 block scales by 2, the next by 1/2.
        let mut q = vec![0x40u8; 64 * 32]; // 2.0 everywhere
        q[0] = 0x00; // first element zero
        let s = vec![128u8, 126];
        let mut out = vec![0f32; 64 * 32];
        fp8_rows(&q, &s, 64, 32, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 4.0);
        assert_eq!(out[31], 4.0);
        // Row 32 is the next block, so its scale is the second one.
        assert_eq!(out[32 * 32], 1.0);
    }

    #[test]
    fn the_fp4_pairs_are_low_nibble_first() {
        // 0x21 = high 2 (1.0), low 1 (0.5); scale 1.0.
        let q = vec![0x21u8; 16];
        let s = vec![127u8];
        let mut out = vec![0f32; 32];
        fp4_rows(&q, &s, 1, 32, &mut out).unwrap();
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[30], 0.5);
        assert_eq!(out[31], 1.0);
    }

    /// Dequantizes row 0 of the checkpoint's first query projection and compares it with the reference's
    /// dequantization (torch, the same tensor and scales):
    ///
    ///   DEEPSEEK_CKPT=/models/deepseekv41-flash DEEPSEEK_DEQUANT=/golden/deepseek-dequant \
    ///     cargo test -p malleus-convert deepseek::tests::the_real_tensor -- --ignored --nocapture
    #[test]
    #[ignore = "reads the real checkpoint"]
    fn the_real_tensor_dequantizes_like_the_reference() {
        use malleus_gguf::TensorSource;
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let dir = std::env::var("DEEPSEEK_DEQUANT").expect("DEEPSEEK_DEQUANT");
        let st = malleus_gguf::safetensors::SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let name = "layers.0.attn.wq_a.weight";
        let info = st.info(name).unwrap();
        let (n, k) = (info.shape[0], info.shape[1]);
        let mut q = vec![0u8; n * k];
        st.read(name, 0, &mut q).unwrap();
        let mut s = vec![0u8; (n / 32) * (k / 32)];
        st.read("layers.0.attn.wq_a.scale", 0, &mut s).unwrap();
        let mut out = vec![0f32; n * k];
        fp8_rows(&q, &s, n, k, &mut out).unwrap();
        let want: Vec<f32> = std::fs::read(format!("{dir}/wq_a_row0.f32"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(want.len(), k);
        let mut worst = 0f32;
        for (i, w) in want.iter().enumerate() {
            worst = worst.max((out[i] - w).abs());
        }
        let amax = out[..k].iter().fold(0f32, |a, v| a.max(v.abs()));
        println!("row 0: {n}x{k}, amax {amax:.6}, worst |diff| {worst:.9}");
        assert!(worst < 1e-9, "the dequantization diverges by {worst}");
    }

    /// The same check for the routed experts' format: pairs in an `I8` tensor, one e8m0 scale per row and 32
    /// columns. The reference row comes from `/tmp/dump_fp4.py`.
    #[test]
    #[ignore = "reads the real checkpoint"]
    fn the_real_expert_dequantizes_like_the_reference() {
        use malleus_gguf::TensorSource;
        use malleus_gguf::safetensors::SafeTensors;
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let dir = std::env::var("DEEPSEEK_DEQUANT").expect("DEEPSEEK_DEQUANT");
        let st = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let name = "layers.0.ffn.experts.0.w1.weight";
        let info = st.info(name).unwrap();
        let (n, half) = (info.shape[0], info.shape[1]);
        let k = half * 2;
        let mut q = vec![0u8; n * half];
        st.read(name, 0, &mut q).unwrap();
        let mut s = vec![0u8; n * (k / 32)];
        st.read("layers.0.ffn.experts.0.w1.scale", 0, &mut s).unwrap();
        let mut out = vec![0f32; n * k];
        fp4_rows(&q, &s, n, k, &mut out).unwrap();
        let want: Vec<f32> = std::fs::read(format!("{dir}/w1_row0.f32"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(want.len(), k);
        let mut worst = 0f32;
        for (i, w) in want.iter().enumerate() {
            worst = worst.max((out[i] - w).abs());
        }
        println!("row 0: {n}x{k}, worst |diff| {worst:.9}");
        assert!(worst < 1e-9, "the dequantization diverges by {worst}");
    }

    #[test]
    fn a_short_tensor_is_refused() {
        let mut out = vec![0f32; 32];
        assert!(fp8_rows(&[0; 31], &[127], 1, 32, &mut out).is_err());
        assert!(fp8_rows(&[0; 32], &[127], 1, 32, &mut out).is_ok());
    }
}
