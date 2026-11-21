//! The packed formats a release uses for its routed experts, read back to f32.
//!
//! NVFP4 stores two e2m1 codes per byte — the low nibble is the even column — with one e4m3 scale per 16 columns
//! and a per-expert outer scale `amax / 2688`. The 2688 is `6 * 448`: the largest e2m1 code times the largest e4m3
//! code, so a block at full scale multiplied by the outer factor lands exactly on the amax of that expert.
//!
//! MXFP8 stores one e4m3 byte per weight with one ue8m0 scale (`2^(byte - 127)`) per 32 columns.
//!
//! Both are decoded here, not in the kernels: a checkpoint in these formats is read once, on the host, to be
//! written back as bf16 or quantised to MLS4.

/// e2m1 code values, in bit order: the sign is the high bit, so 8..15 are the same magnitudes negated.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Columns covered by one NVFP4 scale.
pub const NVFP4_BLOCK: usize = 16;
/// Columns covered by one MXFP8 scale.
pub const MXFP8_BLOCK: usize = 32;
/// What an e2m1 code and an e4m3 scale can hold together: the outer scale of an NVFP4 expert is `amax / this`.
pub const NVFP4_FULL_SCALE: f32 = 6.0 * 448.0;

/// One e4m3 (fn) byte. The only NaN is a saturated exponent with a full mantissa, as in the format the scales of a
/// release are written in.
pub fn e4m3(code: u8) -> f32 {
    let exponent = ((code & 0x7f) >> 3) as i32;
    let mantissa = (code & 7) as f32;
    if (code & 0x7f) == 0x7f {
        return f32::NAN;
    }
    let value = if exponent == 0 {
        // Subnormals: mantissa * 2^-9 (2^-6 of the smallest normal, spread over 8 steps).
        mantissa * (1.0 / 512.0)
    } else {
        (1.0 + mantissa / 8.0) * pow2(exponent - 7)
    };
    if code & 0x80 != 0 { -value } else { value }
}

/// One ue8m0 byte: a bare power of two, `2^(byte - 127)`. 255 is not a number.
pub fn ue8m0(byte: u8) -> f32 {
    pow2(byte as i32 - 127)
}

/// `2^exp` exactly, for the range the two scale formats use.
fn pow2(exp: i32) -> f32 {
    if exp < -149 {
        return 0.0;
    }
    if exp > 127 {
        return f32::INFINITY;
    }
    if exp >= -126 {
        f32::from_bits(((exp + 127) as u32) << 23)
    } else {
        // Subnormal: the mantissa carries the power.
        f32::from_bits(1 << (exp + 149))
    }
}

/// Expands `rows` packed NVFP4 rows (`cols / 2` bytes each) into `out` (`rows * cols` values), with the scales of
/// the same rows (`cols / 16` bytes each) and the outer scale of the expert.
pub fn nvfp4_rows(packed: &[u8], scales: &[u8], outer: f32, cols: usize, out: &mut [f32]) {
    let (per_row, scal_row) = (cols / 2, cols / NVFP4_BLOCK);
    for (r, row) in packed.chunks_exact(per_row).enumerate() {
        let sc = &scales[r * scal_row..(r + 1) * scal_row];
        let o = &mut out[r * cols..(r + 1) * cols];
        for (i, b) in row.iter().enumerate() {
            let scale = e4m3(sc[i / 8]) * outer;
            o[2 * i] = E2M1[(b & 15) as usize] * scale;
            o[2 * i + 1] = E2M1[(b >> 4) as usize] * scale;
        }
    }
}

/// Expands `rows` MXFP8 rows (`cols` bytes each) into `out` (`rows * cols` values), with the power-of-two scales of
/// the same rows (`cols / 32` bytes each).
pub fn mxfp8_rows(values: &[u8], scales: &[u8], cols: usize, out: &mut [f32]) {
    let scal_row = cols / MXFP8_BLOCK;
    for (r, row) in values.chunks_exact(cols).enumerate() {
        let sc = &scales[r * scal_row..(r + 1) * scal_row];
        let o = &mut out[r * cols..(r + 1) * cols];
        for (i, b) in row.iter().enumerate() {
            o[i] = e4m3(*b) * ue8m0(sc[i / MXFP8_BLOCK]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_covers_its_range() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x80), -0.0);
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x01), 1.0 / 512.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert!(e4m3(0x7f).is_nan() && e4m3(0xff).is_nan());
    }

    #[test]
    fn ue8m0_is_a_power_of_two() {
        assert_eq!(ue8m0(127), 1.0);
        assert_eq!(ue8m0(115), 1.0 / 4096.0);
        assert_eq!(ue8m0(0), f32::from_bits(1 << 22));
        assert!(ue8m0(255).is_infinite());
    }

    #[test]
    fn nvfp4_is_the_low_nibble_first() {
        // One row of 16 columns: codes 0, 1, 0xf, 0x9 in the first four columns, then zeros.
        let mut packed = [0u8; 8];
        packed[0] = 0x10; // low 0 -> 0.0, high 1 -> 0.5
        packed[1] = 0x9f; // low 0xf -> -6.0, high 9 -> -0.5
        let scales = [0x38u8]; // 1.0
        let mut out = [0.0f32; 16];
        nvfp4_rows(&packed, &scales, 1.0, 16, &mut out);
        assert_eq!(&out[..4], &[0.0, 0.5, -6.0, -0.5]);
        assert!(out[4..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn nvfp4_outer_scale_reaches_the_amax() {
        // A block whose largest code is 6 and whose scale is the e4m3 maximum lands on the amax it was given.
        let packed = [0xffu8; 8];
        let scales = [0x7eu8];
        let amax = 0.3310525f32;
        let mut out = [0.0f32; 16];
        nvfp4_rows(&packed, &scales, amax / NVFP4_FULL_SCALE, 16, &mut out);
        assert!((out[0] + amax).abs() < 1e-7 && (out[1] + amax).abs() < 1e-7);
    }

    #[test]
    fn mxfp8_applies_one_scale_per_32_columns() {
        let values = [0x38u8; 64]; // 1.0 everywhere
        let scales = [127u8, 119]; // 1.0 then 2^-8, one each per 32 columns
        let mut out = [0.0f32; 64];
        mxfp8_rows(&values, &scales, 64, &mut out);
        assert!(out[..32].iter().all(|v| *v == 1.0));
        assert!(out[32..].iter().all(|v| *v == 1.0 / 256.0));
    }
}
