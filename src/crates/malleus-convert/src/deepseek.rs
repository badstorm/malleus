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

// ---------------------------------------------------------------------------------------------------------------
// The converter: a DeepSeek-V4.1 checkpoint to a `.malleus` file.
// ---------------------------------------------------------------------------------------------------------------

use std::path::Path;

use malleus_gguf::deepseek_alloc::{Allocation, Pick, allocation_key};
use malleus_gguf::gguf::{ARCH_PREFIX, GgufWriter, Value};
use malleus_gguf::safetensors::SafeTensors;
use malleus_gguf::{DType, TensorSource};

use crate::{bf16_rows, quantize_mls2, quantize_mls4, quantize_mls8, quantize_rows_i4};

/// A tensor the file will carry, and how to produce it.
enum Out {
    /// Copied byte for byte: norms, hyper-connection coefficients, the router, the vision tower, DSpark.
    Raw {
        name: String,
        dtype: DType,
        shape: Vec<usize>,
    },
    /// A quantized weight and the scales that go with it.
    Quant {
        name: String,
        pick: Pick,
        shape: Vec<usize>,
        /// The source tensor and its companion scale tensor, when the source is blocked.
        src: String,
        scale: Option<String>,
    },
}

/// The layer order of a name, so the file reads layer by layer: embed and head first, then the layers, then the
/// vision tower and DSpark at the end.
fn order(name: &str) -> (u8, usize, String) {
    if name.starts_with("layers.") {
        let i = name.split('.').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        return (1, i, name.to_string());
    }
    if name.starts_with("vision.") || name.starts_with("aligner.") {
        return (2, 0, name.to_string());
    }
    if name.starts_with("mtp.") {
        return (3, 0, name.to_string());
    }
    (0, 0, name.to_string())
}

/// Converts a DeepSeek-V4.1 checkpoint. The precision of every tensor comes from `alloc`; a tensor the map does
/// not mention is copied as it is.
pub fn convert(model_dir: &Path, out: &Path, alloc: &Allocation, log: &dyn Fn(&str)) -> anyhow::Result<()> {
    let st = SafeTensors::open_dir(model_dir)?;
    let read = |f: &str| std::fs::read_to_string(model_dir.join(f)).ok();
    let config = read("config.json").ok_or_else(|| anyhow::anyhow!("config.json is missing"))?;

    // A slice of the layers, for a smoke test: `MALLEUS_DEEPSEEK_LAYERS=0,1` keeps only those.
    let only: Option<Vec<usize>> = std::env::var("MALLEUS_DEEPSEEK_LAYERS").ok().map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect::<Vec<usize>>()
    });
    let keep = |name: &str| match &only {
        None => true,
        Some(layers) => match name.strip_prefix("layers.").and_then(|r| r.split('.').next()) {
            Some(i) => i.parse().map(|i| layers.contains(&i)).unwrap_or(true),
            None => !name.starts_with("mtp."),
        },
    };

    // The source names, in file order. A `.scale` companion is written with its weight, never on its own.
    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.retain(|n| keep(n));
    names.sort_by_key(|n| order(n));

    let mut plan: Vec<Out> = Vec::new();
    for name in &names {
        if name.ends_with(".scale") {
            continue;
        }
        let info = st
            .info(name)
            .ok_or_else(|| anyhow::anyhow!("{name}: not in the checkpoint"))?;
        let pick = alloc.pick(&allocation_key(name));
        let companion = format!("{}.scale", name.trim_end_matches(".weight"));
        let scale = st.info(&companion).map(|_| companion.clone());
        if pick == Pick::Keep || scale.is_none() {
            plan.push(Out::Raw {
                name: name.clone(),
                dtype: info.dtype,
                shape: info.shape.clone(),
            });
            // A blocked source whose weight stays as it is keeps its scales beside it.
            if let Some(s) = scale {
                let si = st.info(&s).unwrap();
                plan.push(Out::Raw {
                    name: s,
                    dtype: si.dtype,
                    shape: si.shape.clone(),
                });
            }
            continue;
        }
        plan.push(Out::Quant {
            name: name.clone(),
            pick,
            shape: info.shape.clone(),
            src: name.clone(),
            scale,
        });
    }

    let mut w = GgufWriter::new();
    w.meta("general.architecture", Value::Str(format!("{ARCH_PREFIX}deepseek_v41")));
    w.meta(
        "general.name",
        Value::Str(
            model_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("deepseek")
                .into(),
        ),
    );
    w.meta(
        "openmalleus.layout_version",
        Value::U32(malleus_gguf::quant::LAYOUT_VERSION),
    );
    w.meta(
        "openmalleus.hadamard_seed",
        Value::U64(malleus_gguf::quant::HADAMARD_SEED),
    );
    w.meta("openmalleus.hf_config", Value::Str(config));
    if let Some(g) = read("generation_config.json") {
        w.meta("openmalleus.generation_config", Value::Str(g));
    }
    if let Some(p) = read("preprocessor_config.json") {
        w.meta("openmalleus.preprocessor_config", Value::Str(p));
    }
    if let Some(t) = read("tokenizer.json") {
        w.meta("tokenizer.huggingface.json", Value::Str(t));
    }
    if let Some(c) = read("chat_template.jinja") {
        w.meta("tokenizer.chat_template", Value::Str(c));
    }

    let mut bytes = 0usize;
    for o in &plan {
        match o {
            Out::Raw { name, dtype, shape } => {
                bytes += dtype.size_of_elems(shape.iter().product());
                w.add_tensor(name, *dtype, shape);
            }
            Out::Quant { name, pick, shape, .. } => {
                let lead = &shape[..shape.len() - 1];
                let k = shape[shape.len() - 1];
                let dims = |last: &[usize]| [lead, last].concat();
                match pick {
                    Pick::Mls2 | Pick::Mls3 | Pick::Mls4 => {
                        let dt = match pick {
                            Pick::Mls2 => DType::Mls2,
                            Pick::Mls3 => DType::Mls3,
                            _ => DType::Mls4,
                        };
                        w.add_tensor(name, dt, shape);
                        let s = format!("{name}.scales");
                        w.add_tensor(&s, DType::F16, &dims(&[k / 128]));
                        bytes += tensor_size(dt, shape) + dims(&[k / 128]).iter().product::<usize>() * 2;
                    }
                    Pick::Mls8 => {
                        w.add_tensor(name, DType::Mls8, shape);
                        let s = format!("{name}.scales");
                        w.add_tensor(&s, DType::F32, lead);
                        bytes += tensor_size(DType::Mls8, shape) + lead.iter().product::<usize>() * 4;
                    }
                    Pick::RowI4 => {
                        w.add_tensor(name, DType::RowI4, shape);
                        let s = format!("{name}.scales");
                        w.add_tensor(&s, DType::F16, lead);
                        bytes += tensor_size(DType::RowI4, shape) + lead.iter().product::<usize>() * 2;
                    }
                    Pick::Keep => unreachable!("Keep is planned as Raw"),
                }
            }
        }
    }
    log(&format!("{} output tensors, {:.2} GB", plan.len(), bytes as f64 / 1e9));

    let mut sink = w.begin(out)?;
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for o in &plan {
        match o {
            Out::Raw { name, dtype, shape } => {
                let size = dtype.size_of_elems(shape.iter().product());
                let mut buf = vec![0u8; 64 << 20];
                let mut off = 0;
                while off < size {
                    let n = buf.len().min(size - off);
                    st.read(name, off, &mut buf[..n])?;
                    sink.write(&buf[..n])?;
                    off += n;
                }
            }
            Out::Quant {
                name: _,
                pick,
                shape,
                src,
                scale,
            } => {
                let (n, k) = (
                    shape[..shape.len() - 1].iter().product::<usize>(),
                    shape[shape.len() - 1],
                );
                let info = st
                    .info(src)
                    .ok_or_else(|| anyhow::anyhow!("{src}: not in the checkpoint"))?;
                let mut w32 = vec![0f32; n * k];
                read_weight(&st, src, info.dtype, scale.as_deref(), n, k, &mut w32)?;
                match pick {
                    Pick::Mls2 => {
                        let (q, s) = quantize_mls2(&w32, n, k);
                        sink.write(&q)?;
                        sink.write(&s)?;
                    }
                    Pick::Mls3 => {
                        let (q, s) = crate::quantize_mls3(&w32, n, k);
                        sink.write(&q)?;
                        sink.write(&s)?;
                    }
                    Pick::Mls4 => {
                        let (q, s) = quantize_mls4(&w32, n, k);
                        sink.write(&q)?;
                        sink.write(&s)?;
                    }
                    Pick::Mls8 => {
                        let (q, s) = quantize_mls8(&w32, n, k);
                        sink.write(&q)?;
                        sink.write(&s)?;
                    }
                    Pick::RowI4 => {
                        let (q, s) = quantize_rows_i4(&w32, n, k);
                        sink.write(&q)?;
                        sink.write(&s)?;
                    }
                    Pick::Keep => unreachable!(),
                }
            }
        }
        done += 1;
        if done.is_multiple_of(200) || done == plan.len() {
            log(&format!("[{done}/{}] {:.0}s", plan.len(), t0.elapsed().as_secs_f64()));
        }
    }
    sink.finish()?;
    Ok(())
}

/// Reads one weight as f32, dequantizing its block format when the source has one.
fn read_weight(
    st: &SafeTensors,
    name: &str,
    dtype: DType,
    scale: Option<&str>,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> anyhow::Result<()> {
    match dtype {
        DType::BF16 => {
            let mut raw = vec![0u8; n * k * 2];
            st.read(name, 0, &mut raw)?;
            out.copy_from_slice(&bf16_rows(&raw));
        }
        DType::F32 => {
            let mut raw = vec![0u8; n * k * 4];
            st.read(name, 0, &mut raw)?;
            for (o, c) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        DType::F8E4M3 => {
            let s = scale.ok_or_else(|| anyhow::anyhow!("{name}: fp8 without a scale"))?;
            let mut q = vec![0u8; n * k];
            st.read(name, 0, &mut q)?;
            let si = st
                .info(s)
                .ok_or_else(|| anyhow::anyhow!("{s}: not in the checkpoint"))?;
            let mut sc = vec![0u8; si.shape.iter().product()];
            st.read(s, 0, &mut sc)?;
            fp8_rows(&q, &sc, n, k, out)?;
        }
        DType::I8 => {
            let s = scale.ok_or_else(|| anyhow::anyhow!("{name}: fp4 without a scale"))?;
            let mut q = vec![0u8; n * k / 2];
            st.read(name, 0, &mut q)?;
            let si = st
                .info(s)
                .ok_or_else(|| anyhow::anyhow!("{s}: not in the checkpoint"))?;
            let mut sc = vec![0u8; si.shape.iter().product()];
            st.read(s, 0, &mut sc)?;
            fp4_rows(&q, &sc, n, k, out)?;
        }
        d => anyhow::bail!("{name}: {d:?} cannot be quantized"),
    }
    Ok(())
}

fn tensor_size(dtype: DType, shape: &[usize]) -> usize {
    dtype.size_of_elems(shape.iter().product())
}

#[cfg(test)]
mod round_trip {
    use super::*;
    use malleus_gguf::gguf::Gguf;
    use malleus_gguf::quant::{Int2Grid, dequant_int2, unrotate_row};

    /// Reads a quantized tensor back from a converted file and compares it with the source checkpoint's f32.
    /// Run with the slice's file and the checkpoint:
    ///
    ///   DEEPSEEK_FILE=/golden/deepseek-slice/deepseek-v41-slice.malleus DEEPSEEK_CKPT=/models/deepseekv41-flash \
    ///   cargo test -p malleus-convert round_trip -- --ignored --nocapture
    #[test]
    #[ignore = "reads a converted slice and the checkpoint"]
    fn a_slice_round_trips_through_the_file() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let f = Gguf::open(std::path::Path::new(&file)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        println!("the file carries {} tensors", f.names().len());

        // The attention projection is MLS8 in the map: int8 codes with one f32 scale per row, rotated.
        let name = "layers.0.attn.wq_a.weight";
        let info = f.info(name).unwrap();
        let (n, k) = (info.shape[0], info.shape[1]);
        assert_eq!(info.dtype, DType::Mls8, "the map asked for Q8_0");
        let mut q = vec![0u8; n * k];
        f.read(name, 0, &mut q).unwrap();
        let mut s = vec![0u8; n * 4];
        f.read(&format!("{name}.scales"), 0, &mut s).unwrap();
        let scales: Vec<f32> = s.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let signs = malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
        let mut got = vec![0f32; n * k];
        for r in 0..n {
            let row = &mut got[r * k..(r + 1) * k];
            for (o, &b) in row.iter_mut().zip(&q[r * k..(r + 1) * k]) {
                *o = b as i8 as f32 * scales[r];
            }
            unrotate_row(row, &signs);
        }
        let want = source_f32(&src, name, n, k);
        let (cos, worst) = compare(&got, &want);
        println!("{name} (Mls8): cosine {cos:.7}, worst |diff| {worst:.6}");
        assert!(cos > 0.999, "{name} diverges: {cos}");

        // A routed expert is MLS2 in the map: four 2-bit codes per byte, one f16 scale per 128.
        let name = "layers.0.ffn.experts.0.w1.weight";
        let info = f.info(name).unwrap();
        let (n, k) = (info.shape[0], info.shape[1]);
        assert_eq!(info.dtype, DType::Mls2, "the map asked for IQ2_XXS");
        let mut q = vec![0u8; n * k / 4];
        f.read(name, 0, &mut q).unwrap();
        let mut s = vec![0u8; n * (k / 128) * 2];
        f.read(&format!("{name}.scales"), 0, &mut s).unwrap();
        let scales: Vec<f32> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect();
        let mut got = vec![0f32; n * k];
        let g = k / 128;
        for r in 0..n {
            for grp in 0..g {
                let bytes = &q[r * k / 4 + grp * 32..r * k / 4 + grp * 32 + 32];
                let mut blk = [0f32; 128];
                dequant_int2(bytes, scales[r * g + grp], Int2Grid::E2m1, &mut blk);
                got[r * k + grp * 128..r * k + grp * 128 + 128].copy_from_slice(&blk);
            }
            unrotate_row(&mut got[r * k..(r + 1) * k], &signs);
        }
        let want = source_f32(&src, name, n, k);
        let (cos, worst) = compare(&got, &want);
        println!("{name} (Mls2): cosine {cos:.7}, worst |diff| {worst:.6}");
        assert!(cos > 0.90, "{name} diverges: {cos}");
    }

    /// The checkpoint's own value of a tensor, dequantized to f32.
    fn source_f32(src: &SafeTensors, name: &str, n: usize, k: usize) -> Vec<f32> {
        let info = src.info(name).unwrap();
        let mut out = vec![0f32; n * k];
        let mut q = vec![0u8; info.size];
        src.read(name, 0, &mut q).unwrap();
        match info.dtype {
            DType::F8E4M3 => {
                let mut s = vec![0u8; n.div_ceil(32) * (k / 32)];
                src.read(&format!("{}.scale", name.trim_end_matches(".weight")), 0, &mut s)
                    .unwrap();
                fp8_rows(&q, &s, n, k, &mut out).unwrap();
            }
            DType::I8 => {
                let mut s = vec![0u8; n * (k / 32)];
                src.read(&format!("{}.scale", name.trim_end_matches(".weight")), 0, &mut s)
                    .unwrap();
                fp4_rows(&q, &s, n, k, &mut out).unwrap();
            }
            d => panic!("{name} is {d:?}, not a blocked weight"),
        }
        out
    }

    fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
        let (mut dot, mut na, mut nb, mut worst) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
            worst = worst.max((*x - *y).abs() as f64);
        }
        (dot / (na.sqrt() * nb.sqrt()), worst)
    }
}
