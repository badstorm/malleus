//! A checkpoint whose routed experts are packed (NVFP4 / MXFP8) read back as a bf16 safetensors directory.
//!
//! The policy is a straight one: every packed expert tensor is dequantised, its companions (`_weight_scale`,
//! `_weight_amax`, `_weight_outer_scale_denominator`) are dropped, and every other tensor is written as bf16 with
//! its name and shape unchanged — including the row order of the packed layout, so the output is still the release
//! it came from, only in a format the rest of the toolchain reads. The aux files (config, tokenizer, chat template)
//! are copied as they are.
//!
//! The output is what a layer-by-layer comparison runs on. It carries the noise the release's own 4-bit experts
//! already had: an MLS4 conversion on top of it is quantised twice, which the measurements must say.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::DType;
use malleus_gguf::TensorSource;
use malleus_gguf::fpquant::{MXFP8_BLOCK, NVFP4_BLOCK, mxfp8_rows, nvfp4_rows};
use malleus_gguf::safetensors::SafeTensors;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Bytes of one output shard.
const SHARD_BYTES: u64 = 8 << 30;
/// Rows dequantised per step: 256 rows of a 2304-wide expert are 2.4 MB of f32.
const ROW_CHUNK: usize = 256;

pub struct Summary {
    pub tensors: usize,
    pub bytes: u64,
    pub shards: Vec<PathBuf>,
    /// Packed expert tensors dequantised, and the companions folded into them.
    pub experts: usize,
    pub dropped: usize,
    /// Smallest and largest `amax / max |w|` over the dequantised NVFP4 experts. A wrong outer scale (a missing
    /// factor, the wrong denominator) shows up here as a ratio far from 1.
    pub amax_ratio: (f32, f32),
}

/// How the experts of one tensor are packed, with the tensors that describe them.
enum Packed {
    /// The outer scale is `amax[e] / denominator`; the scales are e4m3, one per 16 columns.
    Nvfp4 { amax: String, denominator: String },
    /// The scales are ue8m0, one per 32 columns.
    Mxfp8,
}

enum Source {
    /// One tensor, widened to bf16.
    Plain { name: String, dtype: DType },
    /// `experts` x `rows` x `cols` values behind a packed tensor.
    Experts {
        w: String,
        s: String,
        packed: Packed,
        experts: usize,
        rows: usize,
        cols: usize,
    },
}

struct OutTensor {
    name: String,
    shape: Vec<usize>,
    bytes: u64,
    source: Source,
}

/// Dequantises `src` into `dst`, returning what was written.
pub fn dequantize(src: &Path, dst: &Path) -> Result<Summary> {
    let ckpt = SafeTensors::open_dir(src).with_context(|| src.display().to_string())?;
    let mut names = ckpt.names();
    names.sort_unstable();
    let mut plan: Vec<OutTensor> = Vec::new();
    let mut dropped = 0usize;
    for name in names {
        let info = ckpt.info(name).context("info")?;
        let (dtype, shape) = (info.dtype, info.shape.clone());
        if is_companion(name) {
            dropped += 1;
            continue;
        }
        if is_expert_weight(name) {
            plan.push(expert_plan(&ckpt, name, &shape, dtype)?);
            continue;
        }
        ensure!(
            matches!(dtype, DType::BF16 | DType::F16 | DType::F32),
            "{name}: {dtype:?} is neither a packed expert tensor nor bf16/f16/f32"
        );
        plan.push(OutTensor {
            name: name.to_string(),
            shape: shape.clone(),
            bytes: shape.iter().product::<usize>() as u64 * 2,
            source: Source::Plain {
                name: name.to_string(),
                dtype,
            },
        });
    }

    std::fs::create_dir_all(dst).with_context(|| dst.display().to_string())?;
    let shards = assign_shards(&plan);
    let total = shards.len();
    let mut written = Vec::with_capacity(total);
    let mut experts = 0usize;
    let mut ratio = (f32::INFINITY, 0.0f32);
    for (i, shard) in shards.iter().enumerate() {
        let path = dst.join(shard_name(i, total));
        let mut w = BufWriter::new(File::create(&path).with_context(|| path.display().to_string())?);
        write_shard(&ckpt, shard, &mut w, &mut experts, &mut ratio)?;
        w.flush()?;
        written.push(path);
    }
    if total > 1 {
        write_index(dst, &plan, &shards)?;
    }
    copy_aux(src, dst)?;
    Ok(Summary {
        tensors: plan.len(),
        bytes: plan.iter().map(|t| t.bytes).sum(),
        shards: written,
        experts,
        dropped,
        amax_ratio: ratio,
    })
}

fn is_expert_weight(name: &str) -> bool {
    name.ends_with(".w13_weight") || name.ends_with(".w2_weight")
}

/// The companions of a packed expert tensor: no output of their own, they are folded into it.
fn is_companion(name: &str) -> bool {
    name.ends_with("_weight_scale")
        || name.ends_with("_weight_amax")
        || name.ends_with("_weight_outer_scale_denominator")
}

fn shard_name(i: usize, total: usize) -> String {
    format!("model-{:05}-of-{:05}.safetensors", i + 1, total)
}

/// The plan of one packed expert tensor: its format, its logical shape, and the tensors carrying its scales.
fn expert_plan(ckpt: &SafeTensors, name: &str, shape: &[usize], dtype: DType) -> Result<OutTensor> {
    ensure!(
        shape.len() == 3,
        "{name}: a packed expert tensor is [experts, rows, cols], got {shape:?}"
    );
    let (experts, rows, last) = (shape[0], shape[1], shape[2]);
    let scale = format!("{name}_scale");
    let info = ckpt.info(&scale).with_context(|| format!("{scale} missing"))?;
    let (s_dtype, cols, block) = match dtype {
        // Two e2m1 codes per byte, one e4m3 scale per 16 columns.
        DType::U8 => (DType::F8E4M3, last * 2, NVFP4_BLOCK),
        // One e4m3 byte per weight, one ue8m0 scale per 32 columns.
        DType::F8E4M3 => (DType::U8, last, MXFP8_BLOCK),
        d => bail!("{name}: {d:?} is not a packed expert format"),
    };
    ensure!(
        info.dtype == s_dtype,
        "{scale}: expected {s_dtype:?}, got {:?}",
        info.dtype
    );
    ensure!(
        info.shape == [experts, rows, cols / block],
        "{scale}: shape {:?} does not match {experts} x {rows} x {cols} in blocks of {block}",
        info.shape
    );
    let packed = if dtype == DType::U8 {
        Packed::Nvfp4 {
            amax: one_of(ckpt, name, "_amax", DType::F32, &[experts])?,
            denominator: one_of(ckpt, name, "_outer_scale_denominator", DType::F32, &[1])?,
        }
    } else {
        Packed::Mxfp8
    };
    Ok(OutTensor {
        name: name.to_string(),
        shape: vec![experts, rows, cols],
        bytes: (experts * rows * cols) as u64 * 2,
        source: Source::Experts {
            w: name.to_string(),
            s: scale,
            packed,
            experts,
            rows,
            cols,
        },
    })
}

/// The name of one companion tensor of `name` (`<name>_scale`, `<name>_amax`, ...), checked for dtype and shape.
fn one_of(ckpt: &SafeTensors, name: &str, suffix: &str, dtype: DType, shape: &[usize]) -> Result<String> {
    let full = format!("{name}{suffix}");
    let info = ckpt.info(&full).with_context(|| format!("{full} missing"))?;
    ensure!(
        info.dtype == dtype && info.shape == shape,
        "{full}: expected {dtype:?} {shape:?}, got {:?} {:?}",
        info.dtype,
        info.shape
    );
    Ok(full)
}

fn assign_shards(plan: &[OutTensor]) -> Vec<Vec<&OutTensor>> {
    let mut shards: Vec<Vec<&OutTensor>> = Vec::new();
    let mut cur: Vec<&OutTensor> = Vec::new();
    let mut bytes = 0u64;
    for t in plan {
        if !cur.is_empty() && bytes + t.bytes > SHARD_BYTES {
            shards.push(std::mem::take(&mut cur));
            bytes = 0;
        }
        bytes += t.bytes;
        cur.push(t);
    }
    if !cur.is_empty() {
        shards.push(cur);
    }
    shards
}

fn write_shard(
    ckpt: &SafeTensors,
    tensors: &[&OutTensor],
    w: &mut impl Write,
    experts: &mut usize,
    ratio: &mut (f32, f32),
) -> Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_string(), serde_json::json!({"format": "pt"}));
    let mut offset = 0u64;
    for t in tensors {
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": t.shape,
                "data_offsets": [offset, offset + t.bytes],
            }),
        );
        offset += t.bytes;
    }
    let hb = serde_json::to_vec(&serde_json::Value::Object(header))?;
    w.write_all(&(hb.len() as u64).to_le_bytes())?;
    w.write_all(&hb)?;
    for t in tensors {
        match &t.source {
            Source::Plain { name, dtype } => emit_plain(ckpt, name, *dtype, w)?,
            Source::Experts {
                w: weight,
                s,
                packed,
                experts: ne,
                rows,
                cols,
            } => emit_experts(ckpt, weight, s, packed, *ne, *rows, *cols, w, experts, ratio)?,
        }
    }
    Ok(())
}

fn emit_plain(ckpt: &SafeTensors, name: &str, dtype: DType, w: &mut impl Write) -> Result<()> {
    let total = ckpt.info(name).context("info")?.size;
    let mut buf = vec![0u8; 1 << 20];
    let mut start = 0usize;
    while start < total {
        let n = (total - start).min(buf.len());
        ckpt.read(name, start, &mut buf[..n])?;
        match dtype {
            DType::BF16 => w.write_all(&buf[..n])?,
            DType::F16 => {
                let mut out = Vec::with_capacity(n);
                for b in buf[..n].as_chunks::<2>().0 {
                    let v = half::bf16::from_f32(half::f16::from_le_bytes(*b).to_f32());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                w.write_all(&out)?;
            }
            DType::F32 => {
                let mut out = Vec::with_capacity(n / 2);
                for b in buf[..n].as_chunks::<4>().0 {
                    let v = half::bf16::from_f32(f32::from_le_bytes(*b));
                    out.extend_from_slice(&v.to_le_bytes());
                }
                w.write_all(&out)?;
            }
            d => bail!("{name}: {d:?} cannot be written as bf16"),
        }
        start += n;
    }
    Ok(())
}

/// Dequantises one packed expert tensor, one expert and one row chunk at a time.
#[allow(clippy::too_many_arguments)]
fn emit_experts(
    ckpt: &SafeTensors,
    weight: &str,
    scale: &str,
    packed: &Packed,
    experts: usize,
    rows: usize,
    cols: usize,
    w: &mut impl Write,
    count: &mut usize,
    ratio: &mut (f32, f32),
) -> Result<()> {
    let block = match packed {
        Packed::Nvfp4 { .. } => NVFP4_BLOCK,
        Packed::Mxfp8 => MXFP8_BLOCK,
    };
    let packed_row = match packed {
        Packed::Nvfp4 { .. } => cols / 2,
        Packed::Mxfp8 => cols,
    };
    let scale_row = cols / block;
    let (denominator, amax) = match packed {
        Packed::Nvfp4 { amax, denominator } => {
            let d = ckpt.read_all(denominator)?;
            ensure!(d.len() == 4, "{denominator}: not one f32");
            let d = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
            ensure!(d > 0.0 && d.is_finite(), "{denominator}: {d} is not a positive scale");
            let amax = ckpt.read_all(amax)?;
            ensure!(amax.len() == experts * 4, "{weight}_amax: not one f32 per expert");
            (Some(d), Some(amax))
        }
        Packed::Mxfp8 => (None, None),
    };
    let mut values = vec![0f32; ROW_CHUNK * cols];
    let mut out = Vec::with_capacity(ROW_CHUNK * cols * 2);
    for e in 0..experts {
        // The outer scale multiplies the block scales; the amax it comes from is what the result must not exceed.
        let outer = match (&amax, denominator) {
            (Some(bytes), Some(d)) => {
                let a = f32::from_le_bytes([bytes[e * 4], bytes[e * 4 + 1], bytes[e * 4 + 2], bytes[e * 4 + 3]]);
                ensure!(a >= 0.0 && a.is_finite(), "{weight}_amax expert {e}: {a}");
                Some((a / d, a))
            }
            _ => None,
        };
        let mut peak = 0.0f32;
        for r0 in (0..rows).step_by(ROW_CHUNK) {
            let nr = (rows - r0).min(ROW_CHUNK);
            let mut q = vec![0u8; nr * packed_row];
            let mut s = vec![0u8; nr * scale_row];
            let row = e * rows + r0;
            ckpt.read(weight, row * packed_row, &mut q)?;
            ckpt.read(scale, row * scale_row, &mut s)?;
            let chunk = &mut values[..nr * cols];
            match outer {
                Some((o, _)) => nvfp4_rows(&q, &s, o, cols, chunk),
                None => mxfp8_rows(&q, &s, cols, chunk),
            }
            out.clear();
            for v in chunk.iter() {
                ensure!(v.is_finite(), "{weight} expert {e}: a dequantised value is not finite");
                peak = peak.max(v.abs());
                out.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
            }
            w.write_all(&out)?;
        }
        if let Some((_, a)) = outer {
            ensure!(peak > 0.0, "{weight} expert {e}: every dequantised weight is zero");
            ensure!(
                peak <= a * 1.001,
                "{weight} expert {e}: max |w| {peak} is past its amax {a}"
            );
            let r = a / peak;
            ratio.0 = ratio.0.min(r);
            ratio.1 = ratio.1.max(r);
        }
        *count += 1;
    }
    Ok(())
}

fn write_index(dst: &Path, plan: &[OutTensor], shards: &[Vec<&OutTensor>]) -> Result<()> {
    let mut weight_map = serde_json::Map::new();
    for (i, shard) in shards.iter().enumerate() {
        for t in shard {
            weight_map.insert(t.name.clone(), serde_json::Value::String(shard_name(i, shards.len())));
        }
    }
    let index = serde_json::json!({
        "metadata": {"total_size": plan.iter().map(|t| t.bytes).sum::<u64>()},
        "weight_map": weight_map,
    });
    std::fs::write(
        dst.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(())
}

/// Copies everything that is not a shard: the config, the tokenizer, the chat template.
fn copy_aux(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let (from, name) = (entry.path(), entry.file_name());
        if name == ".cache" {
            continue;
        }
        if from.is_dir() {
            copy_dir(&from, &dst.join(&name))?;
        } else if from.extension().is_none_or(|e| e != "safetensors") {
            std::fs::copy(&from, dst.join(&name))?;
        }
    }
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(name: &str, bytes: u64) -> OutTensor {
        OutTensor {
            name: name.to_string(),
            shape: vec![1],
            bytes,
            source: Source::Plain {
                name: name.to_string(),
                dtype: DType::BF16,
            },
        }
    }

    #[test]
    fn companions_are_recognised() {
        assert!(is_companion("model.llm.layers.0.mlp.experts.w13_weight_scale"));
        assert!(is_companion("model.llm.layers.0.mlp.experts.w2_weight_amax"));
        assert!(is_companion(
            "model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator"
        ));
        assert!(!is_companion("model.llm.layers.0.mlp.experts.w13_weight"));
    }

    #[test]
    fn a_shard_breaks_before_it_overflows() {
        let plan = vec![plain("a", SHARD_BYTES / 2), plain("b", SHARD_BYTES / 2), plain("c", 1)];
        let shards = assign_shards(&plan);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].len(), 2);
    }
}
