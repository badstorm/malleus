//! The DeepSeek-V4.1 text graph, on the host in f32 (issue #83).
//!
//! This is the graph the release's 510 GB forward runs, at the scale that fits a laptop: the loader reads a
//! checkpoint whose tensors are either the release's own storage (e4m3 with e8m0 32x32 scales, e2m1 pairs for the
//! routed experts) or plain bf16/f32 from `scripts/deepseek_ref.py`, and `forward` runs a prefill of one sequence.
//!
//! It exists to be compared with the reference stage by stage before any of it moves to kernels: the shapes, the
//! order of the sub-layers, the Sinkhorn split of the hyper-connections, the sliding window with the compressed KV
//! and its two-level indexer, and the router's two biases.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};

use crate::config::{Config, DeepseekConfig};
use crate::deepseek_engram::EngramState;
use crate::hostmath::{dot, linear, parallel_chunks, rmsnorm, silu};

/// A linear weight, row-major `[n, k]`, applied as `y = x · wᵀ`.
pub struct Lin {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl Lin {
    pub(crate) fn mat(&self, x: &[f32], t: usize) -> Vec<f32> {
        linear(x, t, self.k, &self.w, self.n, None)
    }
}

/// e2m1 levels of a packed nibble: sign in bit 3, the magnitude in bits 0..3.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e8m0(byte: u8) -> f32 {
    // 2^(x - 127): the exponent field alone.
    if byte == 255 {
        f32::NAN
    } else {
        f32::from_bits((byte as u32) << 23)
    }
}

fn bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
        .collect()
}

/// Reads a `[n, k]` matrix whatever the checkpoint stores it as: f32/bf16 as is, e4m3 with an e8m0 scale per
/// 32x32 block, or e2m1 pairs (an `I8` tensor of `[n, k/2]`) with one e8m0 scale per 32 values along K.
pub fn load_matrix(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Vec<f32>> {
    let p = |s: &str| format!("{}.{s}", name.strip_suffix(".weight").unwrap_or(name));
    let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
    match info.dtype {
        DType::BF16 | DType::F16 | DType::F32 => {
            ensure!(
                info.shape == [n, k],
                "{name}: shape {:?}, expected [{n}, {k}]",
                info.shape
            );
            let bytes = src.read_all(name)?;
            Ok(match info.dtype {
                DType::BF16 => bf16_f32(&bytes),
                DType::F16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32())
                    .collect(),
                _ => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            })
        }
        DType::F8E4M3 => {
            ensure!(
                info.shape == [n, k],
                "{name}: shape {:?}, expected [{n}, {k}]",
                info.shape
            );
            let q = src.read_all(name)?;
            let sc = src.read_all(&p("scale"))?;
            ensure!(
                sc.len() == n.div_ceil(32) * k.div_ceil(32),
                "{name}: scale of {} bytes for a {n}x{k} block grid",
                sc.len()
            );
            let mut w = vec![0f32; n * k];
            for r in 0..n {
                for c in 0..k {
                    let s = e8m0(sc[(r / 32) * k.div_ceil(32) + c / 32]);
                    w[r * k + c] = f8_e4m3(q[r * k + c]) * s;
                }
            }
            Ok(w)
        }
        // A routed expert: two e2m1 values per byte, one e8m0 scale per 32 values along K.
        DType::I8 | DType::U8 if info.shape == [n, k / 2] => {
            let q = src.read_all(name)?;
            let sc = src.read_all(&p("scale"))?;
            ensure!(
                sc.len() == n * (k / 32),
                "{name}: scale of {} bytes for {n}x{k}",
                sc.len()
            );
            let mut w = vec![0f32; n * k];
            for r in 0..n {
                for c in 0..k / 2 {
                    let byte = q[r * (k / 2) + c];
                    let s = e8m0(sc[r * (k / 32) + (2 * c) / 32]);
                    let lo = (byte & 0xF) as usize;
                    let hi = (byte >> 4) as usize;
                    w[r * k + 2 * c] = E2M1[lo & 7] * if lo & 8 != 0 { -1.0 } else { 1.0 } * s;
                    let s2 = e8m0(sc[r * (k / 32) + (2 * c + 1) / 32]);
                    w[r * k + 2 * c + 1] = E2M1[hi & 7] * if hi & 8 != 0 { -1.0 } else { 1.0 } * s2;
                }
            }
            Ok(w)
        }
        // Our own quantized formats. The codes are rotated with the Hadamard signs, so a row is dequantised
        // group by group and then unrotated; RowI4 is the Engram table, which is never rotated.
        DType::Mls2 | DType::Mls3 | DType::Mls4 | DType::Mls8 | DType::RowI4 => {
            ensure!(
                info.shape == [n, k],
                "{name}: shape {:?}, expected [{n}, {k}]",
                info.shape
            );
            let codes = src.read_all(name)?;
            let raw = src.read_all(&format!("{name}.scales"))?;
            let signs = malleus_gguf::quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
            let mut w = vec![0f32; n * k];
            match info.dtype {
                DType::RowI4 => {
                    let scales = f16_scales(&raw)?;
                    ensure!(scales.len() == n, "{name}: {} row scales", scales.len());
                    for r in 0..n {
                        let row = &mut w[r * k..(r + 1) * k];
                        malleus_gguf::quant::dequant_rows_i4(&codes[r * k / 2..(r + 1) * k / 2], scales[r], row);
                    }
                }
                DType::Mls8 => {
                    let scales = f32_scales(&raw)?;
                    ensure!(scales.len() == n, "{name}: {} row scales", scales.len());
                    for r in 0..n {
                        let row = &mut w[r * k..(r + 1) * k];
                        for (o, &b) in row.iter_mut().zip(&codes[r * k..(r + 1) * k]) {
                            *o = b as i8 as f32 * scales[r];
                        }
                        malleus_gguf::quant::unrotate_row(row, &signs);
                    }
                }
                DType::Mls2 | DType::Mls3 | DType::Mls4 => {
                    let scales = f16_scales(&raw)?;
                    let groups = k / malleus_gguf::quant::BLOCK;
                    ensure!(scales.len() == n * groups, "{name}: {} group scales", scales.len());
                    // Bytes one 128-wide group takes: 4 codes per byte, 3 per 8 bytes, 2 per byte.
                    let per = match info.dtype {
                        DType::Mls2 => malleus_gguf::quant::BLOCK / 4,
                        DType::Mls3 => malleus_gguf::quant::BLOCK / 8 * 3,
                        _ => malleus_gguf::quant::BLOCK / 2,
                    };
                    for r in 0..n {
                        let row = &mut w[r * k..(r + 1) * k];
                        for g in 0..groups {
                            let at = r * groups * per + g * per;
                            let mut out = [0f32; malleus_gguf::quant::BLOCK];
                            match info.dtype {
                                DType::Mls2 => malleus_gguf::quant::dequant_int2(
                                    &codes[at..at + per],
                                    scales[r * groups + g],
                                    malleus_gguf::quant::Int2Grid::E2m1,
                                    &mut out,
                                ),
                                DType::Mls3 => malleus_gguf::quant::dequant_int3(
                                    &codes[at..at + per],
                                    scales[r * groups + g],
                                    &mut out,
                                ),
                                _ => malleus_gguf::quant::dequant_int4(
                                    &codes[at..at + per],
                                    scales[r * groups + g],
                                    &mut out,
                                ),
                            }
                            row[g * malleus_gguf::quant::BLOCK..(g + 1) * malleus_gguf::quant::BLOCK]
                                .copy_from_slice(&out);
                        }
                        malleus_gguf::quant::unrotate_row(row, &signs);
                    }
                }
                _ => unreachable!(),
            }
            Ok(w)
        }
        other => bail!("{name}: cannot read a matrix stored as {other:?}"),
    }
}

/// e4m3 (bias 7, 3 mantissa bits) to f32, without a hardware cast.
fn f8_e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    if exp == 0 {
        sign * man * (1.0 / 8.0) * 2f32.powi(-6)
    } else if exp == 0xF && man == 7.0 {
        // e4m3fn has no infinity: the only NaN is the all-ones mantissa at the top exponent.
        f32::NAN
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

fn f16_scales(raw: &[u8]) -> Result<Vec<f32>> {
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect())
}

fn f32_scales(raw: &[u8]) -> Result<Vec<f32>> {
    ensure!(raw.len().is_multiple_of(4), "a scale block of {} bytes", raw.len());
    Ok(raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Reads a vector (`[n]`) of f32 or bf16.
pub(crate) fn load_vector(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>> {
    let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(info.shape == [n], "{name}: shape {:?}, expected [{n}]", info.shape);
    let bytes = src.read_all(name)?;
    Ok(match info.dtype {
        DType::BF16 => bf16_f32(&bytes),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32())
            .collect(),
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        other => bail!("{name}: cannot read a vector stored as {other:?}"),
    })
}

/// The hyper-connections of one sub-layer: one projection of the flattened streams, split into pre/post/comb.
pub struct Hc {
    pub(crate) f: Vec<f32>,
    pub(crate) base: Vec<f32>,
    pub(crate) scale: Vec<f32>,
}

struct Compressor {
    ratio: usize,
    norm: Vec<f32>,
    wkv: Lin,
    wgate: Option<Lin>,
}

struct Indexer {
    owns_k: bool,
    is_candidate: bool,
    ratio: usize,
    heads: usize,
    dim: usize,
    topk: usize,
    wq_b: Lin,
    weights_proj: Lin,
    wk: Option<Lin>,
    k_norm: Option<Vec<f32>>,
}

struct Attn {
    heads: usize,
    head_dim: usize,
    rope: usize,
    q_lora: usize,
    o_groups: usize,
    o_lora: usize,
    ratio: usize,
    wq_a: Lin,
    wq_b: Lin,
    wkv: Lin,
    wo_a: Lin,
    wo_b: Lin,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    sink: Vec<f32>,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
}

pub(crate) struct Moe {
    pub(crate) gate: Lin,
    pub(crate) bias: Vec<f32>,
    pub(crate) bias_vl: Option<Vec<f32>>,
    pub(crate) shared: (Lin, Lin, Lin),
    pub(crate) experts: Vec<(Lin, Lin, Lin)>,
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: Hc,
    hc_ffn: Hc,
    attn: Attn,
    moe: Moe,
    engram: Option<Engram>,
}

impl Layer {
    /// One block, from the file. `residents` loads the 384 routed experts too (54 GB in f32); the streamed path
    /// leaves them to `StreamedExperts`, so a block costs what its attention and its shared expert cost.
    pub fn load(src: &dyn TensorSource, c: &Config, i: usize, residents: bool) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let h = c.hidden;
        let p = |s: &str| format!("layers.{i}.{s}");
        let hc = |which: &str| -> Result<Hc> {
            Ok(Hc {
                f: load_mat(
                    src,
                    &p(&format!("{which}_fn")),
                    (2 + d.hc_mult) * d.hc_mult,
                    d.hc_mult * h,
                )?,
                base: load_vector(src, &p(&format!("{which}_base")), (2 + d.hc_mult) * d.hc_mult)?,
                scale: load_vector(src, &p(&format!("{which}_scale")), 3)?,
            })
        };
        let ratio = d.compress_ratios[i] as usize;
        let is_kv_source = d.kv_source_layers.contains(&i);
        let is_index_source = d.index_source_layers.contains(&i);
        let compressor = if is_kv_source {
            Some(Compressor {
                ratio,
                norm: load_vector(src, &p("attn.compressor.norm.weight"), d.head_dim)?,
                wkv: load_lin(src, &p("attn.compressor.wkv.weight"), d.head_dim, h)?,
                wgate: if ratio > 1 {
                    Some(load_lin(src, &p("attn.compressor.wgate.weight"), d.head_dim, h)?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let indexer = if is_index_source {
            let owns_k = is_kv_source;
            Some(Indexer {
                owns_k,
                is_candidate: d.candidate.as_ref().is_some_and(|x| x.source_layer == i),
                ratio,
                heads: d.index.heads,
                dim: d.index.dim,
                topk: d.index.topk,
                wq_b: load_lin(
                    src,
                    &p("attn.indexer.wq_b.weight"),
                    d.index.heads * d.index.dim,
                    d.q_lora,
                )?,
                weights_proj: load_lin(src, &p("attn.indexer.weights_proj.weight"), d.index.heads, h)?,
                wk: if owns_k {
                    Some(load_lin(src, &p("attn.indexer.wk.weight"), d.index.dim, d.head_dim)?)
                } else {
                    None
                },
                k_norm: if owns_k {
                    Some(load_vector(src, &p("attn.indexer.k_norm.weight"), d.index.dim)?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let attn = Attn {
            heads: d.heads,
            head_dim: d.head_dim,
            rope: d.rope,
            q_lora: d.q_lora,
            o_groups: d.o_groups,
            o_lora: d.o_lora,
            ratio,
            wq_a: load_lin(src, &p("attn.wq_a.weight"), d.q_lora, h)?,
            wq_b: load_lin(src, &p("attn.wq_b.weight"), d.heads * d.head_dim, d.q_lora)?,
            wkv: load_lin(src, &p("attn.wkv.weight"), d.head_dim, h)?,
            wo_a: load_lin(
                src,
                &p("attn.wo_a.weight"),
                d.o_groups * d.o_lora,
                d.heads * d.head_dim / d.o_groups,
            )?,
            wo_b: load_lin(src, &p("attn.wo_b.weight"), h, d.o_groups * d.o_lora)?,
            q_norm: load_vector(src, &p("attn.q_norm.weight"), d.q_lora)?,
            kv_norm: load_vector(src, &p("attn.kv_norm.weight"), d.head_dim)?,
            sink: load_vector(src, &p("attn.attn_sink"), d.heads)?,
            compressor,
            indexer,
        };
        let inter = d.moe_inter;
        let gate = Lin {
            n: d.experts,
            k: h,
            w: load_matrix(src, &p("ffn.gate.weight"), d.experts, h)?,
        };
        // The routed experts are 54 GB of one block: the streamed path leaves them to `StreamedExperts`, which
        // reads the few a position selects.
        let mut experts = Vec::new();
        if residents {
            experts.reserve(d.experts);
            for e in 0..d.experts {
                let ep = |s: &str| p(&format!("ffn.experts.{e}.{s}"));
                experts.push((
                    load_lin(src, &ep("w1.weight"), inter, h)?,
                    load_lin(src, &ep("w2.weight"), h, inter)?,
                    load_lin(src, &ep("w3.weight"), inter, h)?,
                ));
            }
        }
        let moe = Moe {
            gate,
            bias: load_vector(src, &p("ffn.gate.bias"), d.experts)?,
            bias_vl: if d.vision.is_some() {
                Some(load_vector(src, &p("ffn.gate.bias_vl"), d.experts)?)
            } else {
                None
            },
            shared: (
                load_lin(src, &p("ffn.shared_experts.w1.weight"), inter, h)?,
                load_lin(src, &p("ffn.shared_experts.w2.weight"), h, inter)?,
                load_lin(src, &p("ffn.shared_experts.w3.weight"), inter, h)?,
            ),
            experts,
        };
        Ok(Layer {
            attn_norm: load_vector(src, &p("attn_norm.weight"), h)?,
            ffn_norm: load_vector(src, &p("ffn_norm.weight"), h)?,
            hc_attn: hc("hc_attn")?,
            hc_ffn: hc("hc_ffn")?,
            attn,
            moe,
            engram: match &d.engram {
                Some(e) if e.layers.contains(&i) => Some(Engram::load(src, c, i)?),
                _ => None,
            },
        })
    }
}

/// The Engram memory of one layer, on the host: the rows a position hashes to go through `wkv` into one key per
/// residual copy plus a shared value, and the gate is a normalised dot of the stream against the key. The table
/// itself is in the companion file and is read a row at a time, so only the projections are held here.
pub struct Engram {
    wkv: Lin,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    /// Table tensor in the companion file, and the position of this layer among `engram_layer_ids` (which is the
    /// index into the hash ids, not the layer's).
    table: String,
    position: usize,
}

/// What an Engram layer needs on top of the backbone file: the hash ids, already computed, and the companion file
/// that holds the tables.
pub struct EngramInput<'a> {
    /// One `[t, (max_ngram - 1) * heads]` list per table, in `engram_layer_ids` order.
    pub hash: &'a [Vec<u64>],
    pub table: &'a dyn TensorSource,
}

impl Engram {
    /// The layer's projections, from the backbone. The table is not here: 98.31B rows stay in the companion file.
    pub fn load(src: &dyn TensorSource, c: &Config, layer: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let e = d.engram.as_ref().context("the config carries no Engram")?;
        let position = e
            .layers
            .iter()
            .position(|&l| l == layer)
            .with_context(|| format!("layer {layer} carries no Engram table"))?;
        let p = |s: &str| format!("layers.{layer}.engram.{s}");
        let cols = (e.max_ngram - 1) * e.n_heads * e.head_dim;
        Ok(Engram {
            wkv: load_lin(src, &p("wkv.weight"), c.hidden * (d.hc_mult + 1), cols)?,
            q_weight: load_matrix(src, &p("q_weight"), d.hc_mult, c.hidden)?,
            k_weight: load_matrix(src, &p("k_weight"), d.hc_mult, c.hidden)?,
            table: p("embed.weight"),
            position,
        })
    }

    /// Writes the lookup into the streams: `h` is `[t, hc, hidden]`, `hash` is `[t, (max_ngram - 1) * heads]` of
    /// table rows and `table` is the file holding them.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        table: &dyn TensorSource,
        h: &mut [f32],
        hash: &[u64],
        t: usize,
        hidden: usize,
        hc: usize,
        eps: f32,
        image_mask: Option<&[bool]>,
    ) -> Result<()> {
        ensure!(
            t > 0 && hash.len().is_multiple_of(t) && h.len() == t * hc * hidden,
            "engram: {} ids and {} values for {t} positions x {hc} copies x {hidden}",
            hash.len(),
            h.len()
        );
        let cols = hash.len() / t;
        ensure!(
            cols > 0 && self.wkv.k.is_multiple_of(cols),
            "engram: {} projected values for {cols} buckets",
            self.wkv.k
        );
        let head_dim = self.wkv.k / cols;
        let stride = (hc + 1) * hidden;
        let clamp = 1e-6f32;
        let inv_sqrt_hidden = 1.0 / (hidden as f32).sqrt();
        // A prompt can be a million positions: read and project them a block at a time, so the rows of the lookup
        // never have to exist all at once.
        const CHUNK: usize = 1024;
        let mut out = vec![0f32; h.len()];
        for (block, chunk) in hash.chunks(CHUNK * cols).enumerate() {
            let n = chunk.len() / cols;
            // The rows of those positions, one read each, then the block in one projection.
            let rows = crate::deepseek_engram::read_rows(table, &self.table, chunk, head_dim)?;
            let kv = self.wkv.mat(&rows, n);
            for b in 0..n {
                let i = block * CHUNK + b;
                let kv = &kv[b * stride..(b + 1) * stride];
                let value = &kv[hc * hidden..];
                for c in 0..hc {
                    let at = (i * hc + c) * hidden;
                    let stream = &h[at..at + hidden];
                    let key = &kv[c * hidden..(c + 1) * hidden];
                    let weight = &self.q_weight[c * hidden..(c + 1) * hidden];
                    let kweight = &self.k_weight[c * hidden..(c + 1) * hidden];
                    let (mut sh, mut sk, mut dot) = (0f32, 0f32, 0f32);
                    for j in 0..hidden {
                        sh += stream[j] * stream[j];
                        sk += key[j] * key[j];
                        dot += stream[j] * weight[j] * kweight[j] * key[j];
                    }
                    // One rstd per copy over `hidden`, not one over the copies: the reference does the same.
                    let rstd = (1.0 / (sh / hidden as f32 + eps).sqrt()) * (1.0 / (sk / hidden as f32 + eps).sqrt());
                    let d = dot * rstd * inv_sqrt_hidden;
                    // Signed sqrt before the sigmoid: the gate keeps the sign of the dot and its loudness.
                    let gate = 1.0 / (1.0 + (-d.abs().max(clamp).sqrt().copysign(d)).exp());
                    // An image span takes no part in an n-gram: its gate is shut and the stream passes through.
                    let gate = match image_mask {
                        Some(mask) if !mask.get(i).copied().unwrap_or(false) => 0.0,
                        _ => gate,
                    };
                    for j in 0..hidden {
                        out[at + j] = stream[j] + gate * value[j];
                    }
                }
            }
        }
        h.copy_from_slice(&out);
        Ok(())
    }
}

/// Everything the text graph reads, on the host in f32.
pub struct Weights {
    pub c: DeepseekConfig,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub n_layers: usize,
    embed: Vec<f32>,
    head: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    /// Backbone layers whose attention input the DSpark draft reads (`dspark_target_layer_ids`), empty without a
    /// draft. Their hidden states are what `Streamer::main_hidden` collects.
    pub target: Vec<usize>,
}

impl Weights {
    pub fn load(src: &dyn TensorSource, c: &Config, progress: &dyn Fn(usize, usize)) -> Result<Self> {
        let mut w = Weights::open(src, c)?;
        let n = w.n_layers;
        w.layers.reserve(n);
        for (i, selected) in w.selected().into_iter().enumerate() {
            w.layers.push(Layer::load(src, c, selected, true)?);
            progress(i + 1, n);
        }
        Ok(w)
    }

    /// One row of the token table, which the draft embeds its own positions with.
    pub(crate) fn embed_row(&self, id: usize) -> &[f32] {
        &self.embed[id * self.hidden..(id + 1) * self.hidden]
    }

    /// The head on an already collapsed stream, `[t, hidden]`: the draft scores its block with it, without the
    /// final hyper-connection step the served graph does.
    pub(crate) fn head_rows(&self, x: &[f32], t: usize) -> Vec<f32> {
        linear(x, t, self.hidden, &self.head, self.vocab, None)
    }

    /// What stays resident whatever the stack: the embedding, the head and the final norm. The loads are 5.3 GB in
    /// f32 — two `[vocab, hidden]` matrices — and the blocks are left to the caller.
    pub fn open(src: &dyn TensorSource, c: &Config) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?.clone();
        ensure!(
            d.compress_ratios.len() == c.n_layers,
            "compress_ratios must cover every layer"
        );
        let target = d.dspark.as_ref().map(|s| s.target_layers.clone()).unwrap_or_default();
        Ok(Weights {
            hidden: c.hidden,
            vocab: c.vocab,
            eps: d.eps,
            n_layers: c.n_layers,
            c: d,
            embed: load_matrix(src, "embed.weight", c.vocab, c.hidden)?,
            head: load_matrix(src, "head.weight", c.vocab, c.hidden)?,
            norm: load_vector(src, "norm.weight", c.hidden)?,
            layers: Vec::new(),
            target,
        })
    }

    /// The block indices to run, in order: all of them, or the slice `MALLEUS_DEEPSEEK_LAYERS` names. One layer is
    /// the stage-by-stage comparison against the reference, and the cheapest way to try the streamed path.
    pub fn selected(&self) -> Vec<usize> {
        let only: Option<Vec<usize>> = std::env::var("MALLEUS_DEEPSEEK_LAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        (0..self.n_layers)
            .filter(|i| only.as_ref().is_none_or(|l| l.contains(i)))
            .collect()
    }

    /// Runs the attention of one loaded layer on an already-normed input. This is the stage gate for the
    /// compressors, the two-level indexer and the sparse attention, which the whole-graph comparison cannot
    /// isolate: an isolated layer is fed a fixed input on both sides.
    pub fn attention_stage(&self, index: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let l = self.layers.get(index).context("no such layer loaded")?;
        let mut shared = Shared::default();
        let mut hook = |_: &str, _: &[f32]| Ok(());
        self.attention(l, x, t, &mut shared, None, &mut hook, index)
    }

    /// Parameters held on the host, for the load report.
    pub fn parameters(&self) -> usize {
        let lin = |l: &Lin| l.n * l.k;
        let engram = |e: &Engram| lin(&e.wkv) + e.q_weight.len() + e.k_weight.len();
        let mut t = self.embed.len() + self.head.len() + self.norm.len();
        for l in &self.layers {
            t += lin(&l.attn.wq_a)
                + lin(&l.attn.wq_b)
                + lin(&l.attn.wkv)
                + lin(&l.attn.wo_a)
                + lin(&l.attn.wo_b)
                + lin(&l.moe.gate)
                + (l.moe.bias.len());
            for (a, b, c) in &l.moe.experts {
                t += lin(a) + lin(b) + lin(c);
            }
            t += lin(&l.moe.shared.0) + lin(&l.moe.shared.1) + lin(&l.moe.shared.2);
            t += l.engram.as_ref().map_or(0, engram);
        }
        t
    }

    /// Bytes held on the host.
    pub fn bytes(&self) -> usize {
        let mut t = (self.embed.len() + self.head.len() + self.norm.len()) * 4;
        for l in &self.layers {
            t += layer_bytes(l);
            t += l.moe.experts.iter().map(expert_bytes).sum::<usize>();
        }
        t
    }
}

fn lin_bytes(l: &Lin) -> usize {
    l.w.len() * 4
}

/// Bytes one routed expert takes: `w1`, `w2` and `w3` in f32.
fn expert_bytes((a, b, c): &(Lin, Lin, Lin)) -> usize {
    lin_bytes(a) + lin_bytes(b) + lin_bytes(c)
}

/// Bytes one block takes without its routed experts: the attention, the router, the shared expert and the Engram
/// projections.
fn layer_bytes(l: &Layer) -> usize {
    let engram = l
        .engram
        .as_ref()
        .map_or(0, |e| lin_bytes(&e.wkv) + e.q_weight.len() * 4 + e.k_weight.len() * 4);
    lin_bytes(&l.attn.wq_a)
        + lin_bytes(&l.attn.wq_b)
        + lin_bytes(&l.attn.wkv)
        + lin_bytes(&l.attn.wo_a)
        + lin_bytes(&l.attn.wo_b)
        + lin_bytes(&l.moe.gate)
        + lin_bytes(&l.moe.shared.0)
        + lin_bytes(&l.moe.shared.1)
        + lin_bytes(&l.moe.shared.2)
        + (l.attn.q_norm.len() + l.attn.kv_norm.len() + l.attn.sink.len()) * 4
        + engram
}

pub(crate) fn load_lin(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Lin> {
    Ok(Lin {
        n,
        k,
        w: load_matrix(src, name, n, k)?,
    })
}

/// `load_matrix` for a tensor whose name has no `.weight` suffix (the hyper-connection projection).
pub(crate) fn load_mat(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Vec<f32>> {
    load_matrix(src, name, n, k)
}

// ------------------------------------------------------------------------------------------------------------------
// The graph.
// ------------------------------------------------------------------------------------------------------------------

/// What attention layers hand down the stack: the last source's compressed KV, index keys, candidate mask and top-k
/// lists. Layers run in order and a source always precedes its consumers, so one slot each is enough.
#[derive(Default)]
pub(crate) struct Shared {
    compress_kv: Vec<f32>,
    compress_len: usize,
    index_k: Vec<f32>,
    index_len: usize,
    candidates: Option<Vec<bool>>,
    candidate_width: usize,
    topk: Option<Vec<i32>>,
}

/// The hyper-connection coefficients of one sub-layer.
pub(crate) struct Mixes {
    pub(crate) pre: Vec<f32>,
    pub(crate) post: Vec<f32>,
    pub(crate) comb: Vec<f32>,
}

impl Hc {
    pub(crate) fn mixes(&self, h: &[f32], t: usize, hidden: usize, hc: usize, iters: usize, eps: f32) -> Mixes {
        let hd = hc * hidden;
        let mut flat = vec![0f32; t * hd];
        for i in 0..t {
            flat[i * hd..(i + 1) * hd].copy_from_slice(&h[i * hd..(i + 1) * hd]);
        }
        let mix_n = (2 + hc) * hc;
        // No bias here: the reference adds `hc_base` inside the split, on each coefficient's own scale.
        let mut mixes = linear(&flat, t, hd, &self.f, mix_n, None);
        // one statistic per token over the whole flattened stream
        for i in 0..t {
            let row = &flat[i * hd..(i + 1) * hd];
            let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let r = 1.0 / (mean + eps).sqrt();
            for v in mixes[i * mix_n..(i + 1) * mix_n].iter_mut() {
                *v *= r;
            }
        }
        let (mut pre, mut post, mut comb) = (vec![0f32; t * hc], vec![0f32; t * hc], vec![0f32; t * hc * hc]);
        for i in 0..t {
            let m = &mixes[i * mix_n..(i + 1) * mix_n];
            for j in 0..hc {
                pre[i * hc + j] = 1.0 / (1.0 + (-(m[j] * self.scale[0] + self.base[j])).exp()) + eps;
                post[i * hc + j] = 2.0 / (1.0 + (-(m[j + hc] * self.scale[1] + self.base[j + hc])).exp());
            }
            // comb = softmax over the row, then doubly stochastic by Sinkhorn
            let mut c = vec![0f32; hc * hc];
            for j in 0..hc {
                for k in 0..hc {
                    let idx = 2 * hc + j * hc + k;
                    c[j * hc + k] = m[idx] * self.scale[2] + self.base[idx];
                }
            }
            for j in 0..hc {
                let max = c[j * hc..(j + 1) * hc]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for k in 0..hc {
                    let v = (c[j * hc + k] - max).exp();
                    c[j * hc + k] = v;
                    sum += v;
                }
                for k in 0..hc {
                    c[j * hc + k] = c[j * hc + k] / sum + eps;
                }
            }
            for k in 0..hc {
                let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
                for j in 0..hc {
                    c[j * hc + k] /= s + eps;
                }
            }
            for _ in 0..iters - 1 {
                for j in 0..hc {
                    let s: f32 = (0..hc).map(|k| c[j * hc + k]).sum();
                    for k in 0..hc {
                        c[j * hc + k] /= s + eps;
                    }
                }
                for k in 0..hc {
                    let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
                    for j in 0..hc {
                        c[j * hc + k] /= s + eps;
                    }
                }
            }
            comb[i * hc * hc..(i + 1) * hc * hc].copy_from_slice(&c);
        }
        Mixes { pre, post, comb }
    }
}

/// Collapse the streams: `[t, hc, hidden]` weighted by `pre [t, hc]` into `[t, hidden]`.
pub(crate) fn hc_pre(h: &[f32], pre: &[f32], t: usize, hc: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * hidden];
    for i in 0..t {
        for c in 0..hc {
            let w = pre[i * hc + c];
            for d in 0..hidden {
                out[i * hidden + d] += w * h[(i * hc + c) * hidden + d];
            }
        }
    }
    out
}

/// Expand a sub-layer output back over the streams and mix the residual in: `y[e] = post[e] * x + Σ_c comb[c, e] *
/// residual[c]`, which is the reference's `post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) *
/// residual.unsqueeze(-2), dim=2)`: the surviving index of that sum is comb's *second* stream, so the output
/// stream is its column while the sum runs over its rows.
pub(crate) fn hc_post(
    x: &[f32],
    res: &[f32],
    post: &[f32],
    comb: &[f32],
    t: usize,
    hc: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; t * hc * hidden];
    for i in 0..t {
        for e in 0..hc {
            let p = post[i * hc + e];
            for d in 0..hidden {
                let mut v = p * x[i * hidden + d];
                for c in 0..hc {
                    v += comb[(i * hc + c) * hc + e] * res[(i * hc + c) * hidden + d];
                }
                out[(i * hc + e) * hidden + d] = v;
            }
        }
    }
    out
}

/// The inverse frequencies of a rotary dimension, with the YaRN blend the checkpoint was trained with.
///
/// `original` is the context the model was trained on: when it is non-zero, the dimensions whose wavelength is
/// longer than `beta_fast` turns of that context are divided by `factor`, those shorter than `beta_slow` are
/// left alone, and the band between the two fades linearly. The reference applies it to the compressed layers'
/// base only; the window layers rotate at their own base with no extrapolation.
pub(crate) fn rope_inv(
    rope: usize,
    theta: f32,
    original: usize,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    let half = rope / 2;
    let mut inv: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f32 / rope as f32))
        .collect();
    if original > 0 && factor > 1.0 {
        let (o, t) = (original as f32, theta);
        let corrected =
            |rotations: f32| rope as f32 * (o / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * t.ln());
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(rope as f32 - 1.0);
        for (j, f) in inv.iter_mut().enumerate() {
            let ramp = ((j as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    inv
}

/// RoPE in place on the last `rope` values of every row of an `[rows, row]` matrix (interleaved pairs).
pub(crate) fn rope_rows(
    x: &mut [f32],
    rows: usize,
    row: usize,
    rope: usize,
    inv: &[f32],
    pos: impl Fn(usize) -> f32,
    inverse: bool,
) {
    let half = rope / 2;
    for i in 0..rows {
        let p = pos(i);
        let base = i * row + row - rope;
        for (j, &f) in inv.iter().enumerate().take(half) {
            let angle = p * f;
            let (s, c) = (angle.sin(), angle.cos());
            let s = if inverse { -s } else { s };
            let (a, b) = (x[base + 2 * j], x[base + 2 * j + 1]);
            x[base + 2 * j] = a * c - b * s;
            x[base + 2 * j + 1] = b * c + a * s;
        }
    }
}

/// The same on the tail of every head of a `[t, heads, head_dim]` tensor: the reference rotates `x[..., -rope:]`.
fn rope_heads(x: &mut [f32], t: usize, heads: usize, head_dim: usize, rope: usize, inv: &[f32], inverse: bool) {
    rope_rows(x, t * heads, head_dim, rope, inv, |i| (i / heads) as f32, inverse);
}

/// The same for a single row at an explicit position: a decode step runs one row whose index is not its position.
pub(crate) fn rope_heads_at(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope: usize,
    inv: &[f32],
    pos: f32,
    inverse: bool,
) {
    rope_rows(x, heads, head_dim, rope, inv, |_| pos, inverse);
}

/// The sliding-window slots each query reads during a prefill of `t` tokens.
fn window_idxs(t: usize, window: usize) -> (Vec<i32>, usize) {
    let m = window.min(t);
    let mut idx = vec![-1i32; t * m];
    for i in 0..t {
        for j in 0..m {
            let at = (i as i32 - window as i32 + 1).max(0) as usize + j;
            idx[i * m + j] = if at > i { -1 } else { at as i32 };
        }
    }
    (idx, m)
}

/// Sparse attention with the learned sink: `q [t, heads, hd]`, `kv [n, hd]`, `idx [t, m]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_attn(
    q: &[f32],
    kv: &[f32],
    n_kv: usize,
    sink: &[f32],
    idx: &[i32],
    t: usize,
    m: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; t * heads * hd];
    parallel_chunks(&mut out, heads * hd, |i, o| {
        let mut scores = vec![0f32; m];
        let mut valid = vec![false; m];
        for h in 0..heads {
            let qr = &q[i * heads * hd + h * hd..][..hd];
            let mut max = f32::NEG_INFINITY;
            for j in 0..m {
                let at = idx[i * m + j];
                valid[j] = at >= 0 && (at as usize) < n_kv;
                scores[j] = if valid[j] {
                    dot(qr, &kv[at as usize * hd..][..hd], hd) * scale
                } else {
                    f32::NEG_INFINITY
                };
                max = max.max(scores[j]);
            }
            if max == f32::NEG_INFINITY {
                continue;
            }
            let mut sum = 0f32;
            for j in 0..m {
                scores[j] = if valid[j] { (scores[j] - max).exp() } else { 0.0 };
                sum += scores[j];
            }
            sum += (sink[h] - max).exp();
            let inv = 1.0 / sum;
            for j in 0..m {
                if !valid[j] {
                    continue;
                }
                let w = scores[j] * inv;
                let v = &kv[idx[i * m + j] as usize * hd..][..hd];
                for d in 0..hd {
                    o[h * hd + d] += w * v[d];
                }
            }
        }
    });
    out
}

/// Level one of the two-level top-k: keep the best blocks, pinning the partial one the query just reached.
fn select_candidate_blocks(
    scores: &[f32],
    compress_lens: &[usize],
    t: usize,
    width: usize,
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let blocks = width.div_ceil(block_size);
    let mut keep = vec![false; t * width];
    for i in 0..t {
        let mut bs = vec![f32::NEG_INFINITY; blocks];
        for (b, score) in bs.iter_mut().enumerate() {
            for k in 0..block_size {
                let at = b * block_size + k;
                if at < width {
                    *score = score.max(scores[i * width + at]);
                }
            }
        }
        let last = (compress_lens[i].max(1) - 1) / block_size;
        bs[last.min(blocks - 1)] = f32::INFINITY;
        let mut order: Vec<usize> = (0..blocks).collect();
        order.sort_by(|a, b| bs[*b].total_cmp(&bs[*a]));
        for &b in order.iter().take(topk_blocks) {
            if bs[b] == f32::NEG_INFINITY {
                continue;
            }
            for k in 0..block_size {
                let at = b * block_size + k;
                if at < width {
                    keep[i * width + at] = true;
                }
            }
        }
    }
    keep
}

impl Weights {
    /// One prefill of `ids` at `start_pos == 0`. Returns the logits of every position, `[t, vocab]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_with(ids, None, |_, _| Ok(()))
    }

    /// The same, calling `hook(name, values)` after each stage the reference dumps. A stack with an Engram layer
    /// needs `engram`: the hash ids and the companion file that holds the tables.
    pub fn forward_with(
        &self,
        ids: &[u32],
        engram: Option<&EngramInput>,
        hook: impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.forward_impl(ids, engram, hook, None)
    }

    /// The same prefill, filling `cache` with the KV of every layer: the positions after this one are read as steps.
    pub fn forward_cached(&self, ids: &[u32], engram: Option<&EngramInput>, cache: &mut Cache) -> Result<Vec<f32>> {
        self.forward_impl(ids, engram, |_, _| Ok(()), Some(cache))
    }

    fn forward_impl(
        &self,
        ids: &[u32],
        engram: Option<&EngramInput>,
        mut hook: impl FnMut(&str, &[f32]) -> Result<()>,
        mut cache: Option<&mut Cache>,
    ) -> Result<Vec<f32>> {
        let d = &self.c;
        let (h_dim, hc, t) = (self.hidden, d.hc_mult, ids.len());
        ensure!(t > 0, "an empty prompt");
        let mut emb = vec![0f32; t * h_dim];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            ensure!(id < self.vocab, "token {id} outside the vocabulary");
            emb[i * h_dim..(i + 1) * h_dim].copy_from_slice(&self.embed[id * h_dim..(id + 1) * h_dim]);
        }
        hook("embed", &emb)?;
        let mut h = vec![0f32; t * hc * h_dim];
        for i in 0..t {
            for c in 0..hc {
                h[(i * hc + c) * h_dim..(i * hc + c + 1) * h_dim].copy_from_slice(&emb[i * h_dim..(i + 1) * h_dim]);
            }
        }
        let mut pre_mix = identity_mix(t, hc);
        let mut shared = Shared::default();
        let mut main: Vec<f32> = Vec::new();
        for (layer, l) in self.layers.iter().enumerate() {
            let mut residents = ResidentExperts(&l.moe.experts);
            self.block(
                l,
                layer,
                &mut h,
                &mut pre_mix,
                t,
                &mut shared,
                &mut residents,
                engram,
                None,
                cache.as_mut().map(|c| &mut c.layers[layer]),
                &mut main,
                &mut hook,
            )?;
        }
        if let Some(c) = cache.as_mut() {
            c.positions = t;
        }
        Ok(self.head_logits(&h, &pre_mix, t))
    }

    /// The last stage: collapse the streams, normalise, project to the vocabulary. `[t, vocab]`, one row per
    /// position.
    fn head_logits(&self, h: &[f32], pre_mix: &[f32], t: usize) -> Vec<f32> {
        let (h_dim, hc) = (self.hidden, self.c.hc_mult);
        let mut x = hc_pre(h, pre_mix, t, hc, h_dim);
        rmsnorm(&mut x, h_dim, &self.norm, self.eps);
        linear(&x, t, h_dim, &self.head, self.vocab, None)
    }

    /// One block on the streams `h`: the Engram lookup, then the attention sub-layer between its hyper-connections,
    /// then the FFN. `experts` is where the routed experts come from — resident, or read out of the file.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        l: &Layer,
        layer: usize,
        h: &mut Vec<f32>,
        pre_mix: &mut Vec<f32>,
        t: usize,
        shared: &mut Shared,
        experts: &mut dyn ExpertBank,
        engram: Option<&EngramInput>,
        image_mask: Option<&[bool]>,
        cache: Option<&mut LayerCache>,
        main: &mut Vec<f32>,
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let d = &self.c;
        let (h_dim, hc) = (self.hidden, d.hc_mult);
        // The Engram lookup is written into the streams before the block's own sub-layers.
        if let Some(e) = &l.engram {
            let input =
                engram.context("this stack carries an Engram table: it needs its hash ids and its companion file")?;
            let hash = input
                .hash
                .get(e.position)
                .with_context(|| format!("engram: no hash ids for table {}", e.position))?;
            e.forward(input.table, h, hash, t, h_dim, hc, self.eps, image_mask)?;
            hook(&format!("engram{layer}"), h)?;
        }
        // The draft reads the streams this block starts from, averaged over the copies, before any sub-layer.
        collect_main(self, layer, h, t, h_dim, hc, main);
        // Attention: its mixes come from the streams, the pre-mix from the previous sub-layer.
        let am = l.hc_attn.mixes(h, t, h_dim, hc, d.hc_iters, d.hc_eps);
        let x = hc_pre(h, pre_mix, t, hc, h_dim);
        let mut xn = x.clone();
        rmsnorm(&mut xn, h_dim, &l.attn_norm, self.eps);
        let attn_out = self.attention(l, &xn, t, shared, cache, hook, layer)?;
        *h = hc_post(&attn_out, h, &am.post, &am.comb, t, hc, h_dim);
        hook(&format!("attn{layer}"), &attn_out)?;

        // FFN: its own mixes, and the pre-mix the attention just produced.
        let fm = l.hc_ffn.mixes(h, t, h_dim, hc, d.hc_iters, d.hc_eps);
        let x = hc_pre(h, &am.pre, t, hc, h_dim);
        hook(&format!("ffn_premix{layer}"), &x)?;
        let mut xn = x.clone();
        rmsnorm(&mut xn, h_dim, &l.ffn_norm, self.eps);
        let ffn_out = moe(
            &l.moe, &xn, t, h_dim, d, d.experts, d.top_k, image_mask, experts, hook, layer,
        )?;
        *h = hc_post(&ffn_out, h, &fm.post, &fm.comb, t, hc, h_dim);
        hook(&format!("ffn{layer}"), &ffn_out)?;
        *pre_mix = fm.pre;
        hook(&format!("block{layer}"), h)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: &Layer,
        x: &[f32],
        t: usize,
        shared: &mut Shared,
        cache: Option<&mut LayerCache>,
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let d = &self.c;
        let a = &l.attn;
        let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
        // The window layers rotate at the plain base, the compressed ones at the longer one, with YaRN.
        let inv = if a.ratio > 0 {
            rope_inv(
                rd,
                d.rope_params.compress_theta,
                d.rope_params.original_seq,
                d.rope_params.factor,
                d.rope_params.beta_fast,
                d.rope_params.beta_slow,
            )
        } else {
            rope_inv(rd, d.rope_params.theta, 0, 1.0, 0.0, 0.0)
        };
        let mut qr = a.wq_a.mat(x, t);
        rmsnorm(&mut qr, a.q_lora, &a.q_norm, self.eps);
        let mut q = a.wq_b.mat(&qr, t);
        rope_heads(&mut q, t, heads, hd, rd, &inv, false);
        let mut kv = a.wkv.mat(x, t);
        rmsnorm(&mut kv, hd, &a.kv_norm, self.eps);
        rope_rows(&mut kv, t, hd, rd, &inv, |i| i as f32, false);
        let (win, wm) = window_idxs(t, d.window);
        let mut kv_all = kv.clone();
        let mut n_kv = t;
        let mut idx = win.clone();
        let mut m = wm;
        if let Some(compress_len) = t.checked_div(a.ratio).filter(|_| a.ratio > 0) {
            // The indexer reads the latent *before* RoPE, so it runs before the cache is written.
            let latent = match &a.compressor {
                Some(c) => {
                    let (lat, groups) = compressor(c, x, t, hd, self.eps);
                    let cidxs = match &a.indexer {
                        Some(ix) => indexer(
                            ix,
                            x,
                            &qr,
                            Some(&lat),
                            groups,
                            t,
                            hd,
                            rd,
                            &inv,
                            d,
                            shared,
                            self.eps,
                            hook,
                            layer,
                        )?,
                        None => shared
                            .topk
                            .clone()
                            .context("an indexed layer without a published top-k list")?,
                    };
                    let mut rotated = lat;
                    rope_rows(&mut rotated, groups, hd, rd, &inv, |i| (i * a.ratio) as f32, false);
                    shared.compress_kv = rotated;
                    shared.compress_len = groups;
                    hook(&format!("ckv{layer}"), &shared.compress_kv)?;
                    Some(cidxs)
                }
                None => {
                    // Not a source: read what the source with this ratio published.
                    let cidxs = match &a.indexer {
                        Some(ix) => indexer(ix, x, &qr, None, 0, t, hd, rd, &inv, d, shared, self.eps, hook, layer)?,
                        None => shared
                            .topk
                            .clone()
                            .context("an indexed layer without a published top-k list")?,
                    };
                    Some(cidxs)
                }
            };
            ensure!(
                shared.compress_len >= compress_len,
                "layer {layer}: {compress_len} compressed positions wanted, {} made",
                shared.compress_len
            );
            kv_all.extend_from_slice(&shared.compress_kv[..compress_len * hd]);
            n_kv += compress_len;
            if let Some(cidxs) = latent {
                let cm = cidxs.len() / t;
                let mut all = vec![-1i32; t * (wm + cm)];
                for i in 0..t {
                    for j in 0..wm {
                        all[i * (wm + cm) + j] = win[i * wm + j];
                    }
                    for j in 0..cm {
                        // The indexer already publishes absolute slots into the concatenated KV: the window
                        // rows first, then the compressed ones.
                        all[i * (wm + cm) + wm + j] = cidxs[i * cm + j];
                    }
                }
                idx = all;
                m = wm + cm;
            }
            // The reference publishes the compressed ids only (the window slots are implicit).
            if let Some(cidxs) = &shared.topk {
                let as_f32: Vec<f32> = cidxs.iter().map(|v| *v as f32).collect();
                hook(&format!("topk{layer}"), &as_f32)?;
            }
        }
        // The cache takes the rows just computed: the token KV, and on the layer that owns the compressed KV its
        // latent and its index keys. A consumer's compressed rows come from the source layer's cache.
        if let Some(lc) = cache {
            lc.kv.clear();
            lc.kv.extend_from_slice(&kv_all[..t * hd]);
            lc.n = t;
            if a.compressor.is_some() {
                lc.ckv.clear();
                lc.ckv.extend_from_slice(&kv_all[t * hd..]);
                lc.g = n_kv - t;
                let keep = a.ratio.min(t);
                lc.xh.clear();
                lc.xh.extend_from_slice(&x[(t - keep) * self.hidden..]);
            }
            if a.indexer.as_ref().is_some_and(|ix| ix.owns_k) {
                lc.ik.clear();
                lc.ik.extend_from_slice(&shared.index_k);
            }
        }
        let out = sparse_attn(&q, &kv_all, n_kv, &a.sink, &idx, t, m, heads, hd);
        Ok(self.out_projection(a, out, t))
    }

    /// The grouped output projection for one row at an explicit position.
    fn out_projection_at(&self, a: &Attn, mut o: Vec<f32>, pos: usize) -> Vec<f32> {
        let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
        let inv = if a.ratio > 0 {
            rope_inv(
                rd,
                self.c.rope_params.compress_theta,
                self.c.rope_params.original_seq,
                self.c.rope_params.factor,
                self.c.rope_params.beta_fast,
                self.c.rope_params.beta_slow,
            )
        } else {
            rope_inv(rd, self.c.rope_params.theta, 0, 1.0, 0.0, 0.0)
        };
        rope_heads_at(&mut o, heads, hd, rd, &inv, pos as f32, true);
        let per_group = heads * hd / a.o_groups;
        let mut g = vec![0f32; a.o_groups * a.o_lora];
        for grp in 0..a.o_groups {
            let src = &o[grp * per_group..][..per_group];
            let w = &a.wo_a.w[grp * a.o_lora * per_group..][..a.o_lora * per_group];
            let out = &mut g[grp * a.o_lora..][..a.o_lora];
            for r in 0..a.o_lora {
                out[r] = dot(src, &w[r * per_group..][..per_group], per_group);
            }
        }
        a.wo_b.mat(&g, 1)
    }

    /// The grouped output projection: `wo_a` is block-diagonal over the head groups, then `wo_b`.
    fn out_projection(&self, a: &Attn, mut o: Vec<f32>, t: usize) -> Vec<f32> {
        let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
        let inv = if a.ratio > 0 {
            rope_inv(
                rd,
                self.c.rope_params.compress_theta,
                self.c.rope_params.original_seq,
                self.c.rope_params.factor,
                self.c.rope_params.beta_fast,
                self.c.rope_params.beta_slow,
            )
        } else {
            rope_inv(rd, self.c.rope_params.theta, 0, 1.0, 0.0, 0.0)
        };
        rope_heads(&mut o, t, heads, hd, rd, &inv, true);
        let per_group = heads * hd / a.o_groups;
        let mut g = vec![0f32; t * a.o_groups * a.o_lora];
        for i in 0..t {
            for grp in 0..a.o_groups {
                let src = &o[i * heads * hd + grp * per_group..][..per_group];
                let w = &a.wo_a.w[grp * a.o_lora * per_group..][..a.o_lora * per_group];
                let out = &mut g[(i * a.o_groups + grp) * a.o_lora..][..a.o_lora];
                for r in 0..a.o_lora {
                    out[r] = dot(src, &w[r * per_group..][..per_group], per_group);
                }
            }
        }
        a.wo_b.mat(&g, t)
    }
}

/// A tensor source the streamer owns: the backbone (and the Engram table) are read for the whole life of a served
/// model, so nothing hangs them off the loader that opened them. `Send` because the model moves to the engine
/// thread, `Sync` because a served run counts the bytes read through a handle shared with the log.
pub type Source = Box<dyn TensorSource + Send + Sync>;

/// What the draft did over a generation: how many passes it ran, how many tokens it proposed and how many of its
/// first proposals the target itself then chose. That number is the acceptance rate a verify would see on the first
/// row, before any speculative loop exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub passes: usize,
    pub proposed: usize,
    pub matched: usize,
}

/// An image span in the prompt: `len` positions starting at `at` whose embeddings come from the vision tower
/// instead of the token table, and whose positions are the ones the Engram gate shuts, the Engram lookback stops at,
/// and the router steers with its VL bias.
#[derive(Clone, Debug)]
pub struct Span {
    /// First position of the span in the prompt.
    pub at: usize,
    /// `[len, hidden]`: the delimiters and the aligner rows in the order `deepseek_vision::span_embeddings`
    /// produces them.
    pub rows: Vec<f32>,
}

impl Span {
    /// Positions the span covers.
    pub fn len(&self, hidden: usize) -> usize {
        self.rows.len() / hidden
    }

    /// Whether the span covers nothing (an empty image is not one, but a caller may pass `None` rows).
    pub fn is_empty(&self, hidden: usize) -> bool {
        self.len(hidden) == 0
    }
}

/// The backbone read a block at a time: its 162.45 GB do not fit, so a block is built from the file, run on the
/// streams and dropped. What stays resident is the embedding, the head, the final norm and one block's non-expert
/// weights — ~6 GB — while the routed experts are read when a position selects them.
pub struct Streamer {
    config: Config,
    graph: Weights,
    src: Source,
    table: Option<Source>,
    state: Option<EngramState>,
    /// Routed experts held at once per block, before the cache is emptied.
    hold: usize,
    /// Experts read from the file over the whole run.
    pub experts_read: usize,
    /// The attention input of the DSpark target layers, concatenated, `[t, target_layers * hidden]`: what a draft
    /// reads. Filled by the last prefill or step, empty when the config carries no draft.
    pub main_hidden: Vec<f32>,
    /// The DSpark draft, when one is attached (`with_draft`).
    draft: Option<crate::deepseek_dspark::Dspark>,
    draft_stats: DraftStats,
    /// The non-expert weights of every block, held resident for decode: 40 blocks of attention and shared experts
    /// are ~26 GB in f32 against 162 GB of backbone, and a step reads them once per position instead of once per
    /// prefill.
    blocks: Option<Vec<Layer>>,
}

impl Streamer {
    /// Takes the backbone as its own: `Model` holds the streamer across requests, so the file cannot be borrowed
    /// from the caller's frame any more.
    pub fn open(src: Source, c: &Config) -> Result<Self> {
        Ok(Streamer {
            graph: Weights::open(src.as_ref(), c)?,
            config: c.clone(),
            src,
            table: None,
            state: None,
            hold: 8,
            experts_read: 0,
            main_hidden: Vec::new(),
            draft: None,
            draft_stats: DraftStats::default(),
            blocks: None,
        })
    }

    /// Holds every block's non-expert weights resident, so a decode step does not read the attention back from the
    /// file. The routed experts stay streamed.
    pub fn with_blocks(mut self) -> Result<Self> {
        let mut blocks = Vec::with_capacity(self.config.n_layers);
        for i in 0..self.config.n_layers {
            blocks.push(Layer::load(self.src.as_ref(), &self.config, i, false)?);
        }
        self.blocks = Some(blocks);
        Ok(self)
    }

    /// A decode cache for this model: the per-layer KV and the experts held across positions.
    pub fn cache(&self, hold: usize) -> Cache {
        Cache::new(&self.config, hold)
    }

    /// One position: the logits of the next token against the cache. `ids` is the whole sequence so far, because the
    /// Engram hashes read the n-grams behind the new position.
    pub fn step(&mut self, cache: &mut Cache, ids: &[u32]) -> Result<Vec<f32>> {
        let t = ids.len();
        ensure!(t > 0, "an empty sequence");
        ensure!(
            cache.len() + 1 == t,
            "the cache holds {} positions, the sequence {t}",
            cache.len()
        );
        let d = self.config.deepseek.clone().context("no deepseek config")?;
        let h_dim = self.graph.hidden;
        let hc = d.hc_mult;
        let mut h = vec![0f32; hc * h_dim];
        let last = *ids.last().unwrap() as usize;
        ensure!(last < self.graph.vocab, "token {last} outside the vocabulary");
        for c in 0..hc {
            h[c * h_dim..(c + 1) * h_dim].copy_from_slice(&self.graph.embed[last * h_dim..(last + 1) * h_dim]);
        }
        let mut pre_mix = identity_mix(1, hc);
        let mut shared = Shared::default();
        let mut main: Vec<f32> = Vec::new();
        // A position is text unless the prefill said otherwise: the mask is what keeps the n-grams of this token
        // from reaching back into an image span.
        ensure!(
            cache.text.len() + 1 == t,
            "the mask holds {} positions, the sequence {t}",
            cache.text.len()
        );
        cache.text.push(true);
        let hash: Vec<Vec<u64>> = match (&self.table, &self.state, &d.engram) {
            (Some(_), Some(state), Some(e)) => {
                // The hash row is `(max_ngram - 1) * heads` wide, the geometry the state carries; the layer's
                // `head_dim` is the table row's width, not the hash's.
                let cols = (state.max_ngram - 1) * state.heads;
                (0..e.layers.len())
                    .map(|i| {
                        let all = state.hash_ids_masked(ids, i, &cache.text);
                        all[all.len() - cols..].to_vec()
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        let engram = match (&self.table, hash.is_empty()) {
            (Some(table), false) => Some(EngramInput {
                hash: &hash,
                table: table.as_ref(),
            }),
            _ => None,
        };
        for layer in self.graph.selected() {
            let owned;
            let l: &Layer = match &self.blocks {
                Some(b) => b
                    .get(layer)
                    .with_context(|| format!("block {layer} outside the stack"))?,
                None => {
                    owned = Layer::load(self.src.as_ref(), &self.config, layer, false)?;
                    &owned
                }
            };
            let source = cache.source_of[layer];
            let (lc, src_lc) = if source == usize::MAX || source == layer {
                (&mut cache.layers[layer], None)
            } else {
                let (before, after) = cache.layers.split_at_mut(layer);
                (&mut after[0], Some(&before[source]))
            };
            let mut bank = HeldExperts {
                src: self.src.as_ref(),
                layer,
                inter: d.moe_inter,
                hidden: h_dim,
                hold: cache.hold,
                tick: &mut cache.tick,
                read: &mut cache.experts_read,
                held: &mut cache.experts[layer],
            };
            block_step(
                &self.graph,
                l,
                layer,
                lc,
                src_lc,
                &mut h,
                &mut pre_mix,
                t - 1,
                t,
                &mut shared,
                &mut bank,
                engram.as_ref(),
                None,
                &mut main,
            )?;
        }
        self.main_hidden = main;
        cache.positions += 1;
        Ok(self.graph.head_logits(&h, &pre_mix, 1))
    }

    /// The DSpark draft: when one is given, a cached generation also drafts after every committed token, so the
    /// proposals can be scored against what the target itself chose.
    pub fn with_draft(mut self, draft: crate::deepseek_dspark::Dspark) -> Self {
        self.draft = Some(draft);
        self
    }

    /// What the draft did over the last cached generation.
    pub fn draft_stats(&self) -> DraftStats {
        self.draft_stats
    }

    /// The Engram tables and the addressing they are hashed with: an Engram layer without them cannot run.
    pub fn with_engram(mut self, table: Source, state: EngramState) -> Self {
        self.table = Some(table);
        self.state = Some(state);
        self
    }

    /// How many experts of one block to hold at once. A position uses six in a row, so a handful is enough; a large
    /// prompt selects hundreds and pays a dequant per use instead of 54 GB of memory.
    pub fn with_experts_held(mut self, hold: usize) -> Self {
        self.hold = hold.max(1);
        self
    }

    /// The weights the streamer reads through, and the file it reads them from: the gates that compare a stage
    /// against the reference open both (the DSpark draft's experts come from the same file).
    #[cfg(test)]
    pub(crate) fn graph(&self) -> &Weights {
        &self.graph
    }

    #[cfg(test)]
    pub(crate) fn source(&self) -> &dyn TensorSource {
        self.src.as_ref()
    }

    /// Bytes the run keeps on the host, minus the experts it is holding: the embedding, the head and the norm,
    /// plus every block's non-expert weights when they are held resident.
    pub fn resident_bytes(&self) -> usize {
        self.graph.bytes()
            + self
                .blocks
                .as_ref()
                .map_or(0, |b| b.iter().map(layer_bytes).sum::<usize>())
    }

    /// One prefill of `ids`: the logits of every position, `[t, vocab]`.
    pub fn logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.logits_with(ids, |_, _| {})
    }

    /// The same, calling `on_layer(layer, seconds)` after each block: a pass over the streamed backbone takes
    /// minutes and silence is indistinguishable from a hang.
    pub fn logits_with(&mut self, ids: &[u32], on_layer: impl FnMut(usize, f64)) -> Result<Vec<f32>> {
        let mut hook = |_: &str, _: &[f32]| Ok(());
        self.run_with(ids, on_layer, &mut hook)
    }

    /// The same, calling `hook(name, values)` after the stages the reference dumps.
    pub fn run_with(
        &mut self,
        ids: &[u32],
        on_layer: impl FnMut(usize, f64),
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.run_with_span(ids, None, on_layer, hook)
    }

    /// The same, with one image span whose rows replace the embeddings of its positions.
    pub fn run_with_span(
        &mut self,
        ids: &[u32],
        span: Option<&Span>,
        mut on_layer: impl FnMut(usize, f64),
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let mut h = self.embed_streams(ids)?;
        let mask = splice(&mut h, ids.len(), span, self.graph.hidden, self.graph.c.hc_mult)?;
        self.stack(&mut h, ids.len(), Some(ids), mask.as_deref(), &mut on_layer, hook)
    }

    /// The stack from a stream somebody else handed over — `x` is `[t, hc, hidden]` — with `ids` for the Engram
    /// layers when the window carries one. This is what a two-block comparison against the reference needs.
    #[allow(clippy::too_many_arguments)]
    pub fn run_from(
        &mut self,
        x: &[f32],
        t: usize,
        ids: Option<&[u32]>,
        mut on_layer: impl FnMut(usize, f64),
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        ensure!(
            h.len() == t * self.graph.c.hc_mult * self.graph.hidden,
            "engram: {} values for {t} positions",
            h.len()
        );
        self.stack(&mut h, t, ids, None, &mut on_layer, hook)
    }

    /// The embedding, expanded to the `hc_mult` copies the streams are carried in.
    fn embed_streams(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (h_dim, hc) = (self.graph.hidden, self.graph.c.hc_mult);
        let t = ids.len();
        ensure!(t > 0, "an empty prompt");
        let mut h = vec![0f32; t * hc * h_dim];
        for (i, &id) in ids.iter().enumerate() {
            ensure!((id as usize) < self.graph.vocab, "token {id} outside the vocabulary");
            let row = &self.graph.embed[id as usize * h_dim..(id as usize + 1) * h_dim];
            for c in 0..hc {
                h[(i * hc + c) * h_dim..(i * hc + c + 1) * h_dim].copy_from_slice(row);
            }
        }
        Ok(h)
    }

    /// Every selected block on the streams `h`, then the last stage.
    #[allow(clippy::too_many_arguments)]
    fn stack(
        &mut self,
        h: &mut Vec<f32>,
        t: usize,
        ids: Option<&[u32]>,
        mask: Option<&[bool]>,
        on_layer: &mut impl FnMut(usize, f64),
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let d = self.graph.c.clone();
        let hc = d.hc_mult;
        // The hash ids of the whole prompt, one list per table: they do not depend on the block that reads them.
        // Inside an image span a position is dead, so no n-gram reaches into it.
        let text: Option<Vec<bool>> = mask.map(|m| m.iter().map(|inside| !inside).collect());
        let hash: Vec<Vec<u64>> = match (&self.state, &d.engram, ids) {
            (Some(state), Some(e), Some(ids)) => (0..e.layers.len())
                .map(|k| match &text {
                    Some(all) => state.hash_ids_masked(ids, k, all),
                    None => state.hash_ids(ids, k),
                })
                .collect(),
            _ => Vec::new(),
        };
        let engram = match (&self.table, hash.is_empty()) {
            (Some(table), false) => Some(EngramInput {
                hash: &hash,
                table: table.as_ref(),
            }),
            _ => None,
        };
        let mut pre_mix = identity_mix(t, hc);
        let mut shared = Shared::default();
        let mut main: Vec<f32> = Vec::new();
        for layer in self.graph.selected() {
            let begin = std::time::Instant::now();
            let l = Layer::load(self.src.as_ref(), &self.config, layer, false)?;
            let read;
            {
                let mut experts = StreamedExperts::new(self.src.as_ref(), &self.config, layer, self.hold);
                self.graph.block(
                    &l,
                    layer,
                    h,
                    &mut pre_mix,
                    t,
                    &mut shared,
                    &mut experts,
                    engram.as_ref(),
                    mask,
                    None,
                    &mut main,
                    hook,
                )?;
                read = experts.read;
            }
            self.experts_read += read;
            on_layer(layer, begin.elapsed().as_secs_f64());
        }
        self.main_hidden = main;
        Ok(self.graph.head_logits(h, &pre_mix, t))
    }

    /// Greedy generation. The host graph has no KV cache yet, so every token is a full pass over the prefix: this is
    /// the slowest sampler there is, and it is here to see the model talk, not to time decode.
    pub fn generate(
        &mut self,
        ids: &[u32],
        tokens: usize,
        mut on_token: impl FnMut(usize, &[u32], &[f32]),
    ) -> Result<Vec<u32>> {
        let mut ids = ids.to_vec();
        let vocab = self.graph.vocab;
        let mut out = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let logits = self.logits(&ids)?;
            let row = &logits[(ids.len() - 1) * vocab..];
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            let next = best as u32;
            ids.push(next);
            out.push(next);
            on_token(step, &out, row);
        }
        Ok(out)
    }

    /// The prefill into the cache: one pass over the prompt, keeping the KV of every layer. The positions after it
    /// are `step`s, so the prompt costs one pass and each token one position.
    pub fn prefill_cached(&mut self, cache: &mut Cache, ids: &[u32]) -> Result<Vec<f32>> {
        self.prefill_cached_span(cache, ids, None)
    }

    /// The same, with one image span: its rows replace the embeddings of its positions, and its positions take no
    /// part in an n-gram and no Engram contribution.
    pub fn prefill_cached_span(&mut self, cache: &mut Cache, ids: &[u32], span: Option<&Span>) -> Result<Vec<f32>> {
        let d = self.graph.c.clone();
        let (h_dim, hc, t) = (self.graph.hidden, d.hc_mult, ids.len());
        ensure!(t > 0, "an empty prompt");
        let mut h = vec![0f32; t * hc * h_dim];
        for (i, &id) in ids.iter().enumerate() {
            ensure!((id as usize) < self.graph.vocab, "token {id} outside the vocabulary");
            let row = &self.graph.embed[id as usize * h_dim..(id as usize + 1) * h_dim];
            for c in 0..hc {
                h[(i * hc + c) * h_dim..(i * hc + c + 1) * h_dim].copy_from_slice(row);
            }
        }
        let mask = splice(&mut h, t, span, h_dim, hc)?;
        cache.text = match &mask {
            Some(m) => m.iter().map(|inside| !inside).collect(),
            None => vec![true; t],
        };
        let mut pre_mix = identity_mix(t, hc);
        let mut shared = Shared::default();
        let mut main: Vec<f32> = Vec::new();
        let hash: Vec<Vec<u64>> = match (&self.state, &d.engram) {
            (Some(state), Some(e)) => (0..e.layers.len())
                .map(|k| state.hash_ids_masked(ids, k, &cache.text))
                .collect(),
            _ => Vec::new(),
        };
        let engram = match (&self.table, hash.is_empty()) {
            (Some(table), false) => Some(EngramInput {
                hash: &hash,
                table: table.as_ref(),
            }),
            _ => None,
        };
        for layer in self.graph.selected() {
            let owned;
            let l: &Layer = match &self.blocks {
                Some(b) => b
                    .get(layer)
                    .with_context(|| format!("block {layer} outside the stack"))?,
                None => {
                    owned = Layer::load(self.src.as_ref(), &self.config, layer, false)?;
                    &owned
                }
            };
            let lc = &mut cache.layers[layer];
            let mut bank = HeldExperts {
                src: self.src.as_ref(),
                layer,
                inter: d.moe_inter,
                hidden: h_dim,
                hold: cache.hold,
                tick: &mut cache.tick,
                read: &mut cache.experts_read,
                held: &mut cache.experts[layer],
            };
            let mut hook = |_: &str, _: &[f32]| Ok(());
            self.graph.block(
                l,
                layer,
                &mut h,
                &mut pre_mix,
                t,
                &mut shared,
                &mut bank,
                engram.as_ref(),
                mask.as_deref(),
                Some(lc),
                &mut main,
                &mut hook,
            )?;
        }
        self.main_hidden = main;
        cache.positions = t;
        Ok(self.graph.head_logits(&h, &pre_mix, t))
    }

    /// Greedy generation against the cache: the prompt in one pass, then a step per token. This is what the served
    /// path runs; the full-prefill `generate` stays as the reference it is compared against.
    pub fn generate_cached(
        &mut self,
        cache: &mut Cache,
        ids: &[u32],
        tokens: usize,
        on_token: impl FnMut(usize, &[u32], &[f32]),
    ) -> Result<Vec<u32>> {
        self.generate_cached_span(cache, ids, tokens, None, on_token)
    }

    /// The same, with one image span in the prompt.
    pub fn generate_cached_span(
        &mut self,
        cache: &mut Cache,
        ids: &[u32],
        tokens: usize,
        span: Option<&Span>,
        mut on_token: impl FnMut(usize, &[u32], &[f32]),
    ) -> Result<Vec<u32>> {
        let vocab = self.graph.vocab;
        let mut ids = ids.to_vec();
        let logits = self.prefill_cached_span(cache, &ids, span)?;
        let mut row = logits[(ids.len() - 1) * vocab..].to_vec();
        self.draft_stats = DraftStats::default();
        // The draft, if there is one, first takes the prompt's hidden into its window, then drafts from the last
        // position of it.
        let main = self.main_hidden.clone();
        if let Some(d) = self.draft.as_mut() {
            d.seed(&self.graph, &main, ids.len())?;
        }
        let mut pending = self.draft_from(&ids, ids.len() - 1, Some(&main))?;
        let mut out = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            let next = best as u32;
            if let Some(p) = &pending {
                self.draft_stats.proposed += 1;
                self.draft_stats.matched += usize::from(p.ids.first() == Some(&next));
            }
            ids.push(next);
            out.push(next);
            on_token(step, &out, &row);
            if step + 1 < tokens {
                row = self.step(cache, &ids)?;
                pending = self.draft_from(&ids, ids.len() - 1, None)?;
            }
        }
        Ok(out)
    }

    /// One draft pass from the last committed token, when a draft is attached: `main` is the hidden of that
    /// position, `[layers * hidden]`, or `None` to take the last row of the step that just ran.
    fn draft_from(
        &mut self,
        ids: &[u32],
        pos: usize,
        main: Option<&[f32]>,
    ) -> Result<Option<crate::deepseek_dspark::Proposal>> {
        let Some(draft) = self.draft.as_mut() else {
            return Ok(None);
        };
        self.draft_stats.passes += 1;
        let last = *ids.get(pos).context("the draft wants the token it drafts from")?;
        let row = match main {
            Some(all) => {
                let layers = self.graph.target.len().max(1);
                all[(pos * layers) * self.graph.hidden..].to_vec()
            }
            None => self.main_hidden.clone(),
        };
        Ok(Some(draft.propose(self.src.as_ref(), &self.graph, &row, last, pos)?))
    }
}

// ------------------------------------------------------------------------------------------------------------------
// The decode cache: one position at a time over the same graph.
// ------------------------------------------------------------------------------------------------------------------

/// What one layer keeps between two positions: the KV rows it has already written, the latent it pooled them into
/// (only on the layer that owns the compressed KV), and the index keys that go with them.
#[derive(Default)]
struct LayerCache {
    /// Token KV rows, `[n, hd]`, rotated.
    kv: Vec<f32>,
    n: usize,
    /// Compressed latent rows, `[g, hd]`, rotated; empty on a layer that does not own the compressed KV.
    ckv: Vec<f32>,
    g: usize,
    /// The last `ratio` normed sub-layer inputs, `[ratio, hidden]`: the compressor pools them when a group closes.
    xh: Vec<f32>,
    /// Index keys, `[g, dim]`, on the layer that owns the compressed KV.
    ik: Vec<f32>,
}

/// The decode cache: the per-layer KV of the sequence so far, and the routed experts each layer holds across
/// positions. A `Cache` is what turns the full-prefill graph into a one-position step.
/// The routed experts one layer holds, keyed by expert id, with the tick each was last read.
type Held = std::collections::HashMap<usize, (u64, (Lin, Lin, Lin))>;

pub struct Cache {
    layers: Vec<LayerCache>,
    /// For every layer, the layer that owns the compressed KV it reads (itself on a source).
    source_of: Vec<usize>,
    /// Experts held per layer, with the tick they were last read.
    experts: Vec<Held>,
    /// How many experts one layer holds before the least recently read leaves.
    hold: usize,
    tick: u64,
    /// Bytes of one routed expert in f32.
    expert_bytes: usize,
    /// Positions the cache holds, set by the prefill and advanced by every step.
    positions: usize,
    /// Whether each position so far is text: the positions of an image span are not, and the Engram lookback stops
    /// at them. Set by the prefill, one `true` per step.
    text: Vec<bool>,
    /// Experts read from the file over the whole run.
    pub experts_read: usize,
}

impl Cache {
    pub fn new(c: &Config, hold: usize) -> Self {
        let d = c.deepseek.as_ref().expect("no deepseek config");
        let n = c.n_layers;
        // The source of a layer: itself when it owns the compressed KV, otherwise the last source before it. The full
        // prefill reads whatever the last source published, so the cache does the same. A window layer (ratio 0)
        // has no source and no index keys.
        let mut source_of = vec![usize::MAX; n];
        for (i, s) in source_of.iter_mut().enumerate() {
            if d.compress_ratios.get(i).copied().unwrap_or(0) == 0 {
                continue;
            }
            *s = if d.kv_source_layers.contains(&i) {
                i
            } else {
                (0..i)
                    .rev()
                    .find(|&j| d.kv_source_layers.contains(&j))
                    .unwrap_or(usize::MAX)
            };
        }
        Cache {
            layers: (0..n).map(|_| LayerCache::default()).collect(),
            source_of,
            experts: (0..n).map(|_| std::collections::HashMap::new()).collect(),
            hold: hold.max(1),
            tick: 0,
            expert_bytes: expert_bytes_of(d, c.hidden),
            positions: 0,
            text: Vec::new(),
            experts_read: 0,
        }
    }

    /// Positions the cache holds.
    pub fn len(&self) -> usize {
        self.positions
    }

    /// Whether the cache holds no position yet.
    pub fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Drops the sequence and the held experts, keeping the allocation.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            *l = LayerCache::default();
        }
        for e in &mut self.experts {
            e.clear();
        }
        self.positions = 0;
        self.experts_read = 0;
        self.text.clear();
    }

    /// Bytes the held experts take on the host, in f32.
    pub fn expert_bytes(&self) -> usize {
        self.experts.iter().map(|e| e.len()).sum::<usize>() * self.expert_bytes
    }
}

/// The three f32 matrices of one routed expert: `2 * inter * hidden + hidden * inter`, four bytes each.
fn expert_bytes_of(d: &DeepseekConfig, hidden: usize) -> usize {
    3 * d.moe_inter * hidden * 4
}

/// The routed experts of one layer, held across positions: a hit costs no dequant, a miss reads the three matrices
/// and dequantises them. The least recently read expert leaves when the layer is full.
struct HeldExperts<'a> {
    src: &'a dyn TensorSource,
    layer: usize,
    inter: usize,
    hidden: usize,
    hold: usize,
    tick: &'a mut u64,
    read: &'a mut usize,
    held: &'a mut std::collections::HashMap<usize, (u64, (Lin, Lin, Lin))>,
}

impl ExpertBank for HeldExperts<'_> {
    fn with_expert(&mut self, e: usize, f: &mut dyn FnMut(&(Lin, Lin, Lin))) -> Result<()> {
        if !self.held.contains_key(&e) {
            if self.held.len() >= self.hold {
                let oldest = self.held.iter().min_by_key(|(_, (tick, _))| *tick).map(|(k, _)| *k);
                if let Some(k) = oldest {
                    self.held.remove(&k);
                }
            }
            let p = |s: &str| format!("layers.{}.ffn.experts.{e}.{s}", self.layer);
            let w = (
                load_lin(self.src, &p("w1.weight"), self.inter, self.hidden)?,
                load_lin(self.src, &p("w2.weight"), self.hidden, self.inter)?,
                load_lin(self.src, &p("w3.weight"), self.inter, self.hidden)?,
            );
            *self.read += 1;
            *self.tick += 1;
            self.held.insert(e, (*self.tick, w));
        }
        let tick = *self.tick;
        let w = self.held.get_mut(&e).expect("just inserted");
        w.0 = tick;
        f(&w.1);
        Ok(())
    }
}

/// Sparse attention over a cached KV: the token rows and the compressed rows are separate buffers, and a slot at or
/// above `n_tok` addresses the compressed one.
#[allow(clippy::too_many_arguments)]
fn sparse_attn_cached(
    q: &[f32],
    kv: &[f32],
    ckv: &[f32],
    n_tok: usize,
    sink: &[f32],
    idx: &[i32],
    m: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; heads * hd];
    for h in 0..heads {
        let qr = &q[h * hd..][..hd];
        let mut scores = vec![f32::NEG_INFINITY; m];
        for j in 0..m {
            let at = idx[j];
            if at < 0 {
                continue;
            }
            let at = at as usize;
            let row = if at < n_tok {
                &kv[at * hd..][..hd]
            } else {
                &ckv[(at - n_tok) * hd..][..hd]
            };
            scores[j] = dot(qr, row, hd) * scale;
        }
        let mut max = f32::NEG_INFINITY;
        for s in &scores {
            max = max.max(*s);
        }
        if max == f32::NEG_INFINITY {
            continue;
        }
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            if *s > f32::NEG_INFINITY {
                *s = (*s - max).exp();
                sum += *s;
            } else {
                *s = 0.0;
            }
        }
        sum += (sink[h] - max).exp();
        let inv = 1.0 / sum;
        for j in 0..m {
            if scores[j] == 0.0 {
                continue;
            }
            let at = idx[j] as usize;
            let row = if at < n_tok {
                &kv[at * hd..][..hd]
            } else {
                &ckv[(at - n_tok) * hd..][..hd]
            };
            let w = scores[j] * inv;
            for d in 0..hd {
                out[h * hd + d] += w * row[d];
            }
        }
    }
    out
}

/// One layer's attention for one position, against the cache. `t` is the total number of positions including this
/// one; `src` is the layer that owns the compressed KV when this layer does not.
#[allow(clippy::too_many_arguments)]
fn attention_step(
    graph: &Weights,
    l: &Layer,
    lc: &mut LayerCache,
    src: Option<&LayerCache>,
    x: &[f32],
    pos: usize,
    t: usize,
    shared: &mut Shared,
    layer: usize,
) -> Result<Vec<f32>> {
    let d = &graph.c;
    let a = &l.attn;
    let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
    let inv = if a.ratio > 0 {
        rope_inv(
            rd,
            d.rope_params.compress_theta,
            d.rope_params.original_seq,
            d.rope_params.factor,
            d.rope_params.beta_fast,
            d.rope_params.beta_slow,
        )
    } else {
        rope_inv(rd, d.rope_params.theta, 0, 1.0, 0.0, 0.0)
    };
    let mut qr = a.wq_a.mat(x, 1);
    rmsnorm(&mut qr, a.q_lora, &a.q_norm, graph.eps);
    let mut q = a.wq_b.mat(&qr, 1);
    rope_heads_at(&mut q, heads, hd, rd, &inv, pos as f32, false);
    let mut kv = a.wkv.mat(x, 1);
    rmsnorm(&mut kv, hd, &a.kv_norm, graph.eps);
    rope_rows(&mut kv, 1, hd, rd, &inv, |_| pos as f32, false);
    lc.kv.extend_from_slice(&kv);
    lc.n += 1;

    // The window slots: the last `window` positions, all of them in the cache by now.
    let wm = d.window.min(t);
    let mut idx: Vec<i32> = (t - wm..t).map(|i| i as i32).collect();
    let mut m = wm;

    // The compressed branch: a source pools its own latent, a consumer reads the source's.
    let ratio = a.ratio;
    if ratio > 0 {
        let compress_len = t.checked_div(ratio).unwrap_or(0);
        if compress_len > 0
            && let Some(c) = &a.compressor
        {
            // The rolling window of normed inputs the compressor pools: the last `ratio` of them.
            lc.xh.extend_from_slice(x);
            if lc.xh.len() > ratio * graph.hidden {
                lc.xh.drain(..graph.hidden);
            }
            if t.is_multiple_of(ratio) {
                let (lat, groups) = compressor(c, &lc.xh, ratio, hd, graph.eps);
                let mut rotated = lat.clone();
                rope_rows(&mut rotated, groups, hd, rd, &inv, |_| (t - ratio) as f32, false);
                lc.ckv.extend_from_slice(&rotated);
                lc.g += groups;
                if let Some(ix) = &a.indexer
                    && ix.owns_k
                {
                    let dim = ix.dim;
                    let mut k = ix.wk.as_ref().expect("a key owner holds wk").mat(&lat, groups);
                    rmsnorm(
                        &mut k,
                        dim,
                        ix.k_norm.as_ref().expect("a key owner holds its norm"),
                        graph.eps,
                    );
                    rope_rows(&mut k, groups, dim, rd, &inv, |_| (t - ratio) as f32, false);
                    lc.ik.extend_from_slice(&k);
                }
            }
        }
        let src = src.unwrap_or(&*lc);
        let n_ix = src.g;
        if n_ix > 0 {
            // The prefill reaches `(row + 1) / ratio` groups, which for this row is `t / ratio`.
            let reach = t / ratio;
            let cidxs = match &a.indexer {
                Some(ix) => indexer_step(ix, &src.ik, x, &qr, pos, reach, rd, &inv, d, shared, graph.eps, layer)?,
                None => shared.topk.clone().unwrap_or_default(),
            };
            let cm = cidxs.len();
            if cm > 0 {
                idx.extend(cidxs.iter().copied());
                m += cm;
            }
        }
    }

    let src = src.unwrap_or(&*lc);
    let out = sparse_attn_cached(&q, &lc.kv, &src.ckv, t, &a.sink, &idx, m, heads, hd);
    Ok(graph.out_projection_at(a, out, pos))
}

/// The indexer for one position: the second level of the two-level top-k, scored against the source's cached keys.
#[allow(clippy::too_many_arguments)]
fn indexer_step(
    ix: &Indexer,
    keys: &[f32],
    x: &[f32],
    qr: &[f32],
    pos: usize,
    reach: usize,
    rd: usize,
    inv: &[f32],
    d: &DeepseekConfig,
    shared: &mut Shared,
    eps: f32,
    layer: usize,
) -> Result<Vec<i32>> {
    let (heads, dim) = (ix.heads, ix.dim);
    let n_ix = keys.len() / dim;
    if n_ix == 0 {
        return Ok(Vec::new());
    }
    let mut q = ix.wq_b.mat(qr, 1);
    rope_heads_at(&mut q, heads, dim, rd, inv, pos as f32, false);
    let w = ix.weights_proj.mat(x, 1);
    let scale = (dim as f32).powf(-0.5);
    let mut scores = vec![0f32; n_ix];
    for h in 0..heads {
        let qh = &q[h * dim..][..dim];
        let wh = w[h] * scale;
        for j in 0..n_ix {
            scores[j] += dot(qh, &keys[j * dim..][..dim], dim).max(0.0) * wh;
        }
    }
    if ix.is_candidate {
        let cand = d.candidate.as_ref().expect("a candidate source has a geometry");
        let keep = select_candidate_blocks(&scores, &[reach], 1, n_ix, cand.topk_blocks, cand.block_size);
        shared.candidates = Some(keep);
        shared.candidate_width = n_ix;
    } else if let Some(cand) = &shared.candidates {
        for j in 0..n_ix {
            if !cand[j] {
                scores[j] = f32::NEG_INFINITY;
            }
        }
    }
    let _ = (eps, layer);
    let topk = ix.topk.min(n_ix);
    let mut order: Vec<usize> = (0..n_ix).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    let mut picked: Vec<usize> = order.into_iter().take(topk).collect();
    picked.sort();
    let mut out = Vec::with_capacity(picked.len());
    for j in picked {
        out.push(if j < reach { (j + pos + 1) as i32 } else { -1 });
    }
    shared.topk = Some(out.clone());
    Ok(out)
}

/// One block for one position: the same sub-layers in the same order as `Weights::block`, against the cache.
#[allow(clippy::too_many_arguments)]
fn block_step(
    graph: &Weights,
    l: &Layer,
    layer: usize,
    lc: &mut LayerCache,
    src: Option<&LayerCache>,
    h: &mut Vec<f32>,
    pre_mix: &mut Vec<f32>,
    pos: usize,
    t: usize,
    shared: &mut Shared,
    experts: &mut dyn ExpertBank,
    engram: Option<&EngramInput>,
    image_mask: Option<&[bool]>,
    main: &mut Vec<f32>,
) -> Result<()> {
    let d = &graph.c;
    let (h_dim, hc) = (graph.hidden, d.hc_mult);
    if let Some(e) = &l.engram {
        let input =
            engram.context("this stack carries an Engram table: it needs its hash ids and its companion file")?;
        let hash = input
            .hash
            .get(e.position)
            .with_context(|| format!("engram: no hash ids for table {}", e.position))?;
        e.forward(input.table, h, hash, 1, h_dim, hc, graph.eps, image_mask)?;
    }
    collect_main(graph, layer, h, 1, h_dim, hc, main);
    let am = l.hc_attn.mixes(h, 1, h_dim, hc, d.hc_iters, d.hc_eps);
    let x = hc_pre(h, pre_mix, 1, hc, h_dim);
    let mut xn = x.clone();
    rmsnorm(&mut xn, h_dim, &l.attn_norm, graph.eps);
    let attn_out = attention_step(graph, l, lc, src, &xn, pos, t, shared, layer)?;
    *h = hc_post(&attn_out, h, &am.post, &am.comb, 1, hc, h_dim);

    let fm = l.hc_ffn.mixes(h, 1, h_dim, hc, d.hc_iters, d.hc_eps);
    let x = hc_pre(h, &am.pre, 1, hc, h_dim);
    let mut xn = x.clone();
    rmsnorm(&mut xn, h_dim, &l.ffn_norm, graph.eps);
    let ffn_out = moe(
        &l.moe,
        &xn,
        1,
        h_dim,
        d,
        d.experts,
        d.top_k,
        image_mask,
        experts,
        &mut |_, _| Ok(()),
        layer,
    )?;
    *h = hc_post(&ffn_out, h, &fm.post, &fm.comb, 1, hc, h_dim);
    *pre_mix = fm.pre;
    Ok(())
}

/// Writes an image span into the streams `h` (`[t, hc, hidden]`): its rows replace the embeddings of the positions
/// it covers, in every copy, and the positions it covers are the ones the Engram gate and the router are told about.
fn splice(h: &mut [f32], t: usize, span: Option<&Span>, h_dim: usize, hc: usize) -> Result<Option<Vec<bool>>> {
    let Some(span) = span else { return Ok(None) };
    let len = span.len(h_dim);
    ensure!(len > 0, "an image span with no positions");
    ensure!(
        span.rows.len() == len * h_dim,
        "the span's rows are {} values for {len} x {h_dim}",
        span.rows.len()
    );
    ensure!(
        span.at + len <= t,
        "the span [{}..{}) does not fit {t} positions",
        span.at,
        span.at + len
    );
    for i in 0..len {
        for c in 0..hc {
            let dst = ((span.at + i) * hc + c) * h_dim;
            h[dst..dst + h_dim].copy_from_slice(&span.rows[i * h_dim..(i + 1) * h_dim]);
        }
    }
    let mut mask = vec![false; t];
    mask[span.at..span.at + len].fill(true);
    Ok(Some(mask))
}

/// Writes the streams a DSpark target layer hands to the draft when `layer` is one of them: the mean over the
/// hyper-connection copies, one row per position, `[t, target_layers * hidden]` with the layers along the last axis
/// — the order the draft's first projection reads (they are concatenated per position, not per layer).
fn collect_main(g: &Weights, layer: usize, h: &[f32], t: usize, hidden: usize, hc: usize, main: &mut Vec<f32>) {
    let Some(at) = g.target.iter().position(|&l| l == layer) else {
        return;
    };
    let layers = g.target.len();
    if main.len() < t * layers * hidden {
        main.resize(t * layers * hidden, 0.0);
    }
    for i in 0..t {
        for j in 0..hidden {
            let mut sum = 0f32;
            for c in 0..hc {
                sum += h[(i * hc + c) * hidden + j];
            }
            main[(i * layers + at) * hidden + j] = sum / hc as f32;
        }
    }
}

pub(crate) fn identity_mix(t: usize, hc: usize) -> Vec<f32> {
    let mut pre = vec![0f32; t * hc];
    for i in 0..t {
        pre[i * hc] = 1.0;
    }
    pre
}

/// The compressor: pool `ratio` tokens into one latent with a learned softmax gate, then norm it (pre-RoPE).
fn compressor(c: &Compressor, x: &[f32], t: usize, hd: usize, eps: f32) -> (Vec<f32>, usize) {
    let kv = c.wkv.mat(x, t);
    if c.ratio == 1 {
        let mut v = kv;
        rmsnorm(&mut v, hd, &c.norm, eps);
        return (v, t);
    }
    let score = c.wgate.as_ref().expect("a ratio above 1 pools with a gate").mat(x, t);
    let groups = t / c.ratio;
    let mut out = vec![0f32; groups * hd];
    for g in 0..groups {
        // The softmax is over the group's rows, so it is recomputed per column of the latent.
        for d in 0..hd {
            let mut max = f32::NEG_INFINITY;
            for j in 0..c.ratio {
                max = max.max(score[(g * c.ratio + j) * hd + d]);
            }
            let mut sum = 0f32;
            let mut acc = 0f32;
            for j in 0..c.ratio {
                let w = (score[(g * c.ratio + j) * hd + d] - max).exp();
                sum += w;
                acc += w * kv[(g * c.ratio + j) * hd + d];
            }
            out[g * hd + d] = acc / sum;
        }
    }
    rmsnorm(&mut out, hd, &c.norm, eps);
    (out, groups)
}

/// The indexer: level two of the top-k, or level one when this layer is the candidate source.
#[allow(clippy::too_many_arguments)]
fn indexer(
    ix: &Indexer,
    x: &[f32],
    qr: &[f32],
    latent: Option<&[f32]>,
    groups: usize,
    t: usize,
    _hd: usize,
    rd: usize,
    inv: &[f32],
    d: &DeepseekConfig,
    shared: &mut Shared,
    eps: f32,
    hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    layer: usize,
) -> Result<Vec<i32>> {
    let (heads, dim) = (ix.heads, ix.dim);
    // A prefix shorter than the ratio has closed no group yet: there is nothing to compress, so there are no index
    // keys and no selection to publish, and the window is all the attention has to read.
    if ix.owns_k && groups == 0 {
        shared.index_k.clear();
        shared.index_len = 0;
        shared.topk = Some(Vec::new());
        return Ok(Vec::new());
    }
    if ix.owns_k {
        let rows = groups;
        let mut k = match (&ix.wk, latent) {
            (Some(wk), Some(lat)) => wk.mat(lat, rows),
            _ => bail!("an index-key owner without a latent"),
        };
        rmsnorm(&mut k, dim, ix.k_norm.as_ref().expect("the owner has a key norm"), eps);
        rope_rows(&mut k, rows, dim, rd, inv, |i| (i * ix.ratio) as f32, false);
        shared.index_k = k;
        shared.index_len = rows;
        hook(&format!("ixk{layer}"), &shared.index_k)?;
    }
    let n_ix = shared.index_len;
    if n_ix == 0 {
        shared.topk = Some(Vec::new());
        return Ok(Vec::new());
    }
    let mut q = ix.wq_b.mat(qr, t);
    rope_heads(&mut q, t, heads, dim, rd, inv, false);
    let w = ix.weights_proj.mat(x, t);
    let scale = (dim as f32).powf(-0.5) * (heads as f32).powf(-0.5);

    // score[t, j] = sum over heads of relu(q·k) * head weight
    let mut score = vec![0f32; t * n_ix];
    for i in 0..t {
        for h in 0..heads {
            let qh = &q[(i * heads + h) * dim..][..dim];
            let wh = w[i * heads + h] * scale;
            for j in 0..n_ix {
                score[i * n_ix + j] += dot(qh, &shared.index_k[j * dim..][..dim], dim).max(0.0) * wh;
            }
        }
    }
    let compress_lens: Vec<usize> = (1..=t).map(|i| i / ix.ratio).collect();
    for i in 0..t {
        for j in compress_lens[i]..n_ix {
            score[i * n_ix + j] = f32::NEG_INFINITY;
        }
    }
    if ix.is_candidate {
        let cand = d.candidate.as_ref().expect("the candidate source has a geometry");
        let keep = select_candidate_blocks(&score, &compress_lens, t, n_ix, cand.topk_blocks, cand.block_size);
        let as_f32: Vec<f32> = keep.iter().map(|v| *v as u8 as f32).collect();
        hook(&format!("cand{layer}"), &as_f32)?;
        shared.candidates = Some(keep);
        shared.candidate_width = n_ix;
    } else if let Some(cand) = &shared.candidates {
        ensure!(
            shared.candidate_width == n_ix,
            "the candidate mask does not fit this indexer"
        );
        for i in 0..t {
            for j in 0..n_ix {
                if !cand[i * n_ix + j] {
                    score[i * n_ix + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // The reference takes exactly `topk` indices and then keeps only the reachable ones, so a position the
    // candidate mask zeroed can still be picked when there are fewer than `topk` candidates: the mask is a
    // ranking, not a cut. Ties are broken by ascending index, which is what its sort produces.
    let topk = ix.topk.min(n_ix);
    let mut out = vec![-1i32; t * topk];
    for i in 0..t {
        let mut order: Vec<usize> = (0..n_ix).collect();
        order.sort_by(|a, b| score[i * n_ix + *b].total_cmp(&score[i * n_ix + *a]).then(a.cmp(b)));
        let mut picked: Vec<usize> = order.iter().take(topk).copied().collect();
        picked.sort();
        for (slot, j) in picked.iter().enumerate() {
            out[i * topk + slot] = if *j < compress_lens[i] { (*j + t) as i32 } else { -1 };
        }
    }
    shared.topk = Some(out.clone());
    Ok(out)
}

/// Where a block's routed experts come from: the f32 matrices the resident loader built, or the file, dequantized
/// when a position selects an expert. Either way `with_expert` hands one expert's three matrices to the caller, so
/// the FFN keeps one implementation and one accumulation order.
pub trait ExpertBank {
    fn with_expert(&mut self, e: usize, f: &mut dyn FnMut(&(Lin, Lin, Lin))) -> Result<()>;
}

/// The resident stack's experts, already in memory.
pub struct ResidentExperts<'a>(pub &'a [(Lin, Lin, Lin)]);

impl ExpertBank for ResidentExperts<'_> {
    fn with_expert(&mut self, e: usize, f: &mut dyn FnMut(&(Lin, Lin, Lin))) -> Result<()> {
        let w = self
            .0
            .get(e)
            .with_context(|| format!("expert {e} of {}", self.0.len()))?;
        f(w);
        Ok(())
    }
}

/// One block's experts straight from the file, kept for the positions that select them again and dropped with the
/// block. 384 experts in f32 are 54 GB; six of them, 850 MB.
pub struct StreamedExperts<'a> {
    src: &'a dyn TensorSource,
    /// `layers.{layer}.ffn.experts.` for the backbone, `mtp.{stage}.ffn.experts.` for a draft stage.
    pub base: String,
    inter: usize,
    hidden: usize,
    /// Experts held at once. Past this, the cache is emptied: a position's own six are used in a row, so the ones it
    /// needs are always the ones just read.
    hold: usize,
    cache: std::collections::HashMap<usize, Box<(Lin, Lin, Lin)>>,
    pub read: usize,
}

impl<'a> StreamedExperts<'a> {
    pub fn new(src: &'a dyn TensorSource, c: &Config, layer: usize, hold: usize) -> Self {
        let d = c.deepseek.as_ref().expect("a deepseek config");
        StreamedExperts {
            src,
            base: format!("layers.{layer}.ffn.experts."),
            inter: d.moe_inter,
            hidden: c.hidden,
            hold: hold.max(1),
            cache: std::collections::HashMap::new(),
            read: 0,
        }
    }

    /// The same reader on another namespace: the draft keeps its stages under `mtp.`.
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn load(&mut self, e: usize) -> Result<Box<(Lin, Lin, Lin)>> {
        let base = self.base.clone();
        let p = |s: &str| format!("{base}{e}.{s}");
        let (inter, hidden) = (self.inter, self.hidden);
        self.read += 1;
        Ok(Box::new((
            load_lin(self.src, &p("w1.weight"), inter, hidden)?,
            load_lin(self.src, &p("w2.weight"), hidden, inter)?,
            load_lin(self.src, &p("w3.weight"), inter, hidden)?,
        )))
    }
}

impl ExpertBank for StreamedExperts<'_> {
    fn with_expert(&mut self, e: usize, f: &mut dyn FnMut(&(Lin, Lin, Lin))) -> Result<()> {
        if !self.cache.contains_key(&e) {
            if self.cache.len() >= self.hold {
                self.cache.clear();
            }
            let w = self.load(e)?;
            self.cache.insert(e, w);
        }
        f(&self.cache[&e]);
        Ok(())
    }
}

/// The FFN: `top_k` routed experts plus the shared one, both under the SwiGLU clamps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe(
    m: &Moe,
    x: &[f32],
    t: usize,
    hidden: usize,
    d: &DeepseekConfig,
    experts: usize,
    top_k: usize,
    image_mask: Option<&[bool]>,
    bank: &mut dyn ExpertBank,
    hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    layer: usize,
) -> Result<Vec<f32>> {
    hook(&format!("ffn_in{layer}"), x)?;
    let mut scores = linear(x, t, hidden, &m.gate.w, experts, None);
    for v in scores.iter_mut() {
        // sqrt(softplus(x)), with torch's threshold: above 20 the exponential would only add noise (and
        // overflow past 88), so softplus is the identity there.
        let x = *v;
        *v = if x > 20.0 {
            x.sqrt()
        } else {
            (1.0 + x.exp()).ln().sqrt()
        };
    }
    // The bias steers the selection only; the weights come from the raw scores.
    let mut out = vec![0f32; t * hidden];
    for i in 0..t {
        // An image-span token routes with the VL bias (`noaux_tc_for_vl`): it steers the selection only.
        let bias = match (&m.bias_vl, image_mask) {
            (Some(vl), Some(mask)) if mask.get(i).copied().unwrap_or(false) => vl.as_slice(),
            _ => m.bias.as_slice(),
        };
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|a, b| (scores[i * experts + *b] + bias[*b]).total_cmp(&(scores[i * experts + *a] + bias[*a])));
        let top: Vec<usize> = order.iter().take(top_k).copied().collect();
        let mut w: Vec<f32> = top.iter().map(|&e| scores[i * experts + e]).collect();
        if d.norm_topk_prob && top_k > 1 {
            let s: f32 = w.iter().sum::<f32>() + 1e-20;
            for v in w.iter_mut() {
                *v /= s;
            }
        }
        for v in w.iter_mut() {
            *v *= d.route_scale;
        }
        let xi = &x[i * hidden..(i + 1) * hidden];
        let limit = d.swiglu_limit;
        for (slot, &e) in top.iter().enumerate() {
            let weight = w[slot];
            let mut add = |w: &(Lin, Lin, Lin)| {
                let e_out = expert(&w.0, &w.1, &w.2, xi, limit);
                for (o, v) in out[i * hidden..(i + 1) * hidden].iter_mut().zip(&e_out) {
                    *o += weight * v;
                }
            };
            bank.with_expert(e, &mut add)?;
        }
        let s_out = expert(&m.shared.0, &m.shared.1, &m.shared.2, xi, d.swiglu_limit);
        for (o, v) in out[i * hidden..(i + 1) * hidden].iter_mut().zip(&s_out) {
            *o += v;
        }
    }
    Ok(out)
}

pub(crate) fn expert(w1: &Lin, w2: &Lin, w3: &Lin, x: &[f32], limit: f32) -> Vec<f32> {
    let gate = linear(x, 1, x.len(), &w1.w, w1.n, None);
    let up = linear(x, 1, x.len(), &w3.w, w3.n, None);
    let mut act = vec![0f32; w2.k];
    for i in 0..w2.k {
        let g = if limit > 0.0 { gate[i].min(limit) } else { gate[i] };
        let u = if limit > 0.0 { up[i].clamp(-limit, limit) } else { up[i] };
        act[i] = silu(g) * u;
    }
    linear(&act, 1, w2.k, &w2.w, w2.n, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use malleus_gguf::gguf::Gguf;
    use malleus_gguf::safetensors::SafeTensors;

    fn f32_file(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn compare(what: &str, got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
        let (mut max, mut sum) = (0f64, 0f64);
        for (a, b) in got.iter().zip(want) {
            dot += (*a as f64) * (*b as f64);
            ng += (*a as f64) * (*a as f64);
            nw += (*b as f64) * (*b as f64);
            let d = (*a as f64 - *b as f64).abs();
            max = max.max(d);
            sum += d;
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        eprintln!(
            "{what}: cosine {cos:.8}, max |diff| {max:.6}, mean |diff| {:.6}",
            sum / got.len() as f64
        );
        cos
    }

    /// The loader reads our own `.malleus` file: every quantized format dequantises and unrotates to what the
    /// checkpoint's source produced, up to the quantizer's own error.
    /// DEEPSEEK_FILE=.../deepseek-v41-flash-mls2.malleus DEEPSEEK_CKPT=/models/deepseekv41-flash
    #[test]
    #[ignore = "reads the converted file and the checkpoint"]
    fn the_loader_reads_our_own_file() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&file)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        for (name, floor) in [
            ("layers.0.attn.wq_a.weight", 0.99),
            ("layers.0.attn.wkv.weight", 0.99),
            ("layers.0.ffn.experts.0.w1.weight", 0.90),
            ("layers.0.ffn.shared_experts.w1.weight", 0.98),
        ] {
            let info = g.info(name).unwrap();
            let (n, k) = (info.shape[0], info.shape[1]);
            let got = load_matrix(&g, name, n, k).unwrap();
            let want = load_matrix(&src, name, n, k).unwrap();
            let cos = compare(name, &got, &want);
            eprintln!("{name} {:?}: cosine {cos:.7}", info.dtype);
            assert!(cos > floor, "{name}: {cos} below {floor}");
        }
    }

    /// Layer 0 on the checkpoint's own weights, against `scripts/deepseek_ref_layer.py`: the embedding, the
    /// attention output, the MoE output, the block output and the logits. `MALLEUS_DEEPSEEK_LAYERS=0` makes the
    /// loader keep one layer, which is what the reference runs.
    ///
    ///   MALLEUS_DEEPSEEK_LAYERS=0 DEEPSEEK_CKPT=/models/deepseekv41-flash DEEPSEEK_REF_LAYER=/golden/deepseek-ref/layer0
    #[test]
    #[ignore = "loads one real layer (54 GB) and compares against the reference"]
    fn layer_zero_matches_the_reference_on_the_real_weights() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let dir = std::env::var("DEEPSEEK_REF_LAYER").expect("DEEPSEEK_REF_LAYER");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let t0 = std::time::Instant::now();
        let w = Weights::load(&src, &c, &|_, _| {}).unwrap();
        assert_eq!(w.layers.len(), 1, "set MALLEUS_DEEPSEEK_LAYERS=0");
        eprintln!(
            "loaded {:.1}B parameters in {:.0}s",
            w.parameters() as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
        let mut worst = 1.0f64;
        let logits = w
            .forward_with(&ids, None, |name, values| {
                let path = format!("{dir}/{name}.f32");
                if std::path::Path::new(&path).exists() {
                    let cos = compare(name, values, &f32_file(&path));
                    worst = worst.min(cos);
                }
                Ok(())
            })
            .unwrap();
        let cos = compare("logits_all", &logits, &f32_file(&format!("{dir}/logits_all.f32")));
        eprintln!("layer 0: worst stage {worst:.6}, logits {cos:.6}");
        assert!(worst > 0.999, "a stage diverges: {worst}");
        assert!(cos > 0.999, "the logits diverge: {cos}");
    }

    /// An isolated layer's attention against the reference: the compressor, the indexer and the sparse attention
    /// on the checkpoint's own weights. `MALLEUS_DEEPSEEK_LAYERS=<layer>` loads that layer alone.
    ///
    ///   MALLEUS_DEEPSEEK_LAYERS=2 DEEPSEEK_CKPT=/models/deepseekv41-flash DEEPSEEK_REF_LAYER=/golden/deepseek-ref/layer2
    #[test]
    #[ignore = "loads one real layer and compares against the reference"]
    fn an_isolated_layer_attention_matches() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let dir = std::env::var("DEEPSEEK_REF_LAYER").expect("DEEPSEEK_REF_LAYER");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let layer: usize = std::env::var("DEEPSEEK_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let x = f32_file(&format!("{dir}/x{layer}.f32"));
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let d = c.deepseek.as_ref().unwrap().clone();
        let hc = d.hc_mult;
        let t = x.len() / (hc * c.hidden);
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let w = Weights::load(&src, &c, &|_, _| {}).unwrap();
        assert_eq!(w.layers.len(), 1, "set MALLEUS_DEEPSEEK_LAYERS={layer}");
        // The reference normalises the collapsed streams before the attention; the same here.
        let mut xn = hc_pre(&x, &identity_mix(t, hc), t, hc, c.hidden);
        rmsnorm(&mut xn, c.hidden, &w.layers[0].attn_norm, w.eps);
        let got = w.attention_stage(0, &xn, t).unwrap();
        let cos = compare(
            &format!("attn{layer}"),
            &got,
            &f32_file(&format!("{dir}/attn{layer}.f32")),
        );
        assert!(cos > 0.999, "attn{layer} diverges: {cos}");
        let _ = manifest;
    }

    /// A sample across the stack, through the loader: every layer has its own rotation and its own scales, and
    /// the indexed layers carry a compressor and an indexer the first two do not. The Engram tables are not
    /// part of this: they are 98.31B elements and are read a row at a time, never as a matrix.
    /// DEEPSEEK_FILE=.../deepseek-v41-flash-mls2.malleus DEEPSEEK_CKPT=/models/deepseekv41-flash
    #[test]
    #[ignore = "reads the converted file and the checkpoint"]
    fn a_sample_across_the_stack_round_trips() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&file)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let mut checked = 0;
        for layer in [0usize, 1, 2, 5, 20, 39] {
            for (suffix, floor) in [
                ("attn.wq_a.weight", 0.999),
                ("attn.wkv.weight", 0.999),
                ("ffn.experts.0.w1.weight", 0.93),
                ("ffn.experts.0.w2.weight", 0.93),
                ("ffn.shared_experts.w1.weight", 0.99),
            ] {
                let name = format!("layers.{layer}.{suffix}");
                let (n, k) = {
                    let i = g.info(&name).unwrap();
                    (i.shape[0], i.shape[1])
                };
                let got = load_matrix(&g, &name, n, k).unwrap();
                let want = load_matrix(&src, &name, n, k).unwrap();
                let cos = compare(&name, &got, &want);
                assert!(cos > floor, "{name}: {cos} below {floor}");
                checked += 1;
            }
        }
        eprintln!("{checked} tensors across six layers agree with the checkpoint");
    }

    /// The text graph against `scripts/deepseek_ref.py`: every stage the reference dumps, on the same weights.
    /// The reference runs on the values it exports (the file the engine reads is bf16), so what is left between
    /// the two is the order of their f32 arithmetic.
    #[test]
    #[ignore = "needs the tiny reference dump"]
    fn the_text_graph_matches_the_reference() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["tokens"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&dir)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&dir)).unwrap();
        let w = Weights::load(&src, &c, &|_, _| {}).unwrap();
        eprintln!("text weights: {:.1} MB on the host", w.bytes() as f64 / 1e6);

        let mut worst: Vec<(String, f64)> = Vec::new();
        let mut last_topk: Vec<f32> = Vec::new();
        let logits = w
            .forward_with(&ids, None, |name, values| {
                if name.starts_with("topk") {
                    last_topk = values.to_vec();
                }
                let path = format!("{dir}/{name}.f32");
                if std::path::Path::new(&path).exists() {
                    let cos = compare(name, values, &f32_file(&path));
                    worst.push((name.to_string(), cos));
                }
                Ok(())
            })
            .unwrap();
        let all_cos = compare("logits_all", &logits, &f32_file(&format!("{dir}/logits_all.f32")));
        let last = &logits[(ids.len() - 1) * w.vocab..];
        let cos_last = compare("logits", last, &f32_file(&format!("{dir}/logits.f32")));
        // The reference's single `topk` dump is the last index source's list.
        compare("topk", &last_topk, &f32_file(&format!("{dir}/topk.f32")));

        // Every smooth stage has to agree to f32, and so does the whole logits matrix up to one selection: the two
        // index sources (layers 2 and 4) publish the same lists, while the layers that read their keys (3 and 5)
        // can flip one slot where two candidates score within the noise of the two dot-product orders. The flipped
        // slot is a real key, so that layer's output moves by ~1e-3 and the logits land at 0.9994, which is what
        // the floor below separates from a structural error (a wrong reach, a wrong offset or key land at 0.6).
        // The checkpoint's own weights are pinned by the gates on the real file.
        let smooth = worst.iter().filter(|(n, _)| !n.starts_with("topk"));
        let worst_cos = smooth.clone().map(|(_, c)| *c).fold(1.0f64, f64::min);
        assert!(
            worst_cos > 0.99,
            "a stage diverges: {:?}",
            smooth.filter(|(_, c)| *c <= 0.99).collect::<Vec<_>>()
        );
        assert!(all_cos > 0.999, "the logits diverge: {all_cos}");
        assert!(cos_last > 0.99, "the last row diverges: {cos_last}");
    }

    /// The streamed stack changes where the bytes come from, not the numbers: on the tiny reference checkpoint the
    /// resident and the streamed paths have to agree bit for bit, layer by layer, expert by expert.
    ///
    ///   DEEPSEEK_REF=/golden/deepseek-ref/tiny
    #[test]
    #[ignore = "reads the tiny reference checkpoint"]
    fn the_streamed_stack_matches_the_resident_one() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["tokens"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&dir)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&dir)).unwrap();
        let w = Weights::load(&src, &c, &|_, _| {}).unwrap();
        let resident = w.forward(&ids).unwrap();
        let mut s = Streamer::open(Box::new(src), &c).unwrap();
        let streamed = s.logits(&ids).unwrap();
        assert_eq!(resident.len(), streamed.len());
        compare("streamed", &streamed, &resident);
        let differing = resident.iter().zip(&streamed).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{differing} of {} logits differ", resident.len());
        eprintln!(
            "{} experts read, {:.1} MB resident",
            s.experts_read,
            s.resident_bytes() as f64 / 1e6
        );
    }

    /// The span's rows replace the embeddings of the positions it covers, in every hyper-connection copy: the same
    /// prompt, spliced by the caller into a stream and run through `run_from`, has to give the same logits, bit for
    /// bit. The tiny checkpoint has no Engram layer and no VL bias, so the mask is a no-op there and the splice is
    /// the only difference; the mask's arithmetic is gated against the reference on the real tables.
    ///
    ///   DEEPSEEK_REF=/golden/deepseek-ref/tiny
    #[test]
    #[ignore = "reads the reference checkpoint"]
    fn the_image_span_is_spliced_into_the_embeddings() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["tokens"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&dir)).unwrap();
        let (hidden, hc) = (c.hidden, c.deepseek.as_ref().unwrap().hc_mult);
        let src = SafeTensors::open_dir(std::path::Path::new(&dir)).unwrap();
        let mut s = Streamer::open(Box::new(src), &c).unwrap();
        // A span of three positions in the middle, whose rows are not any token's embedding.
        let (at, len) = (2usize, 3usize);
        let rows: Vec<f32> = (0..len * hidden).map(|i| ((i % 89) as f32) * 0.011 - 0.5).collect();
        let span = Span { at, rows: rows.clone() };
        // What the caller would have done by hand: the token embeddings with those rows written in.
        let mut x = vec![0f32; ids.len() * hc * hidden];
        for (i, &id) in ids.iter().enumerate() {
            let row = &s.graph.embed[id as usize * hidden..(id as usize + 1) * hidden];
            for copy in 0..hc {
                x[(i * hc + copy) * hidden..][..hidden].copy_from_slice(row);
            }
        }
        for i in 0..len {
            for copy in 0..hc {
                let dst = ((at + i) * hc + copy) * hidden;
                x[dst..dst + hidden].copy_from_slice(&rows[i * hidden..(i + 1) * hidden]);
            }
        }
        let want = s
            .run_from(&x, ids.len(), Some(&ids), |_, _| {}, &mut |_, _| Ok(()))
            .unwrap();
        let mut cache = s.cache(8);
        let got = s.prefill_cached_span(&mut cache, &ids, Some(&span)).unwrap();
        assert_eq!(got.len(), want.len());
        let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{differing} of {} logits differ", got.len());
        // And the span is where it says it is: written one position later, the logits move.
        let shifted = Span {
            at: at + 1,
            rows: rows.clone(),
        };
        let mut other = s.cache(8);
        let moved = s.prefill_cached_span(&mut other, &ids, Some(&shifted)).unwrap();
        assert!(
            moved.iter().zip(&got).any(|(a, b)| a != b),
            "the span's position is ignored"
        );
    }

    /// A prefix shorter than the compressed ratio — the tiny checkpoint's layers 2 and 3 close a group every two
    /// positions — has nothing to compress yet: the attention reads the window alone and the indexer publishes an
    /// empty list. Serving a one-token prompt goes through this, and it used to panic in the indexer.
    ///
    ///   DEEPSEEK_REF=/golden/deepseek-ref/tiny cargo test -p malleus-model a_prefix_shorter_than_the_ratio -- --ignored
    #[test]
    #[ignore = "reads the reference checkpoint"]
    fn a_prefix_shorter_than_the_ratio_attends_on_the_window_alone() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids = [manifest["tokens"][0][0].as_u64().unwrap() as u32];
        let c = Config::from_hf(std::path::Path::new(&dir)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&dir)).unwrap();
        let mut s = Streamer::open(Box::new(src), &c).unwrap();
        let logits = s.logits(&ids).unwrap();
        assert_eq!(logits.len(), c.vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "a one-token prompt does not give a logits row"
        );
    }

    /// The decode path: a prefill fills the cache, then one step per position. Every step's logits must equal the
    /// prefill's row for that position — the same graph, read one position at a time.
    ///   DEEPSEEK_REF=/golden/deepseek-ref/tiny cargo test -p malleus-model the_cached_step -- --ignored --nocapture
    #[test]
    #[ignore = "reads the reference checkpoint"]
    fn the_cached_step_matches_the_prefill() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["tokens"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&dir)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&dir)).unwrap();
        let mut s = Streamer::open(Box::new(src), &c).unwrap();
        let vocab = c.vocab;
        let t = ids.len();
        // The reference: the whole sequence in one prefill.
        let mut whole = s.cache(8);
        let prefill = s.prefill_cached(&mut whole, &ids).unwrap();
        // The same sequence as a short prefill and then one step per position.
        let mut cache = s.cache(8);
        let head = s.prefill_cached(&mut cache, &ids[..2]).unwrap();
        for pos in 0..2 {
            let cos = compare(
                &format!("prefill{pos}"),
                &head[pos * vocab..(pos + 1) * vocab],
                &prefill[pos * vocab..(pos + 1) * vocab],
            );
            assert!(cos > 0.9999, "prefill {pos}: cosine {cos}");
        }
        for pos in 2..t {
            let row = s.step(&mut cache, &ids[..=pos]).unwrap();
            let want = &prefill[pos * vocab..(pos + 1) * vocab];
            let cos = compare(&format!("step{pos}"), &row, want);
            assert!(cos > 0.9999, "step {pos}: cosine {cos}");
        }
        eprintln!("{} positions, {} experts read", t, cache.experts_read);
    }

    /// The 4-bit rows are the checkpoint's own rows, up to the quantizer: a table read by row at run time turns a
    /// decode that is off by a bias into noise in the streams of the two layers that carry one. The Engram forward
    /// gate decodes both sides with this crate's reader, so it cannot see that — this compares against the source.
    ///
    ///   DEEPSEEK_CKPT=... DEEPSEEK_ENGRAM_TABLE=.../deepseek-v41-flash-engram.malleus
    #[test]
    #[ignore = "reads rows of the 100 GB table and of the checkpoint"]
    fn the_engram_table_matches_the_checkpoint() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let table = std::env::var("DEEPSEEK_ENGRAM_TABLE").expect("DEEPSEEK_ENGRAM_TABLE");
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let g = Gguf::open(std::path::Path::new(&table)).unwrap();
        for layer in [1usize, 14] {
            let name = format!("layers.{layer}.engram.embed.weight");
            let scales = format!("layers.{layer}.engram.embed.scale");
            let info = src.info(&name).unwrap();
            let (rows, k) = (info.shape[0], info.shape[1]);
            // The first rows of the table and a few far inside it: the rows a prompt reads are arbitrary.
            let ids: Vec<u64> = [0u64, 1, 7, 4096, 1 << 20, rows as u64 / 2, rows as u64 - 1].to_vec();
            let got = crate::deepseek_engram::read_rows(&g, &name, &ids, k).unwrap();
            let mut want = Vec::with_capacity(ids.len() * k);
            let per_row = k.div_ceil(32);
            let (mut q, mut sc) = (vec![0u8; k], vec![0u8; per_row]);
            for &r in &ids {
                src.read(&name, r as usize * k, &mut q).unwrap();
                src.read(&scales, r as usize * per_row, &mut sc).unwrap();
                for (c, b) in q.iter().enumerate() {
                    want.push(f8_e4m3(*b) * e8m0(sc[c / 32]));
                }
            }
            // 4 bits of absmax per row against the checkpoint's fp8: 0.9395 measured, and the release's own build keeps
            // the table at IQ4_NL. A decode off by the bias falls to -0.59, so the floor separates the two by far.
            let cos = compare(&name, &got, &want);
            assert!(cos > 0.93, "{name}: the rows diverge from the checkpoint: {cos}");
        }
    }

    /// Two real blocks end to end — the embedding, the hyper-connection handoff between them, and the Engram lookup
    /// at layer 1 — against the release's own blocks in bf16: `scripts/deepseek_ref_stack.py --layers 0,1`.
    ///
    ///   MALLEUS_DEEPSEEK_LAYERS=0,1 DEEPSEEK_CKPT=... DEEPSEEK_ENGRAM=... DEEPSEEK_ENGRAM_TABLE=... \
    ///   DEEPSEEK_REF_STACK=/golden/deepseek-ref/stack01
    #[test]
    #[ignore = "streams two real blocks from the checkpoint"]
    fn the_first_blocks_match_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let base = std::env::var("DEEPSEEK_ENGRAM").expect("DEEPSEEK_ENGRAM");
        let table = std::env::var("DEEPSEEK_ENGRAM_TABLE").expect("DEEPSEEK_ENGRAM_TABLE");
        let dir = std::env::var("DEEPSEEK_REF_STACK").expect("DEEPSEEK_REF_STACK");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let layers: Vec<usize> = manifest["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let state = EngramState::from_json(
            &std::fs::read_to_string(format!("{base}-state.json")).unwrap(),
            &std::fs::read(format!("{base}-token-map.i32")).unwrap(),
        )
        .unwrap();
        let tables = Gguf::open(std::path::Path::new(&table)).unwrap();
        let mut s = Streamer::open(Box::new(src), &c)
            .unwrap()
            .with_engram(Box::new(tables), state)
            .with_experts_held(64);
        let mut worst = 1.0f64;
        let logits = s
            .run_with(&ids, |_, _| {}, &mut |name, values| {
                let path = format!("{dir}/{name}.f32");
                if std::path::Path::new(&path).exists() {
                    worst = worst.min(compare(name, values, &f32_file(&path)));
                }
                Ok(())
            })
            .unwrap();
        let cos = compare("logits_all", &logits, &f32_file(&format!("{dir}/logits_all.f32")));
        eprintln!(
            "{layers:?}: worst block {worst:.6}, logits {cos:.6}, {} experts read",
            s.experts_read
        );
        // The reference is in bf16 and this run is f32, so the two differ by the storage, not by the graph.
        // The reference is in bf16 here and this run is in f32, so the floor separates a structural error from the
        // storage: a wrong compressed-KV handoff or a row decode off by eight lands at 0.6, not at 0.99.
        assert!(worst > 0.99, "a block diverges: {worst}");
        assert!(cos > 0.99, "the logits diverge: {cos}");
    }

    /// A window of blocks starting from a fixed stream: `scripts/deepseek_ref_stack.py --layers 2,3 --random-x`.
    /// Two blocks 2 and 3 are a compressed-KV source and the layer that reuses what it published — the handoff the
    /// isolated gates cannot see.
    ///
    ///   MALLEUS_DEEPSEEK_LAYERS=2,3 DEEPSEEK_CKPT=... DEEPSEEK_REF_STACK=/golden/deepseek-ref/stack23
    #[test]
    #[ignore = "streams real blocks from the checkpoint"]
    fn a_window_of_blocks_matches_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let dir = std::env::var("DEEPSEEK_REF_STACK").expect("DEEPSEEK_REF_STACK");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let tokens = manifest["tokens"].as_u64().unwrap() as usize;
        let ids: Vec<u32> = manifest["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        // A window from a fixed stream starts at `x.f32`; one that starts at the prompt embeds its ids.
        let x = std::path::Path::new(&format!("{dir}/x.f32"))
            .exists()
            .then(|| f32_file(&format!("{dir}/x.f32")));
        // A window that carries an Engram layer needs the tables and their addressing.
        let tables: Option<Source> = std::env::var("DEEPSEEK_ENGRAM_TABLE")
            .ok()
            .map(|p| Box::new(Gguf::open(std::path::Path::new(&p)).unwrap()) as Source);
        let mut s = Streamer::open(Box::new(src), &c).unwrap().with_experts_held(64);
        if let (Some(tables), Ok(base)) = (tables, std::env::var("DEEPSEEK_ENGRAM")) {
            let state = EngramState::from_json(
                &std::fs::read_to_string(format!("{base}-state.json")).unwrap(),
                &std::fs::read(format!("{base}-token-map.i32")).unwrap(),
            )
            .unwrap();
            s = s.with_engram(tables, state);
        }
        let mut worst = 1.0f64;
        let mut hook = |name: &str, values: &[f32]| {
            let path = format!("{dir}/{name}.f32");
            if std::path::Path::new(&path).exists() {
                worst = worst.min(compare(name, values, &f32_file(&path)));
            }
            Ok(())
        };
        let logits = match &x {
            Some(x) => s.run_from(x, tokens, Some(&ids), |_, _| {}, &mut hook).unwrap(),
            None => s.run_with(&ids, |_, _| {}, &mut hook).unwrap(),
        };
        let cos = compare("logits_all", &logits, &f32_file(&format!("{dir}/logits_all.f32")));
        eprintln!(
            "worst block {worst:.6}, logits {cos:.6}, {} experts read",
            s.experts_read
        );
        // The reference is in bf16 here and this run is in f32, so the floor separates a structural error from the
        // storage: a wrong compressed-KV handoff or a row decode off by eight lands at 0.6, not at 0.99.
        assert!(worst > 0.99, "a block diverges: {worst}");
        assert!(cos > 0.99, "the logits diverge: {cos}");
    }

    /// The Engram forward against the release's own module: the rows come from our 4-bit table, `wkv` and the two
    /// gate vectors from the checkpoint, and the gate itself is the reference's `Engram` in torch. The hash ids are
    /// recomputed here from the state, so the lookup, the rows and the gate are compared together.
    ///
    ///   DEEPSEEK_CKPT=... DEEPSEEK_ENGRAM=.../golden/deepseek-engram \
    ///   DEEPSEEK_ENGRAM_TABLE=.../deepseek-v41-flash-engram.malleus DEEPSEEK_REF_ENGRAM=.../golden/deepseek-ref/engram1
    #[test]
    #[ignore = "reads the checkpoint and the companion table"]
    fn the_engram_forward_matches_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let base = std::env::var("DEEPSEEK_ENGRAM").expect("DEEPSEEK_ENGRAM");
        let table = std::env::var("DEEPSEEK_ENGRAM_TABLE").expect("DEEPSEEK_ENGRAM_TABLE");
        let dir = std::env::var("DEEPSEEK_REF_ENGRAM").expect("DEEPSEEK_REF_ENGRAM");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let layer = manifest["layer"].as_u64().unwrap() as usize;
        let (t, hc, hidden) = (
            ids.len(),
            manifest["hc"].as_u64().unwrap() as usize,
            manifest["hidden"].as_u64().unwrap() as usize,
        );
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let d = c.deepseek.as_ref().unwrap().clone();
        let src = SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let e = Engram::load(&src, &c, layer).unwrap();
        let state = EngramState::from_json(
            &std::fs::read_to_string(format!("{base}-state.json")).unwrap(),
            &std::fs::read(format!("{base}-token-map.i32")).unwrap(),
        )
        .unwrap();
        let hash = state.hash_ids(&ids, e.position);
        let table = Gguf::open(std::path::Path::new(&table)).unwrap();
        // Stage by stage: the rows out of the table, the projection they go through, the gate's output.
        let head_dim = e.wkv.k / (hash.len() / t);
        let rows = crate::deepseek_engram::read_rows(&table, &e.table, &hash, head_dim).unwrap();
        compare("rows", &rows, &f32_file(&format!("{dir}/rows.f32")));
        let kv = e.wkv.mat(&rows, t);
        compare("kv", &kv, &f32_file(&format!("{dir}/kv.f32")));
        let mut h = f32_file(&format!("{dir}/x.f32"));
        assert_eq!(h.len(), t * hc * hidden);
        e.forward(&table, &mut h, &hash, t, hidden, hc, d.eps, None).unwrap();
        let cos = compare("engram", &h, &f32_file(&format!("{dir}/out.f32")));
        assert!(cos > 0.9999, "the Engram forward diverges: {cos}");
    }
}
