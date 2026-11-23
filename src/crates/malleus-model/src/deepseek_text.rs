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
use crate::hostmath::{dot, linear, parallel_chunks, rmsnorm, silu};

/// A linear weight, row-major `[n, k]`, applied as `y = x · wᵀ`.
pub struct Lin {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl Lin {
    fn mat(&self, x: &[f32], t: usize) -> Vec<f32> {
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
    } else if exp == 0xF {
        if man == 0.0 { sign * f32::INFINITY } else { f32::NAN }
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Reads a vector (`[n]`) of f32 or bf16.
pub fn load_vector(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>> {
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
    f: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
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

struct Moe {
    gate: Lin,
    bias: Vec<f32>,
    bias_vl: Option<Vec<f32>>,
    shared: (Lin, Lin, Lin),
    experts: Vec<(Lin, Lin, Lin)>,
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    hc_attn: Hc,
    hc_ffn: Hc,
    attn: Attn,
    moe: Moe,
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
}

impl Weights {
    pub fn load(src: &dyn TensorSource, c: &Config, progress: &dyn Fn(usize, usize)) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?.clone();
        let h = c.hidden;
        let n = c.n_layers;
        ensure!(d.compress_ratios.len() == n, "compress_ratios must cover every layer");
        let embed = load_matrix(src, "embed.weight", c.vocab, h)?;
        let head = load_matrix(src, "head.weight", c.vocab, h)?;
        let norm = load_vector(src, "norm.weight", h)?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
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
            let gate = Lin {
                n: d.experts,
                k: h,
                w: load_matrix(src, &p("ffn.gate.weight"), d.experts, h)?,
            };
            let mut experts = Vec::with_capacity(d.experts);
            let inter = d.moe_inter;
            for e in 0..d.experts {
                let ep = |s: &str| p(&format!("ffn.experts.{e}.{s}"));
                experts.push((
                    load_lin(src, &ep("w1.weight"), inter, h)?,
                    load_lin(src, &ep("w2.weight"), h, inter)?,
                    load_lin(src, &ep("w3.weight"), inter, h)?,
                ));
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
            layers.push(Layer {
                attn_norm: load_vector(src, &p("attn_norm.weight"), h)?,
                ffn_norm: load_vector(src, &p("ffn_norm.weight"), h)?,
                hc_attn: hc("hc_attn")?,
                hc_ffn: hc("hc_ffn")?,
                attn,
                moe,
            });
            progress(i + 1, n);
        }
        Ok(Weights {
            hidden: h,
            vocab: c.vocab,
            eps: d.eps,
            n_layers: n,
            c: d,
            embed,
            head,
            norm,
            layers,
        })
    }

    /// Bytes held on the host.
    pub fn bytes(&self) -> usize {
        fn lin(l: &Lin) -> usize {
            l.w.len() * 4
        }
        let mut t = (self.embed.len() + self.head.len() + self.norm.len()) * 4;
        for l in &self.layers {
            t += lin(&l.attn.wq_a)
                + lin(&l.attn.wq_b)
                + lin(&l.attn.wkv)
                + lin(&l.attn.wo_a)
                + lin(&l.attn.wo_b)
                + lin(&l.moe.gate)
                + lin(&l.moe.shared.0)
                + lin(&l.moe.shared.1)
                + lin(&l.moe.shared.2)
                + (l.attn.q_norm.len() + l.attn.kv_norm.len() + l.attn.sink.len()) * 4;
            for (a, b, c) in &l.moe.experts {
                t += lin(a) + lin(b) + lin(c);
            }
        }
        t
    }
}

fn load_lin(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Lin> {
    Ok(Lin {
        n,
        k,
        w: load_matrix(src, name, n, k)?,
    })
}

/// `load_matrix` for a tensor whose name has no `.weight` suffix (the hyper-connection projection).
fn load_mat(src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Vec<f32>> {
    load_matrix(src, name, n, k)
}

// ------------------------------------------------------------------------------------------------------------------
// The graph.
// ------------------------------------------------------------------------------------------------------------------

/// What attention layers hand down the stack: the last source's compressed KV, index keys, candidate mask and top-k
/// lists. Layers run in order and a source always precedes its consumers, so one slot each is enough.
#[derive(Default)]
struct Shared {
    compress_kv: Vec<f32>,
    compress_len: usize,
    index_k: Vec<f32>,
    index_len: usize,
    candidates: Option<Vec<bool>>,
    candidate_width: usize,
    topk: Option<Vec<i32>>,
}

/// The hyper-connection coefficients of one sub-layer.
struct Mixes {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl Hc {
    fn mixes(&self, h: &[f32], t: usize, hidden: usize, hc: usize, iters: usize, eps: f32) -> Mixes {
        let hd = hc * hidden;
        let mut flat = vec![0f32; t * hd];
        for i in 0..t {
            flat[i * hd..(i + 1) * hd].copy_from_slice(&h[i * hd..(i + 1) * hd]);
        }
        let mix_n = (2 + hc) * hc;
        let mut mixes = linear(&flat, t, hd, &self.f, mix_n, Some(&self.base));
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
fn hc_pre(h: &[f32], pre: &[f32], t: usize, hc: usize, hidden: usize) -> Vec<f32> {
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

/// Expand a sub-layer output back over the streams and mix the residual in.
fn hc_post(x: &[f32], res: &[f32], post: &[f32], comb: &[f32], t: usize, hc: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * hc * hidden];
    for i in 0..t {
        for c in 0..hc {
            let p = post[i * hc + c];
            for d in 0..hidden {
                let mut v = p * x[i * hidden + d];
                for e in 0..hc {
                    v += comb[(i * hc + c) * hc + e] * res[(i * hc + e) * hidden + d];
                }
                out[(i * hc + c) * hidden + d] = v;
            }
        }
    }
    out
}

/// RoPE in place on the last `rope` values of every row of an `[rows, row]` matrix (interleaved pairs).
fn rope_rows(
    x: &mut [f32],
    rows: usize,
    row: usize,
    rope: usize,
    theta: f32,
    pos: impl Fn(usize) -> f32,
    inverse: bool,
) {
    let half = rope / 2;
    let inv: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f32 / rope as f32))
        .collect();
    for i in 0..rows {
        let p = pos(i);
        let base = i * row + row - rope;
        for j in 0..half {
            let angle = p * inv[j];
            let (s, c) = (angle.sin(), angle.cos());
            let s = if inverse { -s } else { s };
            let (a, b) = (x[base + 2 * j], x[base + 2 * j + 1]);
            x[base + 2 * j] = a * c - b * s;
            x[base + 2 * j + 1] = b * c + a * s;
        }
    }
}

/// The same on the tail of every head of a `[t, heads, head_dim]` tensor: the reference rotates `x[..., -rope:]`.
fn rope_heads(x: &mut [f32], t: usize, heads: usize, head_dim: usize, rope: usize, theta: f32, inverse: bool) {
    rope_rows(x, t * heads, head_dim, rope, theta, |i| (i / heads) as f32, inverse);
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
fn sparse_attn(
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
        self.forward_with(ids, |_, _| Ok(()))
    }

    /// The same, calling `hook(name, values)` after each stage the reference dumps.
    pub fn forward_with(&self, ids: &[u32], mut hook: impl FnMut(&str, &[f32]) -> Result<()>) -> Result<Vec<f32>> {
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
        for (layer, l) in self.layers.iter().enumerate() {
            // Attention: its mixes come from the streams, the pre-mix from the previous sub-layer.
            let am = l.hc_attn.mixes(&h, t, h_dim, hc, d.hc_iters, d.hc_eps);
            let x = hc_pre(&h, &pre_mix, t, hc, h_dim);
            let mut xn = x.clone();
            rmsnorm(&mut xn, h_dim, &l.attn_norm, self.eps);
            let attn_out = self.attention(l, &xn, t, &mut shared, &mut hook, layer)?;
            h = hc_post(&attn_out, &h, &am.post, &am.comb, t, hc, h_dim);
            hook(&format!("attn{layer}"), &attn_out)?;

            // FFN: its own mixes, and the pre-mix the attention just produced.
            let fm = l.hc_ffn.mixes(&h, t, h_dim, hc, d.hc_iters, d.hc_eps);
            let x = hc_pre(&h, &am.pre, t, hc, h_dim);
            let mut xn = x.clone();
            rmsnorm(&mut xn, h_dim, &l.ffn_norm, self.eps);
            let ffn_out = moe(&l.moe, &xn, t, h_dim, d, None)?;
            h = hc_post(&ffn_out, &h, &fm.post, &fm.comb, t, hc, h_dim);
            hook(&format!("ffn{layer}"), &ffn_out)?;
            pre_mix = fm.pre;
            hook(&format!("block{layer}"), &h)?;
        }
        let x = hc_pre(&h, &pre_mix, t, hc, h_dim);
        let mut x = x;
        rmsnorm(&mut x, h_dim, &self.norm, self.eps);
        let logits = linear(&x, t, h_dim, &self.head, self.vocab, None);
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: &Layer,
        x: &[f32],
        t: usize,
        shared: &mut Shared,
        hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let d = &self.c;
        let a = &l.attn;
        let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
        // The window layers rotate at the plain base, the compressed ones at the longer one.
        let theta = if a.ratio > 0 {
            d.rope_params.compress_theta
        } else {
            d.rope_params.theta
        };
        let mut qr = a.wq_a.mat(x, t);
        rmsnorm(&mut qr, a.q_lora, &a.q_norm, self.eps);
        let mut q = a.wq_b.mat(&qr, t);
        rope_heads(&mut q, t, heads, hd, rd, theta, false);
        let mut kv = a.wkv.mat(x, t);
        rmsnorm(&mut kv, hd, &a.kv_norm, self.eps);
        rope_rows(&mut kv, t, hd, rd, theta, |i| i as f32, false);
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
                    rope_rows(&mut rotated, groups, hd, rd, theta, |i| (i * a.ratio) as f32, false);
                    shared.compress_kv = rotated;
                    shared.compress_len = groups;
                    hook(&format!("ckv{layer}"), &shared.compress_kv)?;
                    Some(cidxs)
                }
                None => {
                    // Not a source: read what the source with this ratio published.
                    let cidxs = match &a.indexer {
                        Some(ix) => indexer(ix, x, &qr, None, 0, t, hd, rd, d, shared, self.eps, hook, layer)?,
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
        let out = sparse_attn(&q, &kv_all, n_kv, &a.sink, &idx, t, m, heads, hd);
        Ok(self.out_projection(a, out, t))
    }

    /// The grouped output projection: `wo_a` is block-diagonal over the head groups, then `wo_b`.
    fn out_projection(&self, a: &Attn, mut o: Vec<f32>, t: usize) -> Vec<f32> {
        let (heads, hd, rd) = (a.heads, a.head_dim, a.rope);
        let theta = if a.ratio > 0 {
            self.c.rope_params.compress_theta
        } else {
            self.c.rope_params.theta
        };
        rope_heads(&mut o, t, heads, hd, rd, theta, true);
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

fn identity_mix(t: usize, hc: usize) -> Vec<f32> {
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
    d: &DeepseekConfig,
    shared: &mut Shared,
    eps: f32,
    hook: &mut impl FnMut(&str, &[f32]) -> Result<()>,
    layer: usize,
) -> Result<Vec<i32>> {
    let (heads, dim) = (ix.heads, ix.dim);
    let theta = d.rope_params.compress_theta;
    if ix.owns_k {
        let rows = groups;
        let mut k = match (&ix.wk, latent) {
            (Some(wk), Some(lat)) => wk.mat(lat, rows),
            _ => bail!("an index-key owner without a latent"),
        };
        rmsnorm(&mut k, dim, ix.k_norm.as_ref().expect("the owner has a key norm"), eps);
        rope_rows(&mut k, rows, dim, rd, theta, |i| (i * ix.ratio) as f32, false);
        shared.index_k = k;
        shared.index_len = rows;
        hook(&format!("ixk{layer}"), &shared.index_k)?;
    }
    let n_ix = shared.index_len;
    ensure!(n_ix > 0, "the indexer has no keys");
    let mut q = ix.wq_b.mat(qr, t);
    rope_heads(&mut q, t, heads, dim, rd, theta, false);
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

/// The FFN: `top_k` routed experts plus the shared one, both under the SwiGLU clamps.
fn moe(
    m: &Moe,
    x: &[f32],
    t: usize,
    hidden: usize,
    d: &DeepseekConfig,
    image_mask: Option<&[bool]>,
) -> Result<Vec<f32>> {
    let experts = m.experts.len();
    let mut scores = linear(x, t, hidden, &m.gate.w, experts, None);
    for v in scores.iter_mut() {
        *v = (v.exp() + 1.0).ln().sqrt(); // sqrt(softplus(x))
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
        let top: Vec<usize> = order.iter().take(d.top_k).copied().collect();
        let mut w: Vec<f32> = top.iter().map(|&e| scores[i * experts + e]).collect();
        if d.norm_topk_prob && d.top_k > 1 {
            let s: f32 = w.iter().sum::<f32>() + 1e-20;
            for v in w.iter_mut() {
                *v /= s;
            }
        }
        for v in w.iter_mut() {
            *v *= d.route_scale;
        }
        let xi = &x[i * hidden..(i + 1) * hidden];
        for (slot, &e) in top.iter().enumerate() {
            let (w1, w2, w3) = &m.experts[e];
            let e_out = expert(w1, w2, w3, xi, d.swiglu_limit);
            for (o, v) in out[i * hidden..(i + 1) * hidden].iter_mut().zip(&e_out) {
                *o += w[slot] * v;
            }
        }
        let s_out = expert(&m.shared.0, &m.shared.1, &m.shared.2, xi, d.swiglu_limit);
        for (o, v) in out[i * hidden..(i + 1) * hidden].iter_mut().zip(&s_out) {
            *o += v;
        }
    }
    Ok(out)
}

fn expert(w1: &Lin, w2: &Lin, w3: &Lin, x: &[f32], limit: f32) -> Vec<f32> {
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

    /// The text graph against `scripts/deepseek_ref.py`: every stage the reference dumps, on the same weights.
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
            .forward_with(&ids, |name, values| {
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

        // Every smooth stage has to agree, and the whole logits matrix. The selection lists are exact where the
        // indexer's scores are not within f32 noise (layers 2 and 4); at the other index sources one slot can flip
        // on a near-tie, and because the filler positions are real keys the layer's output moves a little with it.
        // That is the order of the dot products, not the graph: the gates below say so.
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
}
