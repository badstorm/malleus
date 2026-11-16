use crate::kernels::Kernels;
use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{Arg, DeviceBuffer, PinnedBuffer, Stream};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub inter: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub lin_nk: usize,
    pub lin_nv: usize,
    pub lin_dk: usize,
    pub lin_dv: usize,
    pub conv_k: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub rot_dim: usize,
    pub mrope_section: [usize; 3],
    pub full_attn: Vec<bool>,
    pub eos: Vec<u32>,
    /// Sparse MoE FFN in every layer (Qwen3.5-MoE); `inter` is then 0.
    pub moe: Option<MoeConfig>,
    /// Gated hyper-connections (Qwen3.8-Flash-Next): `count` residual streams replace the residual and the
    /// pre-norms.
    pub hc: Option<HcConfig>,
    /// Per-layer embedding of hashed n-grams added to the streams before one layer.
    pub ple: Option<PleConfig>,
    /// Token indexer of the sparse attention (QSA).
    pub indexer: Option<IndexerConfig>,
    /// DeltaNet output gate sigmoid(z) instead of silu(z).
    pub gate_sigmoid: bool,
    /// Which graph the checkpoint has: Qwen3.5 and its MoE variant share one, Ling-3.0-flash has another.
    pub family: Family,
    /// Mixer per layer. `full_attn` is the Qwen-facing view of the same information.
    pub kind: Vec<MixerKind>,
    /// MLA geometry (Ling): the cache holds the latent, K and V are absorbed from it.
    pub mla: Option<MlaConfig>,
    /// KDA decay floor (`kda_lower_bound`): the gate is `lb * sigmoid(raw)`, with `lb` negative.
    pub kda_lower_bound: f32,
    /// RMSNorm weight multiplies by `1 + w` (Qwen) or by `w` (Ling). Folded at load, see ADR 0005.
    pub norm_one_plus_w: bool,
    /// 3-section M-RoPE (Qwen) or plain RoPE on `rot_dim` (Ling).
    pub mrope: bool,
    /// Leading layers with a dense FFN (Ling: `first_k_dense_replace`); the rest use `moe`.
    pub dense_ffn: usize,
    /// Per-layer SwiGLU clamp of Ling-3.0-flash, routed and shared (0 = no clamp). ADR 0004.
    pub swiglu: Option<Swiglu>,
}

/// The graphs Malleus knows. `Ling3` is the one with different mixers (KDA and MLA), see ADR 0005.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Qwen35,
    Qwen4Exp,
    Ling3,
}

impl Family {
    pub fn from_model_type(t: &str) -> Option<Family> {
        Some(match t {
            "qwen3_5" | "qwen3_5_moe" => Family::Qwen35,
            "qwen4_exp" => Family::Qwen4Exp,
            "bailing_hybrid" => Family::Ling3,
            _ => return None,
        })
    }
}

/// The mixer of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    /// Gated attention (Qwen), with the optional QSA indexer.
    Attn,
    /// Gated DeltaNet (Qwen): one scalar decay per value head.
    Delta,
    /// Kimi Delta Attention (Ling): a decay per key channel.
    Kda,
    /// Multi-head latent attention (Ling): the cache holds a 512-wide latent plus 64 rotated dims.
    Mla,
}

/// MLA geometry, from `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim` and `v_head_dim`.
#[derive(Clone, Copy, Debug)]
pub struct MlaConfig {
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v: usize,
}

/// Per-layer SwiGLU clamp, one limit for the routed experts and one for the shared expert (0 = off).
#[derive(Clone, Debug)]
pub struct Swiglu {
    pub routed: Vec<i32>,
    pub shared: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct HcConfig {
    pub count: usize,
    pub rank: usize,
}

#[derive(Clone, Debug)]
pub struct PleConfig {
    /// Layer index (0-based) whose input receives the embedding.
    pub layer: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub dim: usize,
    pub conv_k: usize,
    /// End-of-text id: n-grams do not cross it.
    pub eos: u32,
}

impl PleConfig {
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    /// Rows of normalized inputs the dilated conv keeps between calls (dilation = ngram).
    pub fn conv_state(&self) -> usize {
        (self.conv_k - 1) * self.ngram
    }
}

#[derive(Clone, Debug)]
pub struct IndexerConfig {
    pub heads: usize,
    pub dim: usize,
    pub budget: usize,
    pub ratio: usize,
}

#[derive(Clone, Debug)]
pub struct MoeConfig {
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    /// Router scores are sigmoid (Ling, `score_function: sigmoid`) instead of softmax.
    pub sigmoid: bool,
    /// Group top-k of `noaux_tc`: `topk_group` of `n_group` groups are kept, and the expert scores that decide
    /// the groups carry `expert_bias`. Qwen has neither: one group, no bias.
    pub n_group: usize,
    pub topk_group: usize,
    pub bias: bool,
    /// Multiplier applied to the renormalized weights (Ling: 2,5).
    pub scaling: f32,
    /// The shared expert is gated by a learned scalar (Qwen) or added with weight 1 (Ling).
    pub shared_gate: bool,
}

impl MoeConfig {
    /// Routed experts per token plus the shared one.
    pub fn slots(&self) -> usize {
        self.top_k + 1
    }

    /// Whether the router selects through groups (`noaux_tc`, Ling). `n_group` defaults to 1 for families that
    /// have no groups at all — Qwen parses 1 — so "more than one group" is what distinguishes them, and the
    /// group scores are defined on the expert bias, so a config asking for groups without one is a config error.
    pub fn group_routing(&self) -> bool {
        self.n_group > 1
    }
}

impl Config {
    pub fn from_hf(model_dir: &Path) -> Result<Self> {
        let cfg = std::fs::read_to_string(model_dir.join("config.json")).context("config.json")?;
        let gen_cfg = std::fs::read_to_string(model_dir.join("generation_config.json")).ok();
        Self::from_json(&cfg, gen_cfg.as_deref())
    }

    pub fn from_json(config: &str, generation_config: Option<&str>) -> Result<Self> {
        let cfg: Value = serde_json::from_str(config)?;
        // Ling-3.0-flash keeps its config at the top level; the Qwen checkpoints nest it under text_config.
        let family = Family::from_model_type(cfg["model_type"].as_str().unwrap_or(""))
            .or_else(|| Family::from_model_type(cfg["text_config"]["model_type"].as_str().unwrap_or("")))
            .context("config: unknown or missing model_type")?;
        let ling = family == Family::Ling3;
        let t = if ling { &cfg } else { &cfg["text_config"] };
        let u = |k: &str| {
            t[k].as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("config: {k}"))
        };
        let rope = &t["rope_parameters"];
        let head_dim = u("head_dim")?;
        // Ling spells the rope settings as plain keys, Qwen inside rope_parameters.
        let prf = rope["partial_rotary_factor"]
            .as_f64()
            .or_else(|| t["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0);
        let theta = rope["rope_theta"]
            .as_f64()
            .or_else(|| t["rope_theta"].as_f64())
            .context("config: rope_theta")? as f32;
        let (mrope_section, mrope) = if ling {
            ([0usize, 0, 0], false)
        } else {
            let sec: Vec<usize> = rope["mrope_section"]
                .as_array()
                .context("mrope_section")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            ensure!(sec.len() == 3, "mrope_section must have 3 entries");
            ensure!(
                rope["mrope_interleaved"].as_bool() == Some(true),
                "only interleaved M-RoPE supported"
            );
            ([sec[0], sec[1], sec[2]], true)
        };
        // Ling has no layer_types: the rule is in the code of the checkpoint, MLA every `layer_group_size`
        // (plus a final partial group, which 42 layers never produce) and KDA elsewhere.
        let n_layers = u("num_hidden_layers")?;
        let kind: Vec<MixerKind> = if ling {
            let grp = t["layer_group_size"].as_u64().unwrap_or(6) as usize;
            let tail = n_layers / grp * grp;
            (0..n_layers)
                .map(|i| {
                    if (i + 1) % grp == 0 || i >= tail {
                        MixerKind::Mla
                    } else {
                        MixerKind::Kda
                    }
                })
                .collect()
        } else {
            t["layer_types"]
                .as_array()
                .context("layer_types")?
                .iter()
                .map(|v| {
                    if v.as_str() == Some("full_attention") {
                        MixerKind::Attn
                    } else {
                        MixerKind::Delta
                    }
                })
                .collect()
        };
        let full_attn: Vec<bool> = kind.iter().map(|k| *k == MixerKind::Attn).collect();
        let mut eos = vec![];
        if let Some(g) = generation_config {
            let g: Value = serde_json::from_str(g)?;
            match &g["eos_token_id"] {
                Value::Array(a) => eos.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32)),
                Value::Number(n) => eos.extend(n.as_u64().map(|v| v as u32)),
                _ => {}
            }
        }
        let moe = match t["num_experts"].as_u64() {
            Some(_) => {
                let inter = u("moe_intermediate_size")?;
                let shared = t["shared_expert_intermediate_size"]
                    .as_u64()
                    .or_else(|| t["moe_shared_expert_intermediate_size"].as_u64())
                    .context("config: shared_expert_intermediate_size")? as usize;
                ensure!(shared == inter, "shared expert size must match the routed experts");
                Some(MoeConfig {
                    experts: u("num_experts")?,
                    top_k: u("num_experts_per_tok")?,
                    inter,
                    sigmoid: t["score_function"].as_str() == Some("sigmoid"),
                    n_group: t["n_group"].as_u64().unwrap_or(1) as usize,
                    topk_group: t["topk_group"].as_u64().unwrap_or(1) as usize,
                    bias: t["moe_router_enable_expert_bias"].as_bool().unwrap_or(false),
                    scaling: t["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
                    shared_gate: !ling,
                })
            }
            None => None,
        };
        let qwen4 = t["model_type"].as_str() == Some("qwen4_exp_text");
        let hc = match t["hc_count"].as_u64() {
            Some(n) if qwen4 => Some(HcConfig {
                count: n as usize,
                rank: u("hc_lowrank")?,
            }),
            _ => None,
        };
        let ple = match t["ple_layer_ids"].as_array() {
            Some(ids) if qwen4 && !ids.is_empty() => {
                ensure!(ids.len() == 1, "only one PLE layer is supported");
                let eos = match &t["eos_token_id"] {
                    Value::Array(a) => a.first().and_then(|v| v.as_u64()),
                    v => v.as_u64(),
                }
                .context("config: eos_token_id")?;
                Some(PleConfig {
                    layer: ids[0].as_u64().context("ple_layer_ids")? as usize - 1,
                    ngram: u("ngram_size")?,
                    heads_per_ngram: u("heads_per_ngram")?,
                    dim: t["ple_embed_dim"].as_u64().map_or(u("hidden_size")?, |v| v as usize),
                    conv_k: u("ple_conv_kernel_size")?,
                    eos: eos as u32,
                })
            }
            _ => None,
        };
        let indexer = match t["indexer_n_heads"].as_u64() {
            Some(h) if qwen4 => Some(IndexerConfig {
                heads: h as usize,
                dim: u("indexer_head_dim")?,
                budget: u("indexer_budget")?,
                ratio: u("indexer_compress_ratio")?,
            }),
            _ => None,
        };
        let mla = if ling {
            Some(MlaConfig {
                kv_lora: u("kv_lora_rank")?,
                nope: u("qk_nope_head_dim")?,
                rope: u("qk_rope_head_dim")?,
                v: u("v_head_dim")?,
            })
        } else {
            None
        };
        let swiglu = match (
            t["expert_swiglu_limit_list"].as_array(),
            t["share_expert_swiglu_limit_list"].as_array(),
        ) {
            (Some(r), Some(s)) if ling => Some(Swiglu {
                routed: r.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect(),
                shared: s.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect(),
            }),
            _ => None,
        };
        let c = Config {
            hidden: u("hidden_size")?,
            n_layers,
            // Ling keeps a dense FFN in the first layers and a MoE in the rest: both sizes are live.
            inter: if moe.is_some() && !ling {
                0
            } else {
                u("intermediate_size")?
            },
            vocab: u("vocab_size")?,
            n_heads: u("num_attention_heads")?,
            n_kv: u("num_key_value_heads")?,
            head_dim,
            // KDA reads the same lin_* fields DeltaNet does: 32 heads of 128, conv 4.
            lin_nk: if ling {
                u("num_attention_heads")?
            } else {
                u("linear_num_key_heads")?
            },
            lin_nv: if ling {
                u("num_attention_heads")?
            } else {
                u("linear_num_value_heads")?
            },
            lin_dk: if ling { head_dim } else { u("linear_key_head_dim")? },
            lin_dv: if ling { head_dim } else { u("linear_value_head_dim")? },
            conv_k: if ling {
                u("short_conv_kernel_size")?
            } else {
                u("linear_conv_kernel_dim")?
            },
            eps: t["rms_norm_eps"].as_f64().context("rms_norm_eps")? as f32,
            rope_theta: theta,
            rot_dim: (head_dim as f64 * prf) as usize,
            mrope_section,
            full_attn,
            eos,
            moe,
            hc,
            ple,
            indexer,
            gate_sigmoid: t["output_gate_type"].as_str() == Some("sigmoid"),
            family,
            kind,
            mla,
            kda_lower_bound: t["kda_lower_bound"].as_f64().unwrap_or(0.0) as f32,
            norm_one_plus_w: !ling,
            mrope,
            dense_ffn: if ling { u("first_k_dense_replace")? } else { 0 },
            swiglu,
        };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.full_attn.len() == self.n_layers && self.kind.len() == self.n_layers,
            "layer types length mismatch"
        );
        ensure!(self.conv_k == 4, "reference kernels require conv kernel 4");
        ensure!(
            self.n_heads.is_multiple_of(self.n_kv) && self.lin_nv.is_multiple_of(self.lin_nk),
            "bad head grouping"
        );
        ensure!(
            self.hidden.is_multiple_of(8) && self.inter.is_multiple_of(8),
            "dims must be multiples of 8"
        );
        match self.family {
            Family::Ling3 => {
                let m = self.mla.context("config: MLA geometry")?;
                ensure!(
                    self.kind.contains(&MixerKind::Mla) && self.kind.contains(&MixerKind::Kda),
                    "Ling needs both KDA and MLA layers"
                );
                ensure!(
                    m.nope == self.head_dim && m.v == self.head_dim,
                    "MLA nope and value dims must agree with head_dim"
                );
                ensure!(
                    m.kv_lora.is_multiple_of(64) && m.rope == 64,
                    "MLA kernels need a latent multiple of 64 and 64 rotated dims"
                );
                ensure!(
                    self.lin_dk == 128 && self.lin_dv == 128,
                    "KDA kernels require head dim 128"
                );
                ensure!(self.kda_lower_bound < 0.0, "the KDA gate floor must be negative");
                let s = self.swiglu.as_ref().context("config: swiglu limits")?;
                ensure!(
                    s.routed.len() == self.n_layers && s.shared.len() == self.n_layers,
                    "swiglu limit lists must cover every layer"
                );
                ensure!(self.dense_ffn <= self.n_layers, "first_k_dense_replace too large");
            }
            _ => {
                ensure!(self.head_dim == 256, "reference kernels require head_dim 256");
                ensure!(
                    self.lin_dk == 128 && self.lin_dv == 128,
                    "reference kernels require DeltaNet head dim 128"
                );
                ensure!(self.mrope, "only interleaved M-RoPE supported");
            }
        }
        if let Some(m) = &self.moe {
            ensure!(
                m.inter.is_multiple_of(128) && self.hidden.is_multiple_of(128),
                "MoE kernels need an expert size and hidden multiple of 128"
            );
            ensure!(m.top_k < 64 && m.experts <= 1 << 20, "unsupported MoE routing");
            ensure!(
                m.n_group == 1 || (m.experts.is_multiple_of(m.n_group) && m.topk_group <= m.n_group),
                "group routing needs groups of equal size"
            );
        }
        if let Some(h) = &self.hc {
            ensure!(h.count >= 2 && h.rank > 0, "bad hyper-connections");
        }
        if let Some(p) = &self.ple {
            ensure!(
                self.hc.is_some() && p.layer < self.n_layers,
                "PLE needs hyper-connections"
            );
            ensure!(p.dim.is_multiple_of(p.heads()) && p.ngram >= 2, "bad PLE config");
        }
        Ok(())
    }

    /// Width of the hyper-connection stream state (count * hidden), or hidden without hyper-connections.
    pub fn stream_dim(&self) -> usize {
        self.hc.as_ref().map_or(self.hidden, |h| h.count * self.hidden)
    }

    pub fn lin_qkv_dim(&self) -> usize {
        self.lin_nk * self.lin_dk * 2 + self.lin_nv * self.lin_dv
    }
}

pub(crate) fn pad128(n: usize) -> usize {
    n.next_multiple_of(128)
}

const WMMA_MIN_TOKENS: usize = 8;
/// Up to this many tokens the decode weights are read once by a GEMV for all tokens (speculative verify).
pub const GEMV_MAX_T: usize = 6;
/// MLA prefill is absorbed and projected in tiles of this many tokens (the qt/lat buffers are sized for it).
const MLA_PREFILL_TILE: usize = 2048;

/// Context length of the fp16 KV cache rows (V is stored transposed, padded so 32-key tiles stay in bounds).
pub(crate) fn kv_ctx(max_ctx: usize) -> usize {
    max_ctx.next_multiple_of(32)
}

/// int8 GEMM variant for t tokens: (kernel, token tile, column tile).
fn i8_kernel(t: usize, glu: bool) -> (&'static str, usize, usize) {
    match (t <= 128, glu) {
        (true, false) => ("gemm_i8_m128", 128, 128),
        (false, false) => ("gemm_i8", 256, 64),
        (true, true) => ("gemm_i8_glu_m128", 128, 128),
        (false, true) => ("gemm_i8_glu", 256, 64),
    }
}

pub enum Weights {
    Bf16(DeviceBuffer),
    Mls4 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    /// 3-bit codes with an fp16 scale per group of 128 (Ling's routed gate/up, docs/model-ling3-flash.md).
    Mls3 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
    Mls8 {
        q: DeviceBuffer,
        sc: DeviceBuffer,
    },
}

pub struct Linear {
    pub w: Weights,
    /// Optional MLS8 copy used for prefill (T > 1).
    pub prefill: Option<(DeviceBuffer, DeviceBuffer)>,
    pub n: usize,
    pub k: usize,
}

enum Mixer {
    Attn {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        q_norm: DeviceBuffer,
        k_norm: DeviceBuffer,
        indexer: Option<Indexer>,
        /// Decode: q | k | v | indexer projection in one GEMV.
        dec: Option<Merged>,
    },
    Delta {
        qkv: Linear,
        z: Linear,
        a: Linear,
        b: Linear,
        out: Linear,
        conv: DeviceBuffer,
        a_log: DeviceBuffer,
        dt_bias: DeviceBuffer,
        norm: DeviceBuffer,
        /// Decode: qkv | z in one GEMV, a | b (bf16) in another.
        dec: Option<Merged>,
        ab: Option<Linear>,
    },
    /// Kimi Delta Attention (Ling-3.0-flash): three projections with their own short conv and l2norm, a decay gate
    /// per key channel (`f_proj`), a per-head update gate (`b_proj`) and a gated RMSNorm on the output. The
    /// recurrence is `kda_scan`; the conv, the norms and the gate reuse the DeltaNet passes with different shapes.
    ///
    /// `allow(dead_code)`: the weights load (increment I2) and the forward lands with I4..I7; until then nothing
    /// reads them, and the loader is what proves the tensor mapping and the memory footprint are right.
    #[allow(dead_code)]
    Kda {
        q: Linear,
        k: Linear,
        v: Linear,
        f: Linear,
        g: Linear,
        o: Linear,
        b: Linear,
        /// The q, k and v convolutions concatenated in that order, [3 * heads * dim, conv_k] bf16: DeltaNet has
        /// one fused conv because Qwen ships one fused projection, Ling has three, and `kda_conv_norm` walks them
        /// as one buffer so `conv_state_save` can be called per stream with an offset.
        conv: DeviceBuffer,
        a_log: DeviceBuffer,
        dt_bias: DeviceBuffer,
        o_norm: DeviceBuffer,
    },
    /// Multi-head latent attention (Ling-3.0-flash): the cache holds the normalized latent (`kv_lora` dims) and
    /// the 64 rotated dims; K and V come from `kv_b_proj`, which decode absorbs.
    #[allow(dead_code)] // as above: loaded by I2, read by I4
    Mla {
        q: Linear,
        kv_a: Linear,
        kv_a_norm: DeviceBuffer,
        /// [heads * (nope + v), kv_lora] fp32, dequantized at load: decode absorbs it into the query, which is a
        /// product against its transpose, and a quantized matrix cannot be read that way.
        kv_b: DeviceBuffer,
        /// Per-head output gate, [heads, hidden], never quantized.
        gate: Linear,
        dense: Linear,
    },
}

/// QSA token indexer: q (heads x dim) | k (1 x dim) projection and their RMSNorms.
struct Indexer {
    qk: Linear,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

/// Gated hyper-connection: per-stream RMSNorm, [down; inject] (rows padded to 128) and up (K padded to 128). MLS8
/// `up` rows are reordered at load to j * count + s (stream s of output j), so one GEMM tile holds every stream of
/// its outputs and the mix runs in its epilogue.
struct Hc {
    norm: DeviceBuffer,
    down: Linear,
    up: Linear,
    up_inter: bool,
}

/// PLE layer: key (one per stream) and value projections of the n-gram embedding, norms and the dilated conv.
struct Ple {
    key: Linear,
    value: Linear,
    norm_key: DeviceBuffer,
    norm_query: DeviceBuffer,
    norm_conv: DeviceBuffer,
    conv: DeviceBuffer,
}

/// Steps of at most this many tokens read their n-gram rows on the calling thread (`NgramTable::embed`).
const NGRAM_INLINE_TOKENS: usize = 16;

/// Hashed n-gram embedding table read from the model file on the host (int8 rows + f32 scale, ngram_table).
struct NgramTable {
    file: std::sync::Arc<malleus_gguf::TensorFile>,
    /// Row reads are page-cache or disk reads: many in flight (decode reads 16 rows per token).
    pool: rayon::ThreadPool,
    rows: usize,
    /// Width of one head's row.
    dim: usize,
    mult: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    cfg: PleConfig,
}

impl NgramTable {
    fn load(src: &dyn TensorSource, base: &str, cfg: &PleConfig) -> Result<Self> {
        let name = format!("{base}ngram_table");
        let info = src.info(&name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::I8 && info.shape.len() == 2,
            "{name}: expected I8 [rows, dim + 4]"
        );
        let (rows, dim) = (info.shape[0], info.shape[1] - 4);
        ensure!(
            dim * cfg.heads() == cfg.dim,
            "{name}: row width {dim} for {} heads",
            cfg.heads()
        );
        let i64s = |n: &str, len: usize| -> Result<Vec<i64>> {
            let t = format!("{base}{n}");
            let info = src.info(&t).with_context(|| format!("missing tensor {t}"))?;
            ensure!(
                info.dtype == DType::I64 && info.numel() == len,
                "{t}: expected I64 [{len}]"
            );
            Ok(src
                .read_all(&t)?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect())
        };
        // Opened again so that the hint does not reach the loader's sequential reads: without it every 4 KB row
        // read brings in a readahead window that fills the page cache with rows nobody asked for.
        let file = std::sync::Arc::new(
            src.open_tensor(&name)
                .context("n-gram table: no file handle")?
                .reopened(),
        );
        if let Some((f, _, _)) = file.range(0, 0) {
            malleus_hip::file_hint::random(f);
        }
        if std::env::var("MALLEUS_NGRAM_WARM").is_ok_and(|v| v == "1") {
            // Read the table once in the background so the first lookups hit the page cache. Off by default: with the
            // weights in GTT the table (52 GB) does not fit in the remaining page cache, and streaming it evicts the
            // rows of frequent n-grams that plain lookups keep cached.
            let f = file.clone();
            std::thread::Builder::new().name("ngram-warm".into()).spawn(move || {
                let mut buf = vec![0u8; 8 << 20];
                let mut off = 0;
                while off < f.size {
                    let n = buf.len().min(f.size - off);
                    if f.read_at(off, &mut buf[..n]).is_err() {
                        break;
                    }
                    off += n;
                }
            })?;
        }
        let t = NgramTable {
            file,
            pool: rayon::ThreadPoolBuilder::new()
                // One blocking pread per thread, so this is the queue depth the table is read at: the rows are
                // scattered over 52 GB and the drive, not the CPU, sets the pace. The 94 400 row reads of a 5900
                // token prompt take 253 ms at 32 threads, 152 at 96 and 43 at 256. The stacks stay at rayon's
                // default: 64 KiB overflows inside the work-stealing and error paths.
                .num_threads(
                    std::env::var("MALLEUS_NGRAM_THREADS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or(256),
                )
                .thread_name(|i| format!("ngram-{i}"))
                .build()?,
            rows,
            dim,
            mult: i64s("layer_multipliers", cfg.ngram)?,
            sizes: i64s("ngram_heads_vocab_sizes", cfg.heads())?,
            offsets: i64s("ngram_heads_offsets", cfg.heads())?,
            cfg: cfg.clone(),
        };
        ensure!(
            t.sizes
                .iter()
                .zip(&t.offsets)
                .all(|(s, o)| *s > 0 && *o >= 0 && (o + s) as usize <= rows),
            "{name}: head ranges out of the table"
        );
        Ok(t)
    }

    /// Table row of every head for each token, given the `ngram - 1` tokens before them (`ctx`).
    fn row_ids(&self, ctx: &[u32], tokens: &[u32]) -> Vec<usize> {
        let c = &self.cfg;
        let hist: Vec<u32> = ctx.iter().chain(tokens).copied().collect();
        let mut ids = Vec::with_capacity(tokens.len() * c.heads());
        for p in ctx.len()..hist.len() {
            // Token s positions back, or eos when that crosses an eos (n-grams stay inside one document).
            let mut shifted = vec![hist[p] as i64; c.ngram];
            let mut crossed = false;
            for s in 1..c.ngram {
                crossed |= p < s || hist[p - s] == c.eos;
                shifted[s] = if crossed { c.eos as i64 } else { hist[p - s] as i64 };
            }
            for n in 2..=c.ngram {
                let mut mixed = shifted[0].wrapping_mul(self.mult[0]);
                for (j, sh) in shifted.iter().enumerate().take(n).skip(1) {
                    mixed ^= sh.wrapping_mul(self.mult[j]);
                }
                for h in 0..c.heads_per_ngram {
                    let head = (n - 2) * c.heads_per_ngram + h;
                    ids.push((mixed.rem_euclid(self.sizes[head]) + self.offsets[head]) as usize);
                }
            }
        }
        ids
    }

    /// Starts reading the rows of a decode step: called before the layers that precede the PLE are launched, so
    /// the drive works while the GPU does and `embed` finds the rows cached. A prefill is left to `embed`'s pool.
    fn prefetch(&self, ctx: &[u32], tokens: &[u32]) {
        if tokens.len() > NGRAM_INLINE_TOKENS {
            return;
        }
        let rb = self.dim + 4;
        for id in self.row_ids(ctx, tokens) {
            if let Some((f, offset, len)) = self.file.range(id * rb, rb) {
                malleus_hip::file_hint::will_need(f, offset, len);
            }
        }
    }

    /// Dequantized embeddings [tokens, heads * dim] (f32).
    fn embed(&self, ctx: &[u32], tokens: &[u32]) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let ids = self.row_ids(ctx, tokens);
        let (dim, rb) = (self.dim, self.dim + 4);
        let mut out = vec![0f32; ids.len() * dim];
        let file = &self.file;
        let rows = self.rows;
        if tokens.len() <= NGRAM_INLINE_TOKENS {
            // A decode step: its few rows were asked for by `prefetch`, and waking a pool of blocking readers
            // costs more than the reads (measured: 1-2 ms of idle GPU per token at the PLE layer).
            let mut buf = vec![0u8; rb];
            for (row, &id) in out.chunks_mut(dim).zip(&ids) {
                ensure!(id < rows, "n-gram row {id} out of range");
                file.read_at(id * rb, &mut buf)?;
                let s = f32::from_le_bytes([buf[dim], buf[dim + 1], buf[dim + 2], buf[dim + 3]]);
                for (d, &q) in row.iter_mut().zip(&buf[..dim]) {
                    *d = q as i8 as f32 * s;
                }
            }
            return Ok(out);
        }
        self.pool.install(|| {
            out.par_chunks_mut(dim)
                .zip(ids.par_iter())
                .with_min_len(4)
                .try_for_each_init(
                    || vec![0u8; rb],
                    |buf, (row, &id)| -> Result<()> {
                        ensure!(id < rows, "n-gram row {id} out of range");
                        file.read_at(id * rb, buf)?;
                        let s = f32::from_le_bytes([buf[dim], buf[dim + 1], buf[dim + 2], buf[dim + 3]]);
                        for (d, &q) in row.iter_mut().zip(&buf[..dim]) {
                            *d = q as i8 as f32 * s;
                        }
                        Ok(())
                    },
                )
        })?;
        Ok(out)
    }
}

struct Layer {
    /// Pre-norms (absent with hyper-connections, whose mix normalizes the streams).
    in_norm: Option<DeviceBuffer>,
    post_norm: Option<DeviceBuffer>,
    /// Hyper-connections around the mixer and the FFN.
    hc: Option<[Hc; 2]>,
    ple: Option<Ple>,
    mixer: Mixer,
    ffn: Ffn,
    /// Prefill int8 weights of the projections reading the layer input (q|k|v or qkv|z), fused into one GEMM.
    fin: Option<Fused>,
}

#[allow(clippy::large_enum_variant)]
enum Ffn {
    Dense {
        gate: Linear,
        up: Linear,
        down: Linear,
        /// Prefill int8 gate/up rows interleaved in blocks of 16: one GEMM producing silu(gate) * up.
        fgu: Option<Fused>,
    },
    Moe(Moe),
}

/// MLS4 experts with the shared expert appended as expert number `experts`.
struct Moe {
    /// bf16 [experts + 1, hidden]: router rows, then the shared expert gate.
    router: DeviceBuffer,
    /// [experts + 1, 2 * inter, hidden]: per expert gate rows then up rows. Absent with `MALLEUS_EXPERT_CACHE`:
    /// the experts stay in the file and `cache` decides where the kernels read them from.
    gu: Option<(DeviceBuffer, DeviceBuffer)>,
    /// [experts + 1, hidden, inter].
    down: Option<(DeviceBuffer, DeviceBuffer)>,
    /// The experts are not resident: their slots (decode) and the layer streamed into the staging (prefill).
    cache: Option<Box<crate::expert_cache::Layer>>,
    /// MLS8 copies of `gu` and `down` (int8 + fp32 row scales) for the grouped prefill GEMMs.
    pre: Option<[(DeviceBuffer, DeviceBuffer); 2]>,
    /// `expert_bias` of the group router (Ling, `moe_router_enable_expert_bias`): added to the sigmoid scores
    /// for the selection only, never for the weights. ADR: docs/model-ling3-flash.md, § MoE.
    #[allow(dead_code)] // read by `moe_route_g`, which lands with the Ling FFN wiring (I6)
    bias: Option<DeviceBuffer>,
    /// The shared expert of Ling: no gate, and possibly another weight format from the routed experts, so it
    /// lives in its own Linears and is added to the routed sum with weight 1.
    shared: Option<(Linear, Linear, Linear)>,
    /// Per-layer SwiGLU clamp of ADR 0004: `silu(gate).clamp(max = L) * up.clamp(-L, L)`, 0 = off. Routed and
    /// shared experts have their own list, and the dense FFN of the first layers is never clamped.
    swiglu: (f32, f32),
    /// The stacked experts are MLS3 (Ling) rather than MLS4 (Qwen): different kernels, because the 3-bit codes
    /// are packed 8 per 3 bytes and the core does not fold the input rotation. One flag per projection: a file can
    /// mix them (`ling3-flash-mls3.malleus` has MLS3 gate/up and MLS4 down, and reading the MLS4 rows with the MLS3
    /// stride is silent garbage).
    gu_mls3: bool,
    down_mls3: bool,
}

/// Dense prefill GEMMs on the MLS4 weights with int4 activations (MALLEUS_DENSE_W4A4=1): the int4 WMMA is double
/// rate on gfx11 and the 4-bit weights are already there, but the projections must stop being merged, because
/// `fuse` stacks the MLS8 copies and the merged kernel has no int4 form.
///
/// Opt-in, and the measurement that keeps it that way is on GPTQ-calibrated weights, so it is not the quality
/// argument that ruled it out before. Same model, same command, flag on against flag off: prefill +0.6/+1.9/+2.4 %
/// at 512/4290/17034 tokens, all-position argmax a tie (94.34 against 94.28 %), but last-token top-1 21/21 ->
/// 19/21 and worst KL 0.282 -> 0.435. Two first generated tokens out of 21 are not worth 2 % of prefill while
/// prefill is 20-30 % behind anyway. On round-to-nearest weights it is worse still (93.30 %, KL 1.751).
static ENV_DENSE_W4A4: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("MALLEUS_DENSE_W4A4").is_ok_and(|v| v == "1" || v == "acts" || v == "2" || v == "w4a8" || v == "3")
});

/// Whether the dense prefill GEMMs run W4A4. Read by the CLI to report the mode it chose.
pub fn dense_w4a4() -> bool {
    *ENV_DENSE_W4A4
}

/// `MALLEUS_DENSE_W4A4=acts` (or `2`): the same, extended to the projections whose input needs a transform
/// (`Act::Norm`, `GatedNorm`, `AttnGate` — on Flash-Next exactly `o_proj` and `out_proj`), which then fold it in
/// `rot_quant_i4_row` instead of staying on int8 and on the MLS8 copy of their weights. Measured on Flash-Next at
/// p32768, same binary: prefill **+5,6 %** at 24576 tokens and nothing at 8192, but the gate costs
/// **-1,58 punti** di argmax su tutte le posizioni (94,09 -> 92,51 %) e la KL peggiore raddoppia (0,465 -> 1,246).
/// That is the positive of the "twelve `o_proj` tensors to 8 bits" finding, on this model: without calibration these
/// two tensors are the sensitive ones. Opt-in, as the substrate for a calibrated conversion.
static ENV_DENSE_W4A4_ACTS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_DENSE_W4A4").is_ok_and(|v| v == "acts" || v == "2"));

pub fn dense_w4a4_acts() -> bool {
    *ENV_DENSE_W4A4_ACTS
}

/// `MALLEUS_DENSE_W4A4=w4a8` (or `3`): the same two projections on 4-bit weights, but with **int8** activations per
/// group folded in `rot_quant_i8_row`. What the MLS8 copies cost is the weight bytes, and those go away all the same;
/// the activation format is where the quality of the int4 variant was lost (measured: -1,58 punti di argmax).
static ENV_DENSE_W4A8_ACTS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_DENSE_W4A8_ACTS").map_or(true, |v| v != "0"));

pub fn dense_w4a8_acts() -> bool {
    *ENV_DENSE_W4A8_ACTS
}

/// Rows per tile of the grouped MoE GEMMs (moe_i8*).
pub(crate) const MOE_BM: usize = 64;
/// Average rows per expert from which the W4A8 grouped GEMMs use 128-row tiles (MALLEUS_MOE_BM128 overrides).
/// `MALLEUS_TRACE_ROUTE=<path>`: append the experts the router picked, one line per MoE call and token. It is the
/// input of the expert-cache measurements (scripts/expert_cache.py): how skewed the routing is, how much of the
/// working set a cache of N GiB would hit, and how much a token would ask the disk. Debug: it synchronizes the
/// stream once per layer, so a trace run is slower than a normal one.
static ENV_TRACE_ROUTE: std::sync::LazyLock<Option<String>> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_TRACE_ROUTE").ok());

/// Appends `layer call row id:w ...` for every (token, slot) of one MoE routing decision. The shared expert comes
/// out with id `experts` (Ling): the cache measures the routed ones, so the reader drops ids past the last expert.
fn trace_routes(st: &Stream, ms: &MoeScratch, layer: usize, t: usize, slots: usize) -> Result<()> {
    let Some(path) = ENV_TRACE_ROUTE.as_deref() else {
        return Ok(());
    };
    st.synchronize()?;
    let ids: Vec<i32> = ms.ids.download(t * slots)?;
    let w: Vec<f32> = ms.w.download(t * slots)?;
    let mut text = String::new();
    for row in 0..t {
        text.push_str(&format!("{layer} {t} {row}"));
        for slot in 0..slots {
            let j = row * slots + slot;
            text.push_str(&format!(" {}:{:.6}", ids[j], w[j]));
        }
        text.push('\n');
    }
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(text.as_bytes())?;
    Ok(())
}

/// `MALLEUS_HIDDEN_DUMP=<dir>`: for every attention layer, writes the layer input and the K of the chunk being
/// processed. Diagnostics for #96: it is the material with which one tests whether a block's sensitivity can be
/// learnt. It synchronizes at every layer, so a run with this variable set is very slow: not a path to leave on.
static ENV_HIDDEN_DUMP: std::sync::LazyLock<Option<String>> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_HIDDEN_DUMP").ok());

static MOE_BM128_ROWS_ENV: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("MALLEUS_MOE_BM128")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48)
});

fn load_moe(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    m: &MoeConfig,
    pre: &str,
    layer: usize,
    cache_budget: usize,
) -> Result<Moe> {
    let (e, h, i) = (m.experts, c.hidden, m.inter);
    let l = |s: &str| format!("{pre}mlp.{s}");
    // Upstream spells the shared expert `shared_expert` in Qwen and `shared_experts` in Ling.
    let se = if c.family == Family::Ling3 {
        "shared_experts"
    } else {
        "shared_expert"
    };
    // Zeroed because the row of the shared expert must be zero when there is no gate to load into it: `into`
    // writes only the first e rows. (An `alloc` plus a `memset` of the whole buffer here erases the router.)
    let mut router = DeviceBuffer::zeroed((e + 1) * h * 2)?;
    up.into(src, &l("gate.weight"), DType::BF16, &[e, h], &mut router, 0)?;
    if m.shared_gate {
        up.into(
            src,
            &l("shared_expert_gate.weight"),
            DType::BF16,
            &[1, h],
            &mut router,
            e * h * 2,
        )?;
    }
    // Ling's shared expert has no gate: its row stays zero and `moe_route_g` gives the slot weight 1.
    let bias = match m.bias {
        true => Some(up.typed(src, &l("gate.expert_bias"), DType::F32, &[e])?),
        false => None,
    };
    // `MALLEUS_EXPERT_CACHE`: the experts stay in the file. The int8 prefill copies are not loaded either (the
    // decode copies are what the cache streams), so the prompt runs W4A8 on the MLS3/MLS4 weights.
    let cache = if cache_budget > 0 {
        let names = [
            l("experts.gate_up_proj"),
            l("experts.gate_up_proj.scales"),
            l("experts.down_proj"),
            l("experts.down_proj.scales"),
        ];
        // Families that stack the shared expert into the expert matrix (everything but Ling) keep it in its own
        // tensors, but the router addresses it as expert `experts`: the cache assembles that block from the three.
        let shared = if m.shared_gate {
            Some([
                l(&format!("{se}.gate_proj.weight")),
                l(&format!("{se}.gate_proj.weight.scales")),
                l(&format!("{se}.up_proj.weight")),
                l(&format!("{se}.up_proj.weight.scales")),
                l(&format!("{se}.down_proj.weight")),
                l(&format!("{se}.down_proj.weight.scales")),
            ])
        } else {
            None
        };
        Some(Box::new(crate::expert_cache::Layer::build(
            src,
            &names,
            e,
            cache_budget,
            shared.as_ref(),
        )?))
    } else {
        None
    };
    let use_prefill = up.use_prefill && up.want_prefill(&l("experts")) && cache.is_none();
    // Parts as (tensor, rows, first row in the stacked matrix); every row has k columns. int8: the MLS8 prefill copy.
    // The format is read from the file: Ling's routed gate/up are MLS3 where everything else is MLS4, and one
    // stacked buffer has to be one packing because the kernel decodes it with a single stride.
    // Without a shared gate (Ling) the router still gives slot `topk` the expert id E, and the kernels index the
    // stacked matrices by that id: one expert row of zeros is cheaper than teaching every expert kernel that the
    // last slot may be absent, and the shared expert is added by its own Linears anyway.
    let pad = if m.shared_gate { 0 } else { 1 };
    let mut stack = |parts: &[(String, usize, usize)],
                     k: usize,
                     int8: bool,
                     pad_rows: usize|
     -> Result<(DeviceBuffer, DeviceBuffer)> {
        let g = k / 128;
        let rows: usize = parts.iter().map(|p| p.1).sum::<usize>() + pad_rows;
        let dt = if int8 {
            DType::Mls8
        } else {
            src.info(&parts[0].0).context("no expert tensors to stack")?.dtype
        };
        let (qb, sb) = match dt {
            DType::Mls8 => (k, 4),
            DType::Mls4 => (k / 2, g * 2),
            DType::Mls3 => (k / 8 * 3, g * 2),
            d => bail!("{d:?} is not a weight format for the stacked experts"),
        };
        let mut q = DeviceBuffer::alloc(rows * qb)?;
        let mut sc = DeviceBuffer::alloc(rows * sb)?;
        for (name, n, row) in parts {
            if int8 {
                let pn = format!("{name}.prefill");
                up.into(src, &pn, DType::Mls8, &[*n, k], &mut q, row * qb)?;
                up.into(src, &format!("{pn}.scales"), DType::F32, &[*n], &mut sc, row * sb)?;
            } else {
                up.into(src, name, dt, &[*n, k], &mut q, row * qb)?;
                up.into(src, &format!("{name}.scales"), DType::F16, &[*n, g], &mut sc, row * sb)?;
            }
        }
        if pad_rows > 0 {
            // The tail is a few megabytes: a zeroed scratch of that size and one copy, rather than zeroing a
            // buffer of tens of gigabytes.
            let null = Stream::null();
            let (zq, zs) = (pad_rows * qb, pad_rows * sb);
            let (zqb, zsb) = (DeviceBuffer::zeroed(zq)?, DeviceBuffer::zeroed(zs)?);
            q.copy_from(q.len() - zq, &zqb, 0, zq, &null)?;
            sc.copy_from(sc.len() - zs, &zsb, 0, zs, &null)?;
            null.synchronize()?;
        }
        Ok((q, sc))
    };
    // With a gate the shared expert is expert number E of the same stacked matrix (ADR 0002). Without one —
    // Ling — it is a plain dense MLP added with weight 1, and its tensors can be a different format from the
    // routed ones, so they cannot share the buffer and are loaded as their own Linears.
    let mut gu_parts = vec![(l("experts.gate_up_proj"), e * 2 * i, 0usize)];
    let mut down_parts = vec![(l("experts.down_proj"), e * h, 0usize)];
    if m.shared_gate {
        gu_parts.push((l(&format!("{se}.gate_proj.weight")), i, e * 2 * i));
        gu_parts.push((l(&format!("{se}.up_proj.weight")), i, e * 2 * i + i));
        down_parts.push((l(&format!("{se}.down_proj.weight")), h, e * h));
    }
    let (gu, down) = if cache.is_some() {
        (None, None)
    } else {
        (
            Some(stack(&gu_parts, h, false, pad * 2 * i)?),
            Some(stack(&down_parts, i, false, pad * h)?),
        )
    };
    let has_pre = gu_parts
        .iter()
        .chain(&down_parts)
        .all(|p| src.info(&format!("{}.prefill", p.0)).is_some());
    let pre = if use_prefill && has_pre {
        Some([
            stack(&gu_parts, h, true, pad * 2 * i)?,
            stack(&down_parts, i, true, pad * h)?,
        ])
    } else {
        None
    };
    // After the last `stack` call: the closure holds the uploader while it lives.
    if pre.is_some() {
        up.prefill_loaded = true;
    }
    let shared = if m.shared_gate {
        None
    } else {
        Some((
            up.linear(src, &l(&format!("{se}.gate_proj")), i, h)?,
            up.linear(src, &l(&format!("{se}.up_proj")), i, h)?,
            up.linear(src, &l(&format!("{se}.down_proj")), h, i)?,
        ))
    };
    let swiglu = match &c.swiglu {
        Some(l) => (l.routed[layer] as f32, l.shared[layer] as f32),
        None => (0.0, 0.0),
    };
    let is_mls3 = |name: &str| src.info(name).is_some_and(|i| i.dtype == DType::Mls3);
    let (gu_mls3, down_mls3) = (is_mls3(&l("experts.gate_up_proj")), is_mls3(&l("experts.down_proj")));
    Ok(Moe {
        router,
        gu,
        down,
        cache,
        pre,
        bias,
        shared,
        swiglu,
        gu_mls3,
        down_mls3,
    })
}

/// The geometry the shared staging buffer has to hold: the largest cached MoE layer of the model.
fn max_cache_geom(layers: &[Layer], mtp: Option<&Mtp>) -> Option<crate::expert_cache::Geom> {
    let mut out: Option<crate::expert_cache::Geom> = None;
    for ffn in layers.iter().map(|l| &l.ffn).chain(mtp.map(|m| &m.layer.ffn)) {
        let Ffn::Moe(m) = ffn else { continue };
        let Some(c) = &m.cache else { continue };
        let g = c.geom;
        out = Some(match out {
            None => g,
            Some(a) => crate::expert_cache::Geom {
                experts: a.experts.max(g.experts),
                gu_q: a.gu_q.max(g.gu_q),
                gu_s: a.gu_s.max(g.gu_s),
                down_q: a.down_q.max(g.down_q),
                down_s: a.down_s.max(g.down_s),
            },
        });
    }
    out
}

/// Requantizes an MLS8 matrix (same rotation) to MLS4 with round-to-nearest group scales.
fn mls8_to_mls4(q8: &DeviceBuffer, s8: &DeviceBuffer, n: usize, k: usize) -> Result<Linear> {
    let q: Vec<i8> = q8.download(n * k)?;
    let s: Vec<f32> = s8.download(n)?;
    let g = k / malleus_gguf::quant::BLOCK;
    let mut q4 = vec![0u8; n * k / 2];
    let mut s4 = vec![half::f16::ZERO; n * g];
    let threads = std::thread::available_parallelism().map_or(4, |v| v.get());
    let rows = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for ((qc, sc4), (i, q8c)) in q4
            .chunks_mut(rows * k / 2)
            .zip(s4.chunks_mut(rows * g))
            .zip(q.chunks(rows * k).enumerate())
        {
            let s = &s;
            sc.spawn(move || {
                for r in 0..q8c.len() / k {
                    let row_s = s[i * rows + r];
                    for j in 0..g {
                        let grp = &q8c[r * k + j * 128..r * k + (j + 1) * 128];
                        let amax = grp.iter().fold(0i32, |m, &v| m.max((v as i32).abs())) as f32 * row_s;
                        let gs = half::f16::from_f32(amax / 7.0);
                        sc4[r * g + j] = gs;
                        let inv = if gs.to_f32() > 0.0 { row_s / gs.to_f32() } else { 0.0 };
                        for e in 0..64 {
                            let v = |x: i8| ((x as f32 * inv).round().clamp(-8.0, 7.0) as i8 as u8) & 0xF;
                            qc[r * k / 2 + j * 64 + e] = v(grp[2 * e]) | (v(grp[2 * e + 1]) << 4);
                        }
                    }
                }
            });
        }
    });
    Ok(Linear {
        w: Weights::Mls4 {
            q: DeviceBuffer::from_slice(&q4)?,
            sc: DeviceBuffer::from_slice(&s4)?,
        },
        prefill: None,
        n,
        k,
    })
}

/// The rows `ids` of an MLS4 linear as a linear of their own: same layout `mls8_to_mls4` produces ([n, k / 2]
/// nibbles, [n, k / 128] fp16 scales), gathered rather than requantized. This is what the draft head needs when
/// lm_head itself is already MLS4, or there would be no draft head at all.
fn mls4_rows(q4: &DeviceBuffer, s4: &DeviceBuffer, ids: &[u32], n_src: usize, k: usize) -> Result<Linear> {
    let g = k / malleus_gguf::quant::BLOCK;
    let qb: Vec<u8> = q4.download(n_src * k / 2)?;
    let sb: Vec<u16> = s4.download(n_src * g)?;
    let mut q = Vec::with_capacity(ids.len() * k / 2);
    let mut s = Vec::with_capacity(ids.len() * g);
    for &i in ids {
        let r = i as usize;
        ensure!(r < n_src, "draft head row {r} outside lm_head ({n_src} rows)");
        q.extend_from_slice(&qb[r * k / 2..(r + 1) * k / 2]);
        s.extend_from_slice(&sb[r * g..(r + 1) * g]);
    }
    Ok(Linear {
        w: Weights::Mls4 {
            q: DeviceBuffer::from_slice(&q)?,
            sc: DeviceBuffer::from_slice(&s)?,
        },
        prefill: None,
        n: ids.len(),
        k,
    })
}

/// Multi-token prediction head: fc([norm(embed(next)); norm(h)]), one full-attention layer, norm, shared lm_head.
/// Qwen3.8-Flash-Next: fc_embedding(norm(embed(next))) added to fc_hidden of every stream of norm(streams), one
/// layer with hyper-connections, the MTP stream mixer, shared lm_head; the streams feed the next chained draft.
struct Mtp {
    fc: Option<Linear>,
    fc_e: Option<Linear>,
    fc_h: Option<Linear>,
    pre_e: DeviceBuffer,
    pre_h: DeviceBuffer,
    norm: Option<DeviceBuffer>,
    mixer: Option<Hc>,
    layer: Layer,
}

fn load_layer(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    pre: &str,
    kind: MixerKind,
    layer: usize,
    seed: u64,
    ple: bool,
    cache_budget: usize,
) -> Result<Layer> {
    let l = |s: &str| format!("{pre}{s}");
    let mut mixer = match kind {
        MixerKind::Attn => {
            let qd = c.n_heads * c.head_dim;
            let kvd = c.n_kv * c.head_dim;
            let indexer = match &c.indexer {
                Some(ic) => Some(Indexer {
                    qk: up.linear(
                        src,
                        &l("self_attn.indexer.index_qk_proj"),
                        (ic.heads + 1) * ic.dim,
                        c.hidden,
                    )?,
                    q_norm: up.tensor(src, &l("self_attn.indexer.q_layernorm.weight"), &[ic.dim])?,
                    k_norm: up.tensor(src, &l("self_attn.indexer.k_layernorm.weight"), &[ic.dim])?,
                }),
                None => None,
            };
            Mixer::Attn {
                q: up.linear(src, &l("self_attn.q_proj"), qd * 2, c.hidden)?,
                k: up.linear(src, &l("self_attn.k_proj"), kvd, c.hidden)?,
                v: up.linear(src, &l("self_attn.v_proj"), kvd, c.hidden)?,
                o: up.linear(src, &l("self_attn.o_proj"), c.hidden, qd)?,
                q_norm: up.tensor(src, &l("self_attn.q_norm.weight"), &[c.head_dim])?,
                k_norm: up.tensor(src, &l("self_attn.k_norm.weight"), &[c.head_dim])?,
                indexer,
                dec: None,
            }
        }
        MixerKind::Delta => {
            let vd = c.lin_nv * c.lin_dv;
            Mixer::Delta {
                qkv: up.linear(src, &l("linear_attn.in_proj_qkv"), c.lin_qkv_dim(), c.hidden)?,
                z: up.linear(src, &l("linear_attn.in_proj_z"), vd, c.hidden)?,
                a: up.linear(src, &l("linear_attn.in_proj_a"), c.lin_nv, c.hidden)?,
                b: up.linear(src, &l("linear_attn.in_proj_b"), c.lin_nv, c.hidden)?,
                out: up.linear(src, &l("linear_attn.out_proj"), c.hidden, vd)?,
                conv: up.tensor(src, &l("linear_attn.conv1d.weight"), &[c.lin_qkv_dim(), c.conv_k])?,
                a_log: up.tensor(src, &l("linear_attn.A_log"), &[c.lin_nv])?,
                dt_bias: up.tensor(src, &l("linear_attn.dt_bias"), &[c.lin_nv])?,
                norm: up.tensor(src, &l("linear_attn.norm.weight"), &[c.lin_dv])?,
                dec: None,
                ab: None,
            }
        }
        MixerKind::Kda => {
            // Ling spells the linear mixer `attention`, and its projections are separate tensors where Qwen
            // fuses qkv into one. The gate of the decay (`f_proj`) is one value per key channel.
            let (qd, vd) = (c.lin_nk * c.lin_dk, c.lin_nv * c.lin_dv);
            let cnv = |which: &str| format!("attention.{which}_conv1d.weight");
            Mixer::Kda {
                q: up.linear(src, &l("attention.q_proj"), qd, c.hidden)?,
                k: up.linear(src, &l("attention.k_proj"), qd, c.hidden)?,
                v: up.linear(src, &l("attention.v_proj"), vd, c.hidden)?,
                f: up.linear(src, &l("attention.f_proj"), vd, c.hidden)?,
                g: up.linear(src, &l("attention.g_proj"), vd, c.hidden)?,
                o: up.linear(src, &l("attention.o_proj"), c.hidden, vd)?,
                b: up.linear(src, &l("attention.b_proj"), c.lin_nv, c.hidden)?,
                conv: {
                    let mut cv = DeviceBuffer::zeroed(3 * vd * c.conv_k * 2)?;
                    for (i, which) in ["q", "k", "v"].iter().enumerate() {
                        up.into(
                            src,
                            &l(&cnv(which)),
                            DType::BF16,
                            &[vd, c.conv_k],
                            &mut cv,
                            i * vd * c.conv_k * 2,
                        )?;
                    }
                    cv
                },
                a_log: up.typed(src, &l("attention.A_log"), DType::F32, &[c.lin_nv])?,
                dt_bias: up.typed(src, &l("attention.dt_bias"), DType::F32, &[vd])?,
                o_norm: up.norm(src, &l("attention.o_norm.weight"), c.lin_dv, true)?,
            }
        }
        MixerKind::Mla => {
            let m = c.mla.context("MLA geometry")?;
            let qd = m.nope + m.rope;
            Mixer::Mla {
                q: up.linear(src, &l("attention.q_proj"), c.n_heads * qd, c.hidden)?,
                kv_a: up.linear(src, &l("attention.kv_a_proj_with_mqa"), m.kv_lora + m.rope, c.hidden)?,
                kv_a_norm: up.norm(src, &l("attention.kv_a_layernorm.weight"), m.kv_lora, true)?,
                kv_b: up.linear_f32(
                    src,
                    &l("attention.kv_b_proj"),
                    c.n_heads * (m.nope + m.v),
                    m.kv_lora,
                    seed,
                )?,
                gate: up.linear(src, &l("attention.g_proj"), c.n_heads, c.hidden)?,
                dense: up.linear(src, &l("attention.dense"), c.hidden, c.n_heads * m.v)?,
            }
        }
    };
    match &mut mixer {
        Mixer::Attn {
            q, k, v, indexer, dec, ..
        } => {
            let mut parts = vec![&*q, &*k, &*v];
            if let Some(ix) = indexer {
                parts.push(&ix.qk);
            }
            *dec = merge_mls4(&parts)?;
        }
        Mixer::Delta {
            qkv, z, a, b, dec, ab, ..
        } => {
            *dec = merge_mls4(&[&*qkv, &*z])?;
            if let (Weights::Bf16(wa), Weights::Bf16(wb)) = (&a.w, &b.w) {
                let st = Stream::null();
                let mut m = DeviceBuffer::alloc(wa.len() + wb.len())?;
                m.copy_from(0, wa, 0, wa.len(), &st)?;
                m.copy_from(wa.len(), wb, 0, wb.len(), &st)?;
                st.synchronize()?;
                *ab = Some(Linear {
                    w: Weights::Bf16(m),
                    prefill: None,
                    n: a.n + b.n,
                    k: a.k,
                });
            }
        }
        // The merged decode GEMVs of Ling are a later increment: every projection is read on its own for now.
        _ => {}
    }
    let fin = match &mut mixer {
        Mixer::Attn { q, k, v, .. } => fuse(&mut [q, k, v], false)?,
        Mixer::Delta { qkv, z, .. } if qkv.prefill.is_some() => {
            let names = [l("linear_attn.in_proj_a.weight"), l("linear_attn.in_proj_b.weight")];
            let mut ab = int8_rows(src, &names, c.lin_nv, c.hidden, 128, seed)?;
            fuse(&mut [qkv, z, &mut ab], false)?
        }
        _ => None,
    };
    // Ling replaces the FFN of the first `dense_ffn` layers with a dense one; Qwen is all-dense or all-MoE.
    let dense_ffn = c.moe.is_none() || layer < c.dense_ffn;
    let ffn = match (&c.moe, dense_ffn) {
        (Some(m), false) => Ffn::Moe(load_moe(up, src, c, m, pre, layer, cache_budget)?),
        _ => {
            let mut gate = up.linear(src, &l("mlp.gate_proj"), c.inter, c.hidden)?;
            let mut upl = up.linear(src, &l("mlp.up_proj"), c.inter, c.hidden)?;
            let fgu = fuse(&mut [&mut gate, &mut upl], true)?;
            Ffn::Dense {
                gate,
                up: upl,
                down: up.linear(src, &l("mlp.down_proj"), c.hidden, c.inter)?,
                fgu,
            }
        }
    };
    let (in_norm, post_norm) = if c.hc.is_some() {
        (None, None)
    } else {
        (
            Some(up.norm(src, &l("input_layernorm.weight"), c.hidden, c.norm_one_plus_w)?),
            Some(up.norm(src, &l("post_attention_layernorm.weight"), c.hidden, c.norm_one_plus_w)?),
        )
    };
    let hc = match &c.hc {
        Some(_) => Some([
            load_hc(up, src, c, &l("attn_hyper_connection."))?,
            load_hc(up, src, c, &l("mlp_hyper_connection."))?,
        ]),
        None => None,
    };
    let ple = match &c.ple {
        Some(p) if ple => {
            let sd = c.stream_dim();
            Some(Ple {
                key: up.linear(src, &l("ple.key_proj"), sd, p.dim)?,
                value: up.linear(src, &l("ple.value_proj"), c.hidden, p.dim)?,
                norm_key: up.tensor(src, &l("ple.norm_key.weight"), &[sd])?,
                norm_query: up.tensor(src, &l("ple.norm_query.weight"), &[sd])?,
                norm_conv: up.tensor(src, &l("ple.norm_conv.weight"), &[sd])?,
                conv: up.tensor(src, &l("ple.conv1d.weight"), &[sd, p.conv_k])?,
            })
        }
        _ => None,
    };
    Ok(Layer {
        in_norm,
        post_norm,
        hc,
        ple,
        mixer,
        ffn,
        fin,
    })
}

/// Permutes `rows` weights from stream-major (`s * hidden + j`) to the interleaved (`j * cnt + s`) order the
/// hyper-connection up GEMM reads. Weight and scale rows move as opaque bytes, so this serves MLS8 (`qw` = K int8,
/// `sw` = 4 for one f32) and MLS4 (`qw` = K / 2 nibble bytes, `sw` = 2 * K / 128 for the f16 group scales) alike.
fn permute_hc_rows(
    q: &mut DeviceBuffer,
    sc: &mut DeviceBuffer,
    rows: usize,
    qw: usize,
    sw: usize,
    cnt: usize,
    hidden: usize,
) -> Result<()> {
    let qv: Vec<u8> = q.download(rows * qw)?;
    let sv: Vec<u8> = sc.download(rows * sw)?;
    let (mut q2, mut s2) = (vec![0u8; rows * qw], vec![0u8; rows * sw]);
    for s in 0..cnt {
        for j in 0..hidden {
            let (from, to) = (s * hidden + j, j * cnt + s);
            q2[to * qw..(to + 1) * qw].copy_from_slice(&qv[from * qw..(from + 1) * qw]);
            s2[to * sw..(to + 1) * sw].copy_from_slice(&sv[from * sw..(from + 1) * sw]);
        }
    }
    q.upload(&q2)?;
    sc.upload(&s2)?;
    Ok(())
}

/// Permutes every representation of the hyper-connection `up` weights that the file carries: the decode weights
/// and, when `--hc-mls4` stored them apart, the MLS8 prefill copy. A path that read one and not the other would
/// mix rows of different streams.
fn permute_hc_up(l: &mut Linear, cnt: usize, hidden: usize) -> Result<bool> {
    let k = l.k;
    match &mut l.w {
        Weights::Mls8 { q, sc } => permute_hc_rows(q, sc, l.n, k, 4, cnt, hidden)?,
        Weights::Mls4 { q, sc } => {
            permute_hc_rows(q, sc, l.n, k / 2, 2 * (k / malleus_gguf::quant::BLOCK), cnt, hidden)?
        }
        // Hyper-connections only exist in Qwen4Exp, whose mixing is MLS4 or MLS8.
        Weights::Mls3 { .. } | Weights::Bf16(_) => return Ok(false),
    }
    if let Some((q, sc)) = &mut l.prefill {
        permute_hc_rows(q, sc, l.n, k, 4, cnt, hidden)?;
    }
    Ok(true)
}

/// The int8 weights of a linear to run a prefill GEMM on: the tensor itself when it is MLS8, otherwise the separate
/// MLS8 prefill copy. The hyper-connection mixing has no int4 kernel, so a file converted with `--hc-mls4` runs
/// both of its paths on the copy until one lands.
fn mls8_of(l: &Linear) -> Result<(&DeviceBuffer, &DeviceBuffer)> {
    match (&l.w, &l.prefill) {
        (Weights::Mls8 { q, sc }, _) => Ok((q, sc)),
        (_, Some((q, sc))) => Ok((q, sc)),
        (Weights::Mls4 { .. }, None) => bail!("MLS4 weights and no MLS8 prefill copy"),
        (Weights::Mls3 { .. }, None) => bail!("MLS3 weights have no MLS8 form"),
        (Weights::Bf16(_), None) => bail!("bf16 weights have no MLS8 form"),
    }
}

/// `MALLEUS_ROPE_YARN` (factor, default 1 = native rotation) and `MALLEUS_ROPE_ORIG` (native positions, default
/// 262144, the checkpoint's limit). With a factor, the long-wavelength frequencies are divided by it and the short
/// ones stay native, with a linear ramp in between — YaRN (Peng et al. 2023) in the form Hugging Face also uses. It
/// is needed because past `max_position_embeddings` the native rope extrapolates and the output is wrong: at 262144
/// the model is at home, at 524288 it is not.
fn yarn_factor() -> f32 {
    std::env::var("MALLEUS_ROPE_YARN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|f| *f > 1.0)
        .unwrap_or(1.0)
}

/// The attention scale multiplier YaRN asks for, 1 when the rotation is native.
fn yarn_mscale() -> f32 {
    let f = yarn_factor();
    if f > 1.0 { 0.1 * f.ln() + 1.0 } else { 1.0 }
}

fn rope_invf(c: &Config) -> Result<Option<Vec<f32>>> {
    let factor = yarn_factor();
    if factor <= 1.0 {
        return Ok(None);
    }
    let orig: f32 = std::env::var("MALLEUS_ROPE_ORIG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(262144.0);
    // The ends of the ramp: wavelengths below orig/beta_fast are native, above orig/beta_slow they are scaled.
    let (beta_fast, beta_slow) = (32.0f32, 1.0f32);
    let (lo, hi) = (orig / beta_fast, orig / beta_slow);
    let half = c.rot_dim / 2;
    let mut t = Vec::with_capacity(half);
    for j in 0..half {
        let f0 = 1.0 / c.rope_theta.powf(2.0 * j as f32 / c.rot_dim as f32);
        let wl = 2.0 * std::f32::consts::PI / f0;
        let ramp = ((wl - lo) / (hi - lo)).clamp(0.0, 1.0);
        t.push(f0 * (1.0 - ramp) + f0 / factor * ramp);
    }
    eprintln!(
        "rope: YaRN x{factor} over {orig} positions, attention scale {:.6}, {half} frequencies",
        yarn_mscale()
    );
    Ok(Some(t))
}

fn load_hc(up: &mut Uploader, src: &dyn TensorSource, c: &Config, pre: &str) -> Result<Hc> {
    let sd = c.stream_dim();
    let rows = |n: &str| -> Result<usize> {
        let info = src.info(n).with_context(|| format!("missing tensor {n}"))?;
        Ok(info.shape[0])
    };
    let (down, upn) = (format!("{pre}mix_down"), format!("{pre}mix_up"));
    let nd = rows(&format!("{down}.weight"))?;
    let rp = src.info(&format!("{upn}.weight")).map_or(0, |i| i.shape[1]);
    let mut h = Hc {
        norm: up.tensor(src, &format!("{pre}hc_norm.weight"), &[sd])?,
        down: up.linear(src, &down, nd, sd)?,
        up: up.linear(src, &upn, sd, rp)?,
        up_inter: false,
    };
    // MALLEUS_HC_W4=0 drops the MLS4 weights and keeps decode on the MLS8 copy. A file converted with --hc-mls4
    // carries both, so the 4-bit mixing can be measured against today's behaviour on one file; converting twice
    // would instead also change the dense calibration, because calibrate.py propagates through the weights it has
    // just quantized.
    if std::env::var("MALLEUS_HC_W4").is_ok_and(|v| v == "0") {
        for l in [&mut h.down, &mut h.up] {
            if matches!(l.w, Weights::Mls4 { .. })
                && let Some((q, sc)) = l.prefill.take()
            {
                l.w = Weights::Mls8 { q, sc };
            }
        }
    }
    h.up_inter = permute_hc_up(&mut h.up, c.hc.as_ref().map_or(1, |x| x.count), c.hidden)?;
    Ok(h)
}

/// MLS4 projections sharing an input stacked by rows for decode: one GEMV writing up to 4 output buffers.
struct Merged {
    q: DeviceBuffer,
    sc: DeviceBuffer,
    n: usize,
    k: usize,
    /// First row of segments 1..3 (N for absent segments).
    bounds: [usize; 3],
}

fn merge_mls4(parts: &[&Linear]) -> Result<Option<Merged>> {
    let k = parts[0].k;
    if parts.len() > 4
        || !k.is_multiple_of(32)
        || parts.iter().any(|p| p.k != k || !matches!(p.w, Weights::Mls4 { .. }))
    {
        return Ok(None);
    }
    let n: usize = parts.iter().map(|p| p.n).sum();
    let g = k / 128;
    let st = Stream::null();
    let mut q = DeviceBuffer::alloc(n * k / 2)?;
    let mut sc = DeviceBuffer::alloc(n * g * 2)?;
    let mut bounds = [n; 3];
    let mut row = 0;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            bounds[i - 1] = row;
        }
        let Weights::Mls4 { q: pq, sc: ps } = &p.w else {
            unreachable!()
        };
        q.copy_from(row * k / 2, pq, 0, p.n * k / 2, &st)?;
        sc.copy_from(row * g * 2, ps, 0, p.n * g * 2, &st)?;
        row += p.n;
    }
    st.synchronize()?;
    Ok(Some(Merged { q, sc, n, k, bounds }))
}

struct Fused {
    q: DeviceBuffer,
    sc: DeviceBuffer,
    n: usize,
    k: usize,
    segs: [usize; 2],
    glu: bool,
}

/// Rotated per-row int8 copy of bf16 tensors stacked by rows and zero-padded to `rows` (a prefill-only Linear).
fn int8_rows(
    src: &dyn TensorSource,
    names: &[String],
    n_each: usize,
    k: usize,
    rows: usize,
    seed: u64,
) -> Result<Linear> {
    use malleus_gguf::quant;
    ensure!(names.len() * n_each <= rows, "too many rows");
    let signs = quant::hadamard_signs(seed);
    let mut q = vec![0i8; rows * k];
    let mut sc = vec![0f32; rows];
    for (i, name) in names.iter().enumerate() {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::BF16 && info.size == n_each * k * 2,
            "{name}: expected bf16 [{n_each}, {k}]"
        );
        let raw = src.read_all(name)?;
        for r in 0..n_each {
            let mut row: Vec<f32> = raw[r * k * 2..(r + 1) * k * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect();
            quant::rotate_row(&mut row, &signs);
            let o = i * n_each + r;
            sc[o] = quant::quantize_row_int8(&row, &mut q[o * k..(o + 1) * k]);
        }
    }
    let qb = DeviceBuffer::from_slice(&q)?;
    let sb = DeviceBuffer::from_slice(&sc)?;
    Ok(Linear {
        w: Weights::Mls8 {
            q: DeviceBuffer::from_slice(&q)?,
            sc: DeviceBuffer::from_slice(&sc)?,
        },
        prefill: Some((qb, sb)),
        n: rows,
        k,
    })
}

/// Moves the MLS8 prefill copies of `parts` into one fused weight (None if any part has no copy).
fn fuse(parts: &mut [&mut Linear], glu: bool) -> Result<Option<Fused>> {
    // Without a merged int8 weight the caller falls back to one GEMM per projection, which is what the dense W4A4
    // path needs. The per-tensor MLS8 copies stay, so the GEMMs whose input transform only rot_quant_row can do
    // keep using them.
    if dense_w4a4() || parts.iter().any(|p| p.prefill.is_none()) {
        return Ok(None);
    }
    let k = parts[0].k;
    ensure!(parts.iter().all(|p| p.k == k), "fused projections must share K");
    let n: usize = parts.iter().map(|p| p.n).sum();
    let st = Stream::null();
    let mut q = DeviceBuffer::alloc(n * k)?;
    let mut sc = DeviceBuffer::alloc(n * 4)?;
    if glu {
        ensure!(
            parts.len() == 2 && parts[0].n == parts[1].n && parts[0].n.is_multiple_of(16),
            "bad gate/up"
        );
        for j in 0..parts[0].n / 16 {
            for (h, p) in parts.iter().enumerate() {
                let (pq, ps) = p.prefill.as_ref().unwrap();
                let dst = 32 * j + 16 * h;
                q.copy_from(dst * k, pq, 16 * j * k, 16 * k, &st)?;
                sc.copy_from(dst * 4, ps, 16 * j * 4, 64, &st)?;
            }
        }
    } else {
        let mut row = 0;
        for p in parts.iter() {
            ensure!(p.n.is_multiple_of(128), "fused segments must be multiples of 128 rows");
            let (pq, ps) = p.prefill.as_ref().unwrap();
            q.copy_from(row * k, pq, 0, p.n * k, &st)?;
            sc.copy_from(row * 4, ps, 0, p.n * 4, &st)?;
            row += p.n;
        }
    }
    st.synchronize()?;
    for p in parts.iter_mut() {
        p.prefill = None;
    }
    let segs = if glu {
        [0, 0]
    } else {
        [parts[0].n, parts.get(1).map_or(0, |p| p.n)]
    };
    Ok(Some(Fused { q, sc, n, k, segs, glu }))
}

struct Uploader {
    pinned: PinnedBuffer,
    bytes: usize,
    use_prefill: bool,
    prefill_families: Vec<String>,
    /// Decode on the MLS8 copy when the file has one (MTP head).
    decode_int8: bool,
    /// Whether any `.prefill` tensor was actually read: a single-copy file has none, and then the prefill runs on
    /// the MLS4 nibbles (W4A8) whatever the flags say. Reported at startup and used to word it correctly.
    prefill_loaded: bool,
}

impl Uploader {
    /// Whether the MLS8 prefill copy of `name` is loaded: all of them when the list is unset, none with `none`, else
    /// the ones whose tensor name contains one of the entries (`MALLEUS_PREFILL_INT8_FAMILIES`).
    fn want_prefill(&self, name: &str) -> bool {
        match self.prefill_families.first().map(String::as_str) {
            Some("none") => false,
            None => true,
            _ => self.prefill_families.iter().any(|f| name.contains(f.as_str())),
        }
    }

    const CHUNK: usize = 64 << 20;

    fn new() -> Result<Self> {
        Ok(Uploader {
            pinned: PinnedBuffer::alloc(Self::CHUNK)?,
            bytes: 0,
            use_prefill: true,
            prefill_families: Vec::new(),
            decode_int8: false,
            prefill_loaded: false,
        })
    }

    fn tensor(&mut self, src: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<DeviceBuffer> {
        self.typed(src, name, DType::BF16, shape)
    }

    fn typed(&mut self, src: &dyn TensorSource, name: &str, dtype: DType, shape: &[usize]) -> Result<DeviceBuffer> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(info.dtype == dtype, "{name}: expected {dtype:?}, got {:?}", info.dtype);
        let flat: Vec<usize> = info.shape.iter().copied().filter(|&d| d != 1).collect();
        let want: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
        ensure!(flat == want, "{name}: shape {:?}, expected {:?}", info.shape, shape);
        let mut buf = DeviceBuffer::alloc(info.size)?;
        self.copy(src, name, info.size, &mut buf, 0)?;
        Ok(buf)
    }

    /// Uploads a tensor into `buf` at byte offset `at`; `shape` is checked against the tensor flattened to the same
    /// number of elements and, for matrices, last dimension (expert stacks [E, N, K] load as [E * N, K]).
    fn into(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        dtype: DType,
        shape: &[usize],
        buf: &mut DeviceBuffer,
        at: usize,
    ) -> Result<()> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(info.dtype == dtype, "{name}: expected {dtype:?}, got {:?}", info.dtype);
        ensure!(
            info.shape.iter().product::<usize>() == shape.iter().product::<usize>()
                && (shape.len() < 2 || info.shape.last() == shape.last()),
            "{name}: shape {:?}, expected {:?}",
            info.shape,
            shape
        );
        ensure!(at + info.size <= buf.len(), "{name}: does not fit its buffer");
        self.copy(src, name, info.size, buf, at)
    }

    fn copy(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        size: usize,
        buf: &mut DeviceBuffer,
        at: usize,
    ) -> Result<()> {
        let mut off = 0;
        while off < size {
            let n = Self::CHUNK.min(size - off);
            src.read(name, off, &mut self.pinned.as_mut_slice()[..n])?;
            self.pinned.copy_to(n, buf, at + off)?;
            off += n;
        }
        self.bytes += size;
        Ok(())
    }

    fn linear(&mut self, src: &dyn TensorSource, name: &str, n: usize, k: usize) -> Result<Linear> {
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        let g = k / malleus_gguf::quant::BLOCK;
        let pn = format!("{wn}.prefill");
        let w = match info.dtype {
            _ if self.decode_int8 && src.info(&pn).is_some() => Weights::Mls8 {
                q: self.typed(src, &pn, DType::Mls8, &[n, k])?,
                sc: self.typed(src, &format!("{pn}.scales"), DType::F32, &[n])?,
            },
            DType::BF16 => Weights::Bf16(self.tensor(src, &wn, &[n, k])?),
            DType::Mls4 => Weights::Mls4 {
                q: self.typed(src, &wn, DType::Mls4, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F16, &[n, g])?,
            },
            DType::Mls3 => Weights::Mls3 {
                q: self.typed(src, &wn, DType::Mls3, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F16, &[n, g])?,
            },
            DType::Mls8 => Weights::Mls8 {
                q: self.typed(src, &wn, DType::Mls8, &[n, k])?,
                sc: self.typed(src, &format!("{wn}.scales"), DType::F32, &[n])?,
            },
            d => bail!("{wn}: unsupported dtype {d:?}"),
        };
        let family_ok = self.want_prefill(name);
        let prefill = if self.use_prefill && family_ok && src.info(&pn).is_some() {
            self.prefill_loaded = true;
            Some((
                self.typed(src, &pn, DType::Mls8, &[n, k])?,
                self.typed(src, &format!("{pn}.scales"), DType::F32, &[n])?,
            ))
        } else {
            None
        };
        Ok(Linear { w, prefill, n, k })
    }

    /// An RMSNorm weight. Ling stores `w` where the kernels expect `1 + w` (Qwen's form): folding the one at load
    /// keeps every norm kernel single-flavoured, which is the whole point of ADR 0005. The tensors are a few KB, so
    /// the fold happens on the host while uploading.
    /// An MLS4 tensor read back as plain fp32 [n, k]: dequantize every 128 group and undo the Hadamard rotation.
    /// This is the one place the quantized form has to become a matrix again, and it is here because MLA decode
    /// absorbs `kv_b_proj` into the query — a product against the transpose, which no GEMV can express.
    fn linear_f32(
        &mut self,
        src: &dyn TensorSource,
        name: &str,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<DeviceBuffer> {
        use malleus_gguf::quant::{self, BLOCK};
        let wn = format!("{name}.weight");
        let info = src.info(&wn).with_context(|| format!("missing tensor {wn}"))?;
        ensure!(info.dtype == DType::Mls4, "{wn}: expected MLS4, got {:?}", info.dtype);
        let q = src.read_all(&wn)?;
        let sc = src.read_all(&format!("{wn}.scales"))?;
        let g = k / BLOCK;
        let signs = quant::hadamard_signs(seed);
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            for j in 0..g {
                let s = half::f16::from_le_bytes([sc[r * g + j * 2], sc[r * g + j * 2 + 1]]).to_f32();
                let mut blk = [0f32; BLOCK];
                quant::dequant_int4(&q[r * k / 2 + j * BLOCK / 2..][..BLOCK / 2], s, &mut blk);
                quant::unrotate_block(&mut blk, &signs);
                out[r * k + j * BLOCK..][..BLOCK].copy_from_slice(&blk);
            }
        }
        self.bytes += n * k * 4;
        Ok(DeviceBuffer::from_slice(&out)?)
    }

    /// A norm weight, as the kernels that read it want it.
    ///
    /// Qwen's RMSNorm is `(1 + w) * x / rms(x)` and the kernels add the one, so the tensor goes up as it is
    /// (`raw`). Ling's is `w * x / rms(x)`: rather than teaching every kernel the difference — the one is added
    /// in `rmsnorm`, in the fused GEMM epilogues of `wmma.hip` and in the hyper-connection passes — the one is
    /// taken away here and they put it back. Storing `w - 1` in bf16 is also *more* accurate in absolute terms
    /// than storing `w`, because the rounding is relative to the stored value.
    ///
    /// The gated norms (KDA `o_norm`, MLA `kv_a_layernorm`) are fla's `FusedRMSNormGated`, which uses `w`
    /// directly in a kernel of its own that never adds one: those are `raw` for both families.
    fn norm(&mut self, src: &dyn TensorSource, name: &str, n: usize, raw: bool) -> Result<DeviceBuffer> {
        if raw {
            return self.tensor(src, name, &[n]);
        }
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::BF16 && info.size == n * 2,
            "{name}: expected bf16 [{n}]"
        );
        let bytes = src.read_all(name)?;
        let w: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32() - 1.0)
            .map(|x| half::bf16::from_f32(x).to_le_bytes())
            .map(u16::from_le_bytes)
            .collect();
        self.bytes += w.len() * 2;
        Ok(DeviceBuffer::from_slice(&w)?)
    }
}

/// GEMM input: a fresh buffer, the same buffer as the previous GEMM (its int8 quantization is reused), or
/// silu(gate) * up computed on the fly.
#[derive(Clone, Copy)]
pub enum Act<'a> {
    X(&'a DeviceBuffer),
    Same(&'a DeviceBuffer),
    SiluMul(&'a DeviceBuffer, &'a DeviceBuffer),
    /// RMSNorm(x) with weight (1 + w), applied while quantizing (int8 row path only, as the variants below).
    Norm(&'a DeviceBuffer, &'a DeviceBuffer),
    /// Per-head RMSNorm(x) * w * silu(z) (DeltaNet output).
    GatedNorm(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer),
    /// x * sigmoid(gate), gate interleaved in the q projection output.
    AttnGate(&'a DeviceBuffer, &'a DeviceBuffer),
}

pub struct Model {
    pub cfg: Config,
    pub k: Kernels,
    embed: DeviceBuffer,
    layers: Vec<Layer>,
    /// Final norm, or the final hyper-connection mixer of the streams.
    norm: Option<DeviceBuffer>,
    hc_final: Option<Hc>,
    ngram: Option<NgramTable>,
    lm_head: Linear,
    mtp: Option<Mtp>,
    /// Feed the MTP head the final-normed hidden state instead of the last layer output.
    mtp_post_norm: bool,
    /// Chained drafts start from the MTP output after its final norm.
    mtp_chain_post: bool,
    /// 4-bit copy of the MLS8 lm_head scoring the MTP drafts (half the bytes per draft).
    draft_head: Option<Linear>,
    /// Token id of every draft_head row when drafts are scored on a token list (MALLEUS_DRAFT_VOCAB_FILE).
    draft_map: Option<Vec<u32>>,
    signs: DeviceBuffer,
    /// Inverse RoPE frequencies blended by YaRN (`MALLEUS_ROPE_YARN` > 1): `rot_dim / 2` fp32, or None for the plain
    /// `1 / theta^(2j/rot)` the kernels compute inline. The blend depends on the wavelength of every index, so it is
    /// a table rather than a closed form.
    rope_invf: Option<DeviceBuffer>,
    scratch_cache: std::sync::Mutex<Option<Scratch>>,
    /// The stream every `forward` runs on.
    stream: Stream,
    /// Format of the activations last quantized into the scratch: 0 none, 1 per-row int8, 2 per-group int8.
    quant_mode: std::sync::atomic::AtomicU8,
    /// Prefill GEMMs on int8 activations (W4A8 / W8A8) instead of fp16 activations.
    pub prefill_a8: bool,
    /// Inputs up to this many tokens use the 4-bit weights (W4A8): half the bytes of the int8 copy.
    pub w4_max_t: usize,
    /// Prefill MoE with int4 activations (W4A4): the int4 WMMA is double rate on gfx11 and the expert nibbles
    /// reach it unmodified. MALLEUS_MOE_W4A4=0 keeps W4A8, `glu` limits int4 to the gate/up product.
    pub moe_w4a4: u8,
    /// `y` of the grouped MoE prefill (the expert output of every (token, slot) pair) in fp16 instead of f32,
    /// MALLEUS_MOE_Y_HALF=1: half the bytes of the widest intermediate of the prefill, which the down GEMM writes
    /// and `moe_combine` reads back (755 MB a layer at pp8192, and that kernel measured 72 % of the memory wall).
    pub moe_y_half: bool,
    pub weight_bytes: usize,
    /// `MALLEUS_EXPERT_CACHE`: one full layer of routed experts, streamed from the file at every prefill visit
    /// (a grouped GEMM needs all of them at once). Shared by every MoE layer, allocated once.
    expert_stage: Option<std::sync::Mutex<crate::expert_cache::Stage>>,
    /// The file carries MLS8 copies of the weights for the prefill (`--prefill-int8` at conversion). A single-copy
    /// file runs the prompt on the MLS4 nibbles: faster, ~35 GB lighter on a 35B MoE, and measurably lower quality.
    pub prefill_copies: bool,
    /// Diagnostics `MALLEUS_ATTN_MASS`: (buffer, sampling stride, tiles per row). The attention kernel writes into it,
    /// for one query row every `mstride`, the (max, sum of exp) of every tile of 32 keys, from which the attention
    /// mass per block is recovered. It does not touch the hot path: without the variable it is None and the kernel
    /// receives a null pointer.
    attn_mass: Option<(DeviceBuffer, usize, usize)>,
    pub vision: Option<crate::vision::VisionModel>,
}

pub struct LoadOptions {
    pub prefix: String,
    pub prefill_int8: bool,
    /// Load the MLS8 prefill copy only for linears whose name contains one of these (empty = all).
    pub prefill_families: Vec<String>,
    pub hadamard_seed: u64,
    /// Load the multi-token prediction head (speculative decoding) when the file has it.
    pub mtp: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            prefix: "model.language_model.".into(),
            prefill_int8: true,
            prefill_families: Vec::new(),
            hadamard_seed: malleus_gguf::quant::HADAMARD_SEED,
            mtp: true,
        }
    }
}

/// Per-sequence state: KV cache for attention layers, conv + recurrent state for DeltaNet layers.
pub struct Session {
    pub max_ctx: usize,
    pub len: usize,
    pub next_pos: i64,
    kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    conv: Vec<DeviceBuffer>,
    rec: Vec<DeviceBuffer>,
    /// Per-split partial results of the split-key attention (inputs shorter than ATTN_WMMA_MIN tokens).
    attn_part: DeviceBuffer,
    /// MLA split partials for Ling, sized for GEMV_MAX_T tokens; None for models without MLA.
    mla_part: Option<DeviceBuffer>,
    /// DeltaNet states after each token but the last of a verify step, per layer (rollback).
    conv_snap: Vec<DeviceBuffer>,
    /// Per DeltaNet layer, two buffers of rank-1 update records (verify writes one while reading the other).
    dn_rec: Vec<[DeviceBuffer; 2]>,
    /// Accepted records not yet applied to the recurrent state, and the buffer holding them.
    dn_np: usize,
    dn_par: usize,
    /// Tokens of the last verify step (0: no rollback possible).
    snap_t: usize,
    /// MTP head: its KV cache, rows written, and hidden states (with positions) waiting for their next token.
    mtp_kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    mtp_len: usize,
    mtp_h: Option<DeviceBuffer>,
    mtp_pos: Vec<[i32; 3]>,
    /// Output hidden state and position of the last MTP row (input of a chained draft).
    mtp_out: Option<DeviceBuffer>,
    mtp_out_pos: [i32; 3],
    /// Rows of chained drafts in the MTP KV cache, dropped before the next real rows.
    mtp_chain: usize,
    /// PLE: normalized conv inputs of the last rows and the tokens before the next input.
    ple_state: Option<DeviceBuffer>,
    ngram_ctx: Vec<u32>,
    /// QSA indexer of each attention layer (pooled block keys fp16, raw keys and positions of the last incomplete
    /// block), then the scores of a group of queries and the selection bitsets of the last input (with the unions of
    /// 64 queries), grown on demand.
    idx_kb: Vec<DeviceBuffer>,
    idx_tk: Vec<DeviceBuffer>,
    idx_tp: Vec<DeviceBuffer>,
    idx_scores: Option<DeviceBuffer>,
    idx_sel: Option<(DeviceBuffer, DeviceBuffer, usize)>,
    /// Last selection bitset of each indexer layer, kept for the decode reuse (one row per layer).
    idx_sel_cache: Vec<DeviceBuffer>,
    /// Host-side validity: the context length at which `idx_sel_cache` was written (0 = invalid).
    idx_cache_len: usize,
    /// Indexer of the MTP attention layer (pooled keys, tail keys, tail positions).
    mtp_idx: Option<[DeviceBuffer; 3]>,
    /// Short inputs with sparse attention: the selected blocks of each query as a list (and their counts).
    idx_list: Option<(DeviceBuffer, usize)>,
    /// Speculative verify with PLE: the conv state before the step and the normalized rows of its tokens (rollback),
    /// and the n-gram context before the step with the step's tokens.
    ple_snap: Option<DeviceBuffer>,
    ngram_prev: Vec<u32>,
}

/// Session state at a token boundary besides the KV caches (`Model::snapshot`, `Model::restore`).
pub struct Snapshot {
    pub len: usize,
    next_pos: i64,
    conv: Vec<DeviceBuffer>,
    rec: Vec<DeviceBuffer>,
    ple_state: Option<DeviceBuffer>,
    ngram_ctx: Vec<u32>,
    idx_tk: Vec<DeviceBuffer>,
    idx_tp: Vec<DeviceBuffer>,
    mtp_len: usize,
    mtp_h: Option<DeviceBuffer>,
    mtp_pos: Vec<[i32; 3]>,
    mtp_idx: Option<[DeviceBuffer; 2]>,
}

impl Snapshot {
    /// GPU bytes held.
    pub fn bytes(&self) -> usize {
        let v = |b: &[DeviceBuffer]| b.iter().map(|x| x.len()).sum::<usize>();
        v(&self.conv) + v(&self.rec) + self.ple_state.as_ref().map_or(0, |b| b.len()) + v(&self.idx_tk)
    }
}

/// Copies one state buffer, replacing the destination allocation when the size changed. A pooled snapshot keeps
/// its buffers from one request to the next: on these APUs the GPU memory *is* system RAM, and `hipMalloc` of a
/// buffer the size of one layer's state has to find pages while the weights hold every other page in sight.
fn copy_state(dst: &mut DeviceBuffer, src: &DeviceBuffer, st: &Stream) -> Result<()> {
    if dst.len() != src.len() {
        *dst = DeviceBuffer::alloc(src.len())?;
    }
    dst.copy_from(0, src, 0, src.len(), st)?;
    Ok(())
}

fn copy_states(dst: &mut Vec<DeviceBuffer>, src: &[DeviceBuffer], st: &Stream) -> Result<()> {
    if dst.len() != src.len() {
        *dst = src
            .iter()
            .map(|b| -> Result<DeviceBuffer> { Ok(DeviceBuffer::alloc(b.len())?) })
            .collect::<Result<_>>()?;
    }
    dst.iter_mut().zip(src).try_for_each(|(d, b)| copy_state(d, b, st))
}

fn copy_state_opt(dst: &mut Option<DeviceBuffer>, src: &Option<DeviceBuffer>, st: &Stream) -> Result<()> {
    match src {
        None => {
            *dst = None;
        }
        Some(s) => {
            if dst.as_ref().is_none_or(|d| d.len() != s.len()) {
                *dst = Some(DeviceBuffer::alloc(s.len())?);
            }
            if let Some(d) = dst.as_mut() {
                d.copy_from(0, s, 0, s.len(), st)?;
            }
        }
    }
    Ok(())
}

/// Recurrent and attention state a stack of layers reads and updates.
struct LayerStates<'a> {
    kv: &'a [(DeviceBuffer, DeviceBuffer)],
    conv: &'a [DeviceBuffer],
    rec: &'a [DeviceBuffer],
    len: usize,
    max_ctx: usize,
    attn_part: &'a DeviceBuffer,
    mla_part: Option<&'a DeviceBuffer>,
    conv_snaps: Option<&'a [DeviceBuffer]>,
    /// DeltaNet update records: (buffers, pending count, buffer with the pending ones, record this call).
    dn: Option<(&'a [[DeviceBuffer; 2]], usize, usize, bool)>,
    ple_state: Option<&'a DeviceBuffer>,
    /// Verify: where `ple` saves the conv state and the step's normalized rows.
    ple_snap: Option<&'a DeviceBuffer>,
    idx: Option<IdxStates<'a>>,
}

/// QSA indexer state for a stack of layers; `sel` is set when some query of the input has more complete blocks than
/// the budget: (scores, selections, 64-query unions, words per selection).
struct IdxStates<'a> {
    kb: &'a [DeviceBuffer],
    tk: &'a [DeviceBuffer],
    tp: &'a [DeviceBuffer],
    sel: Option<(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer, usize)>,
    /// Block lists of the selections (rows, then the counts after `rows` rows).
    list: Option<(&'a DeviceBuffer, usize)>,
    /// Decode reuse: last selection bitset per indexer layer, and whether this forward may use it.
    cache: Option<&'a [DeviceBuffer]>,
    reuse: bool,
}

/// Queries scored per pass of the indexer (bounds the score buffer).
const IDX_SUB: usize = 256;

impl IdxStates<'_> {
    fn sel_args(&self) -> [Arg; 3] {
        match self.sel {
            Some((_, sel, uni, _)) => [sel.into(), uni.into(), Arg::U64(0)],
            None => [Arg::U64(0), Arg::U64(0), Arg::U64(0)],
        }
    }
}

/// Inputs from this many tokens use the WMMA prefill attention; shorter ones the split-key kernel.
const ATTN_WMMA_MIN: usize = 16;
const ATTN_SPLIT: usize = 256;
/// Largest union of the four block lists `attention_gather_u` keeps in LDS; must match UMAX in kernels/attn.hip.
const GATHER_UNION_MAX: usize = 2080;

/// i32 slots before the token ids in `Scratch::hdr`: the KV position, padded to 16 bytes.
const STEP_HDR: usize = 4;
/// Largest host copy (in i32) `Scratch::upload_step` makes in one piece; a longer step uploads its three parts.
const STEP_UPLOAD_MAX: usize = 1 << 16;

struct Scratch {
    cap: usize,
    x: DeviceBuffer,
    hn: DeviceBuffer,
    big0: DeviceBuffer,
    big1: DeviceBuffer,
    big2: DeviceBuffer,
    small0: DeviceBuffer,
    small1: DeviceBuffer,
    small2: DeviceBuffer,
    small3: DeviceBuffer,
    xr: DeviceBuffer,
    xh: DeviceBuffer,
    xsc: DeviceBuffer,
    /// The inputs the host writes at every step, in one allocation so that a decode step uploads them with one
    /// host copy instead of three: [len, pad x 3, ids x cap, pos x 3 cap] as i32. `len_dev`, `ids` and `pos` are
    /// windows into it.
    hdr: DeviceBuffer,
    ids: DeviceBuffer,
    pos: DeviceBuffer,
    /// The session's KV position (sess.len) as a single i32 on the device, uploaded once per forward.
    /// Attention kernels read the write/read offset from here instead of a launch immediate, so the
    /// kernel node parameters stay constant across decode steps and the forward can be graph-captured.
    len_dev: DeviceBuffer,
    /// MTP inputs: hidden rows and the [embedding; hidden] concatenation.
    mh: DeviceBuffer,
    cat: DeviceBuffer,
    /// Chunked DeltaNet terms (deltanet_chunk_prep): Wn, Ut0, Qg, P, KdT, gamma_C.
    dn: [DeviceBuffer; 6],
    moe: Option<MoeScratch>,
    hc: Option<HcScratch>,
    /// Indexer projection [T, (heads + 1) * dim], normalized and roped queries, raw keys.
    idx: Option<[DeviceBuffer; 3]>,
    /// KDA (Ling): the conv output of the k and v streams. q's goes to `big1`, and the raw projections are free
    /// again once `conv_state_save` has read them, so two extra rows are enough for the whole mixer.
    kda: Option<[DeviceBuffer; 2]>,
    /// MLA (Ling): absorbed query per (token, head) for one prefill tile, and the weighted latent.
    mla_qt: Option<DeviceBuffer>,
    mla_lat: Option<DeviceBuffer>,
}

/// Hyper-connection buffers: stream state, normalized streams, low-rank down/gate, up, injection weights, the
/// output of a mixer or FFN, and the PLE n-gram embedding.
struct HcScratch {
    hc: DeviceBuffer,
    hcn: DeviceBuffer,
    hd: DeviceBuffer,
    ha: DeviceBuffer,
    hu: DeviceBuffer,
    /// Injection weights, alternating between consecutive mixes (a fused mix reads one and writes the other).
    inj: [DeviceBuffer; 2],
    mo: DeviceBuffer,
    ple_e: DeviceBuffer,
    /// 1 / rms of every stream (hc_quant_row).
    inv: DeviceBuffer,
}

struct MoeScratch {
    /// Router logits [T, E + 1], expert id and weight of each (token, slot) pair.
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    /// `MALLEUS_EXPERT_CACHE`: the same pairs with the expert id replaced by its slot in the layer's pool.
    slot_ids: std::sync::Mutex<DeviceBuffer>,
    w: DeviceBuffer,
    /// Decode: gate|up outputs and rotated silu(gate) * up per pair; prefill: silu(gate) * up per sorted row.
    gu: DeviceBuffer,
    act: DeviceBuffer,
    /// Prefill (moe_sort): row offsets per expert, sorted row → pair, pair → sorted row, tiles and their count.
    offs: DeviceBuffer,
    perm: DeviceBuffer,
    pos: DeviceBuffer,
    tiles: DeviceBuffer,
    ntiles: DeviceBuffer,
    /// Prefill: gathered int8 inputs, quantized expert activations (with row scales), expert outputs per sorted row.
    xg: DeviceBuffer,
    xgs: DeviceBuffer,
    aq: DeviceBuffer,
    aqs: DeviceBuffer,
    y: DeviceBuffer,
}

/// Tokens per chunk of the chunked DeltaNet; shorter inputs use the token-by-token scan.
pub(crate) const DN_CHUNK: usize = 64;

impl Scratch {
    /// Uploads the KV position, the token ids and their positions of a step.
    ///
    /// One host copy for a decode step: each synchronous copy costs tens of microseconds with the GPU idle, and a
    /// speculative round makes this call half a dozen times.
    fn upload_step(&mut self, len: i32, ids: &[i32], pos: &[i32]) -> Result<()> {
        ensure!(
            ids.len() <= self.cap && pos.len() == 3 * ids.len(),
            "step larger than the scratch"
        );
        let n = STEP_HDR + self.cap + pos.len();
        if n > STEP_UPLOAD_MAX {
            self.len_dev.upload(&[len])?;
            self.ids.upload(ids)?;
            self.pos.upload(pos)?;
            return Ok(());
        }
        let mut h = vec![0i32; n];
        h[0] = len;
        h[STEP_HDR..STEP_HDR + ids.len()].copy_from_slice(ids);
        h[STEP_HDR + self.cap..].copy_from_slice(pos);
        Ok(self.hdr.upload(&h)?)
    }
}

fn f32s(n: usize) -> Result<DeviceBuffer> {
    Ok(DeviceBuffer::alloc(n * 4)?)
}

impl Model {
    pub fn load(
        cfg: Config,
        src: &dyn TensorSource,
        opts: &LoadOptions,
        kernels: Kernels,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let c = &cfg;
        // Read before `kernels` moves into the model: the int4 default for the MoE prefill depends on the GPU.
        let arch = kernels.arch.clone();
        // Ling stores its weights under `model.` and calls the embedding `word_embeddings`; Qwen nests them
        // under `model.language_model.`. Everything else in the loader goes through `p`.
        let ling = c.family == Family::Ling3;
        let prefix = if ling { "model." } else { opts.prefix.as_str() };
        let mut up = Uploader::new()?;
        up.use_prefill = opts.prefill_int8;
        up.prefill_families = opts.prefill_families.clone();
        let signs = DeviceBuffer::from_slice(&malleus_gguf::quant::hadamard_signs(opts.hadamard_seed))?;
        let rope_invf = rope_invf(c)?.map(|t| DeviceBuffer::from_slice(&t)).transpose()?;
        let p = |s: &str| format!("{prefix}{s}");
        let embed = up.tensor(
            src,
            &p(if ling {
                "word_embeddings.weight"
            } else {
                "embed_tokens.weight"
            }),
            &[c.vocab, c.hidden],
        )?;
        // Decode on the MLS8 prefill copies of the non-expert linears (quality over speed).
        up.decode_int8 = std::env::var("MALLEUS_DECODE_INT8").is_ok_and(|v| v == "1");
        // `MALLEUS_EXPERT_CACHE=<GiB>` (issue #50): the routed experts stay on disk, one pool per MoE layer.
        // Models whose shared expert is stacked into the same matrix (Qwen's `shared_expert_gate`) are not
        // supported yet: their `experts + 1` rows are not all in the four tensors the cache reads, and guessing
        // wrong there would silently drop the shared expert.
        let cache_gib = crate::expert_cache::gib();
        let n_moe = c
            .n_layers
            .saturating_sub(c.dense_ffn.min(c.n_layers))
            .saturating_add(usize::from(opts.mtp))
            .max(1);
        let cache_budget = if cache_gib > 0.0 {
            (cache_gib * (1u64 << 30) as f64 / n_moe as f64) as usize
        } else {
            0
        };
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            progress(i, c.n_layers);
            layers.push(load_layer(
                &mut up,
                src,
                c,
                &p(&format!("layers.{i}.")),
                c.kind[i],
                i,
                opts.hadamard_seed,
                c.ple.as_ref().is_some_and(|pc| pc.layer == i),
                cache_budget,
            )?);
        }
        let (norm, hc_final) = match &c.hc {
            Some(_) => (None, Some(load_hc(&mut up, src, c, &p("hyper_connection_mixer."))?)),
            None => (
                Some(up.norm(src, &p("norm.weight"), c.hidden, c.norm_one_plus_w)?),
                None,
            ),
        };
        let ngram = match &c.ple {
            Some(pc) => Some(NgramTable::load(
                src,
                &p(&format!("layers.{}.ple.ple_embedding.", pc.layer)),
                pc,
            )?),
            None => None,
        };
        let mut lm_head = up.linear(src, "lm_head", c.vocab, c.hidden)?;
        if std::env::var("MALLEUS_LMHEAD_W4").is_ok_and(|v| v == "1")
            && let Weights::Mls8 { q, sc } = &lm_head.w
        {
            lm_head = mls8_to_mls4(q, sc, c.vocab, c.hidden)?;
        }
        // The draft head reads its 8-bit prefill copy by default: on code, JSON and lists the draft chain proposes
        // two to three times more tokens per round and three quarters of them are accepted, where with the 4-bit
        // weights it stops at the second draft (measured, bench/baseline.md). `MALLEUS_MTP_INT8=0` keeps 4 bits.
        up.decode_int8 = std::env::var("MALLEUS_MTP_INT8").map_or(true, |v| v != "0");
        let mtp = if opts.mtp && src.info("mtp.fc.weight").is_some() {
            Some(Mtp {
                fc: Some(up.linear(src, "mtp.fc", c.hidden, 2 * c.hidden)?),
                fc_e: None,
                fc_h: None,
                pre_e: up.tensor(src, "mtp.pre_fc_norm_embedding.weight", &[c.hidden])?,
                pre_h: up.tensor(src, "mtp.pre_fc_norm_hidden.weight", &[c.hidden])?,
                norm: Some(up.tensor(src, "mtp.norm.weight", &[c.hidden])?),
                mixer: None,
                layer: load_layer(
                    &mut up,
                    src,
                    c,
                    "mtp.layers.0.",
                    MixerKind::Attn,
                    usize::MAX,
                    opts.hadamard_seed,
                    false,
                    cache_budget,
                )?,
            })
        } else if opts.mtp && c.hc.is_some() && src.info("mtp.fc_embedding.weight").is_some() {
            Some(Mtp {
                fc: None,
                fc_e: Some(up.linear(src, "mtp.fc_embedding", c.hidden, c.hidden)?),
                fc_h: Some(up.linear(src, "mtp.fc_hidden", c.hidden, c.hidden)?),
                pre_e: up.tensor(src, "mtp.pre_fc_norm_embedding.weight", &[c.hidden])?,
                pre_h: up.tensor(src, "mtp.pre_fc_norm_hidden.weight", &[c.stream_dim()])?,
                norm: None,
                mixer: Some(load_hc(&mut up, src, c, "mtp.hyper_connection_mixer.")?),
                layer: load_layer(
                    &mut up,
                    src,
                    c,
                    "mtp.layers.0.",
                    MixerKind::Attn,
                    usize::MAX,
                    opts.hadamard_seed,
                    false,
                    cache_budget,
                )?,
            })
        } else {
            None
        };
        // Drafts are scored on the first `MALLEUS_DRAFT_VOCAB` tokens (ids follow the BPE merge order, so the most
        // frequent ones come first): a fraction of the lm_head bytes per draft.
        let draft_vocab = std::env::var("MALLEUS_DRAFT_VOCAB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(c.vocab)
            .clamp(1024, c.vocab);
        // A token list (one id per line, most frequent first) restricts the drafts to those rows of lm_head.
        let draft_list: Option<Vec<u32>> = match std::env::var("MALLEUS_DRAFT_VOCAB_FILE") {
            Ok(f) if mtp.is_some() => {
                let mut ids: Vec<u32> = std::fs::read_to_string(&f)
                    .with_context(|| f.clone())?
                    .lines()
                    .filter_map(|l| l.trim().parse().ok())
                    .filter(|&i: &u32| (i as usize) < c.vocab)
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                Some(ids)
            }
            _ => None,
        };
        let (draft_head, draft_map) = match (&mtp, &lm_head.w) {
            (Some(_), Weights::Mls8 { q, sc }) if std::env::var("MALLEUS_DRAFT_HEAD").map_or(true, |v| v != "int8") => {
                match draft_list {
                    Some(ids) => {
                        let k = c.hidden;
                        let (q8, s8): (Vec<i8>, Vec<f32>) = (q.download(c.vocab * k)?, sc.download(c.vocab)?);
                        let mut qg = Vec::with_capacity(ids.len() * k);
                        let mut sg = Vec::with_capacity(ids.len());
                        for &i in &ids {
                            qg.extend_from_slice(&q8[i as usize * k..(i as usize + 1) * k]);
                            sg.push(s8[i as usize]);
                        }
                        let (qb, sb) = (DeviceBuffer::from_slice(&qg)?, DeviceBuffer::from_slice(&sg)?);
                        (Some(mls8_to_mls4(&qb, &sb, ids.len(), k)?), Some(ids))
                    }
                    None => (Some(mls8_to_mls4(q, sc, draft_vocab, c.hidden)?), None),
                }
            }
            // lm_head is already MLS4 (MALLEUS_LMHEAD_W4): gather only the rows that would make the draft head
            // differ from it. With no token list and a full draft vocabulary it would be a copy of lm_head, and
            // `None` already scores the drafts on lm_head itself.
            (Some(_), Weights::Mls4 { q, sc })
                if std::env::var("MALLEUS_DRAFT_HEAD").map_or(true, |v| v != "int8")
                    && (draft_list.is_some() || draft_vocab < c.vocab) =>
            {
                let ids = draft_list.clone().unwrap_or_else(|| (0..draft_vocab as u32).collect());
                (Some(mls4_rows(q, sc, &ids, c.vocab, c.hidden)?), draft_list)
            }
            _ => (None, None),
        };
        // The cache's shared staging (one full layer) and the slot-id buffer of the decode path.
        let expert_stage = max_cache_geom(&layers, mtp.as_ref())
            .map(crate::expert_cache::Stage::alloc)
            .transpose()?
            .map(std::sync::Mutex::new);
        if let Some(g) = expert_stage.as_ref().map(|s| crate::expert_cache::lock(s).bytes()) {
            let pool: usize = layers
                .iter()
                .chain(mtp.as_ref().map(|m| &m.layer))
                .map(|l| match &l.ffn {
                    Ffn::Moe(m) => m.cache.as_ref().map_or(0, |c| c.bytes()),
                    _ => 0,
                })
                .sum();
            eprintln!(
                "expert cache: {:.1} GiB of pools + {:.1} GiB of staging instead of the routed experts",
                pool as f64 / (1u64 << 30) as f64,
                g as f64 / (1u64 << 30) as f64
            );
        }
        let cache_bytes: usize = layers
            .iter()
            .chain(mtp.as_ref().map(|m| &m.layer))
            .map(|l| match &l.ffn {
                Ffn::Moe(m) => m.cache.as_ref().map_or(0, |c| c.bytes()),
                _ => 0,
            })
            .sum::<usize>()
            + expert_stage
                .as_ref()
                .map_or(0, |s| crate::expert_cache::lock(s).bytes());
        progress(c.n_layers, c.n_layers);
        // `y` in fp16 needs one width for the whole process, and the int8 grouped path (the one a checkpoint with
        // MLS8 expert copies takes) has no fp16 variant: there the flag is ignored rather than half applied. Which
        // path a checkpoint takes is a property of its experts, not of the prompt length.
        let moe_y_half = std::env::var("MALLEUS_MOE_Y_HALF").is_ok_and(|v| v != "0")
            && !layers.iter().any(|l| matches!(&l.ffn, Ffn::Moe(m) if m.pre.is_some()));
        // Diagnostics of the attention concentration (#96): sized for 262144 tokens of context, one sampled row every
        // 4096, a tile of 32 keys. ~4 MB.
        let attn_mass = match std::env::var_os("MALLEUS_ATTN_MASS") {
            Some(_) => {
                let num = |k: &str, d: usize| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|v| *v > 0)
                        .unwrap_or(d)
                };
                let (mstride, ctx_max) = (
                    num("MALLEUS_ATTN_MASS_STRIDE", 4096),
                    num("MALLEUS_ATTN_MASS_MAX", 262_144),
                );
                let mtiles = ctx_max / 32 + 2;
                let rows = ctx_max / mstride + 2;
                Some((DeviceBuffer::zeroed(rows * mtiles * 8)?, mstride, mtiles))
            }
            None => None,
        };
        Ok(Model {
            weight_bytes: up.bytes + cache_bytes,
            expert_stage,
            attn_mass,
            prefill_copies: up.prefill_loaded,
            cfg,
            k: kernels,
            embed,
            layers,
            norm,
            hc_final,
            ngram,
            lm_head,
            mtp,
            mtp_post_norm: std::env::var("MALLEUS_MTP_HIDDEN").is_ok_and(|v| v == "post"),
            mtp_chain_post: std::env::var("MALLEUS_MTP_CHAIN").map_or(true, |v| v != "pre"),
            draft_head,
            draft_map,
            signs,
            rope_invf,
            scratch_cache: std::sync::Mutex::new(None),
            stream: Stream::new()?,
            quant_mode: std::sync::atomic::AtomicU8::new(0),
            prefill_a8: std::env::var("MALLEUS_PREFILL_GEMM").map_or(true, |v| v != "f16"),
            w4_max_t: std::env::var("MALLEUS_W4_MAX_T")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            // int4 activations for the MoE prefill are double rate on gfx1151 and about twice as slow as int8 on
            // gfx1150 for this model's shapes (measured: docs, "Portare il lavoro di gfx1151 su gfx1150"), so the
            // default follows the GPU: a file without the int8 expert copies (W4A8) then runs the fast path here
            // without needing the environment.
            moe_w4a4: match std::env::var("MALLEUS_MOE_W4A4").unwrap_or_default().as_str() {
                "0" => 0,
                "glu" => 2,
                // An empty variable counts as unset: a container that forwards an optional setting passes it as
                // an empty string, and that must not silently pick the int4 path (measured 662 against 1062 tok/s
                // of prefill at pp2048 on gfx1150).
                "" if arch == "gfx1150" => 0,
                _ => 1,
            },
            moe_y_half,
            vision: None,
        })
    }

    pub fn new_session(&self, max_ctx: usize) -> Result<Session> {
        let c = &self.cfg;
        let mut s = Session {
            max_ctx,
            len: 0,
            next_pos: 0,
            kv: vec![],
            conv: vec![],
            rec: vec![],
            attn_part: f32s((ATTN_WMMA_MIN - 1) * c.n_heads * kv_ctx(max_ctx).div_ceil(ATTN_SPLIT) * (c.head_dim + 2))?,
            mla_part: match &c.mla {
                Some(m) => Some(f32s(
                    GEMV_MAX_T * c.n_heads * kv_ctx(max_ctx).div_ceil(ATTN_SPLIT) * (m.kv_lora + 2),
                )?),
                None => None,
            },
            conv_snap: vec![],
            dn_rec: vec![],
            dn_np: 0,
            dn_par: 0,
            snap_t: 0,
            mtp_kv: vec![],
            mtp_len: 0,
            mtp_h: None,
            mtp_pos: vec![],
            mtp_out: None,
            mtp_out_pos: [0; 3],
            mtp_chain: 0,
            ple_state: match &c.ple {
                Some(p) => Some(DeviceBuffer::zeroed(p.conv_state() * c.stream_dim() * 4)?),
                None => None,
            },
            ngram_ctx: c.ple.as_ref().map_or(vec![], |p| vec![p.eos; p.ngram - 1]),
            idx_kb: vec![],
            idx_tk: vec![],
            idx_tp: vec![],
            idx_scores: None,
            idx_sel: None,
            idx_sel_cache: vec![],
            idx_cache_len: 0,
            mtp_idx: None,
            idx_list: None,
            ple_snap: None,
            ngram_prev: vec![],
        };
        if let Some(ic) = &c.indexer {
            let idx_bufs = || -> Result<[DeviceBuffer; 3]> {
                Ok([
                    DeviceBuffer::zeroed((max_ctx / ic.ratio + 1) * ic.dim * 2)?,
                    DeviceBuffer::zeroed(ic.ratio * ic.dim * 4)?,
                    DeviceBuffer::zeroed(ic.ratio * 3 * 4)?,
                ])
            };
            let idx_w = self.idx_words(max_ctx);
            for _ in c.full_attn.iter().filter(|f| **f) {
                let [kb, tk, tp] = idx_bufs()?;
                s.idx_kb.push(kb);
                s.idx_tk.push(tk);
                s.idx_tp.push(tp);
                s.idx_sel_cache.push(DeviceBuffer::zeroed(idx_w * 4)?);
            }
            if self.mtp.is_some() {
                s.mtp_idx = Some(idx_bufs()?);
            }
            s.idx_list = Some((
                DeviceBuffer::alloc(ATTN_WMMA_MIN * (ic.budget / ic.ratio + 1) * 4)?,
                ATTN_WMMA_MIN,
            ));
        }
        if let (Some(p), Some(_)) = (&c.ple, &self.mtp) {
            s.ple_snap = Some(DeviceBuffer::alloc((p.conv_state() + GEMV_MAX_T) * c.stream_dim() * 4)?);
        }
        let kv_bytes = kv_ctx(max_ctx) * c.n_kv * c.head_dim * 2 + 64;
        // With the indexer, the V buffer also holds V row-major after V^T (gathered by the sparse attention).
        let v_bytes = if c.indexer.is_some() { 2 * kv_bytes } else { kv_bytes };
        let (conv_b, rec_b) = (c.lin_qkv_dim() * (c.conv_k - 1) * 4, c.lin_nv * c.lin_dk * c.lin_dv * 4);
        // MLA stores the latent and the rotated dims apart: 512 + 64 fp16 per token instead of the 20 KiB per
        // layer and token the HF code would spend on K and V (docs/model-ling3-flash.md, § MLA).
        let mla_kv = c.mla.map(|m| kv_ctx(max_ctx) * m.kv_lora * 2).unwrap_or(0);
        let mla_rot = c.mla.map(|m| kv_ctx(max_ctx) * m.rope * 2).unwrap_or(0);
        for &kind in &c.kind {
            if kind == MixerKind::Attn || kind == MixerKind::Mla {
                let (kb, vb) = if kind == MixerKind::Attn {
                    (kv_bytes, v_bytes)
                } else {
                    (mla_kv, mla_rot)
                };
                s.kv.push((DeviceBuffer::zeroed(kb)?, DeviceBuffer::zeroed(vb)?));
            } else {
                s.conv.push(DeviceBuffer::zeroed(conv_b)?);
                s.rec.push(DeviceBuffer::zeroed(rec_b)?);
                if self.mtp.is_some() {
                    s.conv_snap.push(DeviceBuffer::alloc((GEMV_MAX_T - 1) * conv_b)?);
                    let rec = (c.lin_nk * 128 + c.lin_nv * 129) * 4 * GEMV_MAX_T;
                    s.dn_rec.push([DeviceBuffer::alloc(rec)?, DeviceBuffer::alloc(rec)?]);
                }
            }
        }
        if self.mtp.is_some() {
            s.mtp_kv
                .push((DeviceBuffer::zeroed(kv_bytes)?, DeviceBuffer::zeroed(v_bytes)?));
            s.mtp_h = Some(f32s(GEMV_MAX_T * c.stream_dim())?);
            s.mtp_out = Some(f32s(c.stream_dim())?);
        }
        Ok(s)
    }

    /// An empty snapshot with one buffer per state of this session, to be refilled by `snapshot_into`. The
    /// scheduler pools these and recycles them (`Model::snapshot_into`) because allocating them per request
    /// dominates the cost of saving a prompt.
    pub fn new_snapshot(&self, s: &Session) -> Result<Snapshot> {
        let a = |b: &DeviceBuffer| -> Result<DeviceBuffer> { Ok(DeviceBuffer::alloc(b.len())?) };
        Ok(Snapshot {
            len: 0,
            next_pos: 0,
            conv: s.conv.iter().map(a).collect::<Result<_>>()?,
            rec: s.rec.iter().map(a).collect::<Result<_>>()?,
            ple_state: s.ple_state.as_ref().map(a).transpose()?,
            ngram_ctx: Vec::new(),
            idx_tk: s.idx_tk.iter().map(a).collect::<Result<_>>()?,
            idx_tp: s.idx_tp.iter().map(a).collect::<Result<_>>()?,
            mtp_len: 0,
            mtp_h: s.mtp_h.as_ref().map(a).transpose()?,
            mtp_pos: Vec::new(),
            mtp_idx: match &s.mtp_idx {
                Some([_, tk, tp]) => Some([a(tk)?, a(tp)?]),
                None => None,
            },
        })
    }

    /// Copies the states of `s` into a snapshot made by `new_snapshot` on a session of the same model, reusing
    /// the buffers it already holds.
    pub fn snapshot_into(&self, s: &Session, snap: &mut Snapshot) -> Result<()> {
        ensure!(s.dn_np == 0 && s.snap_t == 0, "snapshot during a speculative step");
        let st = Stream::null();
        copy_states(&mut snap.conv, &s.conv, &st)?;
        copy_states(&mut snap.rec, &s.rec, &st)?;
        copy_states(&mut snap.idx_tk, &s.idx_tk, &st)?;
        copy_states(&mut snap.idx_tp, &s.idx_tp, &st)?;
        copy_state_opt(&mut snap.ple_state, &s.ple_state, &st)?;
        copy_state_opt(&mut snap.mtp_h, &s.mtp_h, &st)?;
        match &s.mtp_idx {
            Some([_, tk, tp]) => {
                let fresh = snap
                    .mtp_idx
                    .as_ref()
                    .is_none_or(|[a, b]| a.len() != tk.len() || b.len() != tp.len());
                if fresh {
                    snap.mtp_idx = Some([DeviceBuffer::alloc(tk.len())?, DeviceBuffer::alloc(tp.len())?]);
                }
                if let Some([a, b]) = snap.mtp_idx.as_mut() {
                    a.copy_from(0, tk, 0, tk.len(), &st)?;
                    b.copy_from(0, tp, 0, tp.len(), &st)?;
                }
            }
            None => snap.mtp_idx = None,
        }
        snap.len = s.len;
        snap.next_pos = s.next_pos;
        snap.ngram_ctx = s.ngram_ctx.clone();
        snap.mtp_len = s.mtp_len - s.mtp_chain;
        snap.mtp_pos = s.mtp_pos.clone();
        st.synchronize()?;
        Ok(())
    }

    /// Copies the recurrent and auxiliary states of a session (everything the KV caches do not hold) so a later
    /// input sharing its first `len` tokens can restart from here. Not during a speculative step. A caller that
    /// snapshots on every request wants `new_snapshot` + `snapshot_into` on a pooled buffer instead.
    pub fn snapshot(&self, s: &Session) -> Result<Snapshot> {
        let mut snap = self.new_snapshot(s)?;
        self.snapshot_into(s, &mut snap)?;
        Ok(snap)
    }

    /// Restarts a session from a snapshot taken on it; its KV rows [0, len) must still hold the same tokens.
    pub fn restore(&self, s: &mut Session, snap: &Snapshot) -> Result<()> {
        let st = Stream::null();
        let cp = |d: &mut DeviceBuffer, b: &DeviceBuffer| d.copy_from(0, b, 0, b.len(), &st);
        ensure!(
            s.conv.len() == snap.conv.len() && s.rec.len() == snap.rec.len(),
            "snapshot of another model"
        );
        for (d, b) in s.conv.iter_mut().zip(&snap.conv) {
            cp(d, b)?;
        }
        for (d, b) in s.rec.iter_mut().zip(&snap.rec) {
            cp(d, b)?;
        }
        if let (Some(d), Some(b)) = (s.ple_state.as_mut(), &snap.ple_state) {
            cp(d, b)?;
        }
        for (d, b) in s.idx_tk.iter_mut().zip(&snap.idx_tk) {
            cp(d, b)?;
        }
        for (d, b) in s.idx_tp.iter_mut().zip(&snap.idx_tp) {
            cp(d, b)?;
        }
        if let (Some(d), Some(b)) = (s.mtp_h.as_mut(), &snap.mtp_h) {
            cp(d, b)?;
        }
        if let (Some([_, tk, tp]), Some([btk, btp])) = (s.mtp_idx.as_mut(), &snap.mtp_idx) {
            cp(tk, btk)?;
            cp(tp, btp)?;
        }
        st.synchronize()?;
        s.len = snap.len;
        s.next_pos = snap.next_pos;
        s.ngram_ctx = snap.ngram_ctx.clone();
        s.ngram_prev.clear();
        s.mtp_len = snap.mtp_len;
        s.mtp_chain = 0;
        s.mtp_pos = snap.mtp_pos.clone();
        s.snap_t = 0;
        s.dn_np = 0;
        s.idx_cache_len = 0;
        Ok(())
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        let st = Stream::null();
        for b in s.conv.iter().chain(&s.rec).chain(&s.ple_state) {
            b.memset(0, &st)?;
        }
        if let Some(p) = &self.cfg.ple {
            s.ngram_ctx = vec![p.eos; p.ngram - 1];
        }
        st.synchronize()?;
        s.len = 0;
        s.next_pos = 0;
        s.snap_t = 0;
        s.dn_np = 0;
        s.mtp_len = 0;
        s.mtp_pos.clear();
        s.mtp_chain = 0;
        s.idx_cache_len = 0;
        Ok(())
    }

    fn take_scratch(&self, t: usize) -> Result<Scratch> {
        let cached = self.scratch_cache.lock().unwrap().take();
        if let Some(sc) = cached {
            if sc.cap >= t {
                return Ok(sc);
            }
            // The old buffers go before the new ones are asked for: on these APUs a request that grows the scratch
            // used to need the pages of both at once (the `match` kept the old one alive until after the
            // allocation), which is what turned a long prompt into one oversized allocation instead of two.
            drop(sc);
        }
        self.scratch(t.max(16))
    }

    /// The working set of a chunk of `cap` tokens. `memplan::working_set_bytes` mirrors these sizes for the
    /// pre-flight: a buffer added here has to be added there.
    fn scratch(&self, cap: usize) -> Result<Scratch> {
        let c = &self.cfg;
        let big = c.inter.max(c.n_heads * c.head_dim * 2).max(c.lin_qkv_dim());
        let small = (c.n_heads * c.head_dim).max(c.lin_nv * c.lin_dv);
        // Widest GEMM input: also the hyper-connection streams and the PLE embedding.
        let kmax = c
            .inter
            .max(c.hidden)
            .max(c.n_heads * c.head_dim)
            .max(c.stream_dim())
            .max(c.ple.as_ref().map_or(0, |p| p.dim));
        let mla_prefill = std::env::var("MALLEUS_MLA_PREFILL").is_ok_and(|v| v == "1");
        let hdr_bytes = (STEP_HDR + 4 * cap) * 4;
        let [hdr, len_dev, ids, pos]: [DeviceBuffer; 4] = DeviceBuffer::alloc(hdr_bytes)?
            .into_views(&[
                (0, hdr_bytes),
                (0, 4),
                (STEP_HDR * 4, cap * 4),
                ((STEP_HDR + cap) * 4, cap * 12),
            ])
            .try_into()
            .map_err(|_| anyhow::anyhow!("step header views"))?;
        Ok(Scratch {
            cap,
            x: f32s(cap * c.hidden)?,
            hn: f32s(cap * c.hidden)?,
            big0: f32s(cap * big)?,
            big1: f32s(cap * big)?,
            big2: f32s(cap * big)?,
            small0: f32s(cap * small)?,
            small1: f32s(cap * small)?,
            small2: f32s(cap * small)?,
            small3: f32s(cap * small)?,
            kda: if c.kind.contains(&MixerKind::Kda) {
                Some([f32s(cap * c.lin_nv * c.lin_dv)?, f32s(cap * c.lin_nv * c.lin_dv)?])
            } else {
                None
            },
            mla_qt: match &c.mla {
                Some(m) => Some(f32s(
                    (if mla_prefill { MLA_PREFILL_TILE } else { GEMV_MAX_T }) * c.n_heads * m.kv_lora,
                )?),
                None => None,
            },
            mla_lat: if mla_prefill {
                match &c.mla {
                    Some(m) => Some(f32s(MLA_PREFILL_TILE * c.n_heads * m.kv_lora)?),
                    None => None,
                }
            } else {
                None
            },
            xr: f32s(cap * kmax)?,
            xh: DeviceBuffer::alloc(cap * kmax * 2)?,
            xsc: DeviceBuffer::alloc(cap * kmax / 128 * 4)?,
            hdr,
            ids,
            pos,
            len_dev,
            mh: f32s(cap * c.stream_dim())?,
            cat: f32s(cap * 2 * c.hidden)?,
            dn: {
                let blocks = cap.div_ceil(DN_CHUNK) * c.lin_nv;
                let (cc, d) = (DN_CHUNK, c.lin_dv);
                [
                    DeviceBuffer::alloc(blocks * cc * d * 2)?,
                    DeviceBuffer::alloc(blocks * cc * d * 4)?,
                    DeviceBuffer::alloc(blocks * cc * d * 2)?,
                    DeviceBuffer::alloc(blocks * cc * cc * 2)?,
                    DeviceBuffer::alloc(blocks * d * cc * 2)?,
                    DeviceBuffer::alloc(blocks * 4)?,
                ]
            },
            moe: match &c.moe {
                Some(m) => {
                    let (p, e1) = (cap * m.slots(), m.experts + 1);
                    Some(MoeScratch {
                        logits: f32s(cap * e1)?,
                        ids: DeviceBuffer::alloc(p * 4)?,
                        slot_ids: std::sync::Mutex::new(DeviceBuffer::alloc(if self.expert_stage.is_some() {
                            p * 4
                        } else {
                            4
                        })?),
                        w: f32s(p)?,
                        gu: f32s(p * 2 * m.inter)?,
                        act: f32s(p * m.inter)?,
                        offs: DeviceBuffer::alloc((e1 + 1) * 4)?,
                        perm: DeviceBuffer::alloc(p * 4)?,
                        pos: DeviceBuffer::alloc(p * 4)?,
                        tiles: DeviceBuffer::alloc((p.div_ceil(MOE_BM) + 3 * e1 + 1) * 8)?,
                        ntiles: DeviceBuffer::alloc(8)?,
                        xg: DeviceBuffer::alloc(p * c.hidden)?,
                        xgs: f32s(p * c.hidden.div_ceil(128))?,
                        aq: DeviceBuffer::alloc(p * m.inter)?,
                        aqs: f32s(p * m.inter.div_ceil(128))?,
                        y: DeviceBuffer::alloc(p * c.hidden * if self.moe_y_half { 2 } else { 4 })?,
                    })
                }
                None => None,
            },
            idx: match &c.indexer {
                Some(ic) => Some([
                    f32s(cap * (ic.heads + 1) * ic.dim)?,
                    f32s(cap * ic.heads * ic.dim)?,
                    f32s(cap * ic.dim)?,
                ]),
                None => None,
            },
            hc: match &c.hc {
                Some(h) => {
                    let (sd, rp) = (c.stream_dim(), pad128(h.rank + h.count));
                    Some(HcScratch {
                        hc: f32s(cap * sd)?,
                        hcn: f32s(cap * sd)?,
                        hd: f32s(cap * rp)?,
                        ha: f32s(cap * rp)?,
                        hu: f32s(cap * sd)?,
                        inj: [f32s(cap * h.count)?, f32s(cap * h.count)?],
                        mo: f32s(cap * c.hidden)?,
                        ple_e: f32s(cap * c.ple.as_ref().map_or(1, |p| p.dim))?,
                        inv: f32s(cap * h.count)?,
                    })
                }
                None => None,
            },
        })
    }

    /// Rotation + per-row int8 quantization of the GEMM input into (xh, xsc), skipped when `act` is `Same` and the
    /// previous GEMM already quantized it this way.
    fn quant_row(&self, s: &Stream, xh: &DeviceBuffer, xsc: &DeviceBuffer, act: Act, k: usize, t: usize) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if matches!(act, Act::Same(_)) && self.quant_mode.load(Relaxed) == 1 {
            return Ok(());
        }
        ensure!(k <= 8 * 128 * 17, "rot_quant_row supports K up to 17408");
        let (x, u, mode, nw) = match act {
            Act::X(x) | Act::Same(x) => (x, x, 0i32, Arg::U64(0)),
            Act::SiluMul(g, u) => (g, u, 1, Arg::U64(0)),
            Act::Norm(x, w) => (x, x, 0, w.into()),
            Act::GatedNorm(x, z, w) => (x, z, if self.cfg.gate_sigmoid { 4 } else { 2 }, w.into()),
            Act::AttnGate(x, g) => (x, g, 3, Arg::U64(0)),
        };
        let qa = [
            xh.into(),
            xsc.into(),
            x.into(),
            u.into(),
            (&self.signs).into(),
            k.into(),
            mode.into(),
            nw,
            self.cfg.eps.into(),
        ];
        self.k
            .get("rot_quant_row")
            .launch((t as u32, 1, 1), (256, 1, 1), 0, s, &qa)?;
        self.quant_mode.store(1, Relaxed);
        Ok(())
    }

    /// Rotation + quantization of the GEMM input into (xh, xsc) with the input transform folded in, one scale per 128
    /// group: the counterpart of `quant_row` for the projections whose input needs a transform (`Act::Norm`,
    /// `GatedNorm`, `AttnGate`) and that would otherwise stay on int8 and on the MLS8 weight copy. `bits` = 4 (int4,
    /// `MALLEUS_DENSE_W4A4=acts`) or 8 (int8, `MALLEUS_DENSE_W4A4=w4a8`).
    fn quant_row_g(
        &self,
        s: &Stream,
        xh: &DeviceBuffer,
        xsc: &DeviceBuffer,
        act: Act,
        k: usize,
        t: usize,
        bits: u8,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        ensure!(k <= 8 * 128 * 17, "rot_quant_*_row supports K up to 17408");
        let (x, u, mode, nw) = match act {
            Act::X(x) | Act::Same(x) => (x, x, 0i32, Arg::U64(0)),
            Act::SiluMul(g, u) => (g, u, 1, Arg::U64(0)),
            Act::Norm(x, w) => (x, x, 0, w.into()),
            Act::GatedNorm(x, z, w) => (x, z, if self.cfg.gate_sigmoid { 4 } else { 2 }, w.into()),
            Act::AttnGate(x, g) => (x, g, 3, Arg::U64(0)),
        };
        let qa = [
            xh.into(),
            xsc.into(),
            x.into(),
            u.into(),
            (&self.signs).into(),
            k.into(),
            mode.into(),
            nw,
            self.cfg.eps.into(),
        ];
        self.k
            .get(if bits == 4 {
                "rot_quant_i4_row"
            } else {
                "rot_quant_i8_row"
            })
            .launch((t as u32, 1, 1), (256, 1, 1), 0, s, &qa)?;
        self.quant_mode.store(if bits == 4 { 3 } else { 2 }, Relaxed);
        Ok(())
    }

    /// Whether `w` runs on the per-row int8 activation path for t tokens.
    fn row8(&self, w: &Linear, t: usize) -> bool {
        self.fused_ok(t) && (w.prefill.is_some() || matches!(w.w, Weights::Mls8 { .. }))
    }

    fn fused_ok(&self, t: usize) -> bool {
        t >= WMMA_MIN_TOKENS && self.prefill_a8 && t > self.w4_max_t
    }

    /// One int8 GEMM for projections sharing an input: outputs split by segment, or silu(gate) * up for `glu`.
    fn gemm_fused(
        &self,
        s: &Stream,
        sc: &Scratch,
        outs: [&DeviceBuffer; 3],
        act: Act,
        f: &Fused,
        t: usize,
    ) -> Result<()> {
        self.quant_row(s, &sc.xh, &sc.xsc, act, f.k, t)?;
        let (name, bm, bn) = i8_kernel(t, f.glu);
        let grid = ((f.n.div_ceil(bn) * t.div_ceil(bm)) as u32, 1, 1);
        let args = [
            outs[0].into(),
            outs[1].into(),
            outs[2].into(),
            f.segs[0].into(),
            f.segs[1].into(),
            (&sc.xh).into(),
            (&sc.xsc).into(),
            (&f.q).into(),
            (&f.sc).into(),
            t.into(),
            f.n.into(),
            f.k.into(),
            0i32.into(),
        ];
        self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
        Ok(())
    }

    fn gemm(
        &self,
        s: &Stream,
        xr: &DeviceBuffer,
        xh: &DeviceBuffer,
        xsc: &DeviceBuffer,
        out: &DeviceBuffer,
        act: Act,
        w: &Linear,
        t: usize,
        acc: bool,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let a8 = t >= WMMA_MIN_TOKENS && self.prefill_a8 && !matches!(w.w, Weights::Bf16(_));
        // int4 activations. The plain inputs go through the per-chunk quantizer (`rot_quant_i4_w`); an input that needs
        // a transform folds it in `rot_quant_i4_row`, the int4 twin of the int8 row path, so o_proj and out_proj no
        // longer have to stay on int8 and on the MLS8 copy of their weights.
        let plain = matches!(act, Act::X(_) | Act::Same(_) | Act::SiluMul(..));
        let a4 =
            dense_w4a4() && matches!(w.w, Weights::Mls4 { .. }) && (plain || (dense_w4a4_acts() && self.fused_ok(t)));
        // Weights at 4 bits with int8 activations for a transform input: only the weight bytes change, and those are
        // what the MLS8 copy costs.
        let a8g = !a4
            && !plain
            && dense_w4a4()
            && dense_w4a8_acts()
            && matches!(w.w, Weights::Mls4 { .. })
            && self.fused_ok(t);
        let row8 = self.row8(w, t) && !a4 && !a8g;
        let x = match act {
            Act::X(x) | Act::Same(x) => x,
            Act::Norm(..) | Act::GatedNorm(..) | Act::AttnGate(..) if !row8 && !a4 && !a8g => {
                bail!("this input transform needs the row quantizer (int8 or int4)")
            }
            Act::Norm(x, _) | Act::GatedNorm(x, ..) | Act::AttnGate(x, _) => x,
            Act::SiluMul(g, u) => {
                if !row8 {
                    let n = t * w.k;
                    self.ew(s, "silu_mul", n, &[g.into(), g.into(), u.into(), n.into()])?;
                }
                g
            }
        };
        let reuse = |mode: u8| matches!(act, Act::Same(_)) && self.quant_mode.load(Relaxed) == mode;
        let tail = |q: &DeviceBuffer, sc: &DeviceBuffer| -> Vec<Arg> {
            vec![
                out.into(),
                xr.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ]
        };
        if let Weights::Bf16(b) = &w.w {
            let grid = (w.n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1);
            let args = [
                out.into(),
                x.into(),
                b.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ];
            self.k.get("gemm_bf16").launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        let chunks = (t * w.k / 128) as u32;
        if a8 {
            let (q, sc, w4) = match (&w.w, &w.prefill) {
                (Weights::Mls4 { q, sc }, _) if t <= self.w4_max_t || a4 || a8g => (q, sc, true),
                (_, Some((q, sc))) => (q, sc, false),
                (Weights::Mls4 { q, sc }, None) => (q, sc, true),
                (Weights::Mls8 { q, sc }, None) => (q, sc, false),
                (Weights::Mls3 { .. }, None) => {
                    bail!("MLS3 prefill (W3A8) is not wired yet: docs/ling3-engine.md, increment I7")
                }
                (Weights::Bf16(_), _) => unreachable!(),
            };
            debug_assert_eq!(w4, !row8);
            let (name, bm) = if w4 {
                // quant_mode 2 is int8 activations and 3 is int4: the same input quantized the other way is not
                // reusable, so the two paths get their own mode.
                let mode = u8::from(a4) + 2;
                if !reuse(mode) {
                    if a4 && !plain {
                        self.quant_row_g(s, xh, xsc, act, w.k, t, 4)?;
                    } else if a8g {
                        self.quant_row_g(s, xh, xsc, act, w.k, t, 8)?;
                    } else {
                        let qa = [xh.into(), xsc.into(), x.into(), (&self.signs).into(), w.k.into()];
                        self.k
                            .get(if a4 { "rot_quant_i4_w" } else { "rot_quant_i8_w" })
                            .launch((chunks, 1, 1), (32, 1, 1), 0, s, &qa)?;
                        self.quant_mode.store(mode, Relaxed);
                    }
                }
                match (t, a4) {
                    (0..=32, true) => ("gemm_w4a4_m32", 32),
                    (33..=64, true) => ("gemm_w4a4_m64", 64),
                    (_, true) => ("gemm_w4a4", 128),
                    (0..=32, false) => ("gemm_w4a8_m32", 32),
                    (33..=64, false) => ("gemm_w4a8_m64", 64),
                    (_, false) => ("gemm_w4a8", 128),
                }
            } else {
                self.quant_row(s, xh, xsc, act, w.k, t)?;
                // Few rows and a long K (o_proj, out_proj): 128 x 128 tiles are ~25% faster.
                let (name, bm, bn) = if w.n <= 4096 && w.k >= 4096 {
                    ("gemm_i8_m128", 128, 128)
                } else {
                    i8_kernel(t, false)
                };
                let grid = ((w.n.div_ceil(bn) * t.div_ceil(bm)) as u32, 1, 1);
                let args = [
                    out.into(),
                    out.into(),
                    out.into(),
                    w.n.into(),
                    0usize.into(),
                    xh.into(),
                    xsc.into(),
                    q.into(),
                    sc.into(),
                    t.into(),
                    w.n.into(),
                    w.k.into(),
                    (acc as i32).into(),
                ];
                self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
                return Ok(());
            };
            let grid = ((w.n.div_ceil(128) * t.div_ceil(bm)) as u32, 1, 1);
            let args = [
                out.into(),
                xh.into(),
                xsc.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
            ];
            self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        if t >= WMMA_MIN_TOKENS {
            let (q, sc, int8) = match (&w.w, &w.prefill) {
                (_, Some((q, sc))) => (q, sc, 1i32),
                (Weights::Mls4 { q, sc }, None) => (q, sc, 0),
                (Weights::Mls8 { q, sc }, None) => (q, sc, 1),
                (Weights::Mls3 { .. }, None) => {
                    bail!("MLS3 prefill (W3A8) is not wired yet: docs/ling3-engine.md, increment I7")
                }
                (Weights::Bf16(_), _) => unreachable!(),
            };
            let rot = [xh.into(), x.into(), (&self.signs).into(), w.k.into()];
            self.k
                .get("hadamard128_h")
                .launch((chunks, 1, 1), (128, 1, 1), 0, s, &rot)?;
            let (name, bm) = match t {
                0..=32 => ("gemm_wmma_m32", 32),
                33..=64 => ("gemm_wmma_m64", 64),
                _ => ("gemm_wmma", 128),
            };
            let grid = (w.n.div_ceil(128) as u32, t.div_ceil(bm) as u32, 1);
            let args = [
                out.into(),
                xh.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
                int8.into(),
            ];
            self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
            return Ok(());
        }
        // MLS4 GEMVs of a few tokens rotate their input while staging it.
        let fused_rot =
            matches!(&w.w, Weights::Mls4 { .. }) && (1..=GEMV_MAX_T).contains(&t) && w.k.is_multiple_of(128);
        if !fused_rot {
            let rot = [xr.into(), x.into(), (&self.signs).into(), w.k.into()];
            self.k
                .get("hadamard128")
                .launch((chunks, 1, 1), (128, 1, 1), 0, s, &rot)?;
        }
        let tail_rot = |q: &DeviceBuffer, sc: &DeviceBuffer| -> Vec<Arg> {
            vec![
                out.into(),
                x.into(),
                q.into(),
                sc.into(),
                t.into(),
                w.n.into(),
                w.k.into(),
                (acc as i32).into(),
                (&self.signs).into(),
            ]
        };
        const L: [&str; GEMV_MAX_T] = [
            "gemv_mls4_l1",
            "gemv_mls4_l2",
            "gemv_mls4_l3",
            "gemv_mls4_l4",
            "gemv_mls4_l5",
            "gemv_mls4_l6",
        ];
        const L8: [&str; GEMV_MAX_T] = [
            "gemv_mls8",
            "gemv_mls8_t2",
            "gemv_mls8_t3",
            "gemv_mls8_t4",
            "gemv_mls8_t5",
            "gemv_mls8_t6",
        ];
        let (name, q, sc) = match (&w.w, &w.prefill, t) {
            (Weights::Mls4 { q, sc }, _, 1..=GEMV_MAX_T) if fused_rot => (L[t - 1], q, sc),
            (Weights::Mls4 { q, sc }, _, 1) => ("gemv_mls4", q, sc),
            // The MLS3 GEMV/GEMM take already-rotated activations and the same argument list as MLS4, but there
            // is no fused-rotation variant of them yet (docs/model-ling3-flash.md, § MLS3: da fare).
            (Weights::Mls3 { q, sc }, _, 1) => ("gemv_mls3", q, sc),
            (Weights::Mls3 { q, sc }, _, _) => ("gemm_mls3", q, sc),
            (Weights::Mls8 { q, sc }, _, 1..=GEMV_MAX_T) => (L8[t - 1], q, sc),
            (_, Some((q, sc)), _) => ("gemm_mls8", q, sc),
            (Weights::Mls4 { q, sc }, _, _) => ("gemm_mls4", q, sc),
            (Weights::Mls8 { q, sc }, _, _) => ("gemm_mls8", q, sc),
            (Weights::Bf16(_), _, _) => unreachable!(),
        };
        let grid = if t <= GEMV_MAX_T {
            (w.n.div_ceil(32) as u32, 1, 1)
        } else {
            (w.n.div_ceil(16) as u32, t.div_ceil(8) as u32, 1)
        };
        if name == "gemv_mls4_l1" && w.n <= 4096 {
            // Few rows (o_proj, out_proj): one row per warp gives the GPU 4x the blocks.
            let g = (w.n.div_ceil(8) as u32, 1, 1);
            self.k
                .get("gemv_mls4_r1")
                .launch(g, (256, 1, 1), 0, s, &tail_rot(q, sc))?;
            return Ok(());
        }
        let args = if fused_rot { tail_rot(q, sc) } else { tail(q, sc) };
        self.k.get(name).launch(grid, (256, 1, 1), 0, s, &args)?;
        Ok(())
    }

    fn rmsnorm(
        &self,
        s: &Stream,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        w: &DeviceBuffer,
        rows: usize,
        n: usize,
    ) -> Result<()> {
        self.k.get("rmsnorm").launch(
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                out.into(),
                x.into(),
                w.into(),
                n.into(),
                self.cfg.eps.into(),
                1i32.into(),
            ],
        )?;
        Ok(())
    }

    /// Hyper-connection mix of the streams x [t, C * H] into out [t, H]; with `inj_out` also the injection weights of
    /// the paired combine. `inject` (block output, its injection weights) is first added to the streams.
    #[allow(clippy::too_many_arguments)]
    fn hc_mix(
        &self,
        st: &Stream,
        sc: &Scratch,
        h: &Hc,
        x: &DeviceBuffer,
        out: &DeviceBuffer,
        inj_out: Option<&DeviceBuffer>,
        t: usize,
        inject: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> Result<()> {
        let c = &self.cfg;
        let hcfg = c.hc.as_ref().context("no hyper-connections")?;
        let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
        let (cnt, hd) = (hcfg.count, c.hidden);
        let rp = h.down.n;
        let sd = cnt * hd;
        const SPLIT: usize = 4;
        if let (Weights::Mls8 { q: dq, sc: ds }, Weights::Mls8 { q: uq, sc: us }) = (&h.down.w, &h.up.w)
            && t <= GEMV_MAX_T
            && cnt <= 4
            && sd.is_multiple_of(SPLIT * 512)
            && rp.is_multiple_of(128)
        {
            // Single token only: with more tokens every block of the fused down kernel normalizes all of them.
            if t == 1 && hd <= 2560 && rp <= 384 && std::env::var("MALLEUS_HC_FUSED").map_or(true, |v| v != "0") {
                // Two kernels: inject + norm + rotation + split-K down per stream; gate + up + mix + state update.
                let (o, ij) = match inject {
                    Some((o, ij)) => (o.into(), ij.into()),
                    None => (Arg::U64(0), Arg::U64(0)),
                };
                self.k.get("hc_down_fused").launch(
                    (rp.div_ceil(8) as u32, cnt as u32, 1),
                    (256, 1, 1),
                    (t * hd * 2) as u32,
                    st,
                    &[
                        (&hs.hu).into(),
                        (&hs.inv).into(),
                        x.into(),
                        o,
                        ij,
                        (&h.norm).into(),
                        dq.into(),
                        ds.into(),
                        (&self.signs).into(),
                        t.into(),
                        rp.into(),
                        cnt.into(),
                        hd.into(),
                        c.eps.into(),
                    ],
                )?;
                ensure!(h.up_inter, "hyper-connection up rows not reordered");
                // Gate once (rotated a in hc.ha, injection weights), read by every block of the up kernel.
                self.k.get("hc_gate_dec").launch(
                    ((t * rp / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&hs.ha).into(),
                        inj_out.map_or(Arg::U64(0), |b| b.into()),
                        (&hs.hu).into(),
                        t.into(),
                        hcfg.rank.into(),
                        rp.into(),
                        cnt.into(),
                        cnt.into(),
                        (&self.signs).into(),
                    ],
                )?;
                self.k.get("hc_up_fused").launch(
                    (hd.div_ceil(8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        x.into(),
                        o,
                        ij,
                        Arg::U64(0),
                        (&hs.hu).into(),
                        uq.into(),
                        us.into(),
                        (&h.norm).into(),
                        (&hs.inv).into(),
                        (&self.signs).into(),
                        t.into(),
                        hcfg.rank.into(),
                        rp.into(),
                        cnt.into(),
                        hd.into(),
                        (&hs.ha).into(),
                    ],
                )?;
                return Ok(());
            }
            if let Some((o, ij)) = inject {
                self.hc_inject(st, x, o, ij, t)?;
            }
            // Few tokens: norm + rotation, split-K down, gate + rotation, up with the mix in its epilogue.
            self.k.get("hc_pre_dec").launch(
                ((t * sd / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&sc.xr).into(),
                    (&hs.hcn).into(),
                    x.into(),
                    (&h.norm).into(),
                    (&self.signs).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
            self.k.get("hc_down_dec").launch(
                (rp.div_ceil(8) as u32, SPLIT as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hu).into(),
                    (&sc.xr).into(),
                    dq.into(),
                    ds.into(),
                    t.into(),
                    rp.into(),
                    sd.into(),
                    (sd / SPLIT).into(),
                ],
            )?;
            let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
            self.k.get("hc_gate_dec").launch(
                ((t * rp / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&hs.ha).into(),
                    inj_arg,
                    (&hs.hu).into(),
                    t.into(),
                    hcfg.rank.into(),
                    rp.into(),
                    cnt.into(),
                    SPLIT.into(),
                    (&self.signs).into(),
                ],
            )?;
            self.k.get("hc_up_mix_dec").launch(
                (hd.div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    out.into(),
                    (&hs.hcn).into(),
                    (&hs.ha).into(),
                    uq.into(),
                    us.into(),
                    t.into(),
                    cnt.into(),
                    hd.into(),
                    rp.into(),
                ],
            )?;
            self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if self.row8(&h.down, t) && self.row8(&h.up, t) && cnt <= 4 && sd <= 8 * 128 * 10 {
            // Many tokens: inject + per-stream norm + rotation + int8 in one pass over the streams, the int8 GEMMs,
            // and the mix normalizing the streams again from the saved 1 / rms.
            let (o, ij) = match inject {
                Some((o, ij)) => (o.into(), ij.into()),
                None => (Arg::U64(0), Arg::U64(0)),
            };
            self.k.get("hc_quant_row").launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    (&hs.inv).into(),
                    x.into(),
                    o,
                    ij,
                    (&h.norm).into(),
                    (&self.signs).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
            // mix_down has few rows: split K over 4 slices (partials in hc.hu, summed by hc_gate).
            let (dq, ds) = mls8_of(&h.down)?;
            let (uq, us) = mls8_of(&h.up)?;
            const SK: usize = 4;
            self.k.get("gemm_i8_sk").launch(
                ((rp.div_ceil(128) * t.div_ceil(256)) as u32, SK as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hu).into(),
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    dq.into(),
                    ds.into(),
                    t.into(),
                    rp.into(),
                    (sd / SK).into(),
                    sd.into(),
                ],
            )?;
            let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
            self.ew(
                st,
                "hc_gate",
                t * rp,
                &[
                    (&hs.ha).into(),
                    inj_arg,
                    (&hs.hu).into(),
                    t.into(),
                    hcfg.rank.into(),
                    rp.into(),
                    cnt.into(),
                    SK.into(),
                ],
            )?;
            self.quant_row(st, &sc.xh, &sc.xsc, Act::X(&hs.ha), rp, t)?;
            ensure!(h.up_inter, "hyper-connection up rows not reordered");
            self.k.get("gemm_i8_hcmix").launch(
                ((sd.div_ceil(128) * t.div_ceil(128)) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    out.into(),
                    (&sc.xh).into(),
                    (&sc.xsc).into(),
                    uq.into(),
                    us.into(),
                    t.into(),
                    sd.into(),
                    rp.into(),
                    x.into(),
                    (&hs.inv).into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                ],
            )?;
            self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if let Some((o, ij)) = inject {
            self.k.get("hc_inject_norm").launch(
                ((t * cnt) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hcn).into(),
                    x.into(),
                    o.into(),
                    ij.into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
        } else {
            self.k.get("hc_norm").launch(
                ((t * cnt) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&hs.hcn).into(),
                    x.into(),
                    (&h.norm).into(),
                    cnt.into(),
                    hd.into(),
                    c.eps.into(),
                ],
            )?;
        }
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hd, Act::X(&hs.hcn), &h.down, t, false)?;
        let inj_arg: Arg = inj_out.map_or(Arg::U64(0), |b| b.into());
        self.ew(
            st,
            "hc_gate",
            t * rp,
            &[
                (&hs.ha).into(),
                inj_arg,
                (&hs.hd).into(),
                t.into(),
                hcfg.rank.into(),
                rp.into(),
                cnt.into(),
                1usize.into(),
            ],
        )?;
        ensure!(h.up.k == rp, "hyper-connection up K {} != down rows {rp}", h.up.k);
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hu, Act::X(&hs.ha), &h.up, t, false)?;
        self.ew(
            st,
            "hc_mix",
            t * hd,
            &[
                out.into(),
                (&hs.hu).into(),
                (&hs.hcn).into(),
                t.into(),
                cnt.into(),
                hd.into(),
                (h.up_inter as i32).into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// x[t, s] += o[t] * inj[t, s].
    fn hc_inject(&self, st: &Stream, x: &DeviceBuffer, o: &DeviceBuffer, inj: &DeviceBuffer, t: usize) -> Result<()> {
        let c = &self.cfg;
        let cnt = c.hc.as_ref().context("no hyper-connections")?.count;
        self.ew(
            st,
            "hc_inject",
            t * cnt * c.hidden,
            &[x.into(), o.into(), inj.into(), t.into(), cnt.into(), c.hidden.into()],
        )
    }

    /// PLE: streams += gv + silu(conv(norm(gv))), gv = sigmoid(gate(key_s(e), stream_s)) * value(e), with the n-gram
    /// embedding e in sc.hc.ple_e.
    fn ple(
        &self,
        st: &Stream,
        sc: &Scratch,
        p: &Ple,
        state: &DeviceBuffer,
        snap: Option<&DeviceBuffer>,
        t: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let (hcfg, pc) = (c.hc.as_ref().context("no hc")?, c.ple.as_ref().context("no ple")?);
        let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
        let (cnt, hd, sd) = (hcfg.count, c.hidden, c.stream_dim());
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &hs.hu, Act::X(&hs.ple_e), &p.key, t, false)?;
        self.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &hs.mo,
            Act::Same(&hs.ple_e),
            &p.value,
            t,
            false,
        )?;
        self.k.get("ple_gate_norm").launch(
            ((t * cnt) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&hs.hcn).into(),
                (&hs.hu).into(),
                (&hs.hu).into(),
                (&hs.hc).into(),
                (&hs.mo).into(),
                (&p.norm_key).into(),
                (&p.norm_query).into(),
                (&p.norm_conv).into(),
                cnt.into(),
                hd.into(),
                c.eps.into(),
            ],
        )?;
        self.ew(
            st,
            "ple_conv",
            t * sd,
            &[
                (&hs.hc).into(),
                (&hs.hcn).into(),
                (&hs.hu).into(),
                state.into(),
                (&p.conv).into(),
                t.into(),
                sd.into(),
                pc.conv_k.into(),
                pc.ngram.into(),
            ],
        )?;
        if let Some(snap) = snap {
            // Verify: keep the state before the step and the step's normalized rows for `rollback`.
            let sl = pc.conv_state() * sd;
            ensure!(snap.len() >= (sl + t * sd) * 4, "PLE snapshot too small");
            self.ew(st, "copy_f32", sl, &[snap.into(), state.into(), sl.into()])?;
            let tail = Arg::U64(snap.addr() + sl as u64 * 4);
            self.ew(st, "copy_f32", t * sd, &[tail, (&hs.hu).into(), (t * sd).into()])?;
        }
        self.ew(
            st,
            "ple_state_save",
            sd,
            &[
                state.into(),
                (&hs.hu).into(),
                t.into(),
                sd.into(),
                pc.conv_state().into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Decode GEMV of merged projections on sc.hn (rotated into sc.xr).
    fn gemv_merged(&self, st: &Stream, sc: &Scratch, m: &Merged, outs: [&DeviceBuffer; 4], t: usize) -> Result<()> {
        const M: [&str; GEMV_MAX_T] = [
            "gemv_mls4_m1",
            "gemv_mls4_m2",
            "gemv_mls4_m3",
            "gemv_mls4_m4",
            "gemv_mls4_m5",
            "gemv_mls4_m6",
        ];
        self.k.get(M[t - 1]).launch(
            (m.n.div_ceil(32) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                outs[0].into(),
                outs[1].into(),
                outs[2].into(),
                outs[3].into(),
                m.bounds[0].into(),
                m.bounds[1].into(),
                m.bounds[2].into(),
                (&sc.hn).into(),
                (&m.q).into(),
                (&m.sc).into(),
                t.into(),
                m.n.into(),
                m.k.into(),
                (&self.signs).into(),
            ],
        )?;
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Words of a block-selection bitset for a session.
    fn idx_words(&self, max_ctx: usize) -> usize {
        self.cfg
            .indexer
            .as_ref()
            .map_or(0, |ic| (max_ctx / ic.ratio + 1).div_ceil(32))
    }

    /// Whether an input of t tokens after `len` has a query with more complete blocks than the indexer budget.
    fn idx_sparse(&self, len: usize, t: usize) -> bool {
        self.cfg
            .indexer
            .as_ref()
            .is_some_and(|ic| (len + t) / ic.ratio > ic.budget / ic.ratio)
    }

    /// Grows the indexer score and selection buffers of the session for an input of t tokens after `len`.
    fn idx_grow_len(&self, sess: &mut Session, len: usize, t: usize) -> Result<()> {
        if !self.idx_sparse(len, t) {
            return Ok(());
        }
        let w = self.idx_words(sess.max_ctx);
        if sess.idx_scores.is_none() {
            sess.idx_scores = Some(f32s(IDX_SUB * w * 32)?);
        }
        if let Some(ic) = &self.cfg.indexer
            && sess.idx_list.as_ref().is_none_or(|l| l.1 < t)
        {
            let rows = t.max(ATTN_WMMA_MIN);
            sess.idx_list = Some((DeviceBuffer::alloc(rows * (ic.budget / ic.ratio + 1) * 4)?, rows));
        }
        if sess.idx_sel.as_ref().is_none_or(|s| s.2 < t) {
            let rows = t.max(64);
            sess.idx_sel = Some((
                DeviceBuffer::alloc(rows * w * 4)?,
                DeviceBuffer::alloc(rows.div_ceil(16) * w * 4)?,
                rows,
            ));
        }
        Ok(())
    }

    /// The selection buffers for an input of t tokens after `len`, when it needs sparse attention.
    #[allow(clippy::type_complexity)]
    fn idx_sel_bufs<'a>(
        &self,
        sess: &'a Session,
        len: usize,
        t: usize,
    ) -> Result<Option<(&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer, usize)>> {
        if !self.idx_sparse(len, t) {
            return Ok(None);
        }
        let (sel, uni, _) = sess.idx_sel.as_ref().context("indexer buffers")?;
        Ok(Some((
            sess.idx_scores.as_ref().context("indexer buffers")?,
            sel,
            uni,
            self.idx_words(sess.max_ctx),
        )))
    }

    fn idx_states<'a>(&self, sess: &'a Session, t: usize, reuse: bool) -> Result<Option<IdxStates<'a>>> {
        if self.cfg.indexer.is_none() {
            return Ok(None);
        }
        Ok(Some(IdxStates {
            kb: &sess.idx_kb,
            tk: &sess.idx_tk,
            tp: &sess.idx_tp,
            sel: self.idx_sel_bufs(sess, sess.len, t)?,
            list: sess.idx_list.as_ref().map(|(b, r)| (b, *r)),
            cache: Some(&sess.idx_sel_cache),
            reuse,
        }))
    }

    /// QSA indexer of attention layer `li` for t tokens after `len` (input in sc.hn): pooled keys of the completed
    /// blocks, the tail of the last incomplete block and, when sparse, the block selection of every query.
    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &self,
        st: &Stream,
        sc: &Scratch,
        ix: &Indexer,
        is: &IdxStates,
        li: usize,
        len: usize,
        t: usize,
        proj_ready: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let ic = c.indexer.as_ref().context("no indexer config")?;
        let [proj, qi, kr] = sc.idx.as_ref().context("no indexer scratch")?;
        let r = ic.ratio;
        let [_, sh, sw] = c.mrope_section;
        if !proj_ready {
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, proj, Act::X(&sc.hn), &ix.qk, t, false)?;
        }
        self.k.get("idx_prep").launch(
            (t as u32, (ic.heads + 1) as u32, 1),
            (128, 1, 1),
            0,
            st,
            &[
                qi.into(),
                kr.into(),
                proj.into(),
                (&ix.q_norm).into(),
                (&sc.pos).into(),
                ic.heads.into(),
                c.eps.into(),
                c.rope_theta.into(),
                c.rot_dim.into(),
                sh.into(),
                sw.into(),
                self.rope_invf.as_ref().map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
            ],
        )?;
        // Pooling the keys of every block that completes this step. The kernel reads len from the device buffer so the
        // node params stay constant for a future graph capture; the grid still follows the host-side new-block count
        // (making it capture-constant belongs with the capture wiring, since a fixed max grid is a pre-capture cost).
        let nb_new = (len + t) / r - len / r;
        if nb_new > 0 {
            self.k.get("idx_pool").launch(
                (nb_new as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&is.kb[li]).into(),
                    kr.into(),
                    (&is.tk[li]).into(),
                    (&is.tp[li]).into(),
                    (&sc.pos).into(),
                    (&ix.k_norm).into(),
                    (&sc.len_dev).into(),
                    r.into(),
                    c.eps.into(),
                    c.rope_theta.into(),
                    c.rot_dim.into(),
                    sh.into(),
                    sw.into(),
                    self.rope_invf.as_ref().map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
                ],
            )?;
        }
        self.k.get("idx_tail").launch(
            (1, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&is.tk[li]).into(),
                (&is.tp[li]).into(),
                kr.into(),
                (&sc.pos).into(),
                (&sc.len_dev).into(),
                t.into(),
                r.into(),
            ],
        )?;
        if let Some((scores, sel, uni, w)) = is.sel {
            let ld = w * 32;
            let mut reused = false;
            if is.reuse
                && let Some(cache) = is.cache.and_then(|c| c.get(li))
            {
                sel.copy_from_shared(0, cache, 0, w * 4, st)?;
                reused = true;
            }
            if !reused {
                for q0 in (0..t).step_by(IDX_SUB) {
                    let nq = IDX_SUB.min(t - q0);
                    let maxb = (len + q0 + nq) / r;
                    let sel12 = std::env::var("MALLEUS_IDX_SELECT")
                        .map_or(self.k.arch == "gfx1151" && maxb >= 12288, |v| v == "12");
                    // One score/selection per group of four query rows: the representative is the last row, and
                    // the gather masks the future keys it may name for the earlier rows. At 256K this is 232 -> 185 s
                    // and makes the 32K-256K curve flat (1 453 -> 1 422 tok/s); MALLEUS_IDX_GROUP=1 restores
                    // per-query selection.
                    let want = std::env::var("MALLEUS_IDX_GROUP")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| (1..=16).contains(&v))
                        .unwrap_or(if self.k.arch == "gfx1151" { 4 } else { 1 });
                    // Decode and verify have nq = 1..6: there is no group to share, and the score tile's
                    // representative rows would not even be written. The group is a prefill-only shape and must
                    // divide the sub-batch exactly, so the partial last sub-batch also falls back to per-query.
                    let group = if sel12 && want > 1 && nq >= want && nq.is_multiple_of(want) {
                        want
                    } else {
                        1
                    };
                    let tiled = std::env::var("MALLEUS_IDX_SCORE_TILE").map_or(self.k.arch == "gfx1151", |v| v == "1")
                        && ic.heads == 4;
                    if tiled {
                        // 16 query rows read each 64-key tile once: 7.95 -> 5.27 s at 98K tokens, scores bit-identical
                        // to `idx_score` (same (c, e) order per score). Prefill reuses the q tile over 8 key tiles;
                        // decode needs the key parallelism more than the q reuse, so one key tile per block there.
                        let kt = std::env::var("MALLEUS_IDX_KT")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .filter(|&v| v > 0)
                            .unwrap_or(if nq >= 64 { 8 } else { 1 });
                        self.k.get("idx_score_tile").launch(
                            (maxb.div_ceil(64 * kt) as u32, nq.div_ceil(16) as u32, 1),
                            (1024, 1, 1),
                            0,
                            st,
                            &[
                                scores.into(),
                                qi.into(),
                                (&is.kb[li]).into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                ld.into(),
                                nq.into(),
                                kt.into(),
                                group.into(),
                            ],
                        )?;
                    } else {
                        // Tiled over queries (IDX_TQ rows share one read of the pooled keys): see `idx_score`.
                        const IDX_TQ: usize = 4;
                        self.k.get("idx_score").launch(
                            (maxb.div_ceil(256) as u32, nq.div_ceil(IDX_TQ) as u32, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                scores.into(),
                                qi.into(),
                                (&is.kb[li]).into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                ic.heads.into(),
                                r.into(),
                                ld.into(),
                                nq.into(),
                            ],
                        )?;
                    }
                    // The 4096-bin radix pays 16 KB of LDS and three histogram sweeps; it wins from ~64K blocks
                    // (a 64K-token context) and loses below, where the 256-bin four-pass kernel is cheaper. At 98K
                    // it is 3.67 -> 2.51 s with the same bitset.
                    if sel12 {
                        // One query (decode/verify) is one block on one CU: 1024 threads hide the row's latency;
                        // prefill keeps the 256-thread tile.
                        let (name, threads) = if nq == 1 {
                            ("idx_select12_1024", 1024)
                        } else {
                            ("idx_select12", 256)
                        };
                        self.k.get(name).launch(
                            (nq.div_ceil(group) as u32, 1, 1),
                            (threads, 1, 1),
                            0,
                            st,
                            &[
                                sel.into(),
                                scores.into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                (ic.budget / r).into(),
                                ld.into(),
                                w.into(),
                                group.into(),
                                nq.into(),
                            ],
                        )?;
                    } else {
                        self.k.get("idx_select").launch(
                            (nq as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                sel.into(),
                                scores.into(),
                                (&sc.len_dev).into(),
                                q0.into(),
                                r.into(),
                                (ic.budget / r).into(),
                                ld.into(),
                                w.into(),
                            ],
                        )?;
                    }
                    if let Ok(dir) = std::env::var("MALLEUS_IDX_DUMP_SEL")
                        && q0 + nq == t
                    {
                        // Debug for the recall gate: the selection bitsets of the last sub-batch, one file per
                        // indexer layer. Rows are q0..t of the current chunk.
                        st.synchronize()?;
                        let v: Vec<u32> = sel.download_at(q0 * w * 4, nq * w)?;
                        std::fs::create_dir_all(&dir)?;
                        let mut out = format!("{len} {q0} {nq} {w}\n");
                        for i in 0..nq {
                            for j in 0..w {
                                out.push_str(&format!("{:08x} ", v[i * w + j]));
                            }
                            out.push('\n');
                        }
                        std::fs::write(format!("{dir}/L{li}.txt"), out)?;
                    }
                }
            }
            if !reused
                && t == 1
                && let Some(cache) = is.cache.and_then(|c| c.get(li))
            {
                cache.copy_from_shared(0, sel, 0, w * 4, st)?;
            }
            if !reused && std::env::var("MALLEUS_IDX_STATS").is_ok_and(|v| v == "all" || li == 0) {
                // Debug: density of the selections and of their unions over 16 and 64 queries.
                st.synchronize()?;
                let v: Vec<u32> = sel.download(t * w)?;
                let pop = |r: &[u32]| r.iter().map(|x| x.count_ones() as usize).sum::<usize>();
                let ncb = (len + t) / r;
                let uni = |g: usize| -> f64 {
                    let mut tot = 0;
                    for q0 in (0..t).step_by(g) {
                        let mut u = vec![0u32; w];
                        for q in q0..(q0 + g).min(t) {
                            for (a, b) in u.iter_mut().zip(&v[q * w..(q + 1) * w]) {
                                *a |= b;
                            }
                        }
                        tot += pop(&u);
                    }
                    tot as f64 / t.div_ceil(g) as f64
                };
                eprintln!(
                    "idx stats layer {li} len {len} t {t} blocks {ncb}: per query {:.0}, union4 {:.0}, union16 {:.0}, \
                     union64 {:.0}",
                    pop(&v) as f64 / t as f64,
                    uni(4),
                    uni(16),
                    uni(64)
                );
            }
            if t >= ATTN_WMMA_MIN {
                self.k.get("idx_union").launch(
                    (w.div_ceil(256) as u32, t.div_ceil(16) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[uni.into(), sel.into(), t.into(), w.into()],
                )?;
            }
        }
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn ew(&self, s: &Stream, name: &str, n: usize, args: &[Arg]) -> Result<()> {
        self.k
            .get(name)
            .launch((n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, s, args)?;
        Ok(())
    }

    /// Runs `tokens` through the model, appending to the session. Text positions are assigned automatically
    /// unless `positions` (T, H, W per token) is given. Returns logits for the last `n_logits` tokens.
    pub fn forward(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
    ) -> Result<Vec<f32>> {
        self.forward_mm(sess, tokens, positions, n_logits, &[])
    }

    /// Like `forward`, with rows `[start, start + n)` of the input embeddings replaced by `buf` ([n, hidden] f32).
    pub fn forward_mm(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
        inserts: &[(usize, &DeviceBuffer)],
    ) -> Result<Vec<f32>> {
        self.forward_impl(sess, tokens, positions, n_logits, inserts, false, false)
    }

    /// One-token greedy step: the argmax is computed on the device and only the token id comes back. The
    /// logits never cross the bus; `forward` keeps the full row for sampling and verification.
    pub fn forward_argmax(&self, sess: &mut Session, tokens: &[u32]) -> Result<u32> {
        let v = self.forward_impl(sess, tokens, None, 1, &[], false, true)?;
        Ok(v[0] as u32)
    }

    /// Speculative verify: runs 2..=GEMV_MAX_T tokens on the decode weights, keeping the DeltaNet state after each
    /// token so `rollback` can drop the rejected tail. Returns the logits of every token.
    pub fn forward_verify(&self, sess: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            (2..=GEMV_MAX_T).contains(&tokens.len()) && self.mtp.is_some(),
            "bad verify step"
        );
        self.forward_impl(sess, tokens, None, tokens.len(), &[], true, false)
    }

    /// Writes to `path` the attention mass collected with `MALLEUS_ATTN_MASS` (#96): a header
    /// `mstride tile_k mtiles` and then one `row tile max sum` line per non-empty tile. The exact mass is recovered
    /// afterwards: `lse = max + ln(sum)`, then `exp(lse - logsumexp(lse))`.
    pub fn dump_attn_mass(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write as _;
        let Some((buf, mstride, mtiles)) = &self.attn_mass else {
            return Ok(());
        };
        self.stream.synchronize()?;
        let rows = buf.len() / (mtiles * 8);
        let data: Vec<f32> = buf.download::<f32>(buf.len() / 4)?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "{mstride} 32 {mtiles}")?;
        for r in 0..rows {
            for t in 0..*mtiles {
                let (mx, sum) = (data[(r * mtiles + t) * 2], data[(r * mtiles + t) * 2 + 1]);
                if sum > 0.0 && mx.is_finite() {
                    writeln!(f, "{r} {t} {mx} {sum}")?;
                }
            }
        }
        Ok(())
    }

    /// Drops the last `n` tokens of the preceding `forward_verify`.
    pub fn rollback(&self, sess: &mut Session, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        ensure!(
            n < sess.snap_t,
            "rollback of {n} tokens after a {}-token verify",
            sess.snap_t
        );
        let keep = sess.snap_t - n;
        let st = Stream::null();
        let c = &self.cfg;
        if let (Some(p), Some(snap), Some(state)) = (&c.ple, &sess.ple_snap, &mut sess.ple_state) {
            let sd = c.stream_dim();
            state.copy_from(0, snap, keep * sd * 4, p.conv_state() * sd * 4, &st)?;
            let h = &sess.ngram_prev;
            ensure!(h.len() >= n + p.ngram - 1, "n-gram history of the verify step missing");
            sess.ngram_ctx = h[h.len() - n - (p.ngram - 1)..h.len() - n].to_vec();
        }
        let conv_b = c.lin_qkv_dim() * (c.conv_k - 1) * 4;
        for (dst, src) in sess.conv.iter_mut().zip(&sess.conv_snap) {
            dst.copy_from(0, src, (keep - 1) * conv_b, conv_b, &st)?;
        }
        st.synchronize()?;
        sess.dn_np = keep;
        sess.len -= n;
        sess.next_pos -= n as i64;
        sess.mtp_pos.truncate(keep);
        sess.snap_t = 0;
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Hidden rows waiting for their next token before the MTP head can use them.
    pub fn mtp_pending(&self, sess: &Session) -> usize {
        sess.mtp_pos.len()
    }

    /// Feeds the pending hidden rows with their next tokens to the MTP head; returns its greedy draft for the token
    /// after `next.last()` and the draft's probability.
    pub fn mtp_draft(&self, sess: &mut Session, next: &[u32]) -> Result<(u32, f32)> {
        let k = next.len();
        ensure!(
            k > 0 && k == sess.mtp_pos.len(),
            "MTP: {k} tokens for {} pending rows",
            sess.mtp_pos.len()
        );
        let c = &self.cfg;
        let st = Stream::null();
        let mut sc = self.take_scratch(k * c.stream_dim() / c.hidden)?;
        sc.mh.copy_from(
            0,
            sess.mtp_h.as_ref().context("no MTP state")?,
            0,
            k * c.stream_dim() * 4,
            &st,
        )?;
        let pos: Vec<i32> = sess.mtp_pos.drain(..).flatten().collect();
        let logits = self.mtp_rows(sess, &mut sc, next, &pos, Some(false), &st)?;
        *self.scratch_cache.lock().unwrap() = Some(sc);
        logits.context("MTP logits")
    }

    /// Next draft (and its probability) from the MTP head applied to its own last output and the previous draft.
    pub fn mtp_chain(&self, sess: &mut Session, draft: u32) -> Result<(u32, f32)> {
        let c = &self.cfg;
        let st = Stream::null();
        let mut sc = self.take_scratch(c.stream_dim() / c.hidden)?;
        sc.mh.copy_from(
            0,
            sess.mtp_out.as_ref().context("no MTP state")?,
            0,
            c.stream_dim() * 4,
            &st,
        )?;
        let pos = sess.mtp_out_pos.map(|p| p + 1);
        let logits = self.mtp_rows(sess, &mut sc, &[draft], &pos, Some(true), &st)?;
        *self.scratch_cache.lock().unwrap() = Some(sc);
        logits.context("MTP logits")
    }

    /// Runs the MTP head on the rows of `sc.mh` paired with `ids` and appends them to its KV cache.
    fn mtp_rows(
        &self,
        sess: &mut Session,
        sc: &mut Scratch,
        ids: &[u32],
        pos: &[i32],
        logits: Option<bool>,
        st: &Stream,
    ) -> Result<Option<(u32, f32)>> {
        let c = &self.cfg;
        let m = self.mtp.as_ref().context("no MTP head")?;
        let r = ids.len();
        let chain = logits == Some(true);
        if !chain {
            sess.mtp_len -= sess.mtp_chain;
            sess.mtp_chain = 0;
        }
        if sess.mtp_len + r > sess.max_ctx {
            return Ok(None);
        }
        let idv: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        // The MTP head runs through run_layers like the main model, so its attention kernels read the
        // KV offset from sc.len_dev too — here the head's own cache position, not the main model's.
        sc.upload_step(sess.mtp_len as i32, &idv, pos)?;
        self.k.get("embed_bf16").launch(
            (r as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&sc.hn).into(), (&self.embed).into(), (&sc.ids).into(), c.hidden.into()],
        )?;
        let normed = f32s(c.hidden)?;
        if let (Some(fc_e), Some(fc_h), Some(mixer), Some(hs)) = (&m.fc_e, &m.fc_h, &m.mixer, &sc.hc) {
            // Streams of the main model in sc.mh [r, C * H]: full-vector RMSNorm, fc_hidden on each stream, plus
            // fc_embedding(RMSNorm(embed(next))) on every stream.
            let sd = c.stream_dim();
            let cnt = sd / c.hidden;
            ensure!(sc.cap >= r * cnt, "MTP scratch too small");
            self.rmsnorm(st, &sc.small0, &sc.hn, &m.pre_e, r, c.hidden)?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &sc.small1,
                Act::X(&sc.small0),
                fc_e,
                r,
                false,
            )?;
            self.rmsnorm(st, &hs.hcn, &sc.mh, &m.pre_h, r, sd)?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &hs.hc,
                Act::X(&hs.hcn),
                fc_h,
                r * cnt,
                false,
            )?;
            self.ew(
                st,
                "hc_bcast_add",
                r * sd,
                &[
                    (&hs.hc).into(),
                    (&sc.small1).into(),
                    r.into(),
                    cnt.into(),
                    c.hidden.into(),
                ],
            )?;
            self.idx_grow_len(sess, sess.mtp_len, r)?;
            let mi = sess.mtp_idx.as_ref();
            let ls = LayerStates {
                kv: &sess.mtp_kv,
                conv: &[],
                rec: &[],
                len: sess.mtp_len,
                max_ctx: sess.max_ctx,
                attn_part: &sess.attn_part,
                mla_part: sess.mla_part.as_ref(),
                conv_snaps: None,
                dn: None,
                ple_state: None,
                ple_snap: None,
                idx: match mi {
                    Some([kb, tk, tp]) => Some(IdxStates {
                        kb: std::slice::from_ref(kb),
                        tk: std::slice::from_ref(tk),
                        tp: std::slice::from_ref(tp),
                        sel: self.idx_sel_bufs(sess, sess.mtp_len, r)?,
                        list: sess.idx_list.as_ref().map(|(b, r)| (b, *r)),
                        cache: None,
                        reuse: false,
                    }),
                    None => None,
                },
            };
            self.run_layers(std::slice::from_ref(&m.layer), &ls, sc, r, st, c.n_layers)?;
            sess.mtp_len += r;
            if chain {
                sess.mtp_chain += r;
            }
            if logits.is_none() {
                return Ok(None);
            }
            let out_h = sess.mtp_out.as_mut().context("no MTP state")?;
            out_h.copy_from(0, &hs.hc, (r - 1) * sd * 4, sd * 4, st)?;
            sess.mtp_out_pos = [pos[3 * (r - 1)], pos[3 * (r - 1) + 1], pos[3 * (r - 1) + 2]];
            let out_ro: &DeviceBuffer = out_h;
            self.hc_mix(st, sc, mixer, out_ro, &normed, None, 1, None)?;
        } else {
            if self.mtp_post_norm {
                let norm = self.norm.as_ref().context("no final norm")?;
                self.rmsnorm(st, &sc.small0, &sc.mh, norm, r, c.hidden)?;
                sc.mh.copy_from(0, &sc.small0, 0, r * c.hidden * 4, st)?;
            }
            self.k.get("mtp_cat").launch(
                (r as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&sc.cat).into(),
                    (&sc.hn).into(),
                    (&sc.mh).into(),
                    (&m.pre_e).into(),
                    (&m.pre_h).into(),
                    c.hidden.into(),
                    c.eps.into(),
                ],
            )?;
            self.gemm(
                st,
                &sc.xr,
                &sc.xh,
                &sc.xsc,
                &sc.x,
                Act::X(&sc.cat),
                m.fc.as_ref().context("no MTP fc")?,
                r,
                false,
            )?;
            let ls = LayerStates {
                kv: &sess.mtp_kv,
                conv: &[],
                rec: &[],
                len: sess.mtp_len,
                max_ctx: sess.max_ctx,
                attn_part: &sess.attn_part,
                mla_part: sess.mla_part.as_ref(),
                conv_snaps: None,
                dn: None,
                ple_state: None,
                ple_snap: None,
                idx: None,
            };
            self.run_layers(std::slice::from_ref(&m.layer), &ls, sc, r, st, c.n_layers)?;
            sess.mtp_len += r;
            if chain {
                sess.mtp_chain += r;
            }
            if logits.is_none() {
                return Ok(None);
            }
            let mut last = f32s(c.hidden)?;
            last.copy_from(0, &sc.x, (r - 1) * c.hidden * 4, c.hidden * 4, st)?;
            self.rmsnorm(st, &normed, &last, m.norm.as_ref().context("no MTP norm")?, 1, c.hidden)?;
            let out_h = sess.mtp_out.as_mut().context("no MTP state")?;
            out_h.copy_from(
                0,
                if self.mtp_chain_post { &normed } else { &last },
                0,
                c.hidden * 4,
                st,
            )?;
            sess.mtp_out_pos = [pos[3 * (r - 1)], pos[3 * (r - 1) + 1], pos[3 * (r - 1) + 2]];
        }
        let head = self.draft_head.as_ref().unwrap_or(&self.lm_head);
        let out = f32s(head.n)?;
        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &out, Act::X(&normed), head, 1, false)?;
        let top = f32s(2)?;
        self.k.get("logits_top1").launch(
            (1, 1, 1),
            (1024, 1, 1),
            0,
            st,
            &[(&top).into(), (&out).into(), head.n.into()],
        )?;
        st.synchronize()?;
        let t: Vec<f32> = top.download(2)?;
        let idx = t[0].to_bits();
        let tok = self.draft_map.as_ref().map_or(idx, |m| m[idx as usize]);
        Ok(Some((tok, t[1])))
    }

    fn forward_impl(
        &self,
        sess: &mut Session,
        tokens: &[u32],
        positions: Option<&[[i32; 3]]>,
        n_logits: usize,
        inserts: &[(usize, &DeviceBuffer)],
        verify: bool,
        argmax_only: bool,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let t = tokens.len();
        ensure!(t > 0, "empty input");
        let len0 = sess.len;
        // Decode selection reuse: one query, no complete block has appeared since the previous token. The cache
        // holds the previous token's bitset; the gather already masks keys after the current position.
        let reuse_enabled = std::env::var("MALLEUS_IDX_DECODE_REUSE").map_or(true, |v| v != "0");
        let idx_reuse = reuse_enabled
            && c.indexer.as_ref().is_some_and(|ic| {
                t == 1
                    && sess.idx_sel.is_some()
                    && sess.idx_cache_len == len0
                    && (len0 + 1) / ic.ratio == len0 / ic.ratio
            });
        ensure!(
            sess.len + t <= sess.max_ctx,
            "context full ({} + {} > {})",
            sess.len,
            t,
            sess.max_ctx
        );
        ensure!(n_logits >= 1 && n_logits <= t, "bad n_logits");
        // The model's own stream, made once: creating and destroying one per call cost about a millisecond of
        // idle GPU per decode step. It is a blocking stream, so it still orders against the synchronous host
        // copies and the null-stream sites. Results are read back only after `st.synchronize()` below.
        let st = &self.stream;
        if let Some(ng) = &self.ngram {
            ng.prefetch(&sess.ngram_ctx, tokens);
        }
        let mut sc = self.take_scratch(t)?;
        let ids: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
        let pos: Vec<i32> = match positions {
            Some(p) => {
                ensure!(p.len() == t, "positions length mismatch");
                p.iter().flatten().copied().collect()
            }
            None => (0..t).flat_map(|i| [(sess.next_pos + i as i64) as i32; 3]).collect(),
        };
        let max_pos = pos.iter().copied().max().unwrap_or(0) as i64;
        // The KV position this forward starts at is device-resident, so the attention kernels read it from
        // memory instead of a per-step launch immediate.
        sc.upload_step(sess.len as i32, &ids, &pos)?;
        self.k.get("embed_bf16").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&sc.x).into(), (&self.embed).into(), (&sc.ids).into(), c.hidden.into()],
        )?;
        for (start, buf) in inserts {
            let rows = buf.len() / (c.hidden * 4);
            ensure!(start + rows <= t, "embedding insert out of range");
            sc.x.copy_from(start * c.hidden * 4, buf, 0, rows * c.hidden * 4, st)?;
        }
        // The n-gram context before this input (its rows are read while the GPU runs the layers before PLE).
        let ng_ctx = sess.ngram_ctx.clone();
        if self.ngram.is_some() {
            let mut hist = std::mem::take(&mut sess.ngram_ctx);
            let keep = hist.len();
            hist.extend_from_slice(tokens);
            sess.ngram_ctx = hist[hist.len() - keep..].to_vec();
            sess.ngram_prev = if verify { hist } else { vec![] };
        }
        self.idx_grow_len(sess, sess.len, t)?;
        if let (Some(hcfg), Some(hs)) = (&c.hc, &sc.hc) {
            self.ew(
                st,
                "hc_expand",
                t * c.stream_dim(),
                &[
                    (&hs.hc).into(),
                    (&sc.x).into(),
                    t.into(),
                    hcfg.count.into(),
                    c.hidden.into(),
                ],
            )?;
        }
        let ls = LayerStates {
            kv: &sess.kv,
            conv: &sess.conv,
            rec: &sess.rec,
            len: sess.len,
            max_ctx: sess.max_ctx,
            attn_part: &sess.attn_part,
            mla_part: sess.mla_part.as_ref(),
            conv_snaps: verify.then_some(&sess.conv_snap[..]),
            dn: (!sess.dn_rec.is_empty()).then_some((&sess.dn_rec[..], sess.dn_np, sess.dn_par, verify)),
            ple_state: sess.ple_state.as_ref(),
            ple_snap: if verify { sess.ple_snap.as_ref() } else { None },
            idx: self.idx_states(sess, t, idx_reuse)?,
        };
        match (&self.ngram, &c.ple) {
            (Some(ng), Some(pc)) => {
                let split = pc.layer;
                let a0 = c.full_attn[..split].iter().filter(|f| **f).count();
                self.run_layers_from(&self.layers[..split], &ls, &sc, t, st, 0, (0, 0), false)?;
                let te = std::time::Instant::now();
                let e = ng.embed(&ng_ctx, tokens)?;
                if std::env::var_os("MALLEUS_TIMING").is_some() {
                    eprintln!("ngram embed {t} tokens: {:.1} ms", te.elapsed().as_secs_f64() * 1e3);
                }
                sc.hc
                    .as_mut()
                    .context("no hyper-connection scratch")?
                    .ple_e
                    .upload(&e)?;
                self.run_layers_from(&self.layers[split..], &ls, &sc, t, st, split, (a0, split - a0), true)?;
            }
            _ => self.run_layers(&self.layers, &ls, &sc, t, st, 0)?,
        }
        let first = t - n_logits;
        let normed = f32s(n_logits * c.hidden)?;
        match (&self.norm, &self.hc_final, &sc.hc) {
            (Some(norm), _, _) => {
                let mut last = f32s(n_logits * c.hidden)?;
                last.copy_from(0, &sc.x, first * c.hidden * 4, n_logits * c.hidden * 4, st)?;
                self.rmsnorm(st, &normed, &last, norm, n_logits, c.hidden)?;
            }
            (None, Some(hf), Some(hs)) => {
                let sd = c.stream_dim();
                let mut last = f32s(n_logits * sd)?;
                last.copy_from(0, &hs.hc, first * sd * 4, n_logits * sd * 4, st)?;
                self.hc_mix(st, &sc, hf, &last, &normed, None, n_logits, None)?;
            }
            _ => bail!("no final norm"),
        }
        let logits = f32s(n_logits * c.vocab)?;
        self.gemm(
            st,
            &sc.xr,
            &sc.xh,
            &sc.xsc,
            &logits,
            Act::X(&normed),
            &self.lm_head,
            n_logits,
            false,
        )?;
        if self.mtp.is_some() {
            self.mtp_after_forward(sess, &mut sc, tokens, &pos, verify, st)?;
        }
        st.synchronize()?;
        sess.len += t;
        if c.indexer.is_some() {
            sess.idx_cache_len = if t == 1 && self.idx_sparse(len0, 1) {
                len0 + 1
            } else {
                0
            };
        }
        sess.next_pos = max_pos + 1;
        sess.snap_t = if verify { t } else { 0 };
        if verify {
            sess.dn_par ^= 1;
            sess.dn_np = t;
        } else {
            sess.dn_np = 0;
        }
        *self.scratch_cache.lock().unwrap() = Some(sc);
        if argmax_only {
            ensure!(n_logits == 1, "forward_argmax needs one logits row");
            let out = DeviceBuffer::alloc(4)?;
            self.k.get("logits_argmax").launch(
                (1, 1, 1),
                (1024, 1, 1),
                0,
                st,
                &[(&out).into(), (&logits).into(), c.vocab.into()],
            )?;
            st.synchronize()?;
            let id: Vec<i32> = out.download(1)?;
            return Ok(vec![id[0] as f32]);
        }
        Ok(logits.download(n_logits * c.vocab)?)
    }

    /// After a forward (hidden rows in sc.x): a verify keeps all rows pending; otherwise the rows whose next token
    /// is known (the previous pending row and all but the last new one) go through the MTP head, the last waits.
    fn mtp_after_forward(
        &self,
        sess: &mut Session,
        sc: &mut Scratch,
        tokens: &[u32],
        pos: &[i32],
        verify: bool,
        st: &Stream,
    ) -> Result<()> {
        let c = &self.cfg;
        let t = tokens.len();
        let row = c.stream_dim() * 4;
        let rows_pos = |a: usize, b: usize| (a..b).map(|i| [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
        let mut mtp_h = sess.mtp_h.take().context("no MTP state")?;
        // MTP rows per call: the hyper-connection variant runs fc_hidden on C rows per token.
        let per = sc.cap / (c.stream_dim() / c.hidden);
        let res = (|| -> Result<()> {
            // The main model's output rows: its streams with hyper-connections, else the residual.
            let out_rows: &DeviceBuffer = sc.hc.as_ref().map_or(&sc.x, |hs| &hs.hc);
            if verify {
                mtp_h.copy_from(0, out_rows, 0, t * row, st)?;
                sess.mtp_pos = rows_pos(0, t).collect();
                return Ok(());
            }
            let pend = usize::from(sess.mtp_pos.len() == 1);
            let mut p: Vec<i32> = if pend == 1 { sess.mtp_pos[0].to_vec() } else { vec![] };
            p.extend(rows_pos(0, t - 1).flatten());
            let ids = &tokens[1 - pend..];
            let n = ids.len();
            // MTP input row j: the pending row, then the new rows but the last. The MTP layer overwrites the
            // streams, so with several calls the rows are copied first.
            let main = if n > per {
                let mut b = DeviceBuffer::alloc(t * row)?;
                b.copy_from(0, out_rows, 0, t * row, st)?;
                Some(b)
            } else {
                None
            };
            let j1 = per.min(n);
            {
                let src_rows: &DeviceBuffer = match &main {
                    Some(b) => b,
                    None => sc.hc.as_ref().map_or(&sc.x, |hs| &hs.hc),
                };
                if pend == 1 {
                    sc.mh.copy_from(0, &mtp_h, 0, row, st)?;
                }
                if j1 > pend {
                    sc.mh.copy_from(pend * row, src_rows, 0, (j1 - pend) * row, st)?;
                }
                mtp_h.copy_from(0, src_rows, (t - 1) * row, row, st)?;
            }
            sess.mtp_pos = rows_pos(t - 1, t).collect();
            if n == 0 {
                return Ok(());
            }
            self.mtp_rows(sess, sc, &ids[..j1], &p[..3 * j1], None, st)?;
            let mut j0 = j1;
            while j0 < n {
                let j1 = (j0 + per).min(n);
                let b = main.as_ref().context("MTP rows")?;
                sc.mh.copy_from(0, b, (j0 - pend) * row, (j1 - j0) * row, st)?;
                self.mtp_rows(sess, sc, &ids[j0..j1], &p[3 * j0..3 * j1], None, st)?;
                j0 = j1;
            }
            Ok(())
        })();
        sess.mtp_h = Some(mtp_h);
        res
    }

    /// MoE FFN: out += Σ_slots w · expert(x), shared expert in the last slot; x is RMSNorm(sc.x) with `norm`,
    /// otherwise the input already in sc.hn.
    fn moe(
        &self,
        st: &Stream,
        sc: &Scratch,
        norm: Option<&DeviceBuffer>,
        m: &Moe,
        layer: usize,
        t: usize,
        out: &DeviceBuffer,
        accumulate: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let mc = c.moe.as_ref().context("no MoE config")?;
        let ms = sc.moe.as_ref().context("no MoE scratch")?;
        let (h, i, slots, e1) = (c.hidden, mc.inter, mc.slots(), mc.experts + 1);
        let p = t * slots;
        if let Some(norm) = norm {
            self.rmsnorm(st, &sc.hn, &sc.x, norm, t, h)?;
        }
        if t >= 16 && h.is_multiple_of(32) {
            self.k.get("gemm_bf16_wmma").launch(
                (e1.div_ceil(64) as u32, t.div_ceil(64) as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.logits).into(),
                    (&sc.hn).into(),
                    (&m.router).into(),
                    t.into(),
                    e1.into(),
                    h.into(),
                ],
            )?;
        } else {
            self.k.get("gemm_bf16").launch(
                (e1.div_ceil(8) as u32, t.div_ceil(8) as u32, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.logits).into(),
                    (&sc.hn).into(),
                    (&m.router).into(),
                    t.into(),
                    e1.into(),
                    h.into(),
                    0i32.into(),
                ],
            )?;
        }
        if mc.group_routing() {
            // Ling's `noaux_tc`: groups are chosen on sigmoid + expert_bias, experts within the chosen groups
            // are chosen on the same, and the weights come from the sigmoid without the bias (ADR:
            // docs/model-ling3-flash.md, § MoE). The bias is a separate fp32 tensor, never quantized.
            let bias = m.bias.as_ref().context("group routing without expert_bias")?;
            self.k.get("moe_route_g").launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    (&ms.ids).into(),
                    (&ms.w).into(),
                    (&ms.logits).into(),
                    bias.into(),
                    mc.experts.into(),
                    mc.top_k.into(),
                    mc.n_group.into(),
                    mc.topk_group.into(),
                    mc.scaling.into(),
                ],
            )?;
        } else {
            let (route, rt) = if mc.experts <= 1024 && mc.top_k <= 32 {
                ("moe_route_w", 32)
            } else {
                ("moe_route", 256)
            };
            self.k.get(route).launch(
                (t as u32, 1, 1),
                (rt, 1, 1),
                0,
                st,
                &[
                    (&ms.ids).into(),
                    (&ms.w).into(),
                    (&ms.logits).into(),
                    mc.experts.into(),
                    mc.top_k.into(),
                ],
            )?;
        }
        if ENV_TRACE_ROUTE.is_some() {
            trace_routes(st, ms, layer, t, slots)?;
        }
        // Where the kernels read the routed experts from: the resident buffers, this layer's pool (decode) or the
        // shared staging (prefill, which needs all the experts of the layer at once). The weights are the same
        // bytes in all three, so the output is bit-identical; with the cache the decode path is handed slot
        // indices in place of the expert ids, and the kernels index the pool with them (`q + e * N * K / 2`).
        let cache = m.cache.as_deref();
        // The staging is needed when the kernels will read the layer by *expert id*: the grouped GEMMs sort the
        // tokens per expert and index the stacked matrix by id, and the per-pair path has to as well when this
        // call's pairs do not fit the pool (a slot reused inside one call would invalidate an earlier pair).
        // Otherwise the pool is read with slot indices.
        let pool_fits = cache.is_none_or(|c| p <= c.expert_slots());
        let staged = match cache {
            Some(_) if !pool_fits || (self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3) => {
                let cell = self
                    .expert_stage
                    .as_ref()
                    .context("expert cache without a staging buffer")?;
                cache.unwrap().stage_into(&mut crate::expert_cache::lock(cell))?;
                // The layer is on the device anyway: fill the pool with what this chunk routed most, so the decode
                // does not pay the working set from the disk a second time.
                st.synchronize()?;
                let routed: Vec<i32> = ms.ids.download(p)?;
                cache
                    .unwrap()
                    .warm_from_stage(&routed, &crate::expert_cache::lock(cell), st)?;
                Some(crate::expert_cache::lock(cell))
            }
            _ => None,
        };
        let (gu_q, gu_s, dn_q, dn_s) = match (&m.gu, &m.down, &staged) {
            (Some(g), Some(d), _) => ((&g.0).into(), (&g.1).into(), (&d.0).into(), (&d.1).into()),
            (None, None, Some(s)) => (
                Arg::Ptr(s.gu_q.addr()),
                Arg::Ptr(s.gu_s.addr()),
                Arg::Ptr(s.down_q.addr()),
                Arg::Ptr(s.down_s.addr()),
            ),
            _ => {
                let c = cache.context("routed experts neither resident nor cached")?;
                let (gq, gs) = c.gu();
                let (dq, ds) = c.down();
                (Arg::Ptr(gq), Arg::Ptr(gs), Arg::Ptr(dq), Arg::Ptr(ds))
            }
        };
        let ids: Arg = match (cache, &staged) {
            (Some(c), None) => {
                // The router chose on the device; the cache needs the choice on the host to know what to read.
                // One synchronization per layer, which is the price of the design (see issue #50).
                st.synchronize()?;
                let routed: Vec<i32> = ms.ids.download(p)?;
                let mut out = crate::expert_cache::lock(&ms.slot_ids);
                c.prepare(&routed, &mut out)?;
                Arg::Ptr(out.addr())
            }
            _ => (&ms.ids).into(),
        };
        match &m.pre {
            Some([(guq, gus), (dq, ds)]) if self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3 => {
                self.quant_row(st, &sc.xh, &sc.xsc, Act::X(&sc.hn), h, t)?;
                self.k.get("moe_sort").launch(
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.offs).into(),
                        (&ms.perm).into(),
                        (&ms.pos).into(),
                        (&ms.tiles).into(),
                        (&ms.ntiles).into(),
                        (&ms.ids).into(),
                        p.into(),
                        e1.into(),
                        MOE_BM.into(),
                        // HIP has no default arguments: BM2 and OFF2 (the remainder tiles of the split path, added
                        // for the W4A8 kernels) must be passed here too. Omitting them made hipModuleLaunchKernel
                        // read past the argument list and crash on every MoE model with int8 expert copies.
                        0usize.into(),
                        0usize.into(),
                    ],
                )?;
                self.k.get("moe_gather").launch(
                    (p as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.xg).into(),
                        (&ms.xgs).into(),
                        (&sc.xh).into(),
                        (&sc.xsc).into(),
                        (&ms.perm).into(),
                        slots.into(),
                        h.into(),
                    ],
                )?;
                let max_tiles = p.div_ceil(MOE_BM) + e1;
                let grouped = |name: &str,
                               out: &DeviceBuffer,
                               x: &DeviceBuffer,
                               xs: &DeviceBuffer,
                               q: &DeviceBuffer,
                               s: &DeviceBuffer,
                               n: usize,
                               k: usize| {
                    self.k.get(name).launch(
                        ((max_tiles * n.div_ceil(128)) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            out.into(),
                            x.into(),
                            xs.into(),
                            q.into(),
                            s.into(),
                            (&ms.offs).into(),
                            (&ms.tiles).into(),
                            (&ms.ntiles).into(),
                            n.into(),
                            k.into(),
                        ],
                    )
                };
                grouped("moe_i8_glu", &ms.act, &ms.xg, &ms.xgs, guq, gus, 2 * i, h)?;
                self.quant_row(st, &ms.aq, &ms.aqs, Act::X(&ms.act), i, p)?;
                // f32: `moe_y_half` is off for every checkpoint that reaches this branch (see `Model::load`).
                grouped("moe_i8", &ms.y, &ms.aq, &ms.aqs, dq, ds, h, i)?;
                self.k.get("moe_combine").launch(
                    (t as u32, h.div_ceil(1024) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.y).into(),
                        (&ms.w).into(),
                        (&ms.pos).into(),
                        slots.into(),
                        h.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
            // MLS3 experts have no grouped prefill kernel yet (no W3A8: docs/ling3-engine.md, I7),
            // so a long prompt walks the per-pair GEMV path. Correct and slow.
            None if self.fused_ok(t) && !m.gu_mls3 && !m.down_mls3 => {
                // MLS4 experts without an int8 copy: grouped GEMMs on activations quantized per 128-group, int8
                // (W4A8) or int4 (W4A4, double-rate WMMA and no widening of the expert nibbles).
                // gate/up averages over K = hidden (20 groups of 128), down over K = inter (5). `glu` keeps the
                // shorter product on int8: about a third of the speed-up and of the argmax cost of doing both.
                let (a4g, a4d) = (self.moe_w4a4 >= 1, self.moe_w4a4 == 1);
                let rq = |out: &DeviceBuffer, scl: &DeviceBuffer, x: &DeviceBuffer, rows: usize, k: usize, i4| {
                    self.k.get(if i4 { "rot_quant_i4_w" } else { "rot_quant_i8_w" }).launch(
                        ((rows * k / 128) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        st,
                        &[out.into(), scl.into(), x.into(), (&self.signs).into(), k.into()],
                    )
                };
                rq(&sc.xh, &sc.xsc, &sc.hn, t, h, a4g)?;
                // Row tiles of 128 once experts average more than ~48 rows (weights read once per tile).
                let bm = if p / mc.experts >= *MOE_BM128_ROWS_ENV {
                    128
                } else {
                    MOE_BM
                };
                // With 128-row tiles and few rows per expert (padding up to ~1/3 of the work at 120 rows), the
                // remainder of each expert (< 128 rows) goes to the 64-row kernel.
                // Likewise 64-row tiles with remainders on the 32-row kernel.
                let split =
                    (bm == 64 || p / mc.experts < 200) && std::env::var("MALLEUS_MOE_SPLIT").map_or(true, |v| v != "0");
                let off2 = p.div_ceil(bm) + 1;
                self.k.get("moe_sort").launch(
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.offs).into(),
                        (&ms.perm).into(),
                        (&ms.pos).into(),
                        (&ms.tiles).into(),
                        (&ms.ntiles).into(),
                        (&ms.ids).into(),
                        p.into(),
                        e1.into(),
                        bm.into(),
                        (if split { bm / 2 } else { 0 }).into(),
                        off2.into(),
                    ],
                )?;
                self.k.get(if a4g { "moe_gather_g4" } else { "moe_gather_g" }).launch(
                    (p as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.xg).into(),
                        (&ms.xgs).into(),
                        (&sc.xh).into(),
                        (&sc.xsc).into(),
                        (&ms.perm).into(),
                        slots.into(),
                        h.into(),
                    ],
                )?;
                let max_tiles = p.div_ceil(bm) + e1;
                let max_tiles = if split { p / bm } else { max_tiles };
                // `MALLEUS_DEBUG_MOE=1`: how many tiles each of the two lists carries, and how many rows they cover.
                // The copy synchronizes, so it is a diagnostic and not a path to leave on (issue #92).
                if std::env::var_os("MALLEUS_DEBUG_MOE").is_some() {
                    st.synchronize()?;
                    let nt: Vec<i32> = ms.ntiles.download(2)?;
                    let (n0, n1) = (nt[0].max(0) as usize, nt[1].max(0) as usize);
                    eprintln!(
                        "moe layer {layer}: t={t} p={p} bm={bm} split={split} \
                         tiles[big]={n0} ({:5} righe) tiles[small]={n1} ({:5} righe)",
                        n0 * bm,
                        n1 * (bm / 2)
                    );
                }
                let grouped = |name: &str,
                               out: &DeviceBuffer,
                               x: &DeviceBuffer,
                               xs: &DeviceBuffer,
                               w: (Arg, Arg),
                               n: usize,
                               k: usize,
                               second: bool| {
                    let (tiles, cnt, nt) = if second {
                        (ms.tiles.addr() + 8 * off2 as u64, ms.ntiles.addr() + 4, 2 * e1)
                    } else {
                        (ms.tiles.addr(), ms.ntiles.addr(), max_tiles)
                    };
                    if nt == 0 {
                        return Ok(());
                    }
                    self.k.get(name).launch(
                        ((nt * n.div_ceil(128)) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            out.into(),
                            x.into(),
                            xs.into(),
                            w.0,
                            w.1,
                            (&ms.offs).into(),
                            Arg::U64(tiles),
                            Arg::U64(cnt),
                            n.into(),
                            k.into(),
                        ],
                    )
                };
                let (glu, _) = match (bm == 128, a4g) {
                    (true, true) => ("moe_w4a4_glu_m128", "moe_w4a4_m128"),
                    (true, false) => ("moe_w4a8_glu_m128", "moe_w4a8_m128"),
                    (false, true) => ("moe_w4a4_glu", "moe_w4a4"),
                    (false, false) => ("moe_w4a8_glu", "moe_w4a8"),
                };
                grouped(glu, &ms.act, &ms.xg, &ms.xgs, (gu_q, gu_s), 2 * i, h, false)?;
                let (glu2, _) = match (bm == 128, a4g) {
                    (true, true) => ("moe_w4a4_glu", "moe_w4a4"),
                    (true, false) => ("moe_w4a8_glu", "moe_w4a8"),
                    (false, true) => ("moe_w4a4_glu_m32", "moe_w4a4_m32"),
                    (false, false) => ("moe_w4a8_glu_m32", "moe_w4a8_m32"),
                };
                if split {
                    grouped(glu2, &ms.act, &ms.xg, &ms.xgs, (gu_q, gu_s), 2 * i, h, true)?;
                }
                rq(&ms.aq, &ms.aqs, &ms.act, p, i, a4d)?;
                let (_, dn) = match (bm == 128, a4d) {
                    (true, true) => ("", "moe_w4a4_m128"),
                    (true, false) => ("", "moe_w4a8_m128"),
                    (false, true) => ("", "moe_w4a4"),
                    (false, false) => ("", "moe_w4a8"),
                };
                let (yh, dn_h) = (self.moe_y_half, format!("{dn}_yh"));
                grouped(
                    if yh { dn_h.as_str() } else { dn },
                    &ms.y,
                    &ms.aq,
                    &ms.aqs,
                    (dn_q, dn_s),
                    h,
                    i,
                    false,
                )?;
                if split {
                    let (_, dn2) = match (bm == 128, a4d) {
                        (true, true) => ("", "moe_w4a4"),
                        (true, false) => ("", "moe_w4a8"),
                        (false, true) => ("", "moe_w4a4_m32"),
                        (false, false) => ("", "moe_w4a8_m32"),
                    };
                    let dn2_h = format!("{dn2}_yh");
                    grouped(
                        if yh { dn2_h.as_str() } else { dn2 },
                        &ms.y,
                        &ms.aq,
                        &ms.aqs,
                        (dn_q, dn_s),
                        h,
                        i,
                        true,
                    )?;
                }
                self.k.get(if yh { "moe_combine_yh" } else { "moe_combine" }).launch(
                    (t as u32, h.div_ceil(1024) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.y).into(),
                        (&ms.w).into(),
                        (&ms.pos).into(),
                        slots.into(),
                        h.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
            _ => {
                // MLS3 experts want their input rotated once, in a buffer of their own: `gemv4_hl` folds the
                // rotation into its staging and `mls3_core` does not, and `sc.hn` is still needed unrotated by
                // the shared expert of the same layer.
                let gate_in = if m.gu_mls3 {
                    self.k.get("hadamard128").launch(
                        ((t * h / 128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &[(&sc.xr).into(), (&sc.hn).into(), (&self.signs).into(), h.into()],
                    )?;
                    &sc.xr
                } else {
                    &sc.hn
                };
                // The two kernels differ by one parameter: MLS4 folds the input rotation into its staging and so
                // takes the signs, MLS3 gets them rotated by `hadamard128` above.
                let mut args = vec![
                    (&ms.gu).into(),
                    (gate_in).into(),
                    gu_q,
                    gu_s,
                    ids,
                    slots.into(),
                    (2 * i).into(),
                    h.into(),
                ];
                if !m.gu_mls3 {
                    args.push((&self.signs).into());
                }
                self.k
                    .get(if m.gu_mls3 { "moe_gate_up3" } else { "moe_gate_up" })
                    .launch(((2 * i).div_ceil(32) as u32, p as u32, 1), (256, 1, 1), 0, st, &args)?;
                self.k.get("moe_act").launch(
                    ((p * i / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &[
                        (&ms.act).into(),
                        (&ms.gu).into(),
                        (&self.signs).into(),
                        i.into(),
                        m.swiglu.0.into(),
                    ],
                )?;
                self.k.get(if m.down_mls3 { "moe_down3" } else { "moe_down" }).launch(
                    (h.div_ceil(32) as u32, t as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &[
                        out.into(),
                        (&ms.act).into(),
                        dn_q,
                        dn_s,
                        ids,
                        (&ms.w).into(),
                        slots.into(),
                        h.into(),
                        i.into(),
                        (accumulate as i32).into(),
                    ],
                )?;
            }
        }
        // Ling's shared expert: a dense FFN on the same normalized input, added with weight 1 and its own clamp
        // (docs/model-ling3-flash.md, § MoE — it is not in the renormalization of the routed weights).
        if let Some((gate, up, down)) = &m.shared {
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), gate, t, false)?;
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&sc.hn), up, t, false)?;
            let act = if m.swiglu.1 > 0.0 {
                // The clamp is part of the activation, not of the GEMM epilogue: one pass over the two halves,
                // then the product is the input of `down`.
                self.ew(
                    st,
                    "swiglu_clamp",
                    t * i,
                    &[
                        (&sc.big0).into(),
                        (&sc.big0).into(),
                        (&sc.big1).into(),
                        (t * i).into(),
                        m.swiglu.1.into(),
                    ],
                )?;
                Act::X(&sc.big0)
            } else {
                Act::SiluMul(&sc.big0, &sc.big1)
            };
            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, out, act, down, t, accumulate)?;
        }
        self.quant_mode.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Runs `layers`, the first of which is layer `base` of the model: the index only names the layers in the
    /// routing trace (`MALLEUS_TRACE_ROUTE`), so that the MTP head (passed as layer `cfg.layers`) is not confused
    /// with layer 0.
    fn run_layers(
        &self,
        layers: &[Layer],
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        base: usize,
    ) -> Result<()> {
        self.run_layers_from(layers, ls, sc, t, st, base, (0, 0), true)
    }

    /// Runs `layers`, whose first attention / DeltaNet layers use states `first` of `ls`; `all` when they reach the
    /// last layer (the state counts are then checked).
    #[allow(clippy::too_many_arguments)]
    fn run_layers_from(
        &self,
        layers: &[Layer],
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        base: usize,
        first: (usize, usize),
        all: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let (mut ai, mut di) = first;
        ensure!(sc.cap >= t, "scratch too small");
        // The MoE output of the previous layer not yet added to the streams (done by the next mix).
        let mut pending = false;
        // Injection weights buffer the next mix writes (the other holds the last ones written).
        let mut ib = 0usize;
        let dump = std::env::var_os("MALLEUS_DUMP_DIR");
        // Debug: writes a buffer of `n` floats as `{tag}.f32` when MALLEUS_DUMP_DIR is set, for the layer-by-layer
        // comparison against `ling_ref.py layers`.
        let dump_buf = |tag: &str, buf: &DeviceBuffer, n: usize| -> Result<()> {
            match &dump {
                Some(dir) => {
                    st.synchronize()?;
                    let v: Vec<f32> = buf.download(n)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(std::path::Path::new(dir).join(format!("{tag}.f32")), bytes)?;
                    Ok(())
                }
                None => Ok(()),
            }
        };
        for (local, layer) in layers.iter().enumerate() {
            let layer_index = base + local;
            let fused = self.fused_ok(t) && layer.fin.is_some();
            // Mixer and FFN outputs: added to the residual, or written for the hyper-connection injection.
            let (res, res_acc) = match &sc.hc {
                Some(hs) if layer.hc.is_some() => (&hs.mo, false),
                _ => (&sc.x, true),
            };
            let x_in = match (&layer.hc, &layer.in_norm) {
                (Some(hcw), _) => {
                    let hs = sc.hc.as_ref().context("no hyper-connection scratch")?;
                    if pending && layer.ple.is_some() {
                        self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
                        pending = false;
                    }
                    if let Some(p) = &layer.ple {
                        self.ple(st, sc, p, ls.ple_state.context("no PLE state")?, ls.ple_snap, t)?;
                    }
                    let inject = pending.then_some((&hs.mo, &hs.inj[ib ^ 1]));
                    pending = false;
                    self.hc_mix(
                        st,
                        sc,
                        hcw.first().unwrap(),
                        &hs.hc,
                        &sc.hn,
                        Some(&hs.inj[ib]),
                        t,
                        inject,
                    )?;
                    ib ^= 1;
                    // Debug: the mixer input, the mixer output, the FFN input and the FFN output of every layer
                    // (`X<layer>_{hn,mo,hn2,ffn}_t<tokens>`), named by the layer's own index so that the MTP head's
                    // layer does not overwrite layer 0. `malleus spec-exact` with MALLEUS_DUMP_DIR set leaves the
                    // serial step and row 0 of the verify side by side: the first file that differs is the kernel.
                    dump_buf(&format!("X{layer_index:02}_hn_t{t}"), &sc.hn, t * c.hidden)?;
                    Act::X(&sc.hn)
                }
                (None, Some(w)) => {
                    if !fused {
                        self.rmsnorm(st, &sc.hn, &sc.x, w, t, c.hidden)?;
                    }
                    Act::Norm(&sc.x, w)
                }
                (None, None) => bail!("layer without pre-norm"),
            };
            match &layer.mixer {
                // KDA (Ling): five projections, three short convs with their L2 norms, the gated delta rule with
                // a decay per key channel, and a gated RMSNorm on the output. Prefill is a later increment: the
                // recurrence here is the recurrent one, which is correct for any t but costs a wave per head.
                Mixer::Kda {
                    q,
                    k,
                    v,
                    f,
                    g,
                    o,
                    b,
                    conv,
                    a_log,
                    dt_bias,
                    o_norm,
                } => {
                    let (cd, nv) = (c.lin_nk * c.lin_dk, c.lin_nv);
                    let (conv_state, rec_state) = (&ls.conv[di], &ls.rec[di]);
                    di += 1;
                    let kd = sc.kda.as_ref().context("no KDA scratch")?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), q, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small0, Act::Same(&sc.hn), k, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small1, Act::Same(&sc.hn), v, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small2, Act::Same(&sc.hn), f, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small3, Act::Same(&sc.hn), g, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big2, Act::Same(&sc.hn), b, t, false)?;
                    let st_at = |i: usize| Arg::U64(conv_state.addr() + (i * cd * (c.conv_k - 1) * 4) as u64);
                    self.k.get("kda_conv_norm").launch(
                        ((3 * t * (cd / 128)).div_ceil(8) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            (&sc.big1).into(),
                            (&kd[0]).into(),
                            (&kd[1]).into(),
                            (&sc.big0).into(),
                            (&sc.small0).into(),
                            (&sc.small1).into(),
                            conv.into(),
                            st_at(0),
                            t.into(),
                            cd.into(),
                        ],
                    )?;
                    // The state keeps the last conv_k - 1 raw inputs of the stream, so it is saved after the conv
                    // read the previous one, once per stream at its own offset.
                    for (i, raw) in [(&sc.big0), &sc.small0, &sc.small1].into_iter().enumerate() {
                        self.ew(
                            st,
                            "conv_state_save",
                            cd,
                            &[raw.into(), st_at(i), t.into(), cd.into(), Arg::U64(0)],
                        )?;
                    }
                    // beta in place over the b_proj rows, g in place over f: both read one element and write it.
                    self.ew(
                        st,
                        "kda_gates",
                        t * nv,
                        &[
                            (&sc.big2).into(),
                            (&sc.small2).into(),
                            (&sc.small2).into(),
                            (&sc.big2).into(),
                            a_log.into(),
                            dt_bias.into(),
                            t.into(),
                            nv.into(),
                            c.kda_lower_bound.into(),
                        ],
                    )?;
                    self.k.get("kda_scan").launch(
                        ((nv * 8) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        st,
                        &[
                            (&sc.big0).into(),
                            (&sc.big1).into(),
                            (&kd[0]).into(),
                            (&kd[1]).into(),
                            (&sc.small2).into(),
                            (&sc.big2).into(),
                            rec_state.into(),
                            t.into(),
                            nv.into(),
                        ],
                    )?;
                    self.k.get("rmsnorm_gated").launch(
                        ((t * nv) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &[
                            (&sc.small0).into(),
                            (&sc.big0).into(),
                            (&sc.small3).into(),
                            o_norm.into(),
                            c.eps.into(),
                            1.into(),
                        ],
                    )?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, Act::X(&sc.small0), o, t, res_acc)?;
                }
                // MLA (Ling): the cache is the normalized latent plus the rotated dims, and decode absorbs
                // kv_b_proj into the query so the latent is what the dot products run on.
                Mixer::Mla {
                    q,
                    kv_a,
                    kv_a_norm,
                    kv_b,
                    gate,
                    dense,
                } => {
                    let m = c.mla.context("MLA geometry")?;
                    let (nh, nope, kvl, rd) = (c.n_heads, m.nope, m.kv_lora, m.rope);
                    ensure!(
                        kvl == 512 && rd == 64 && nope == 128,
                        "MLA decode is written for 128 nope / 512 latent / 64 rope dims"
                    );
                    let (ckv, krot) = &ls.kv[ai];
                    let li = ai;
                    ai += 1;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), q, t, false)?;
                    self.gemm(
                        st,
                        &sc.xr,
                        &sc.xh,
                        &sc.xsc,
                        &sc.small0,
                        Act::Same(&sc.hn),
                        kv_a,
                        t,
                        false,
                    )?;
                    self.gemm(
                        st,
                        &sc.xr,
                        &sc.xh,
                        &sc.xsc,
                        &sc.small1,
                        Act::Same(&sc.hn),
                        gate,
                        t,
                        false,
                    )?;
                    self.ew(
                        st,
                        "rope_ilv",
                        t * (nh + 1) * (rd / 2),
                        &[
                            (&sc.big0).into(),
                            (&sc.small0).into(),
                            (&sc.pos).into(),
                            t.into(),
                            nh.into(),
                            nope.into(),
                            kvl.into(),
                            rd.into(),
                            c.rope_theta.into(),
                        ],
                    )?;
                    self.ew(
                        st,
                        "mla_cache_write",
                        t * (kvl + rd),
                        &[
                            ckv.into(),
                            krot.into(),
                            (&sc.small0).into(),
                            kv_a_norm.into(),
                            c.eps.into(),
                            ls.len.into(),
                            t.into(),
                            kvl.into(),
                            rd.into(),
                        ],
                    )?;
                    let scale = (1.0f32 / (nope + rd) as f32).sqrt();
                    if t <= GEMV_MAX_T {
                        // Split-key MLA: each warp owns 256 keys of one (token, head), so a 200K context no longer
                        // runs in four blocks of one warp each. `mla_absorb` computes the absorbed query once per
                        // token and head; `mla_combine` merges the splits and applies the V half of kv_b_proj.
                        let qt = sc.mla_qt.as_ref().context("no MLA absorbed-query scratch")?;
                        let part = ls.mla_part.context("no MLA split buffer")?;
                        let nsplit = (ls.len + t).div_ceil(ATTN_SPLIT);
                        self.k.get("mla_absorb").launch(
                            ((t * nh).div_ceil(8) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                qt.into(),
                                (&sc.big0).into(),
                                kv_b.into(),
                                t.into(),
                                nh.into(),
                                nope.into(),
                                kvl.into(),
                                rd.into(),
                            ],
                        )?;
                        self.k.get("mla_split").launch(
                            ((t * nh * nsplit).div_ceil(8) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                part.into(),
                                qt.into(),
                                (&sc.big0).into(),
                                ckv.into(),
                                krot.into(),
                                ls.len.into(),
                                t.into(),
                                nh.into(),
                                nope.into(),
                                kvl.into(),
                                rd.into(),
                                scale.into(),
                                nsplit.into(),
                            ],
                        )?;
                        self.k.get("mla_combine").launch(
                            ((t * nh) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.big1).into(),
                                part.into(),
                                kv_b.into(),
                                nh.into(),
                                nope.into(),
                                kvl.into(),
                                nsplit.into(),
                            ],
                        )?;
                    } else if std::env::var("MALLEUS_MLA_PREFILL").is_ok_and(|v| v == "1") {
                        // Opt-in and unmeasured end to end: the first attempt at this path made the host freeze,
                        // so it stays behind MALLEUS_MLA_PREFILL. W_KB and W_VB are read once per head per tile,
                        // not once per token; the key loop is still per token.
                        let qt = sc.mla_qt.as_ref().context("no MLA prefill qt scratch")?;
                        let lat = sc.mla_lat.as_ref().context("no MLA prefill latent scratch")?;
                        let mut t0 = 0usize;
                        while t0 < t {
                            let tt = (t - t0).min(MLA_PREFILL_TILE);
                            let qoff = t0 * nh * (nope + rd) * 4;
                            let ooff = t0 * nh * nope * 4;
                            self.k.get("mla_absorb_t").launch(
                                ((nh * 8) as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                st,
                                &[
                                    qt.into(),
                                    Arg::U64(sc.big0.addr() + qoff as u64),
                                    kv_b.into(),
                                    tt.into(),
                                    nh.into(),
                                    nope.into(),
                                    kvl.into(),
                                    rd.into(),
                                ],
                            )?;
                            self.k.get("mla_prefill").launch(
                                ((tt * nh).div_ceil(8) as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                st,
                                &[
                                    lat.into(),
                                    qt.into(),
                                    Arg::U64(sc.big0.addr() + qoff as u64),
                                    ckv.into(),
                                    krot.into(),
                                    (ls.len + t0).into(),
                                    tt.into(),
                                    nh.into(),
                                    nope.into(),
                                    kvl.into(),
                                    rd.into(),
                                    scale.into(),
                                ],
                            )?;
                            self.k.get("mla_project").launch(
                                ((nh * (nope / 16)) as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                st,
                                &[
                                    Arg::U64(sc.big1.addr() + ooff as u64),
                                    lat.into(),
                                    kv_b.into(),
                                    tt.into(),
                                    nh.into(),
                                    nope.into(),
                                    kvl.into(),
                                ],
                            )?;
                            t0 += tt;
                        }
                    } else {
                        self.k.get("mla_decode").launch(
                            ((t * nh).div_ceil(8) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.big1).into(),
                                (&sc.big0).into(),
                                kv_b.into(),
                                ckv.into(),
                                krot.into(),
                                ls.len.into(),
                                t.into(),
                                nh.into(),
                                nope.into(),
                                kvl.into(),
                                rd.into(),
                                scale.into(),
                                c.eps.into(),
                            ],
                        )?;
                    }
                    self.ew(
                        st,
                        "mla_gate",
                        t * nh * nope,
                        &[
                            (&sc.big1).into(),
                            (&sc.small1).into(),
                            (t * nh * nope).into(),
                            nope.into(),
                        ],
                    )?;
                    let _ = li;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, Act::X(&sc.big1), dense, t, res_acc)?;
                }
                Mixer::Attn {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    indexer,
                    dec,
                } => {
                    let (kc, vc) = &ls.kv[ai];
                    let li = ai;
                    ai += 1;
                    let mut proj_ready = false;
                    match (&layer.fin, dec) {
                        (Some(f), _) if fused => {
                            self.gemm_fused(st, sc, [&sc.big0, &sc.small0, &sc.small1], x_in, f, t)?
                        }
                        (_, Some(m)) if t <= GEMV_MAX_T => {
                            let proj = sc.idx.as_ref().map_or(&sc.small2, |b| &b[0]);
                            self.gemv_merged(st, sc, m, [&sc.big0, &sc.small0, &sc.small1, proj], t)?;
                            proj_ready = indexer.is_some();
                        }
                        _ => {
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), q, t, false)?;
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small0, Act::Same(&sc.hn), k, t, false)?;
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small1, Act::Same(&sc.hn), v, t, false)?;
                        }
                    }
                    let [_, sh, sw] = c.mrope_section;
                    let ctx = kv_ctx(ls.max_ctx);
                    let kv_bytes = ctx * c.n_kv * c.head_dim * 2 + 64;
                    let vr_arg = if c.indexer.is_some() {
                        Arg::U64(vc.addr() + kv_bytes as u64)
                    } else {
                        Arg::U64(0)
                    };
                    // `MALLEUS_ATTN_MASS`: where the attention kernel writes the (max, sum) of every tile, and the
                    // distance between sampled rows. Null (no cost) when the diagnostics are off.
                    let mass_args = match &self.attn_mass {
                        Some((b, stride, tiles)) => (Arg::from(b), *stride, *tiles),
                        None => (Arg::U64(0), 0usize, 0usize),
                    };
                    self.k.get("attn_prep").launch(
                        (t as u32, (c.n_heads + c.n_kv) as u32, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            (&sc.small2).into(),
                            kc.into(),
                            vc.into(),
                            (&sc.big0).into(),
                            (&sc.small0).into(),
                            (&sc.small1).into(),
                            q_norm.into(),
                            k_norm.into(),
                            (&sc.pos).into(),
                            (&sc.len_dev).into(),
                            c.n_heads.into(),
                            c.n_kv.into(),
                            c.eps.into(),
                            c.rope_theta.into(),
                            c.rot_dim.into(),
                            sh.into(),
                            sw.into(),
                            ctx.into(),
                            vr_arg,
                            self.rope_invf.as_ref().map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
                        ],
                    )?;
                    // Diagnostica `MALLEUS_HIDDEN_DUMP=<dir>` (#96): per layer, l'ingresso del layer (`sc.hn`) e il
                    // K del chunk appena scritto. Sono le due rappresentazioni da cui un indexer dovrebbe stimare
                    // quanto un blocco conta: servono a provare se la sensibilita' e' imparabile.
                    if let Some(dir) = ENV_HIDDEN_DUMP.as_deref() {
                        st.synchronize()?;
                        let hidden = c.hidden;
                        let kvw = c.n_kv * c.head_dim;
                        let hn: Vec<f32> = sc.hn.download(t * hidden)?;
                        let kk: Vec<u16> = kc.download_at(ls.len * kvw * 2, t * kvw)?;
                        let mut b = Vec::with_capacity(hn.len() * 4);
                        for v in &hn {
                            b.extend_from_slice(&v.to_le_bytes());
                        }
                        std::fs::write(format!("{dir}/hn-{li}-{}x{t}.f32", ls.len), &b)?;
                        let mut b = Vec::with_capacity(kk.len() * 2);
                        for v in &kk {
                            b.extend_from_slice(&v.to_le_bytes());
                        }
                        std::fs::write(format!("{dir}/k-{li}-{}x{t}.f16", ls.len), &b)?;
                    }
                    let mut sel_args = [Arg::U64(0), Arg::U64(0), Arg::U64(0)];
                    let (mut words, mut ratio) = (0usize, 1usize);
                    if let (Some(ix), Some(is)) = (indexer, &ls.idx) {
                        self.indexer(st, sc, ix, is, li, ls.len, t, proj_ready)?;
                        sel_args = is.sel_args();
                        if let Some((_, _, _, w)) = is.sel {
                            words = w;
                        }
                        ratio = c.indexer.as_ref().map_or(1, |ic| ic.ratio);
                    }
                    let scale = 1.0 / (c.head_dim as f32).sqrt() * yarn_mscale();
                    let gather = match (ls.idx.as_ref(), &c.indexer) {
                        (Some(is), Some(ic))
                            if t >= ATTN_WMMA_MIN
                                // Below ~5x the budget the masked WMMA kernel is as fast. `MALLEUS_GATHER_MIN` is
                                // that factor, in units of the indexer budget, so the crossover can be measured
                                // instead of assumed.
                                && ls.len + t
                                    >= std::env::var("MALLEUS_GATHER_MIN")
                                        .ok()
                                        .and_then(|v| v.parse::<usize>().ok())
                                        .unwrap_or(5)
                                        * ic.budget
                                && is.sel.is_some()
                                && is.list.is_some()
                                && std::env::var("MALLEUS_ATTN_GATHER").map_or(true, |v| v != "0") =>
                        {
                            Some((is, ic))
                        }
                        _ => None,
                    };
                    if let Some((is, ic)) = gather {
                        // Queries whose complete blocks fit the budget attend densely (WMMA kernel); the others
                        // gather their selected blocks and tail.
                        let l = ic.budget / ic.ratio;
                        let nd = ((l + 1) * ic.ratio - 1).saturating_sub(ls.len).min(t);
                        let row = c.n_heads * c.head_dim * 4;
                        if nd > 0 {
                            self.k.get("attention_wmma").launch(
                                ((nd.div_ceil(64) * c.n_heads) as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                st,
                                &[
                                    (&sc.small3).into(),
                                    (&sc.small2).into(),
                                    kc.into(),
                                    vc.into(),
                                    nd.into(),
                                    ls.len.into(),
                                    c.n_heads.into(),
                                    c.n_kv.into(),
                                    scale.into(),
                                    ctx.into(),
                                    Arg::U64(0),
                                    Arg::U64(0),
                                    0usize.into(),
                                    1usize.into(),
                                    mass_args.0,
                                    mass_args.1.into(),
                                    mass_args.2.into(),
                                ],
                            )?;
                        }
                        if nd < t {
                            let ns = t - nd;
                            let (_, sel, _, w) = is.sel.context("selection")?;
                            let (list, lrows) = is.list.context("block lists")?;
                            let cnt = Arg::U64(list.addr() + (lrows * l * 4) as u64);
                            self.k.get("idx_compact").launch(
                                (ns as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                st,
                                &[
                                    list.into(),
                                    cnt,
                                    Arg::U64(sel.addr() + (nd * w * 4) as u64),
                                    w.into(),
                                    l.into(),
                                ],
                            )?;
                            // Three shapes, measured on the same 25 574-token prompt (2026-11-14, gfx1151): the
                            // union of four query lists 2 940 ms, one query per block with two waves 2 520-3 034,
                            // one query per block with four waves (`attention_gather_q1`) 1 892-1 958. The union
                            // was 6-9% faster than the per-query kernel on gfx1150 at 12/24K (2026-11-09), so it
                            // stays the default there; on gfx1151 the extra CUs make the narrower block win.
                            // MALLEUS_ATTN_GATHER=1 selects the per-query kernel, MALLEUS_GATHER_Q1 overrides
                            // the architecture default in either direction.
                            let per = std::env::var("MALLEUS_ATTN_GATHER").is_ok_and(|v| v == "1");
                            let q1 = std::env::var("MALLEUS_GATHER_Q1")
                                .map_or(!per && self.k.arch == "gfx1151", |v| v == "1");
                            let share = !q1 && !per && 4 * l <= GATHER_UNION_MAX;
                            self.k
                                .get(if q1 {
                                    "attention_gather_q1"
                                } else if share {
                                    "attention_gather_u"
                                } else {
                                    "attention_gather"
                                })
                                .launch(
                                    (if q1 { ns as u32 } else { ns.div_ceil(4) as u32 }, c.n_kv as u32, 1),
                                    (if q1 { 128 } else { 256 }, 1, 1),
                                    0,
                                    st,
                                    &[
                                        Arg::U64(sc.small3.addr() + (nd * row) as u64),
                                        Arg::U64(sc.small2.addr() + (nd * row) as u64),
                                        kc.into(),
                                        vr_arg,
                                        list.into(),
                                        cnt,
                                        ns.into(),
                                        (ls.len + nd).into(),
                                        ctx.into(),
                                        c.n_heads.into(),
                                        c.n_kv.into(),
                                        l.into(),
                                        ic.ratio.into(),
                                        scale.into(),
                                    ],
                                )?;
                        }
                    } else if t >= ATTN_WMMA_MIN {
                        self.k.get("attention_wmma").launch(
                            ((t.div_ceil(64) * c.n_heads) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.small3).into(),
                                (&sc.small2).into(),
                                kc.into(),
                                vc.into(),
                                t.into(),
                                ls.len.into(),
                                c.n_heads.into(),
                                c.n_kv.into(),
                                scale.into(),
                                ctx.into(),
                                sel_args[0],
                                sel_args[1],
                                words.into(),
                                ratio.into(),
                                mass_args.0,
                                mass_args.1.into(),
                                mass_args.2.into(),
                            ],
                        )?;
                    } else if let (Some(is), Some(ic)) = (ls.idx.as_ref().filter(|is| is.sel.is_some()), &c.indexer)
                        && let Some((list, lrows)) = is.list
                        && std::env::var("MALLEUS_ATTN_SEL").map_or(true, |v| v != "0")
                    {
                        // Sparse decode / verify: only the listed blocks and the tail (~budget keys per query).
                        let (_, sel, _, w) = is.sel.context("selection")?;
                        let l = ic.budget / ic.ratio;
                        let cnt = Arg::U64(list.addr() + (lrows * l * 4) as u64);
                        self.k.get("idx_compact").launch(
                            (t as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[list.into(), cnt, sel.into(), w.into(), l.into()],
                        )?;
                        let nsplit = (l * ic.ratio + ic.ratio).div_ceil(ATTN_SPLIT);
                        self.k.get("attention_sel").launch(
                            ((t * c.n_heads) as u32, nsplit as u32, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                ls.attn_part.into(),
                                (&sc.small2).into(),
                                kc.into(),
                                vc.into(),
                                list.into(),
                                cnt,
                                (&sc.len_dev).into(),
                                c.n_heads.into(),
                                c.n_kv.into(),
                                scale.into(),
                                ctx.into(),
                                nsplit.into(),
                                ic.ratio.into(),
                                l.into(),
                                vr_arg,
                            ],
                        )?;
                        self.k.get("attention_combine").launch(
                            ((t * c.n_heads) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[(&sc.small3).into(), ls.attn_part.into(), nsplit.into()],
                        )?;
                    } else {
                        let nsplit = (ls.len + t).div_ceil(ATTN_SPLIT);
                        self.k.get("attention_split").launch(
                            ((t * c.n_heads) as u32, nsplit as u32, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                ls.attn_part.into(),
                                (&sc.small2).into(),
                                kc.into(),
                                vc.into(),
                                (&sc.len_dev).into(),
                                c.n_heads.into(),
                                c.n_kv.into(),
                                scale.into(),
                                ctx.into(),
                                nsplit.into(),
                                sel_args[0],
                                words.into(),
                                ratio.into(),
                            ],
                        )?;
                        self.k.get("attention_combine").launch(
                            ((t * c.n_heads) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[(&sc.small3).into(), ls.attn_part.into(), nsplit.into()],
                        )?;
                    }
                    if self.row8(o, t) {
                        let act = Act::AttnGate(&sc.small3, &sc.big0);
                        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, act, o, t, res_acc)?;
                    } else {
                        self.ew(
                            st,
                            "attn_gate",
                            t * c.n_heads * c.head_dim,
                            &[(&sc.small3).into(), (&sc.big0).into(), t.into(), c.n_heads.into()],
                        )?;
                        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, Act::X(&sc.small3), o, t, res_acc)?;
                    }
                }
                Mixer::Delta {
                    qkv,
                    z,
                    a,
                    b,
                    out,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
                    dec,
                    ab,
                } => {
                    let conv_state = &ls.conv[di];
                    let rec_state = &ls.rec[di];
                    let conv_snap: Arg = ls.conv_snaps.map_or(Arg::U64(0), |cs| (&cs[di]).into());
                    let (pin, np, pout): (Arg, usize, Arg) = match ls.dn {
                        Some((recs, np, par, record)) => (
                            (&recs[di][par]).into(),
                            np,
                            if record {
                                (&recs[di][par ^ 1]).into()
                            } else {
                                Arg::U64(0)
                            },
                        ),
                        None => (Arg::U64(0), 0, Arg::U64(0)),
                    };
                    di += 1;
                    let cd = c.lin_qkv_dim();
                    let ab_merged = !fused && t <= GEMV_MAX_T && dec.is_some() && ab.is_some();
                    match (&layer.fin, dec) {
                        (Some(f), _) if fused => {
                            self.gemm_fused(st, sc, [&sc.big0, &sc.small0, &sc.small1], x_in, f, t)?
                        }
                        (_, Some(m)) if t <= GEMV_MAX_T => {
                            self.gemv_merged(st, sc, m, [&sc.big0, &sc.small0, &sc.small0, &sc.small0], t)?;
                            match ab {
                                Some(ab) => {
                                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small1, Act::X(&sc.hn), ab, t, false)?
                                }
                                None => {
                                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small1, Act::X(&sc.hn), a, t, false)?;
                                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small2, Act::Same(&sc.hn), b, t, false)?;
                                }
                            }
                        }
                        _ => {
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), qkv, t, false)?;
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small0, Act::Same(&sc.hn), z, t, false)?;
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small1, Act::Same(&sc.hn), a, t, false)?;
                            self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.small2, Act::Same(&sc.hn), b, t, false)?;
                        }
                    }
                    self.k.get("conv_qk_norm").launch(
                        ((t * cd / 128).div_ceil(8) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &[
                            (&sc.big1).into(),
                            (&sc.big0).into(),
                            conv.into(),
                            conv_state.into(),
                            t.into(),
                            cd.into(),
                            c.lin_nk.into(),
                        ],
                    )?;
                    self.ew(
                        st,
                        "conv_state_save",
                        cd,
                        &[(&sc.big0).into(), conv_state.into(), t.into(), cd.into(), conv_snap],
                    )?;
                    let (beta, g) = (&sc.big2, &sc.small3);
                    self.ew(
                        st,
                        "deltanet_gates",
                        t * c.lin_nv,
                        &[
                            beta.into(),
                            g.into(),
                            (&sc.small1).into(),
                            if fused || ab_merged { &sc.small1 } else { &sc.small2 }.into(),
                            a_log.into(),
                            dt_bias.into(),
                            t.into(),
                            c.lin_nv.into(),
                            if fused {
                                128
                            } else if ab_merged {
                                2 * c.lin_nv
                            } else {
                                c.lin_nv
                            }
                            .into(),
                            if fused || ab_merged { c.lin_nv } else { 0 }.into(),
                        ],
                    )?;
                    let scan = |t: usize, pout: Arg| -> Result<()> {
                        self.k.get("deltanet_scan").launch(
                            ((c.lin_nv * 8) as u32, 1, 1),
                            (32, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.big0).into(),
                                (&sc.big1).into(),
                                beta.into(),
                                g.into(),
                                rec_state.into(),
                                t.into(),
                                c.lin_nv.into(),
                                c.lin_nk.into(),
                                pin,
                                np.into(),
                                pout,
                            ],
                        )?;
                        Ok(())
                    };
                    if t >= DN_CHUNK && std::env::var_os("MALLEUS_DN_SCAN").is_none() {
                        if np > 0 {
                            scan(0, Arg::U64(0))?;
                        }
                        ensure!(
                            c.lin_dk == 128 && c.lin_dv == 128,
                            "chunked DeltaNet needs 128-wide heads"
                        );
                        let d = &sc.dn;
                        self.k.get("deltanet_chunk_prep").launch(
                            ((t.div_ceil(DN_CHUNK) * c.lin_nv) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                (&d[0]).into(),
                                (&d[1]).into(),
                                (&d[2]).into(),
                                (&d[3]).into(),
                                (&d[4]).into(),
                                (&d[5]).into(),
                                (&sc.big1).into(),
                                beta.into(),
                                g.into(),
                                t.into(),
                                c.lin_nv.into(),
                                c.lin_nk.into(),
                            ],
                        )?;
                        self.k.get("deltanet_chunk_scan").launch(
                            (c.lin_nv as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.big0).into(),
                                rec_state.into(),
                                (&d[0]).into(),
                                (&d[1]).into(),
                                (&d[2]).into(),
                                (&d[3]).into(),
                                (&d[4]).into(),
                                (&d[5]).into(),
                                t.into(),
                                c.lin_nv.into(),
                            ],
                        )?;
                    } else {
                        scan(t, pout)?;
                    }
                    if self.row8(out, t) {
                        let act = Act::GatedNorm(&sc.big0, &sc.small0, norm);
                        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, act, out, t, res_acc)?;
                    } else {
                        self.k.get("rmsnorm_gated").launch(
                            ((t * c.lin_nv) as u32, 1, 1),
                            (128, 1, 1),
                            0,
                            st,
                            &[
                                (&sc.big1).into(),
                                (&sc.big0).into(),
                                (&sc.small0).into(),
                                norm.into(),
                                c.eps.into(),
                                (c.gate_sigmoid as i32).into(),
                            ],
                        )?;
                        self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, res, Act::X(&sc.big1), out, t, res_acc)?;
                    }
                }
            }
            if dump.is_some() {
                // The residual with the mixer added and the FFN not yet: the delta to the dump at the end of the
                // layer is the FFN output, which is what the reference's `moe` has to match.
                dump_buf(&format!("L{:02}_preffn_t{}", ai + di - 1, t), res, t * c.hidden)?;
                dump_buf(&format!("X{layer_index:02}_mo_t{t}"), res, t * c.hidden)?;
            }
            if let (Some(hcw), Some(hs)) = (&layer.hc, &sc.hc) {
                let attn_out = Some((&hs.mo, &hs.inj[ib ^ 1]));
                self.hc_mix(st, sc, &hcw[1], &hs.hc, &sc.hn, Some(&hs.inj[ib]), t, attn_out)?;
                ib ^= 1;
                dump_buf(&format!("X{layer_index:02}_hn2_t{t}"), &sc.hn, t * c.hidden)?;
                let Ffn::Moe(m) = &layer.ffn else {
                    bail!("hyper-connections need the MoE FFN")
                };
                self.moe(st, sc, None, m, layer_index, t, &hs.mo, false)?;
                dump_buf(&format!("X{layer_index:02}_ffn_t{t}"), &hs.mo, t * c.hidden)?;
                pending = true;
                if let Some(dir) = &dump {
                    self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
                    pending = false;
                    // Debug: the stream state after every layer, as scripts/ref_layerwise.py --dump writes it.
                    st.synchronize()?;
                    let v: Vec<f32> = hs.hc.download(t * c.stream_dim())?;
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(std::path::Path::new(dir).join(format!("L{}.f32", ai + di - 1)), bytes)?;
                }
                continue;
            }
            let post_norm = layer.post_norm.as_ref().context("layer without post-norm")?;
            match &layer.ffn {
                Ffn::Moe(m) => self.moe(st, sc, Some(post_norm), m, layer_index, t, &sc.x, true)?,
                Ffn::Dense { fgu: Some(f), down, .. } if self.fused_ok(t) => {
                    let x_in = Act::Norm(&sc.x, post_norm);
                    self.gemm_fused(st, sc, [&sc.big0, &sc.big0, &sc.big0], x_in, f, t)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.x, Act::X(&sc.big0), down, t, true)?;
                }
                Ffn::Dense { gate, up, down, .. } => {
                    self.rmsnorm(st, &sc.hn, &sc.x, post_norm, t, c.hidden)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&sc.hn), gate, t, false)?;
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&sc.hn), up, t, false)?;
                    let act = Act::SiluMul(&sc.big0, &sc.big1);
                    self.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.x, act, down, t, true)?;
                }
            }
            if let Some(dir) = &dump {
                // Debug: the residual stream after every layer, for the layer-by-layer comparison the reference
                // writes with `ling_ref.py layers` (MIGRATE_MODELS.md §4.3). One file per layer, [t, hidden] f32.
                st.synchronize()?;
                let v: Vec<f32> = sc.x.download(t * c.hidden)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(
                    std::path::Path::new(dir).join(format!("L{:02}_t{}.f32", ai + di - 1, t)),
                    bytes,
                )?;
            }
        }
        if pending && let Some(hs) = &sc.hc {
            self.hc_inject(st, &hs.hc, &hs.mo, &hs.inj[ib ^ 1], t)?;
        }
        if all && (ai != ls.kv.len() || di != ls.conv.len()) {
            bail!("layer/state mismatch");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometry of `inclusionAI/Ling-3.0-flash`, keys and values taken from its `config.json`.
    const LING3: &str = r#"{"model_type":"bailing_hybrid","hidden_size":2560,"num_hidden_layers":42,"head_dim":128,"num_attention_heads":32,"num_key_value_heads":32,"intermediate_size":6144,"moe_intermediate_size":768,"moe_shared_expert_intermediate_size":768,"num_experts":512,"num_experts_per_tok":8,"n_group":8,"topk_group":4,"score_function":"sigmoid","routed_scaling_factor":2.5,"moe_router_enable_expert_bias":true,"first_k_dense_replace":2,"layer_group_size":6,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"q_lora_rank":null,"short_conv_kernel_size":4,"kda_lower_bound":-5.0,"rms_norm_eps":1e-06,"rope_theta":6000000,"partial_rotary_factor":0.5,"rope_interleave":true,"num_nextn_predict_layers":1,"vocab_size":157184,"eos_token_id":156895,"expert_swiglu_limit_list":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,4],"share_expert_swiglu_limit_list":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,7,7]}"#;

    /// Geometry of `Qwen/Qwen3.6-35B-A3B`, the MoE model in production on gfx1150.
    const QWEN35_MOE: &str = r#"{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":2048,"num_hidden_layers":40,"head_dim":256,"num_attention_heads":16,"num_key_value_heads":2,"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"rms_norm_eps":1e-06,"vocab_size":248320,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}}"#;

    #[test]
    fn ling3_geometry() {
        let c = Config::from_json(LING3, None).unwrap();
        assert_eq!(c.family, Family::Ling3);
        assert_eq!((c.hidden, c.n_layers, c.vocab, c.inter), (2560, 42, 157184, 6144));
        assert_eq!(c.dense_ffn, 2);
        // MLA every 6th layer (5, 11, ... 41): 7 of 42, KDA the rest.
        assert_eq!(c.kind[0], MixerKind::Kda);
        assert_eq!(c.kind[5], MixerKind::Mla);
        assert_eq!(c.kind[41], MixerKind::Mla);
        assert_eq!(c.kind.iter().filter(|k| **k == MixerKind::Mla).count(), 7);
        assert_eq!(c.full_attn.iter().filter(|f| **f).count(), 0);
        // KDA borrows the DeltaNet fields: 32 heads of 128, conv 4, gate floor -5.
        assert_eq!(
            (c.lin_nk, c.lin_nv, c.lin_dk, c.lin_dv, c.conv_k),
            (32, 32, 128, 128, 4)
        );
        assert_eq!(c.kda_lower_bound, -5.0);
        let m = c.mla.unwrap();
        assert_eq!((m.kv_lora, m.nope, m.rope, m.v), (512, 128, 64, 128));
        // Plain interleaved RoPE on 64 of the 128 dims, and the norms multiply by w, not 1 + w.
        assert!(!c.mrope && !c.norm_one_plus_w);
        assert_eq!((c.rope_theta, c.rot_dim), (6e6, 64));
        let m = c.moe.unwrap();
        assert_eq!(
            (m.experts, m.top_k, m.inter, m.n_group, m.topk_group),
            (512, 8, 768, 8, 4)
        );
        assert!(m.group_routing() && m.bias && !m.shared_gate);
        assert!(m.sigmoid && m.bias && !m.shared_gate);
        assert_eq!(m.scaling, 2.5);
        let s = c.swiglu.unwrap();
        assert_eq!((s.routed[34], s.routed[35], s.routed[41]), (0, 4, 4));
        assert_eq!((s.shared[33], s.shared[34], s.shared[40]), (0, 5, 7));
    }

    #[test]
    fn qwen35_moe_geometry_is_unchanged() {
        let c = Config::from_json(QWEN35_MOE, None).unwrap();
        assert_eq!(c.family, Family::Qwen35);
        assert_eq!((c.hidden, c.n_layers, c.inter), (2048, 40, 0));
        assert_eq!(c.kind[0], MixerKind::Delta);
        assert_eq!(c.kind[3], MixerKind::Attn);
        assert!(c.norm_one_plus_w && c.mrope);
        assert_eq!(c.mrope_section, [11, 11, 10]);
        assert_eq!((c.head_dim, c.rot_dim, c.rope_theta), (256, 64, 1e7));
        let m = c.moe.unwrap();
        assert_eq!((m.experts, m.top_k, m.inter), (256, 8, 512));
        assert!(!m.sigmoid && !m.bias && m.shared_gate);
        assert_eq!((m.n_group, m.topk_group, m.scaling), (1, 1, 1.0));
        // Qwen parses n_group = 1 when the field is absent: that is "no groups", and the router must take the
        // plain path (it has no expert_bias to route through). Getting this wrong broke every Qwen MoE model.
        assert!(!m.group_routing());
        assert!(c.swiglu.is_none() && c.mla.is_none());
        assert_eq!(c.dense_ffn, 0);
    }

    #[test]
    fn unknown_family_is_rejected() {
        assert!(Family::from_model_type("llama").is_none());
        assert!(Config::from_json(r#"{"model_type":"llama"}"#, None).is_err());
    }
}
