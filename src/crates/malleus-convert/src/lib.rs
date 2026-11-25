#![forbid(unsafe_code)]

//! safetensors (bf16) → Malleus model file (`.malleus`: GGUF container with MLS4 / MLS8 / bf16 tensors).
//!
//! The converter first builds a plan of output tensors. Most are one source tensor, copied or quantized; some are
//! built from several sources or padded (rows and columns past the sources are zero):
//! - `<hc>.mix_down.weight` (qwen4_exp hyper-connections): `input_mix_weight_down` rows, then the
//!   `block_inject_weight` rows, padded to a multiple of 128 rows; `<hc>.mix_up.weight`: `input_mix_weight_up` with
//!   its K padded to the same width. Both MLS8.
//! - `<ple>.ngram_table` (qwen4_exp PLE): the shards of the hashed n-gram embedding stacked in order, one int8 row
//!   per embedding row followed by its f32 scale (little endian): [rows, dim + 4] I8. Read on the host, never uploaded.
//! - `model.layers.N.mlp.experts.{gate_up_proj,down_proj}` (Ling-3.0-flash): the per-expert tensors fused into one
//!   [E, rows, K] tensor per layer; in `gate_up` the gate rows of each expert come before its up rows (ADR 0002).

pub mod codebook;
pub mod deepseek;
pub mod dequant;
pub mod grid;

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::fpquant::{MXFP8_BLOCK, NVFP4_BLOCK, mxfp8_rows, nvfp4_rows};
use malleus_gguf::gguf::{ARCH_PREFIX, GgufWriter, Value};
use malleus_gguf::quant::{self, BLOCK, HADAMARD_SEED, Int2Grid, LAYOUT_VERSION};
use malleus_gguf::safetensors::SafeTensors;
use malleus_gguf::{DType, TensorSource};
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Raw,
    Mls2,
    Mls3,
    Mls4,
    Mls8,
    /// int8 rows with a trailing f32 scale each (n-gram lookup table).
    RowI8,
    /// 4-bit rows read on the host, with one f16 scale per row (`<name>.scales`). Used for the Engram memory
    /// tables, which are looked up by row and never feed a GEMM: 96 wide, so the 128-wide rotation of MLS4 does
    /// not apply and the rows stay unrotated.
    RowI4,
}

/// Which MoE layers store `gate`/`up` at 3 bit. MIGRATE_MODELS.md §2.4 keeps the sensitive last layers at 4 bit;
/// the exact range is a per-model decision, measured with `malleus eval` and written in its document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mls3Layers {
    /// The range the family's model document chose, derived from `config.json`.
    #[default]
    Family,
    /// Every expert at 4 bit.
    Off,
    /// Inclusive layer range.
    Range(usize, usize),
}

/// Which MoE layers store both routed expert projections at 2 bit (MLS2). The step from 4 to 2 bits is a quality
/// decision, not a default: the layer range is measured per family, and the last layers can stay at 4 bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mls2Layers {
    /// The range the family's model document chose, derived from `config.json`.
    #[default]
    Family,
    /// Every routed expert at 4 bit.
    Off,
    /// Inclusive layer range.
    Range(usize, usize),
}

/// Layers at the end of the stack that stay at 4 bit even with MLS3 on. Both model documents that use MLS3 promote
/// their last layers (Ling-3.0-flash: 35..41 plus the MTP; Qwen3.8-27B keeps its last 8 in MLS8 for prefill).
const MLS3_SKIP_LAST_LAYERS: usize = 8;

pub struct Options {
    /// Also store an MLS8 copy (`<name>.prefill`) of every MLS4 linear for prefill. **Off by default**: the copy
    /// doubles the file (~35 GB more for a 35B MoE) and buys prompt quality, not decode speed (measured: eval top-1
    /// 93.87 % without, 97.02 % with, prefill +11…29 % without). The name of the output should say which one it is:
    /// `-mls4` for a single copy, `-mls4-mls8` for the double one.
    pub prefill_int8: bool,
    /// Skip that copy for the routed experts only: prefill then reads their MLS4 nibbles with int8 activations
    /// (W4A8), halving the bytes of the widest block of the prefill and ~32 GB of device memory, while the dense
    /// linears keep their W8A8 path. Only worth it when those nibbles are GPTQ-calibrated: round-to-nearest loses
    /// ~3.9 points of top-1 on the eval set.
    pub no_prefill_int8_experts: bool,
    /// Layers whose FFN stays MLS8 in decode too.
    pub keep_int8_layers: Vec<usize>,
    /// Directory with calibrated MLS4 nibbles and scales (scripts/calibrate.py), used instead of round-to-nearest.
    pub mls4_from: Option<std::path::PathBuf>,
    /// Layers whose MoE `gate`/`up` go to MLS3; MLS3 tensors are always round-to-nearest, calibration is MLS4 only.
    pub mls3_layers: Mls3Layers,
    /// Layers whose routed `gate`/`up` go to MLS2 (2 bits per weight); `down` follows only with `mls2_down`. MLS2
    /// is round-to-nearest: the calibration tool reports its reconstruction error, but the GPTQ files are MLS4 only.
    pub mls2_layers: Mls2Layers,
    /// Also store the routed `down` projection at 2 bits. The default keeps it at 4: measured on
    /// Qwen3.8-Flash-Next, 2 bits on both projections cost 6.6 points of all-position argmax against MLS4, while
    /// gate/up alone keeps most of it and still reads a third less per routed expert.
    pub mls2_down: bool,
    /// Also store the hyper-connection `mix_down`/`mix_up` as MLS4 for decode, keeping the MLS8 copy for prefill.
    /// Decode then reads 4 bit per weight instead of 8 on the 0.78 GB per token the mixing costs; `prefill` keeps
    /// working unchanged because `gemm_i8_hcmix` and `gemm_i8_sk` have no int4 form.
    pub hc_mls4: bool,
}

/// A model family `convert` can write: it fixes `general.architecture`, the name mapping and which tensor class
/// gets which format (MIGRATE_MODELS.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen3.5 and Qwen3.5-MoE (`qwen3_5`, `qwen3_5_moe`).
    Qwen35,
    /// Qwen3.8-Flash-Next (`qwen4_exp`): hyper-connections, PLE n-gram embedding, QSA indexer.
    Qwen4Exp,
    /// Ling-3.0-flash (`bailing_hybrid`, `BailingMoeV3`): KDA + MLA + 512 experts.
    Ling3,
    /// A DFlash speculative draft (`architectures: ["DFlashDraftModel"]`): 6 layers that propose a block of tokens
    /// for a target model to verify, fed by the target's hidden states and sharing its embedding and `lm_head`.
    /// Not a standalone model: `dflash_config.target_layer_ids` names the target layers it reads.
    Dflash,
    /// Hy4-flash (`hy_v4_flash`): iHC streams, gated MLA with a learned sink, hybrid sparse/sliding layers, a
    /// compressed indexer, latent MoE, Engram n-gram memories and a vision tower.
    Hy4,
    /// Inkling-2 (`inkling2_loop`): 24 layers of which 12 run three times, attention residuals per block, short
    /// convolutions and a relative-position bias, 128 experts per layer, a vision tower and a loop-aware MTP head.
    /// Its routed experts arrive packed (NVFP4 or MXFP8) and are dequantised while they are read.
    Inkling2,
    /// DeepSeek-V4.1-flash (`deepseek_v41`): 40 layers, 384 routed experts, Engram n-gram tables, hyper-connections,
    /// a compressed indexer and a vision tower. Its precision map comes from the released 2.00 bpw build, and its
    /// weights are fp8 blocks and packed fp4, so the converter reads them with its own dequantizers. Issue #83.
    Deepseek41,
}

impl Family {
    /// Picks the family from `model_type` in `config.json`.
    pub fn from_model_type(t: &str) -> Result<Self> {
        Ok(match t {
            "qwen3_5" | "qwen3_5_moe" => Family::Qwen35,
            "qwen4_exp" => Family::Qwen4Exp,
            "bailing_hybrid" => Family::Ling3,
            "hy_v4_flash" => Family::Hy4,
            "inkling2" | "inkling2_loop" | "inkling2_text" | "inkling2_loop_text" => Family::Inkling2,
            "deepseek_v41" => Family::Deepseek41,
            t => bail!("unsupported model_type `{t}`"),
        })
    }

    /// Family of a checkpoint, from `architectures` first: a DFlash draft declares `model_type: "qwen3"` (the model it
    /// drafts for), so the architecture name is the only thing that tells it apart.
    pub fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        let arch = |name: &str| {
            cfg["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(name)))
        };
        if arch("DFlashDraftModel") {
            return Ok(Family::Dflash);
        }
        Self::from_model_type(cfg["model_type"].as_str().context("config.json has no model_type")?)
    }

    /// Value of `general.architecture` without the `openmalleus.` prefix; `malleus-model` dispatches on it.
    pub fn arch(self) -> &'static str {
        match self {
            Family::Qwen35 => "qwen35",
            Family::Qwen4Exp => "qwen4exp",
            Family::Ling3 => "ling3",
            Family::Dflash => "dflash",
            Family::Hy4 => "hy4",
            Family::Inkling2 => "inkling2",
            Family::Deepseek41 => "deepseek_v41",
        }
    }

    /// Whether an MLS8 prefill copy of an MLS4 tensor fits the memory budget. qwen4_exp (123 B expert parameters)
    /// keeps it only outside the experts; Ling-3.0-flash (124 GB more) has none and prefills on W4A8. Hy4-flash is
    /// the same shape as qwen4_exp: the routed experts alone are ~75 GB, so only the dense linears get the copy
    /// and the experts never do.
    fn prefill_copy(self, expert: bool) -> bool {
        match self {
            Family::Qwen35 => true,
            Family::Qwen4Exp => !expert,
            Family::Ling3 => false,
            // 738 MB of draft: it never runs on more than a block of positions, so a prefill copy is dead weight.
            Family::Dflash => false,
            Family::Hy4 => !expert,
            // Inkling-2 has no measurement of what an MLS8 prefill would buy: single copy until one exists.
            Family::Inkling2 => false,
            // DeepSeek-V4.1 already carries a 2.00 bpw map chosen per tensor; a second copy would not fit.
            Family::Deepseek41 => false,
        }
    }

    /// Format of one output tensor. `mls3` is the resolved inclusive layer range for the MoE `gate`/`up`; `mls2`
    /// is the same for both routed projections at 2 bits.
    pub fn classify(
        self,
        name: &str,
        shape: &[usize],
        opts: &Options,
        mls3: Option<(usize, usize)>,
        mls2: Option<(usize, usize)>,
    ) -> Kind {
        if name == "lm_head.weight" || (self == Family::Inkling2 && name.ends_with(".unembed.weight")) {
            return Kind::Mls8;
        }
        // The vision tower is 0.9 GB and runs on a handful of positions per image: it stays bf16 as stored.
        if name.contains("visual.") || name.starts_with("vit.") {
            return Kind::Raw;
        }
        if self != Family::Ling3 && MLS8_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            return Kind::Mls8;
        }
        let mls4 = match self {
            Family::Qwen35 | Family::Qwen4Exp => QWEN_MLS4_SUFFIXES,
            Family::Ling3 => LING3_MLS4_SUFFIXES,
            Family::Dflash => DFLASH_MLS4_SUFFIXES,
            Family::Hy4 => HY4_MLS4_SUFFIXES,
            Family::Inkling2 => INKLING2_MLS4_SUFFIXES,
            // DeepSeek-V4.1 does not come through `classify`: its precision map is read from the released build.
            Family::Deepseek41 => &[],
        };
        // The Engram key projections are one 2-D linear per hyper-connection channel (`key_projs.N.weight`), so a
        // suffix does not name them.
        let key_proj = self == Family::Hy4 && name.contains(".engram_layer.key_projs.") && name.ends_with(".weight");
        if !key_proj && !mls4.iter().any(|s| name.ends_with(s)) {
            return Kind::Raw;
        }
        // Inkling-2's MTP head keeps its own layers at bf16. The implementation the head is checked against refuses
        // anything else there, and 4 bits cost it ~4 % of the hidden state its argmax is made of (issue #110): the
        // block is 60M parameters, 0.13 GB of a 15 GB file, bought back by the tokens the draft has to earn.
        if self == Family::Inkling2 && name.starts_with("mtp.") && shape.len() == 2 {
            return Kind::Raw;
        }
        // The MLA output gate of Ling is [num_heads, hidden]: a gate, never quantized. The KDA one is [4096, 2560].
        if self == Family::Ling3 && name.ends_with("attention.g_proj.weight") && shape.first().is_some_and(|&n| n <= 64)
        {
            return Kind::Raw;
        }
        if name.contains(".mlp.")
            && !name.starts_with("mtp.")
            && layer_index(name).is_some_and(|i| opts.keep_int8_layers.contains(&i))
        {
            return Kind::Mls8;
        }
        // MLS2 is the routed-expert format. A family that stacks its shared expert into the same matrix (Qwen's
        // `shared_expert_gate`) has to keep it in the same packing: one stacked buffer has one stride, and the
        // shared expert is expert E of it. Dense linears keep 4 or 8 bits. The MTP layer is deliberately left at
        // 4 bits: it drafts, and a worse draft costs accepted tokens on every step.
        let routed =
            name.ends_with("mlp.experts.gate_up_proj") || (opts.mls2_down && name.ends_with("mlp.experts.down_proj"));
        let shared = name.ends_with("mlp.shared_expert.gate_proj.weight")
            || name.ends_with("mlp.shared_expert.up_proj.weight")
            || (opts.mls2_down && name.ends_with("mlp.shared_expert.down_proj.weight"));
        if let Some((from, to)) = mls2
            && !name.starts_with("mtp.")
            && (routed || shared)
            && layer_index(name).is_some_and(|i| (from..=to).contains(&i))
        {
            return Kind::Mls2;
        }
        // Only the fused gate/up of the routed experts goes to 3 bit: `down` and the shared expert stay at 4.
        if let Some((from, to)) = mls3
            && name.ends_with("mlp.experts.gate_up_proj")
            && layer_index(name).is_some_and(|i| (from..=to).contains(&i))
        {
            return Kind::Mls3;
        }
        Kind::Mls4
    }

    /// The MLS3 layer range the family's model document chose, or `None` if it keeps every expert at 4 bit.
    fn default_mls3(self, cfg: &serde_json::Value) -> Option<(usize, usize)> {
        match self {
            // docs/model-ling3-flash.md, configuration B: 3 bit for the routed gate/up except the last
            // MLS3_SKIP_LAST_LAYERS MoE layers, which dominate output quality and stay at 4.
            Family::Ling3 => {
                let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
                let from = u("first_k_dense_replace");
                let last = u("num_hidden_layers") + u("num_nextn_predict_layers");
                last.checked_sub(1 + MLS3_SKIP_LAST_LAYERS).map(|to| (from, to))
            }
            Family::Qwen35
            | Family::Qwen4Exp
            | Family::Dflash
            | Family::Hy4
            | Family::Inkling2
            | Family::Deepseek41 => None,
        }
    }

    /// The MLS2 layer range the family chose. Only Qwen3.8-Flash-Next is measured with it; a family that has not been
    /// checked keeps every expert at 4 bit.
    fn default_mls2(self, cfg: &serde_json::Value) -> Option<(usize, usize)> {
        match self {
            Family::Qwen4Exp => {
                let t = if cfg["text_config"].is_object() {
                    &cfg["text_config"]
                } else {
                    cfg
                };
                t["num_hidden_layers"].as_u64().map(|n| (0, n as usize - 1))
            }
            Family::Qwen35 | Family::Ling3 | Family::Dflash | Family::Hy4 | Family::Inkling2 | Family::Deepseek41 => {
                None
            }
        }
    }
}

/// Geometry of a MoE block, read from `config.json`; needed to fuse per-expert tensors.
#[derive(Clone, Copy, Debug)]
struct MoeGeom {
    experts: usize,
    inter: usize,
    hidden: usize,
}

/// Linears of a DFlash draft: the projection of the target's features plus the 6 layers. Norms and qk-norms stay bf16.
const DFLASH_MLS4_SUFFIXES: &[&str] = &[
    "fc.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

const QWEN_MLS4_SUFFIXES: &[&str] = &[
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.index_qk_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "mtp.fc.weight",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
];

/// Hy4-flash linears that go to MLS4. Everything else stays raw: the norms, the router (`mlp.gate.weight`) and its
/// correction bias, the learned attention sinks, the iHC parameters (f32), the indexer's fp32 pooling projections and
/// its `intra_group_ape`, the Engram normalization and gate parameters, and the vision tower.
///
/// The compressed indexer's three projections stay raw too, and that is deliberate: they do not compute an output,
/// they *choose* which groups a query recalls, and 4 bits move that choice. Measured at 2100 tokens of context
/// (scripts/ref_indexer_replay.py, layer 0): at rel-L2 0.104 (MLS4) 47 of the 52 queries with a real selection pick
/// a different group than the bf16 reference; perturbing the reference's own weights by the same amount reproduces
/// 46 of them. The three tensors are 5.1 MB per layer in bf16 against 1.3 in MLS4, i.e. 46 MB of disk over the 12
/// indexed layers and +0.2 ms on a decode token — and the selection is exact instead of one group off in 90 % of
/// them. (They were MLS4 in the first conversion of this file.)
const HY4_MLS4_SUFFIXES: &[&str] = &[
    // gated MLA (`*_proj` of the sliding layers carry the wider head shape, same names)
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.linear_gate.weight",
    // The compressed indexer's `query_projection`, `token_key_projection` and `head_weight_projection` are kept raw
    // (bf16) on purpose: see the note above, 4 bits there flips the group selection.
    // dense layer 0
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    // latent MoE: the experts run in 1536 lanes, the two projections around them in the hidden width
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.routed_expert_down_proj.weight",
    "mlp.routed_expert_up_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    // Engram: the value projection; the key projections are matched separately
    "engram_layer.value_proj.weight",
    // DFly2 draft: its MLA, its FFN, the fusion and the corrected-hidden projections
    "mtp_fusion.eh_proj.weight",
    "hidden_correction.gate_proj.weight",
    "hidden_correction.up_proj.weight",
    "hidden_correction.down_proj.weight",
];

/// Ling-3.0-flash linears that go to MLS4. `mlp.gate.weight` (the router) and `mlp.gate.expert_bias` do not end
/// with `mlp.gate_proj.weight` and stay raw, as MIGRATE_MODELS.md §2.4 requires; so do `attention.b_proj`, the three
/// `*_conv1d`, `A_log`, `dt_bias`, `attention.o_norm`, `kv_a_layernorm` and every other norm.
const LING3_MLS4_SUFFIXES: &[&str] = &[
    // KDA (35 layers)
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.f_proj.weight",
    "attention.g_proj.weight",
    "attention.o_proj.weight",
    // MLA (7 layers + MTP); `attention.q_proj.weight` is shared with KDA
    "attention.kv_a_proj_with_mqa.weight",
    "attention.kv_b_proj.weight",
    "attention.dense.weight",
    // Dense FFN (layers 0 and 1)
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    // MoE, fused by this converter into one 3-D tensor per layer
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    // MTP
    "eh_proj.weight",
];

/// qwen4_exp tensors that stay MLS8 in decode too.
const MLS8_SUFFIXES: &[&str] = &["ple.key_proj.weight", "ple.value_proj.weight"];

/// Inkling-2 linears that go to MLS4: the attention projections and the routed experts. Everything else stays raw:
/// the norms, the short-convolution kernels, the L2 clip scales, the attention sink, the router (`mlp.gate`) and its
/// bias, the attention-residual queries (f32), the loop controllers and the vision tower.
///
/// `attn.rel_logits_proj.proj` stays raw on purpose: it does not compute an output, it *scores* relative positions,
/// and 4 bits there move the scores that decide what the attention reads.
const INKLING2_MLS4_SUFFIXES: &[&str] = &[
    "attn.wq_du.weight",
    "attn.wk_dv.weight",
    "attn.wv_dv.weight",
    "attn.wo_ud.weight",
    "attn.wr_du.weight",
    "attn.gate_attn.weight",
    // The MTP head's own layers and its two dense projections.
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    // Routed experts: packed in the checkpoint, fused already, dequantised while the file is written.
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
];

fn layer_index(name: &str) -> Option<usize> {
    let rest = name.split("layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
}

/// Rows `[row0, row0 + rows)` of an output tensor come from source tensor `src` (its first `rows` rows, `k` columns).
#[derive(Clone, Debug)]
struct Part {
    src: String,
    row0: usize,
    rows: usize,
    k: usize,
}

#[derive(Clone, Debug)]
struct Out {
    name: String,
    kind: Kind,
    /// Output shape; quantized tensors are [.., N, K] with K a multiple of 128.
    shape: Vec<usize>,
    parts: Vec<Part>,
    /// Store an MLS8 prefill copy next to an MLS4 tensor.
    prefill: bool,
}

/// Source rows converted per step (bf16 bytes): bounds the RAM used by large expert tensors.
const CHUNK_BYTES: usize = 256 << 20;

/// (rows, K) of a 2-D tensor or of a 3-D expert stack [E, N, K] seen as E * N rows.
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    let k = *shape.last().unwrap_or(&0);
    (shape.iter().rev().skip(1).product(), k)
}

/// Whether calibrated MLS4 tensors are available for `o`: under its own output name, or, for a composite the
/// converter assembles out of several sources, under the names of its parts.
fn is_calibrated(c: &SafeTensors, o: &Out) -> bool {
    c.info(&o.name).is_some() || o.parts.iter().any(|p| c.info(&p.src).is_some())
}

/// Shape and dtype of one calibrated tensor: `rows` x `qn` nibbles and `rows` x `sn` f16 scales.
fn check_calibrated(c: &SafeTensors, name: &str, rows: usize, qn: usize, sn: usize) -> Result<()> {
    let q = c.info(name).context("info")?;
    let s = c
        .info(&format!("{name}.scales"))
        .with_context(|| format!("{name}.scales missing"))?;
    ensure!(
        rows_cols(&q.shape) == (rows, qn) && rows_cols(&s.shape) == (rows, sn) && s.dtype == DType::F16,
        "{name}: calibrated tensors have the wrong shape"
    );
    Ok(())
}

/// Calibrated nibbles and scales of `o`, whose output is `rows` x `k`. A plain tensor is read by output name. A
/// composite is assembled from the calibrated parts at their row offsets, because `scripts/calibrate.py` only ever
/// sees the source tensors — `input_mix_weight_down` and `block_inject_weight` — while the padded `mix_down` row
/// layout is known here alone. Rows no part covers stay zero, which is what the writer pads them with anyway.
fn read_calibrated(c: &SafeTensors, o: &Out, rows: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let (qn, sn) = (k / 2, k / BLOCK);
    if c.info(&o.name).is_some() {
        check_calibrated(c, &o.name, rows, qn, sn)?;
        return Ok((c.read_all(&o.name)?, c.read_all(&format!("{}.scales", o.name))?));
    }
    let (mut q, mut s) = (vec![0u8; rows * qn], vec![0u8; rows * sn * 2]);
    for p in &o.parts {
        if c.info(&p.src).is_none() {
            continue;
        }
        check_calibrated(c, &p.src, p.rows, qn, sn)?;
        q[p.row0 * qn..(p.row0 + p.rows) * qn].copy_from_slice(&c.read_all(&p.src)?);
        let sc = c.read_all(&format!("{}.scales", p.src))?;
        s[p.row0 * sn * 2..(p.row0 + p.rows) * sn * 2].copy_from_slice(&sc);
    }
    Ok((q, s))
}

/// Reads `rows` rows starting at flat row `row0` of a part as f32 values. A source in a packed expert format
/// (NVFP4: U8 codes with one e4m3 scale per 16 columns and an outer `amax / denominator`; MXFP8: e4m3 codes with
/// one ue8m0 scale per 32) is dequantised on the way, so the caller sees the weights the checkpoint means.
fn read_part(st: &SafeTensors, p: &Part, row0: usize, rows: usize) -> Result<Vec<f32>> {
    let info = st
        .info(&p.src)
        .with_context(|| format!("{}: not in the checkpoint", p.src))?;
    match info.dtype {
        DType::BF16 => {
            let mut raw = vec![0u8; rows * p.k * 2];
            st.read(&p.src, row0 * p.k * 2, &mut raw)?;
            Ok(bf16_rows(&raw))
        }
        DType::U8 | DType::F8E4M3 => {
            let nvfp4 = info.dtype == DType::U8;
            ensure!(
                info.shape.len() == 3,
                "{}: a packed expert tensor is [experts, rows, cols], got {:?}",
                p.src,
                info.shape
            );
            let per = info.shape[1];
            let cols = if nvfp4 { info.shape[2] * 2 } else { info.shape[2] };
            ensure!(cols == p.k, "{}: {cols} columns, the part says {}", p.src, p.k);
            let (row_bytes, block) = if nvfp4 {
                (cols / 2, NVFP4_BLOCK)
            } else {
                (cols, MXFP8_BLOCK)
            };
            let scale = format!("{}_scale", p.src);
            let mut out = vec![0f32; rows * cols];
            let (mut rel, mut done) = (row0, 0);
            while done < rows {
                let (e, r) = (rel / per, rel % per);
                let nr = (per - r).min(rows - done);
                let mut q = vec![0u8; nr * row_bytes];
                let mut s = vec![0u8; nr * (cols / block)];
                st.read(&p.src, (e * per + r) * row_bytes, &mut q)?;
                st.read(&scale, (e * per + r) * (cols / block), &mut s)?;
                let chunk = &mut out[done * cols..(done + nr) * cols];
                if nvfp4 {
                    nvfp4_rows(&q, &s, outer_scale(st, &p.src, e)?, cols, chunk);
                } else {
                    mxfp8_rows(&q, &s, cols, chunk);
                }
                rel += nr;
                done += nr;
            }
            Ok(out)
        }
        d => bail!("{}: {d:?} cannot be read as rows", p.src),
    }
}

/// The outer scale of expert `e` of a packed NVFP4 tensor: `amax / denominator`, from its companions.
fn outer_scale(st: &SafeTensors, name: &str, e: usize) -> Result<f32> {
    let amax = st.read_all(&format!("{name}_amax"))?;
    let den = st.read_all(&format!("{name}_outer_scale_denominator"))?;
    ensure!(den.len() == 4, "{name}_outer_scale_denominator: not one f32");
    ensure!(amax.len() >= (e + 1) * 4, "{name}_amax: shorter than its expert count");
    let d = f32::from_le_bytes([den[0], den[1], den[2], den[3]]);
    let a = f32::from_le_bytes([amax[e * 4], amax[e * 4 + 1], amax[e * 4 + 2], amax[e * 4 + 3]]);
    ensure!(d > 0.0 && d.is_finite(), "{name}_outer_scale_denominator: {d}");
    ensure!(a >= 0.0 && a.is_finite(), "{name}_amax expert {e}: {a}");
    Ok(a / d)
}

pub(crate) fn bf16_rows(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect()
}

pub(crate) fn quantize_mls4(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let g = k / BLOCK;
    let rows: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = Vec::with_capacity(k / 2);
            let mut s = Vec::with_capacity(g * 2);
            for grp in row.as_chunks::<BLOCK>().0 {
                let (nib, sc) = quant::quantize_group_int4(grp);
                q.extend_from_slice(&nib);
                s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
            }
            (q, s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k / 2);
    let mut s = Vec::with_capacity(n * g * 2);
    for (a, b) in rows {
        q.extend(a);
        s.extend(b);
    }
    (q, s)
}

pub(crate) fn quantize_mls3(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let g = k / BLOCK;
    let rows: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = Vec::with_capacity(k / 8 * 3);
            let mut s = Vec::with_capacity(g * 2);
            for grp in row.as_chunks::<BLOCK>().0 {
                let (bits, sc) = quant::quantize_group_int3(grp);
                q.extend_from_slice(&bits);
                s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
            }
            (q, s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k / 8 * 3);
    let mut s = Vec::with_capacity(n * g * 2);
    for (a, b) in rows {
        q.extend(a);
        s.extend(b);
    }
    (q, s)
}

pub(crate) fn quantize_mls2(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let g = k / BLOCK;
    let rows: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = Vec::with_capacity(k / 4);
            let mut s = Vec::with_capacity(g * 2);
            for grp in row.as_chunks::<BLOCK>().0 {
                let (bits, sc) = quant::quantize_group_int2(grp, Int2Grid::E2m1);
                q.extend_from_slice(&bits);
                s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
            }
            (q, s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k / 4);
    let mut s = Vec::with_capacity(n * g * 2);
    for (a, b) in rows {
        q.extend(a);
        s.extend(b);
    }
    (q, s)
}

pub(crate) fn quantize_mls8(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let rows: Vec<(Vec<u8>, f32)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = vec![0i8; k];
            let s = quant::quantize_row_int8(&row, &mut q);
            (q.into_iter().map(|v| v as u8).collect(), s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k);
    let mut s = Vec::with_capacity(n * 4);
    for (a, b) in rows {
        q.extend(a);
        s.extend_from_slice(&b.to_le_bytes());
    }
    (q, s)
}

/// Unrotated int8 rows with a trailing f32 scale: [n, k + 4] bytes.
pub(crate) fn quantize_rows_i8(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * (k + 4)];
    out.par_chunks_mut(k + 4).enumerate().for_each(|(r, o)| {
        let mut q = vec![0i8; k];
        let s = quant::quantize_row_int8(&w[r * k..(r + 1) * k], &mut q);
        for (d, v) in o[..k].iter_mut().zip(q) {
            *d = v as u8;
        }
        o[k..].copy_from_slice(&s.to_le_bytes());
    });
    out
}

/// Unrotated 4-bit rows, one f16 scale per row: `n * k / 2` bytes of nibbles (LSB first, `value + 8`) and `n`
/// scales. The Engram rows are 96 wide and are read one at a time on the host, so neither the 128-wide rotation
/// of MLS4 nor its per-block scale applies; per-row absmax keeps them at 52 bytes for a 192-byte bf16 row.
pub(crate) fn quantize_rows_i4(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let packed = k.div_ceil(2);
    let rows: Vec<(Vec<u8>, f32)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let row = &w[r * k..(r + 1) * k];
            let a = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sc = if a > 0.0 { a / 7.0 } else { 1.0 };
            let mut q = vec![0u8; packed];
            for (i, v) in row.iter().enumerate() {
                let qv = (v / sc).round().clamp(-8.0, 7.0) as i32 + 8;
                q[i / 2] |= (qv as u8 & 0xF) << (4 * (i % 2));
            }
            (q, sc)
        })
        .collect();
    let mut q = Vec::with_capacity(n * packed);
    let mut s = Vec::with_capacity(n * 2);
    for (a, sc) in rows {
        q.extend(a);
        s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
    }
    (q, s)
}

/// Hyper-connection low-rank width padded for the 128-wide rotation blocks of MLS8.
pub(crate) fn pad128(n: usize) -> usize {
    n.next_multiple_of(BLOCK)
}

fn shard_index(name: &str) -> Option<usize> {
    name.split("ngram_embedding.shard_")
        .nth(1)?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Output tensors for the sources, in source-name order (a composite tensor takes the place of its first source).
fn plan(
    st: &SafeTensors,
    names: &[String],
    opts: &Options,
    fam: Family,
    geom: Option<MoeGeom>,
    mls3: Option<(usize, usize)>,
    mls2: Option<(usize, usize)>,
) -> Result<Vec<Out>> {
    let mut outs = Vec::new();
    let mut seen_tables = std::collections::HashSet::new();
    let mut fused_layers = std::collections::HashSet::new();
    for name in names {
        let info = st.info(name).context("info")?;
        if let Some(g) = geom
            && let Some(prefix) = expert_layer(name)
        {
            // The first expert tensor of a layer emits both fused tensors; the others are skipped.
            if fused_layers.insert(prefix.clone()) {
                for gate_up in [true, false] {
                    let which = if gate_up { "gate_up_proj" } else { "down_proj" };
                    let name = format!("{prefix}.mlp.experts.{which}");
                    let shape = fused_shape(g, gate_up);
                    let kind = fam.classify(&name, &shape, opts, mls3, mls2);
                    ensure!(kind != Kind::Raw, "{name}: experts must be quantized");
                    let parts = expert_parts(g, &prefix, gate_up);
                    for p in &parts {
                        let pi = st
                            .info(&p.src)
                            .with_context(|| format!("{}: not in the checkpoint", p.src))?;
                        ensure!(
                            pi.dtype == DType::BF16 && rows_cols(&pi.shape) == (p.rows, p.k),
                            "{}: expected bf16 [{}, {}]",
                            p.src,
                            p.rows,
                            p.k
                        );
                    }
                    outs.push(Out {
                        name,
                        kind,
                        shape,
                        parts,
                        prefill: kind == Kind::Mls4 && opts.prefill_int8 && fam.prefill_copy(true),
                    });
                }
            }
            continue;
        }
        let one = |kind: Kind| -> Out {
            let (n, k) = rows_cols(&info.shape);
            Out {
                name: name.clone(),
                kind,
                shape: info.shape.clone(),
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows: n,
                    k,
                }],
                prefill: false,
            }
        };
        // Inkling-2 packs its routed experts per layer into two tensors (`w13` = gate+up, `w2` = down), already in
        // the fused shape and row order: the plan only renames them. The companions (`_scale`, `_amax`,
        // `_outer_scale_denominator`) are read by the dequantiser and dropped here.
        if name.ends_with(".mlp.experts.w13_weight") || name.ends_with(".mlp.experts.w2_weight") {
            let gate_up = name.ends_with("w13_weight");
            let prefix = name
                .strip_suffix(if gate_up { "w13_weight" } else { "w2_weight" })
                .unwrap();
            // The packed tensor's last dimension counts codes, not columns: two per byte for NVFP4.
            ensure!(
                info.shape.len() == 3 && matches!(info.dtype, DType::U8 | DType::F8E4M3),
                "{name}: a packed expert tensor is [E, rows, codes] in U8 or F8_E4M3, got {:?} {:?}",
                info.dtype,
                info.shape
            );
            let last = info.shape[2];
            let cols = if info.dtype == DType::U8 { last * 2 } else { last };
            let rows = info.shape[0] * info.shape[1];
            let shape = vec![info.shape[0], info.shape[1], cols];
            let out_name = format!("{prefix}{}", if gate_up { "gate_up_proj" } else { "down_proj" });
            let kind = fam.classify(&out_name, &shape, opts, mls3, mls2);
            ensure!(kind != Kind::Raw, "{out_name}: experts must be quantized");
            outs.push(Out {
                name: out_name,
                kind,
                shape,
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows,
                    k: cols,
                }],
                prefill: false,
            });
            continue;
        }
        if name.ends_with("_weight_scale")
            || name.ends_with("_weight_amax")
            || name.ends_with("_weight_outer_scale_denominator")
        {
            continue;
        }
        if let Some(hc) = name.strip_suffix("input_mix_weight_down.weight") {
            let (r, k) = rows_cols(&info.shape);
            let inject = format!("{hc}block_inject_weight.weight");
            let mut parts = vec![Part {
                src: name.clone(),
                row0: 0,
                rows: r,
                k,
            }];
            if let Some(ii) = st.info(&inject) {
                let (ri, ki) = rows_cols(&ii.shape);
                ensure!(ki == k, "{inject}: K {ki} != {k}");
                parts.push(Part {
                    src: inject,
                    row0: r,
                    rows: ri,
                    k,
                });
            }
            let rows = parts.iter().map(|p| p.row0 + p.rows).max().unwrap_or(0);
            outs.push(Out {
                name: format!("{hc}mix_down.weight"),
                kind: if opts.hc_mls4 { Kind::Mls4 } else { Kind::Mls8 },
                shape: vec![pad128(rows), k],
                parts,
                prefill: opts.hc_mls4 && opts.prefill_int8,
            });
            continue;
        }
        if name.ends_with("block_inject_weight.weight") {
            ensure!(
                st.info(&name.replace("block_inject_weight", "input_mix_weight_down"))
                    .is_some(),
                "{name} without its input_mix_weight_down"
            );
            continue;
        }
        if let Some(hc) = name.strip_suffix("input_mix_weight_up.weight") {
            let (n, k) = rows_cols(&info.shape);
            let down = st
                .info(&format!("{hc}input_mix_weight_down.weight"))
                .with_context(|| format!("{name} without its input_mix_weight_down"))?;
            ensure!(rows_cols(&down.shape).0 == k, "{name}: K {k} != down rows");
            outs.push(Out {
                name: format!("{hc}mix_up.weight"),
                kind: if opts.hc_mls4 { Kind::Mls4 } else { Kind::Mls8 },
                shape: vec![n, pad128(k)],
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows: n,
                    k,
                }],
                prefill: opts.hc_mls4 && opts.prefill_int8,
            });
            continue;
        }
        // The Engram memory table: 164.9M rows of 96 per layer, ~127 GB in bf16 and ~33 GB at 4 bits per row.
        if name.ends_with("engram_layer.multi_head_embedding.embedding.weight") {
            let (n, k) = rows_cols(&info.shape);
            outs.push(Out {
                name: name.clone(),
                kind: Kind::RowI4,
                shape: vec![n, k],
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows: n,
                    k,
                }],
                prefill: false,
            });
            continue;
        }
        if let Some(i) = name.find("ngram_embedding.shard_") {
            let base = &name[..i];
            if !seen_tables.insert(base.to_string()) {
                continue;
            }
            let mut shards: Vec<(usize, String)> = names
                .iter()
                .filter(|n| n.starts_with(base) && n.contains("ngram_embedding.shard_"))
                .map(|n| (shard_index(n).unwrap_or(usize::MAX), n.clone()))
                .collect();
            shards.sort();
            ensure!(
                shards.iter().enumerate().all(|(j, s)| s.0 == j),
                "{base}: n-gram shards are not numbered 0..{}",
                shards.len()
            );
            let mut parts = Vec::new();
            let mut row0 = 0;
            let mut dim = 0;
            for (_, s) in &shards {
                let si = st.info(s).context("info")?;
                let (r, k) = rows_cols(&si.shape);
                ensure!(dim == 0 || dim == k, "{s}: width {k} != {dim}");
                dim = k;
                parts.push(Part {
                    src: s.clone(),
                    row0,
                    rows: r,
                    k,
                });
                row0 += r;
            }
            outs.push(Out {
                name: format!("{base}ngram_table"),
                kind: Kind::RowI8,
                shape: vec![row0, dim + 4],
                parts,
                prefill: false,
            });
            continue;
        }
        let kind = fam.classify(name, &info.shape, opts, mls3, mls2);
        let mut o = one(kind);
        if kind != Kind::Raw {
            ensure!(
                info.dtype == DType::BF16,
                "{name}: only bf16 input is supported for now"
            );
            ensure!(
                matches!(info.shape.len(), 2 | 3),
                "{name}: quantized tensors must be 2-D or 3-D (experts)"
            );
            let k = *info.shape.last().unwrap_or(&0);
            if k % BLOCK != 0 {
                bail!("{name}: K={k} not a multiple of {BLOCK}");
            }
            let routed = name.contains(".mlp.experts.");
            let expert = routed || name.contains(".mlp.shared_expert");
            o.prefill = kind == Kind::Mls4
                && opts.prefill_int8
                && fam.prefill_copy(expert)
                && !(routed && opts.no_prefill_int8_experts);
        }
        outs.push(o);
    }
    Ok(outs)
}

pub fn convert(model_dir: &Path, out: &Path, opts: &Options, log: &dyn Fn(&str)) -> Result<()> {
    let st = SafeTensors::open_dir(model_dir)?;
    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    let read = |f: &str| std::fs::read_to_string(model_dir.join(f)).with_context(|| f.to_string());
    let config = read("config.json")?;
    let cfg: serde_json::Value = serde_json::from_str(&config)?;
    let fam = Family::from_config(&cfg)?;
    // DeepSeek-V4.1 has its own converter: its precision map comes from the released 2.00 bpw build and its
    // weights are blocked fp8 and packed fp4, neither of which the generic planner knows.
    if fam == Family::Deepseek41 {
        let path = std::env::var("MALLEUS_DEEPSEEK_ALLOC").map_err(|_| {
            anyhow::anyhow!("deepseek_v41: set MALLEUS_DEEPSEEK_ALLOC to the released build's allocation text")
        })?;
        let text = std::fs::read_to_string(&path).with_context(|| path.clone())?;
        let alloc = malleus_gguf::deepseek_alloc::Allocation::parse(&text)?;
        return deepseek::convert(model_dir, out, &alloc, log);
    }
    // Ling-3.0-flash stores the experts one tensor at a time: they are fused, one 3-D tensor per layer and projection.
    // Qwen stores them stacked, except the MTP layer of some checkpoints (Ornith): the geometry is only used for
    // names that match the per-expert pattern, and its keys live under `text_config` there.
    let geom = match fam {
        Family::Ling3 => {
            let u = |k: &str| -> Result<usize> {
                Ok(cfg[k].as_u64().with_context(|| format!("config.json has no {k}"))? as usize)
            };
            Some(MoeGeom {
                experts: u("num_experts")?,
                inter: u("moe_intermediate_size")?,
                hidden: u("hidden_size")?,
            })
        }
        // A DFlash draft has no MoE.
        Family::Dflash => None,
        // Hy4 stores its routed experts fused (`[E, 2 * latent, latent]`), latent MoE included.
        Family::Hy4 => None,
        // Inkling-2 already stores its routed experts fused, one 3-D tensor per projection and layer.
        Family::Inkling2 => None,
        // DeepSeek-V4.1 returns above, before the generic planner runs.
        Family::Deepseek41 => None,
        Family::Qwen35 | Family::Qwen4Exp => {
            let t = if cfg["text_config"].is_object() {
                &cfg["text_config"]
            } else {
                &cfg
            };
            let u = |k: &str| t[k].as_u64().map(|v| v as usize);
            match (u("num_experts"), u("moe_intermediate_size"), u("hidden_size")) {
                (Some(experts), Some(inter), Some(hidden)) => Some(MoeGeom { experts, inter, hidden }),
                _ => None,
            }
        }
    };
    if opts.prefill_int8 && !fam.prefill_copy(true) {
        log("no MLS8 prefill copy of the experts: it does not fit the memory budget of this family");
    }
    let mut w = GgufWriter::new();
    w.meta(
        "general.architecture",
        Value::Str(format!("{ARCH_PREFIX}{}", fam.arch())),
    );
    w.meta(
        "general.name",
        Value::Str(model_dir.file_name().and_then(|s| s.to_str()).unwrap_or("model").into()),
    );
    w.meta("openmalleus.layout_version", Value::U32(LAYOUT_VERSION));
    w.meta("openmalleus.hadamard_seed", Value::U64(HADAMARD_SEED));
    w.meta("openmalleus.hf_config", Value::Str(config));
    if let Ok(g) = read("generation_config.json") {
        w.meta("openmalleus.generation_config", Value::Str(g));
    }
    if let Ok(p) = read("preprocessor_config.json") {
        w.meta("openmalleus.preprocessor_config", Value::Str(p));
    }
    // A speculative draft has no tokenizer of its own (it borrows the target's), so its absence is expected there
    // and fatal everywhere else: a model file that cannot tokenize is not a model.
    match (read("tokenizer.json"), read("chat_template.jinja")) {
        (Ok(tok), Ok(tpl)) => {
            w.meta("tokenizer.huggingface.json", Value::Str(tok));
            w.meta("tokenizer.chat_template", Value::Str(tpl));
        }
        (Err(e), _) if fam == Family::Dflash => {
            log(&format!("no tokenizer ({e}): a draft uses the target's, as expected"));
        }
        (Err(e), _) => return Err(e),
        (Ok(_), Err(e)) => return Err(e),
    }

    let mls3 = match opts.mls3_layers {
        Mls3Layers::Off => None,
        Mls3Layers::Range(from, to) => Some((from, to)),
        Mls3Layers::Family => fam.default_mls3(&cfg),
    };
    let mls2 = match opts.mls2_layers {
        Mls2Layers::Off => None,
        Mls2Layers::Range(from, to) => Some((from, to)),
        Mls2Layers::Family => fam.default_mls2(&cfg),
    };
    match mls3 {
        Some((from, to)) => log(&format!("MoE gate/up at MLS3 for layers {from}..={to}")),
        None => log("every expert at MLS4"),
    }
    match mls2 {
        Some((from, to)) => log(&format!("routed experts at MLS2 for layers {from}..={to}")),
        None => log("no routed expert at MLS2"),
    }

    let plan = plan(&st, &names, opts, fam, geom, mls3, mls2)?;
    let calib = opts.mls4_from.as_deref().map(SafeTensors::open_dir).transpose()?;
    let mut calibrated = 0;
    let mut bytes = 0usize;
    let mut tensors = 0usize;
    let mut add = |name: &str, dtype: DType, shape: &[usize]| {
        bytes += dtype.size_of_elems(shape.iter().product());
        tensors += 1;
        w.add_tensor(name, dtype, shape);
    };
    for o in &plan {
        let (lead, k) = o.shape.split_at(o.shape.len() - 1);
        let k = k[0];
        let dims = |last: &[usize]| [lead, last].concat();
        match o.kind {
            Kind::Raw => {
                let info = st.info(&o.name).context("info")?;
                add(&o.name, info.dtype, &info.shape);
            }
            Kind::Mls2 => {
                add(&o.name, DType::Mls2, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, &dims(&[k / BLOCK]));
            }
            Kind::Mls3 => {
                add(&o.name, DType::Mls3, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, &dims(&[k / BLOCK]));
            }
            Kind::Mls4 => {
                add(&o.name, DType::Mls4, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, &dims(&[k / BLOCK]));
                if o.prefill {
                    add(&format!("{}.prefill", o.name), DType::Mls8, &o.shape);
                    add(&format!("{}.prefill.scales", o.name), DType::F32, lead);
                }
            }
            Kind::Mls8 => {
                add(&o.name, DType::Mls8, &o.shape);
                add(&format!("{}.scales", o.name), DType::F32, lead);
            }
            Kind::RowI8 => add(&o.name, DType::I8, &o.shape),
            Kind::RowI4 => {
                add(&o.name, DType::RowI4, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, lead);
            }
        }
    }
    log(&format!(
        "{} plan entries -> {tensors} output tensors, {:.2} GB ({})",
        plan.len(),
        bytes as f64 / 1e9,
        if opts.prefill_int8 {
            "MLS4 for decode + MLS8 copies for prefill"
        } else {
            "MLS4 only: the prefill reads the same 4-bit weights"
        }
    ));
    // The name is the only thing a directory listing shows, so it has to say whether the file carries the second
    // copy. A warning and not an error: the file is correct either way, and renaming is the caller's business.
    let name = out.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if opts.prefill_int8 && !name.contains("mls8") {
        log(&format!(
            "warning: {name} carries the MLS8 prefill copies: name it `<model>-mls4-mls8.malleus` so a listing \
             does not hide that it is twice the size"
        ));
    } else if !opts.prefill_int8 && name.contains("mls8") {
        log(&format!(
            "warning: {name} says mls8 but carries a single MLS4 copy: name it `<model>-mls4.malleus`"
        ));
    }
    if let Some(c) = &calib {
        // Validate every calibrated tensor up front: a shape mismatch found after 100 GB of writing is useless.
        for o in &plan {
            if o.kind != Kind::Mls4 || !is_calibrated(c, o) {
                continue;
            }
            let (n, k) = rows_cols(&o.shape);
            let (qn, sn) = (k / 2, k / BLOCK);
            if c.info(&o.name).is_some() {
                check_calibrated(c, &o.name, n, qn, sn)?;
            } else {
                for p in &o.parts {
                    if c.info(&p.src).is_some() {
                        check_calibrated(c, &p.src, p.rows, qn, sn)?;
                    }
                }
            }
            calibrated += 1;
        }
        let total = plan.iter().filter(|p| p.kind == Kind::Mls4).count();
        log(&format!(
            "calibrated MLS4: {calibrated}/{total} tensors, the rest round-to-nearest"
        ));
        w.meta("openmalleus.mls4_calibrated", Value::U32(calibrated));
    }
    let mut sink = w.begin(out)?;
    let total = plan.len();
    let t0 = std::time::Instant::now();
    for (i, o) in plan.iter().enumerate() {
        if o.kind == Kind::Raw {
            let info = st.info(&o.name).context("info")?;
            let mut buf = vec![0u8; 64 << 20];
            let mut off = 0;
            while off < info.size {
                let n = buf.len().min(info.size - off);
                st.read(&o.name, off, &mut buf[..n])?;
                sink.write(&buf[..n])?;
                off += n;
            }
            continue;
        }
        let (n, k) = rows_cols(&o.shape);
        if o.kind == Kind::RowI4 {
            // Written in row chunks: the table is ~33 GB.
            let step = (CHUNK_BYTES / (2 * k)).max(1);
            let (mut q, mut s) = (Vec::new(), Vec::new());
            for r0 in (0..n).step_by(step) {
                let rows = step.min(n - r0);
                let mut raw = vec![0u8; rows * k * 2];
                st.read(&o.name, r0 * k * 2, &mut raw)?;
                let (a, b) = quantize_rows_i4(&bf16_rows(&raw), rows, k);
                q.extend(a);
                s.extend(b);
            }
            sink.write(&q)?;
            sink.write(&s)?;
            log(&format!(
                "[{i}/{total}] {} ({n} rows, {:.0}s)",
                o.name,
                t0.elapsed().as_secs_f64()
            ));
            continue;
        }
        if o.kind == Kind::RowI8 {
            // Written part by part: the table is ~50 GB.
            let kk = k - 4;
            for p in &o.parts {
                let step = (CHUNK_BYTES / (2 * kk)).max(1);
                for r0 in (0..p.rows).step_by(step) {
                    let rows = step.min(p.rows - r0);
                    let mut raw = vec![0u8; rows * kk * 2];
                    st.read(&p.src, r0 * kk * 2, &mut raw)?;
                    sink.write(&quantize_rows_i8(&bf16_rows(&raw), rows, kk))?;
                }
            }
            log(&format!(
                "[{i}/{total}] {} ({n} rows, {:.0}s)",
                o.name,
                t0.elapsed().as_secs_f64()
            ));
            continue;
        }
        // Calibration files hold MLS4 tensors only: an MLS3 tensor is always round-to-nearest.
        let cal = calib.as_ref().filter(|c| o.kind == Kind::Mls4 && is_calibrated(c, o));
        let (mut qd, mut sd, mut q8, mut s8) = (vec![], vec![], vec![], vec![]);
        let step = (CHUNK_BYTES / (2 * k)).max(1);
        for r0 in (0..n).step_by(step) {
            let rows = step.min(n - r0);
            let mut wf = vec![0f32; rows * k];
            for p in &o.parts {
                let (a, b) = (p.row0.max(r0), (p.row0 + p.rows).min(r0 + rows));
                if a >= b {
                    continue;
                }
                let src = read_part(&st, p, a - p.row0, b - a)?;
                for r in 0..b - a {
                    let d = (a - r0 + r) * k;
                    wf[d..d + p.k].copy_from_slice(&src[r * p.k..(r + 1) * p.k]);
                }
            }
            if matches!(o.kind, Kind::Mls2 | Kind::Mls3 | Kind::Mls4) && cal.is_none() {
                let (q, s) = match o.kind {
                    Kind::Mls2 => quantize_mls2(&wf, rows, k),
                    Kind::Mls3 => quantize_mls3(&wf, rows, k),
                    _ => quantize_mls4(&wf, rows, k),
                };
                qd.extend(q);
                sd.extend(s);
            }
            if o.kind == Kind::Mls8 || o.prefill {
                let (q, s) = quantize_mls8(&wf, rows, k);
                q8.extend(q);
                s8.extend(s);
            }
        }
        if matches!(o.kind, Kind::Mls2 | Kind::Mls3 | Kind::Mls4) {
            if let Some(c) = cal {
                (qd, sd) = read_calibrated(c, o, n, k)?;
            }
            sink.write(&qd)?;
            sink.write(&sd)?;
        }
        if !q8.is_empty() {
            sink.write(&q8)?;
            sink.write(&s8)?;
        }
        if i % 25 == 0 {
            log(&format!(
                "[{i}/{total}] {} ({:.0}s)",
                o.name,
                t0.elapsed().as_secs_f64()
            ));
        }
    }
    sink.finish()?;
    Ok(())
}

/// Layer index of a per-expert tensor (`model.layers.5.mlp.experts.17.gate_proj.weight` → 5), or `None` for the
/// already-fused 3-D tensors of the other families (`model.layers.5.mlp.experts.gate_up_proj`).
/// Prefix of a layer whose routed experts come one tensor per expert: `model.layers.N` (Ling-3.0-flash) or
/// `mtp.layers.N` (the MTP layer of some Qwen checkpoints, e.g. Ornith, where the main stack is fused but the MTP is
/// not). The prefix names the fused output too, which is what the loader reads.
fn expert_layer(name: &str) -> Option<String> {
    let parse = |prefix: &str| -> Option<String> {
        let rest = name.strip_prefix(prefix)?;
        let (layer, tail) = rest.split_once('.')?;
        tail.strip_prefix("mlp.experts.")?
            .split('.')
            .next()?
            .parse::<usize>()
            .ok()?;
        layer.parse::<usize>().ok()?;
        Some(format!("{prefix}{layer}"))
    };
    parse("model.layers.").or_else(|| parse("mtp.layers."))
}

/// Shape of the tensor that fuses the experts of one layer: `[E, rows_per_expert, K]`.
fn fused_shape(g: MoeGeom, gate_up: bool) -> Vec<usize> {
    let (rows, k) = if gate_up {
        (2 * g.inter, g.hidden)
    } else {
        (g.hidden, g.inter)
    };
    vec![g.experts, rows, k]
}

/// Sources of a fused expert tensor in row order: per expert, gate then up (`gate_up`) or down.
fn expert_parts(g: MoeGeom, prefix: &str, gate_up: bool) -> Vec<Part> {
    let mut parts = Vec::new();
    for e in 0..g.experts {
        let p = format!("{prefix}.mlp.experts.{e}");
        if gate_up {
            for (j, w) in ["gate_proj", "up_proj"].iter().enumerate() {
                parts.push(Part {
                    src: format!("{p}.{w}.weight"),
                    row0: (2 * e + j) * g.inter,
                    rows: g.inter,
                    k: g.hidden,
                });
            }
        } else {
            parts.push(Part {
                src: format!("{p}.down_proj.weight"),
                row0: e * g.hidden,
                rows: g.hidden,
                k: g.inter,
            });
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Options {
        Options {
            prefill_int8: false,
            no_prefill_int8_experts: false,
            keep_int8_layers: vec![],
            mls4_from: None,
            mls3_layers: Mls3Layers::Off,
            mls2_layers: Mls2Layers::Off,
            mls2_down: false,
            hc_mls4: false,
        }
    }

    #[test]
    fn family_from_model_type() {
        let f = |t: &str| Family::from_model_type(t).unwrap();
        assert_eq!(f("qwen3_5"), Family::Qwen35);
        assert_eq!(f("qwen3_5_moe"), Family::Qwen35);
        assert_eq!(f("qwen4_exp"), Family::Qwen4Exp);
        assert_eq!(f("bailing_hybrid"), Family::Ling3);
        assert_eq!(f("hy_v4_flash"), Family::Hy4);
        assert_eq!(Family::Hy4.arch(), "hy4");
        assert!(!Family::Hy4.prefill_copy(true) && Family::Hy4.prefill_copy(false));
        assert_eq!(Family::Ling3.arch(), "ling3");
        assert_eq!(Family::Qwen4Exp.arch(), "qwen4exp");
        assert!(Family::Qwen35.prefill_copy(true) && !Family::Ling3.prefill_copy(true));
        assert!(Family::Qwen4Exp.prefill_copy(false) && !Family::Qwen4Exp.prefill_copy(true));
        assert!(Family::from_model_type("llama").is_err());
    }

    /// Inkling-2: the text stack's linears go to 4 bits, the MTP head's own layers stay bf16 (it is the draft, and
    /// 4 bits cost it 4 % of the hidden state its argmax is made of), and its context projections were never in the
    /// 4-bit list.
    #[test]
    fn inkling2_mtp_head_stays_bf16() {
        let o = opts();
        let c = |name: &str, shape: &[usize]| Family::Inkling2.classify(name, shape, &o, None, None);
        assert_eq!(c("model.layers.0.attn.wq_du.weight", &[3072, 2304]), Kind::Mls4);
        for l in 0..3 {
            let p = format!("mtp.layers.{l}.");
            for (name, shape) in [
                ("self_attn.q_proj.weight", vec![3072, 2304]),
                ("self_attn.k_proj.weight", vec![1024, 2304]),
                ("self_attn.o_proj.weight", vec![2304, 3072]),
                ("mlp.gate_proj.weight", vec![9216, 2304]),
                ("mlp.down_proj.weight", vec![2304, 9216]),
            ] {
                assert_eq!(c(&format!("{p}{name}"), &shape), Kind::Raw, "{p}{name}");
            }
            assert_eq!(c(&format!("{p}self_attn.k_proj_ctx.weight"), &[1024, 2304]), Kind::Raw);
            assert_eq!(c(&format!("{p}self_attn.q_norm.weight"), &[128]), Kind::Raw);
        }
        assert_eq!(c("mtp.markov_head.markov_w1.weight", &[201024, 256]), Kind::Raw);
    }

    /// Hy4-flash: the linears go to 4 bits, the learned sinks, the iHC parameters in fp32, the indexer's fp32 pooling
    /// projections, the Engram normalization and the whole vision tower stay as stored.
    #[test]
    fn hy4_classification() {
        let o = opts();
        let c = |name: &str, shape: &[usize]| Family::Hy4.classify(name, shape, &o, None, None);
        let l = "model.layers.0.";
        // gated MLA: the linear gate is [8192, 3072], the head sink [32] f32
        assert_eq!(
            c(&format!("{l}self_attn.linear_gate.weight"), &[8192, 3072]),
            Kind::Mls4
        );
        assert_eq!(c(&format!("{l}self_attn.q_b_proj.weight"), &[8192, 1024]), Kind::Mls4);
        assert_eq!(c(&format!("{l}self_attn.kv_b_proj.weight"), &[14336, 512]), Kind::Mls4);
        assert_eq!(c(&format!("{l}self_attn.o_proj.weight"), &[3072, 8192]), Kind::Mls4);
        assert_eq!(c(&format!("{l}self_attn.q_a_layernorm.weight"), &[1024]), Kind::Raw);
        assert_eq!(c(&format!("{l}self_attn.post_proj_norm.weight"), &[3072]), Kind::Raw);
        assert_eq!(c(&format!("{l}self_attn.learnable_sink_param"), &[32]), Kind::Raw);
        // compressed indexer: the three projections that decide the selection stay raw, as do the fp32 pooling
        // projections and the apex
        assert_eq!(
            c(
                &format!("{l}self_attn.compressed_indexer.query_projection.weight"),
                &[2048, 1024]
            ),
            Kind::Raw
        );
        assert_eq!(
            c(
                &format!("{l}self_attn.compressed_indexer.token_key_projection.weight"),
                &[128, 3072]
            ),
            Kind::Raw
        );
        assert_eq!(
            c(
                &format!("{l}self_attn.compressed_indexer.head_weight_projection.weight"),
                &[16, 3072]
            ),
            Kind::Raw
        );
        assert_eq!(
            c(
                &format!("{l}self_attn.compressed_indexer.key_value_projection.weight"),
                &[128, 128]
            ),
            Kind::Raw
        );
        assert_eq!(
            c(&format!("{l}self_attn.compressed_indexer.intra_group_ape"), &[4, 128]),
            Kind::Raw
        );
        // latent MoE: the fused experts in the 1536 lanes, the router and the correction bias raw
        assert_eq!(
            c("model.layers.1.mlp.experts.gate_up_proj", &[512, 3072, 1536]),
            Kind::Mls4
        );
        assert_eq!(
            c("model.layers.1.mlp.experts.down_proj", &[512, 1536, 1536]),
            Kind::Mls4
        );
        assert_eq!(
            c("model.layers.1.mlp.routed_expert_down_proj.weight", &[1536, 3072]),
            Kind::Mls4
        );
        assert_eq!(c("model.layers.1.mlp.gate.weight", &[512, 3072]), Kind::Raw);
        assert_eq!(c("model.layers.1.mlp.combine_output_norm.weight", &[3072]), Kind::Raw);
        // iHC: the mixing is fp32 and stays where it is
        assert_eq!(c("model.layers.0.hc_attn_layer.hc_pre.hc_fn", &[8, 12288]), Kind::Raw);
        assert_eq!(c("model.hc_head.hc_head_fn", &[4, 12288]), Kind::Raw);
        // Engram: the table is the converter's RowI4, the projections are 4 bit, the norms raw
        assert_eq!(c("model.layers.1.engram_layer.lookup_table", &[120818]), Kind::Raw);
        assert_eq!(
            c("model.layers.1.engram_layer.value_proj.weight", &[3072, 1536]),
            Kind::Mls4
        );
        assert_eq!(
            c("model.layers.5.engram_layer.key_projs.0.weight", &[3072, 1536]),
            Kind::Mls4
        );
        assert_eq!(c("model.layers.1.engram_layer.unified_pre_gamma", &[3072]), Kind::Raw);
        // vision and lm_head
        assert_eq!(c("vit.layers.0.self_attn.q_proj.weight", &[1152, 1152]), Kind::Raw);
        assert_eq!(c("vit.perceive.proj.0.weight", &[4608, 1152, 2, 2]), Kind::Raw);
        assert_eq!(c("lm_head.weight", &[120832, 3072]), Kind::Mls8);
    }

    /// Ling-3.0-flash: what stays raw is exactly what MIGRATE_MODELS.md §2.4 forbids to quantize, plus the MLA gate.
    #[test]
    fn ling3_classification() {
        let o = opts();
        let c = |name: &str, shape: &[usize]| Family::Ling3.classify(name, shape, &o, None, None);
        let l2 = "model.layers.2.";
        let l5 = "model.layers.5.";
        // KDA (layer 2)
        for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj"] {
            assert_eq!(
                c(&format!("{l2}attention.{p}.weight"), &[4096, 2560]),
                Kind::Mls4,
                "{p}"
            );
        }
        assert_eq!(c(&format!("{l2}attention.b_proj.weight"), &[32, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.q_conv1d.weight"), &[4096, 1, 4]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.A_log"), &[32]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.dt_bias"), &[4096]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.o_norm.weight"), &[128]), Kind::Raw);
        // MLA (layer 5): the head gate is [32, 2560] and stays raw, the KDA one is [4096, 2560] and does not
        assert_eq!(c(&format!("{l5}attention.q_proj.weight"), &[6144, 2560]), Kind::Mls4);
        assert_eq!(
            c(&format!("{l5}attention.kv_a_proj_with_mqa.weight"), &[576, 2560]),
            Kind::Mls4
        );
        assert_eq!(c(&format!("{l5}attention.kv_b_proj.weight"), &[8192, 512]), Kind::Mls4);
        assert_eq!(c(&format!("{l5}attention.dense.weight"), &[2560, 4096]), Kind::Mls4);
        assert_eq!(c(&format!("{l5}attention.g_proj.weight"), &[32, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l5}attention.kv_a_layernorm.weight"), &[512]), Kind::Raw);
        // MoE
        assert_eq!(c(&format!("{l2}mlp.gate.weight"), &[512, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l2}mlp.gate.expert_bias"), &[512]), Kind::Raw);
        assert_eq!(
            c(&format!("{l2}mlp.experts.gate_up_proj"), &[512, 1536, 2560]),
            Kind::Mls4
        );
        assert_eq!(c(&format!("{l2}mlp.experts.down_proj"), &[512, 2560, 768]), Kind::Mls4);
        assert_eq!(
            c(&format!("{l2}mlp.shared_experts.gate_proj.weight"), &[768, 2560]),
            Kind::Mls4
        );
        assert_eq!(
            c(&format!("{l2}mlp.shared_experts.down_proj.weight"), &[2560, 768]),
            Kind::Mls4
        );
        // Dense FFN (layers 0 and 1)
        assert_eq!(c("model.layers.0.mlp.gate_proj.weight", &[6144, 2560]), Kind::Mls4);
        assert_eq!(c("model.layers.0.mlp.down_proj.weight", &[2560, 6144]), Kind::Mls4);
        // MTP and globals
        assert_eq!(c("model.layers.42.eh_proj.weight", &[2560, 5120]), Kind::Mls4);
        assert_eq!(c("model.layers.42.enorm.weight", &[2560]), Kind::Raw);
        assert_eq!(c("model.norm.weight", &[2560]), Kind::Raw);
        assert_eq!(c("model.word_embeddings.weight", &[157184, 2560]), Kind::Raw);
        assert_eq!(c("lm_head.weight", &[157184, 2560]), Kind::Mls8);
    }

    #[test]
    fn qwen35_classification_is_unchanged() {
        let o = opts();
        let c = |name: &str| Family::Qwen35.classify(name, &[4096, 2048], &o, None, None);
        assert_eq!(c("model.language_model.layers.0.self_attn.q_proj.weight"), Kind::Mls4);
        assert_eq!(c("model.language_model.layers.3.mlp.experts.gate_up_proj"), Kind::Mls4);
        assert_eq!(
            c("model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"),
            Kind::Mls4
        );
        assert_eq!(c("model.language_model.layers.0.linear_attn.A_log"), Kind::Raw);
        assert_eq!(c("model.language_model.layers.0.mlp.gate.weight"), Kind::Raw);
        assert_eq!(c("lm_head.weight"), Kind::Mls8);
        assert_eq!(c("model.visual.patch_embed.proj.weight"), Kind::Raw);
        let keep = Options {
            keep_int8_layers: vec![3],
            ..opts()
        };
        assert_eq!(
            Family::Qwen35.classify(
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                &[1, 1, 128],
                &keep,
                None,
                None
            ),
            Kind::Mls8
        );
    }

    /// Configuration B of docs/model-ling3-flash.md: 3 bit for the routed gate/up except the last 8 MoE layers.
    #[test]
    fn mls3_layer_policy() {
        let cfg = serde_json::json!({
            "first_k_dense_replace": 2,
            "num_hidden_layers": 42,
            "num_nextn_predict_layers": 1,
        });
        assert_eq!(Family::Ling3.default_mls3(&cfg), Some((2, 34)));
        assert_eq!(Family::Qwen35.default_mls3(&cfg), None);
        assert_eq!(Family::Qwen4Exp.default_mls3(&cfg), None);
        // A model with fewer layers than the promoted tail keeps everything at 4 bit instead of underflowing.
        let small = serde_json::json!({"first_k_dense_replace": 2, "num_hidden_layers": 6});
        assert_eq!(Family::Ling3.default_mls3(&small), None);

        let o = opts();
        let m3 = Some((2, 34));
        let c = |name: &str, shape: &[usize]| Family::Ling3.classify(name, shape, &o, m3, None);
        let gu = &[512, 1536, 2560][..];
        let dn = &[512, 2560, 768][..];
        assert_eq!(c("model.layers.2.mlp.experts.gate_up_proj", gu), Kind::Mls3);
        assert_eq!(c("model.layers.34.mlp.experts.gate_up_proj", gu), Kind::Mls3);
        // The promoted tail, the MTP layer, `down` and the shared expert stay at 4 bit.
        assert_eq!(c("model.layers.35.mlp.experts.gate_up_proj", gu), Kind::Mls4);
        assert_eq!(c("model.layers.42.mlp.experts.gate_up_proj", gu), Kind::Mls4);
        assert_eq!(c("model.layers.2.mlp.experts.down_proj", dn), Kind::Mls4);
        assert_eq!(
            c("model.layers.2.mlp.shared_experts.gate_proj.weight", &[768, 2560]),
            Kind::Mls4
        );
        assert_eq!(c("model.layers.2.attention.q_proj.weight", &[4096, 2560]), Kind::Mls4);
        assert_eq!(c("model.layers.0.mlp.gate_proj.weight", &[6144, 2560]), Kind::Mls4);
        // An explicit empty range turns MLS3 off everywhere.
        let none = |name: &str| Family::Ling3.classify(name, gu, &o, Some((1, 0)), None);
        assert_eq!(none("model.layers.2.mlp.experts.gate_up_proj"), Kind::Mls4);

        // 48 bytes of codes + 2 of scale per group of 128 = 3.125 bpw.
        let n = 512 * 1536 * 2560;
        assert_eq!(DType::Mls3.size_of_elems(n), n / 8 * 3);
        assert_eq!(
            DType::Mls3.size_of_elems(n) + DType::F16.size_of_elems(n / BLOCK),
            n * 25 / 64
        );
    }

    #[test]
    fn mls2_covers_both_routed_projections_and_nothing_else() {
        let cfg = serde_json::json!({"text_config": {"num_hidden_layers": 48}});
        assert_eq!(Family::Qwen4Exp.default_mls2(&cfg), Some((0, 47)));
        assert_eq!(Family::Qwen35.default_mls2(&cfg), None);
        assert_eq!(Family::Ling3.default_mls2(&cfg), None);

        let o = opts();
        let m2 = Some((0, 47));
        let c = |name: &str, shape: &[usize]| Family::Qwen4Exp.classify(name, shape, &o, None, m2);
        let gu = &[512, 1280, 2560][..];
        let dn = &[512, 2560, 640][..];
        assert_eq!(
            c("model.language_model.layers.0.mlp.experts.gate_up_proj", gu),
            Kind::Mls2
        );
        // The default keeps `down` at 4 bit: it is a third of the routed parameters and the half of the matrix the
        // 2-bit codes hurt most in the golden.
        assert_eq!(
            c("model.language_model.layers.47.mlp.experts.down_proj", dn),
            Kind::Mls4
        );
        assert_eq!(
            c(
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                &[640, 2560]
            ),
            Kind::Mls2
        );
        assert_eq!(
            c(
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                &[2560, 640]
            ),
            Kind::Mls4
        );
        assert_eq!(
            c(
                "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                &[1, 2560]
            ),
            Kind::Raw
        );
        // With `--mls2-down` the whole routed expert follows.
        let both = Options {
            mls2_down: true,
            ..opts()
        };
        let c2 = |name: &str, shape: &[usize]| Family::Qwen4Exp.classify(name, shape, &both, None, m2);
        assert_eq!(
            c2("model.language_model.layers.47.mlp.experts.down_proj", dn),
            Kind::Mls2
        );
        assert_eq!(
            c2(
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                &[2560, 640]
            ),
            Kind::Mls2
        );
        assert_eq!(
            c("model.language_model.layers.0.self_attn.q_proj.weight", &[4096, 2560]),
            Kind::Mls4
        );
        assert_eq!(c("mtp.layers.0.mlp.experts.gate_up_proj", gu), Kind::Mls4);
        assert_eq!(c("mtp.layers.0.mlp.experts.down_proj", dn), Kind::Mls4);
        // An explicit range is inclusive, and an empty one turns MLS2 off.
        let one = |name: &str| Family::Qwen4Exp.classify(name, gu, &o, None, Some((3, 3)));
        assert_eq!(
            one("model.language_model.layers.3.mlp.experts.gate_up_proj"),
            Kind::Mls2
        );
        assert_eq!(
            one("model.language_model.layers.4.mlp.experts.gate_up_proj"),
            Kind::Mls4
        );
        let none = |name: &str| Family::Qwen4Exp.classify(name, gu, &o, None, Some((1, 0)));
        assert_eq!(
            none("model.language_model.layers.3.mlp.experts.gate_up_proj"),
            Kind::Mls4
        );

        // 32 bytes of codes + 2 of scale per group of 128 = 2.125 bpw.
        let n = 512 * 1280 * 2560;
        assert_eq!(DType::Mls2.size_of_elems(n), n / 4);
        assert_eq!(
            DType::Mls2.size_of_elems(n) + DType::F16.size_of_elems(n / BLOCK),
            n * 17 / 64
        );
    }

    #[test]
    fn expert_tensor_detection() {
        let prefix = |s: &str| Some(s.to_string());
        assert_eq!(
            expert_layer("model.layers.41.mlp.experts.511.down_proj.weight"),
            prefix("model.layers.41")
        );
        assert_eq!(
            expert_layer("model.layers.0.mlp.experts.0.gate_proj.weight"),
            prefix("model.layers.0")
        );
        // Ornith stores the MTP experts one tensor at a time: same fusion, different prefix.
        assert_eq!(
            expert_layer("mtp.layers.0.mlp.experts.255.up_proj.weight"),
            prefix("mtp.layers.0")
        );
        assert_eq!(expert_layer("model.layers.5.mlp.experts.gate_up_proj"), None);
        assert_eq!(expert_layer("model.layers.5.mlp.shared_experts.gate_proj.weight"), None);
        assert_eq!(expert_layer("model.layers.5.mlp.gate.weight"), None);
        assert_eq!(
            expert_layer("model.language_model.layers.5.mlp.experts.3.gate_proj.weight"),
            None
        );
        let g = MoeGeom {
            experts: 512,
            inter: 768,
            hidden: 2560,
        };
        assert_eq!(fused_shape(g, true), vec![512, 1536, 2560]);
        assert_eq!(fused_shape(g, false), vec![512, 2560, 768]);
        let gu = expert_parts(g, "model.layers.2", true);
        assert_eq!(gu.len(), 1024);
        assert_eq!(gu.iter().map(|p| p.rows).sum::<usize>(), 512 * 1536);
        assert_eq!(gu[0].src, "model.layers.2.mlp.experts.0.gate_proj.weight");
        assert_eq!(
            (gu[1].src.as_str(), gu[1].row0),
            ("model.layers.2.mlp.experts.0.up_proj.weight", 768)
        );
        assert_eq!(
            (gu[2].src.as_str(), gu[2].row0),
            ("model.layers.2.mlp.experts.1.gate_proj.weight", 1536)
        );
        let d = expert_parts(g, "model.layers.2", false);
        assert_eq!(d.len(), 512);
        assert_eq!((d[1].row0, d[1].rows, d[1].k), (2560, 2560, 768));
    }
}
