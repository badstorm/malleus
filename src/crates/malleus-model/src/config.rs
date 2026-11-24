//! The checkpoint config: geometry and the family tag, parsed from `config.json`.

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;

use crate::ling3;
use crate::qwen4exp;
use crate::qwen35;

#[derive(Clone)]
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
    /// Hy4-flash geometry: iHC, the two attention shapes, the compressed indexer, the latent MoE and Engram.
    pub hy4: Option<Hy4Config>,
    /// Inkling-2 geometry: attention residuals, short convolutions, the relative-position bias, the loop, the
    /// vision tower and the MTP head.
    pub inkling2: Option<Inkling2Config>,
    /// DeepSeek-V4.1 geometry: window + compressed KV, the two-level indexer, hyper-connections, Engram
    /// memories, 384 experts, the DSpark draft stages and the ViT tower. Issue #83.
    pub deepseek: Option<DeepseekConfig>,
}

/// DeepSeek-V4.1 (`deepseek_v41`, issue #83): every layer attends over a sliding window of raw KV and, when
/// `compress_ratios[layer] > 0`, over the `index_topk` compressed positions the indexer picked for that query.
#[derive(Clone, Debug)]
pub struct DeepseekConfig {
    /// Query heads, their width, the rotated tail, and the low-rank input projection.
    pub heads: usize,
    pub head_dim: usize,
    pub nope: usize,
    pub rope: usize,
    pub q_lora: usize,
    /// The output projection is block-diagonal over `o_groups`, each group `o_lora` wide.
    pub o_groups: usize,
    pub o_lora: usize,
    /// Raw keys a query always sees.
    pub window: usize,
    /// Per layer: 0 = sliding window only, `r` = one compressed latent per `r` tokens.
    pub compress_ratios: Vec<u8>,
    /// Layers that own the compressed KV the layers sharing their ratio read, and the ones that own an indexer.
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index: DeepseekIndex,
    /// The first level of the two-level top-k, held by one layer.
    pub candidate: Option<DeepseekCandidate>,
    /// Routed experts, the ones a token takes, their width, the SwiGLU clamp and the router scaling.
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    /// The router score is `sqrt(softplus(x))` when true, `sigmoid(x)` otherwise; the bias steers the selection.
    pub sqrt_softplus: bool,
    pub norm_topk_prob: bool,
    /// Hyper-connections: parallel residual streams, Sinkhorn iterations and epsilon.
    pub hc_mult: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    pub rope_params: DeepseekRope,
    pub engram: Option<DeepseekEngram>,
    pub vision: Option<DeepseekVision>,
    pub dspark: Option<DeepseekDspark>,
    /// Token id filling every position of an image span (0 without vision).
    pub image_token: u32,
    pub eps: f32,
}

/// The indexer: scoring heads of `head_dim` against one shared key per compressed position.
#[derive(Clone, Debug)]
pub struct DeepseekIndex {
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
}

/// Level one: keep the best `topk_blocks` blocks of `block_size` compressed positions, so level two only has to
/// score inside them.
#[derive(Clone, Debug)]
pub struct DeepseekCandidate {
    pub source_layer: usize,
    pub topk_blocks: usize,
    pub block_size: usize,
}

/// RoPE: the base of the window layers, the base of the compressed latent, and the YaRN blend that carries the
/// 65 536 training window out to 1M.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekRope {
    pub theta: f32,
    pub compress_theta: f32,
    pub original_seq: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// Engram: an n-gram hash table per listed layer, `max_ngram - 1` orders of `heads` rows added into the streams.
#[derive(Clone, Debug)]
pub struct DeepseekEngram {
    /// 0-based layer indices carrying a table, in the order of `engram_num_embeddings`.
    pub layers: Vec<usize>,
    /// Rows of each layer's table.
    pub num_embeddings: Vec<usize>,
    pub max_ngram: usize,
    /// Bucket size each (n-gram order, head) starts searching primes from, and the compressed vocabulary the hash
    /// multipliers derive from.
    pub vocab: usize,
    pub compressed_vocab: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pad: u32,
}

/// The ViT: `downsample_ratio` squared patches merged into one text position by the aligner.
#[derive(Clone, Debug)]
pub struct DeepseekVision {
    pub layers: usize,
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub patch: usize,
    pub downsample: usize,
    pub rope_theta: f32,
    pub max_tokens: usize,
    pub min_pixels: usize,
}

/// DSpark: `stages` draft blocks of `block` positions, each with its own 128-expert MoE, plus the markov and
/// confidence heads of the last stage.
#[derive(Clone, Debug)]
pub struct DeepseekDspark {
    pub stages: usize,
    pub block: usize,
    pub noise_token: u32,
    /// Backbone layers whose attention input feeds the drafter.
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub experts: usize,
    pub top_k: usize,
}

/// Inkling-2 (`inkling2_loop`, issue #62): 24 layers of which the middle 12 run `passes` times, one attention
/// residual per block of `attn_res_block` layers, a short convolution on every layer input and output, a
/// relative-position bias in the full layers, 128 experts per layer, a vision tower and a loop-aware MTP head.
#[derive(Clone, Debug)]
pub struct Inkling2Config {
    /// Query heads, their width, and the shared key/value heads of both attention kinds.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Backward distance the relative-position bias of the full layers covers, and the width of the per-head
    /// relative state mixed into it.
    pub rel_extent: usize,
    pub d_rel: usize,
    /// Position from which queries and relative states are scaled up logarithmically, and how strongly.
    pub log_floor: usize,
    pub log_alpha: f32,
    /// Largest L2 norm of a query/key head vector and of a relative state; longer ones are scaled down.
    pub qk_clip: f32,
    pub rel_clip: f32,
    /// Layers summed into one attention-residual block (`n_layers_per_attn_res_block`).
    pub attn_res_block: usize,
    /// Short-convolution kernel (`sconv_kernel_size`) and the window of the sliding layers.
    pub sconv_k: usize,
    pub window: usize,
    /// Routed experts per layer, how many a token takes, their width, and the router scaling.
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    /// Softcaps of the experts' gate and up projections, and of the output logits.
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub logit_softcap: f32,
    /// Rows of the unembedding that hold real tokens (`unpadded_vocab_size`).
    pub unpadded_vocab: usize,
    /// The loop: the inclusive layer range it repeats, how many passes, and the rank of its gates.
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    pub loop_rank: usize,
    /// The vision tower, when the checkpoint carries one.
    pub vision: Option<Inkling2Vision>,
    /// The MTP head (`mtp_config`), when the checkpoint carries one.
    pub mtp: Option<Inkling2Mtp>,
}

/// Inkling-2's vision tower (`vision_config`): a 23-layer ViT with axial RoPE, then a two-layer merger.
#[derive(Clone, Debug)]
pub struct Inkling2Vision {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub layers: usize,
    /// Side of a native patch, the temporal patch size, and the side of the square merged into one token.
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    /// Side of the learned position grid.
    pub grid: usize,
    /// Width of the projector between the tower and the language model.
    pub proj_hidden: usize,
    pub eps: f32,
    /// Token id of an image placeholder in the prompt.
    pub image_token: u32,
}

/// The MTP head (`mtp_config`): 3 layers that propose a block of tokens, with a Markov correction on the logits.
/// Its attention is the full/sliding pair of the text stack at its own head geometry (24 query heads here).
#[derive(Clone, Debug)]
pub struct Inkling2Mtp {
    pub layers: usize,
    pub inter: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub window: usize,
    /// Layers of the head that slide; the rest are full attention.
    pub local: Vec<usize>,
    /// Tokens per proposed block (`block_size`).
    pub block: usize,
    /// Width of the Markov embedding mixed into the logits (`markov_rank`).
    pub markov_rank: usize,
    /// Layer applications whose hidden states the head reads.
    pub target_layers: Vec<usize>,
}

/// Hy4-flash (`hy_v4_flash`, issue #40). The two attention shapes and the two RoPE bases are per layer type, so
/// they live here rather than in `Config`; `Config` keeps only what the shared graph reads.
#[derive(Clone, Debug)]
pub struct Hy4Config {
    /// Streams (`hc_mult`) and the magnitude of the iHC post-gate.
    pub hc_mult: usize,
    pub hc_magnitude: f32,
    pub hc_eps: f32,
    /// `gated_mla`: elementwise output gate, and the learned sink logit per head.
    pub sink: bool,
    pub q_lora: usize,
    pub kv_lora: usize,
    /// Global (indexed) layers: heads, and the 192 nope / 64 rope / 256 value split.
    pub heads: usize,
    pub nope: usize,
    pub rope: usize,
    pub v: usize,
    /// Local (sliding) layers: their own width, and how many keys they see.
    pub swa_heads: usize,
    pub swa_nope: usize,
    pub swa_rope: usize,
    pub swa_v: usize,
    pub window: usize,
    /// RoPE base per layer type: the global one is the long-context base.
    pub theta_indexed: f32,
    pub theta_sliding: f32,
    pub mrope_section: [usize; 3],
    /// Compressed indexer: tokens per pooled group, the recall budget, and the scoring heads and width.
    pub index_pool: usize,
    pub index_topk: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    /// Width the routed experts run in (`routed_expert_hidden_size`), and the SwiGLU clamp of their activation.
    pub latent: usize,
    /// Routed experts per layer, the experts a token takes, and their intermediate width.
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub swiglu_limit: f32,
    pub routed_scaling: f32,
    /// Per-layer FFN kind (`mlp_layer_types`): true where the FFN is a latent MoE, false for the dense layer.
    pub sparse: Vec<bool>,
    /// Engram memories, when the checkpoint carries them.
    pub engram: Option<Hy4Engram>,
    /// The vision tower, when the checkpoint carries one.
    pub vision: Option<Hy4Vision>,
    /// Token id of an image placeholder in the prompt; 0 when the checkpoint has no vision.
    pub image_token: u32,
    /// The DFly2 speculative drafter the checkpoint carries (`model.dfly_layers.*`), if any.
    pub dfly: Option<Hy4DFly>,
}

/// DFly2: `layers` MLA + SwiGLU blocks over a block of `mask`-filled slots, layer i reading the KV cache and the
/// indexer selection of target layer `kv_layers[i]`.
#[derive(Debug, Clone)]
pub struct Hy4DFly {
    pub layers: usize,
    pub mask: u32,
    pub kv_layers: Vec<usize>,
    pub index_layers: Vec<usize>,
}

/// The vision tower's geometry (`vision_config`): HunYuan's ViT plus Hy4's patch merger.
#[derive(Clone, Debug)]
pub struct Hy4Vision {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub patch: usize,
    /// Edge of the learned position grid (`max_image_size / patch_size`).
    pub edge: usize,
    pub merge: usize,
    /// Width the merger projects to: the text model's hidden size.
    pub out: usize,
}

/// The hashed n-gram memory of Hy4: which layers carry it and the shape of the hashes.
#[derive(Clone, Debug)]
pub struct Hy4Engram {
    /// 0-based layer indices.
    pub layers: Vec<usize>,
    /// Hash heads per n-gram order (2..=max_ngram), the width of each row, and the head width.
    pub heads: usize,
    pub embed: usize,
    pub max_ngram: usize,
    pub pad: u32,
    /// Size of the frozen normalized-vocabulary map the hashes go through.
    pub lookup: usize,
    pub seed: u64,
    /// fp8 (e4m3) round trip of the looked-up rows, as the memory was trained.
    pub fp8: bool,
    /// Per layer (same order as `layers`), the hash-head vocabulary sizes by n-gram order.
    pub head_vocab_sizes: Vec<Vec<Vec<usize>>>,
}

/// The graphs Malleus knows. `Ling3` is the one with different mixers (KDA and MLA), see ADR 0005.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Qwen35,
    Qwen4Exp,
    Ling3,
    /// Hy4-flash (`hy_v4_flash`): iHC streams, gated MLA with a learned sink, hybrid indexed/sliding attention,
    /// latent MoE and the Engram n-gram memories. Issue #40.
    Hy4,
    /// Inkling-2 (`inkling2_loop`): attention residuals, short convolutions, a relative-position bias, a 12-layer
    /// loop with cross-pass messages, 128 experts, a vision tower and an MTP head. Issue #62.
    Inkling2,
    /// DeepSeek-V4.1 (`deepseek_v41`): window + compressed KV, a two-level indexer, hyper-connections, Engram
    /// memories, 384 experts, DSpark draft stages and a ViT tower. Issue #83.
    Deepseek41,
}

impl Family {
    pub fn from_model_type(t: &str) -> Option<Family> {
        Some(match t {
            "qwen3_5" | "qwen3_5_moe" => Family::Qwen35,
            "qwen4_exp" => Family::Qwen4Exp,
            "bailing_hybrid" => Family::Ling3,
            "hy_v4_flash" => Family::Hy4,
            "inkling2" | "inkling2_loop" | "inkling2_text" | "inkling2_loop_text" => Family::Inkling2,
            "deepseek_v41" => Family::Deepseek41,
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
    /// Hy4's global layer: gated MLA plus the compressed indexer that recalls groups of `index_pool` tokens.
    Hy4Indexed,
    /// Hy4's local layer: gated MLA over the last `window` tokens, with its own (wider) head geometry.
    Hy4Sliding,
    /// Inkling-2's global layer: full attention with a relative-position bias, a learned sink and an output gate,
    /// instead of the sliding window of its other layers.
    AttnFull,
    /// DeepSeek-V4.1: the sliding window plus, on a layer whose `compress_ratios` entry is non-zero, the compressed
    /// positions its indexer selected.
    Deepseek,
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
    /// The rows of the unembedding that are real tokens. Some checkpoints pad it (Inkling-2: 201 024 rows against
    /// 200 058 tokens) and the engine runs the head on every row, so the rows past this one are not tokens: anything
    /// that samples has to ignore them.
    pub fn real_vocab(&self) -> usize {
        self.inkling2.as_ref().map_or(self.vocab, |i| i.unpadded_vocab)
    }

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
        if family == Family::Hy4 {
            return crate::hy4::config(&cfg, generation_config);
        }
        if family == Family::Inkling2 {
            return crate::inkling2::config(&cfg, generation_config);
        }
        if family == Family::Deepseek41 {
            return crate::deepseek::config(&cfg, generation_config);
        }
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
        let (mrope_section, mrope) = match family {
            Family::Ling3 => ling3::rope(),
            _ => qwen35::rope(t)?,
        };
        // Ling has no layer_types: the rule is in the code of the checkpoint, MLA every `layer_group_size`
        // (plus a final partial group, which 42 layers never produce) and KDA elsewhere.
        let n_layers = u("num_hidden_layers")?;
        let kind = match family {
            Family::Ling3 => ling3::mixer_kinds(n_layers, t),
            _ => qwen35::mixer_kinds(t)?,
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
        let hc = qwen4exp::hc(t, &u)?;
        let ple = qwen4exp::ple(t, &u)?;
        let indexer = qwen4exp::indexer(t, &u)?;
        let mla = match family {
            Family::Ling3 => Some(ling3::mla(&u)?),
            _ => None,
        };
        let swiglu = match family {
            Family::Ling3 => ling3::swiglu(t),
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
            hy4: None,
            inkling2: None,
            deepseek: None,
        };
        c.validate()?;
        Ok(c)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.full_attn.len() == self.n_layers && self.kind.len() == self.n_layers,
            "layer types length mismatch"
        );
        if self.family == Family::Hy4 {
            return crate::hy4::validate(self);
        }
        if self.family == Family::Inkling2 {
            return crate::inkling2::validate(self);
        }
        if self.family == Family::Deepseek41 {
            return crate::deepseek::validate(self);
        }
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
            Family::Ling3 => ling3::validate(self)?,
            _ => qwen35::validate(self)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometry of `inclusionAI/Ling-3.0-flash`, keys and values taken from its `config.json`.
    pub(crate) const LING3: &str = r#"{"model_type":"bailing_hybrid","hidden_size":2560,"num_hidden_layers":42,"head_dim":128,"num_attention_heads":32,"num_key_value_heads":32,"intermediate_size":6144,"moe_intermediate_size":768,"moe_shared_expert_intermediate_size":768,"num_experts":512,"num_experts_per_tok":8,"n_group":8,"topk_group":4,"score_function":"sigmoid","routed_scaling_factor":2.5,"moe_router_enable_expert_bias":true,"first_k_dense_replace":2,"layer_group_size":6,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"q_lora_rank":null,"short_conv_kernel_size":4,"kda_lower_bound":-5.0,"rms_norm_eps":1e-06,"rope_theta":6000000,"partial_rotary_factor":0.5,"rope_interleave":true,"num_nextn_predict_layers":1,"vocab_size":157184,"eos_token_id":156895,"expert_swiglu_limit_list":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,4],"share_expert_swiglu_limit_list":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,7,7]}"#;

    /// Geometry of `Qwen/Qwen3.6-35B-A3B`, the MoE model in production on gfx1150.
    pub(crate) const QWEN35_MOE: &str = r#"{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":2048,"num_hidden_layers":40,"head_dim":256,"num_attention_heads":16,"num_key_value_heads":2,"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"rms_norm_eps":1e-06,"vocab_size":248320,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}}"#;

    #[test]
    pub(crate) fn ling3_geometry() {
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
    pub(crate) fn qwen35_moe_geometry_is_unchanged() {
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
    pub(crate) fn unknown_family_is_rejected() {
        assert!(Family::from_model_type("llama").is_none());
        assert!(Config::from_json(r#"{"model_type":"llama"}"#, None).is_err());
    }
}
