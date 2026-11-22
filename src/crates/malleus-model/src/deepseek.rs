//! DeepSeek-V4.1 (`deepseek_v41`, issue #83): the config, the map of the checkpoint's tensors and the inventory
//! `malleus inspect` prints.
//!
//! The graph is not the Qwen/Ling one. Every layer attends over a sliding window of `window` raw latents and, when
//! its `compress_ratios` entry is non-zero, over the `index_topk` compressed positions a small side attention picked
//! for that query. The compressed KV is produced by four *source* layers and read by the layers sharing their ratio;
//! the indexer has its own source layers, and one of them pre-filters with candidate blocks. The residual stream is
//! `hc_mult` parallel copies mixed by Sinkhorn-normalised hyper-connections, two layers read n-gram hash tables, and
//! three DSpark stages draft a block of tokens.
//!
//! This module holds the geometry and the map; the loader and the forward are the next increments.

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use serde_json::Value;

use crate::config::{
    Config, DeepseekCandidate, DeepseekConfig, DeepseekDspark, DeepseekEngram, DeepseekIndex, DeepseekRope,
    DeepseekVision, Family, MixerKind, MoeConfig,
};

/// The fp8 weights are quantized in 32x32 blocks with one e8m0 scale each.
const FP8_BLOCK: usize = 32;
/// The routed experts are e2m1 pairs in a byte, one e8m0 scale per 32 values along K.
const FP4_BLOCK: usize = 32;

/// The `text_config` of a DeepSeek-V4.1 checkpoint, as `Config` plus the family geometry.
pub(crate) fn config(cfg: &Value, generation_config: Option<&str>) -> Result<Config> {
    let t = &cfg["text_config"];
    let u = |k: &str| -> Result<usize> {
        t[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: {k}"))
    };
    let f = |k: &str| -> Result<f32> { t[k].as_f64().map(|v| v as f32).with_context(|| format!("config: {k}")) };
    let list = |k: &str| -> Result<Vec<usize>> {
        t[k].as_array()
            .with_context(|| format!("config: {k}"))?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize).with_context(|| format!("config: {k}")))
            .collect()
    };

    let n_layers = u("num_hidden_layers")?;
    let heads = u("num_attention_heads")?;
    let head_dim = u("head_dim")?;
    let rope = u("qk_rope_head_dim")?;
    ensure!(rope > 0 && rope < head_dim, "config: qk_rope_head_dim");

    // One entry per layer, MTP layers included in the checkpoint's own list.
    let mut compress_ratios: Vec<u8> = t["compress_ratios"]
        .as_array()
        .context("config: compress_ratios")?
        .iter()
        .map(|v| {
            let r = v.as_u64().context("config: compress_ratios")?;
            ensure!(r <= u8::MAX as u64, "config: compress ratio out of range");
            Ok(r as u8)
        })
        .collect::<Result<_>>()?;
    ensure!(
        compress_ratios.len() >= n_layers,
        "config: compress_ratios shorter than the stack"
    );
    compress_ratios.truncate(n_layers);

    let kv_source_layers = list("kv_source_layer_ids")?;
    let index_source_layers = list("index_source_layer_ids")?;
    let candidate = match t["candidate_source_layer_id"].as_i64() {
        Some(v) if v >= 0 => Some(DeepseekCandidate {
            source_layer: v as usize,
            topk_blocks: u("candidate_topk_blocks")?,
            block_size: u("candidate_block_size")?,
        }),
        _ => None,
    };

    let rope_scaling = &t["rope_scaling"];
    let rope_params = DeepseekRope {
        theta: f("rope_theta")?,
        compress_theta: t["compress_rope_theta"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(f("rope_theta")?),
        original_seq: rope_scaling["original_max_position_embeddings"].as_u64().unwrap_or(0) as usize,
        factor: rope_scaling["factor"].as_f64().unwrap_or(1.0) as f32,
        beta_fast: rope_scaling["beta_fast"].as_f64().unwrap_or(32.0) as f32,
        beta_slow: rope_scaling["beta_slow"].as_f64().unwrap_or(1.0) as f32,
    };

    let engram = {
        let layers = list("engram_layer_ids")?;
        if layers.is_empty() {
            None
        } else {
            let num_embeddings = list("engram_num_embeddings")?;
            ensure!(
                num_embeddings.len() == layers.len(),
                "config: engram_num_embeddings for {} layers",
                layers.len()
            );
            Some(DeepseekEngram {
                layers,
                num_embeddings,
                max_ngram: u("engram_max_ngram_size")?,
                vocab: u("engram_vocab_size")?,
                compressed_vocab: u("engram_compressed_vocab_size")?,
                n_heads: u("engram_n_heads")?,
                head_dim: u("engram_head_dim")?,
                pad: u("engram_pad_token_id")? as u32,
            })
        }
    };

    let vision = match cfg["vision_config"]["hidden_size"].as_u64() {
        Some(hidden) => {
            let v = &cfg["vision_config"];
            let vu = |k: &str| -> Result<usize> {
                v[k].as_u64()
                    .map(|x| x as usize)
                    .with_context(|| format!("config: vision_config.{k}"))
            };
            Some(DeepseekVision {
                layers: vu("num_hidden_layers")?,
                hidden: hidden as usize,
                inter: vu("intermediate_size")?,
                heads: vu("num_attention_heads")?,
                patch: vu("patch_size")?,
                downsample: vu("downsample_ratio")?,
                rope_theta: v["rope_theta"].as_f64().unwrap_or(1e4) as f32,
                max_tokens: v["max_image_tokens"].as_u64().unwrap_or(0) as usize,
                min_pixels: v["min_pixels"].as_u64().unwrap_or(0) as usize,
            })
        }
        None => None,
    };

    let dspark = match t["num_nextn_predict_layers"].as_u64() {
        Some(0) | None => None,
        Some(stages) => Some(DeepseekDspark {
            stages: stages as usize,
            block: u("dspark_block_size")?,
            noise_token: u("dspark_noise_token_id")? as u32,
            target_layers: list("dspark_target_layer_ids")?,
            markov_rank: u("dspark_markov_rank")?,
            experts: u("dspark_n_routed_experts")?,
            top_k: u("dspark_num_experts_per_tok")?,
        }),
    };

    let mut eos = vec![];
    if let Some(g) = generation_config {
        let g: Value = serde_json::from_str(g)?;
        match &g["eos_token_id"] {
            Value::Array(a) => eos.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32)),
            Value::Number(n) => eos.extend(n.as_u64().map(|v| v as u32)),
            _ => {}
        }
    }
    if eos.is_empty()
        && let Some(v) = t["eos_token_id"].as_u64()
    {
        eos.push(v as u32);
    }

    let deepseek = DeepseekConfig {
        heads,
        head_dim,
        nope: head_dim - rope,
        rope,
        q_lora: u("q_lora_rank")?,
        o_groups: u("o_groups")?,
        o_lora: u("o_lora_rank")?,
        window: u("sliding_window")?,
        compress_ratios,
        kv_source_layers,
        index_source_layers,
        index: DeepseekIndex {
            heads: u("index_n_heads")?,
            dim: u("index_head_dim")?,
            topk: u("index_topk")?,
        },
        candidate,
        experts: u("n_routed_experts")?,
        top_k: u("num_experts_per_tok")?,
        moe_inter: u("moe_intermediate_size")?,
        swiglu_limit: f("swiglu_limit")?,
        route_scale: f("routed_scaling_factor")?,
        sqrt_softplus: t["scoring_func"].as_str() == Some("sqrtsoftplus"),
        norm_topk_prob: t["norm_topk_prob"].as_bool().unwrap_or(false),
        hc_mult: u("hc_mult")?,
        hc_iters: u("hc_sinkhorn_iters")?,
        hc_eps: f("hc_eps")?,
        rope_params,
        engram,
        vision,
        dspark,
        image_token: cfg["image_token_id"].as_u64().unwrap_or(0) as u32,
        eps: f("rms_norm_eps")?,
    };

    let kind = vec![MixerKind::Deepseek; n_layers];
    let c = Config {
        hidden: u("hidden_size")?,
        n_layers,
        // No dense FFN: every layer routes.
        inter: 0,
        vocab: u("vocab_size")?,
        n_heads: heads,
        // One KV head: every query head reads the same 512-wide latent.
        n_kv: u("num_key_value_heads")?,
        head_dim,
        lin_nk: 0,
        lin_nv: 0,
        lin_dk: 0,
        lin_dv: 0,
        conv_k: 0,
        eps: deepseek.eps,
        // The compressed latent rotates at the longer base; the window layers use `rope_theta`.
        rope_theta: deepseek.rope_params.compress_theta,
        rot_dim: rope,
        mrope_section: [0, 0, 0],
        full_attn: deepseek.compress_ratios.iter().map(|r| *r > 0).collect(),
        eos,
        moe: Some(MoeConfig {
            experts: deepseek.experts,
            top_k: deepseek.top_k,
            inter: deepseek.moe_inter,
            sigmoid: !deepseek.sqrt_softplus,
            n_group: 1,
            topk_group: 1,
            bias: true,
            scaling: deepseek.route_scale,
            shared_gate: false,
        }),
        hc: None,
        ple: None,
        indexer: None,
        gate_sigmoid: false,
        family: Family::Deepseek41,
        kind,
        mla: None,
        kda_lower_bound: 0.0,
        norm_one_plus_w: false,
        mrope: false,
        dense_ffn: 0,
        swiglu: None,
        hy4: None,
        inkling2: None,
        deepseek: Some(deepseek),
    };
    c.validate()?;
    Ok(c)
}

/// The geometry the kernels will require of a DeepSeek-V4.1 checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let d = c.deepseek.as_ref().context("no deepseek config")?;
    ensure!(c.family == Family::Deepseek41, "bad deepseek family");
    ensure!(
        c.kind.len() == c.n_layers && c.full_attn.len() == c.n_layers,
        "layer kind count"
    );
    ensure!(
        d.compress_ratios.len() == c.n_layers,
        "compress_ratios must cover every layer"
    );
    ensure!(d.heads > 0 && d.head_dim == d.nope + d.rope, "bad head geometry");
    ensure!(
        d.head_dim.is_multiple_of(128),
        "the attention kernels need a head width multiple of 128"
    );
    ensure!(
        d.o_groups > 0 && (d.heads * d.head_dim).is_multiple_of(d.o_groups),
        "bad o_groups"
    );
    ensure!(d.window > 0, "the sliding window must not be empty");
    ensure!(
        d.kv_source_layers.iter().all(|&i| i < c.n_layers) && d.index_source_layers.iter().all(|&i| i < c.n_layers),
        "a source layer is outside the stack"
    );
    ensure!(
        d.kv_source_layers.iter().all(|&i| d.compress_ratios[i] > 0),
        "a KV source layer must have a compress ratio"
    );
    ensure!(
        d.index_source_layers.iter().all(|&i| d.compress_ratios[i] > 0),
        "an index source layer must have a compress ratio"
    );
    if let Some(cand) = &d.candidate {
        ensure!(
            d.index_source_layers.contains(&cand.source_layer),
            "the candidate source is not an index source"
        );
        ensure!(cand.topk_blocks > 0 && cand.block_size > 0, "bad candidate blocks");
    }
    ensure!(d.index.topk > 0 && d.index.dim > 0 && d.index.heads > 0, "bad indexer");
    ensure!(
        d.experts > 0 && d.top_k > 0 && d.top_k <= d.experts,
        "bad expert routing"
    );
    ensure!(
        d.moe_inter.is_multiple_of(128) && c.hidden.is_multiple_of(128),
        "MoE widths"
    );
    ensure!(d.hc_mult >= 2 && d.hc_iters > 0, "bad hyper-connections");
    ensure!(d.rope_params.factor > 0.0, "bad rope factor");
    if let Some(e) = &d.engram {
        ensure!(
            e.layers.len() == e.num_embeddings.len() && e.max_ngram >= 2 && e.n_heads > 0,
            "bad engram layout"
        );
        ensure!(
            e.layers.iter().all(|&i| i < c.n_layers),
            "an engram layer is outside the stack"
        );
        ensure!(
            (e.pad as usize) < e.compressed_vocab,
            "the pad token is outside the compressed vocab"
        );
    }
    if let Some(v) = &d.vision {
        ensure!(
            v.layers > 0 && v.hidden > 0 && v.patch > 0 && v.downsample > 0,
            "bad vision tower"
        );
        ensure!(d.image_token > 0, "vision without an image token");
    }
    if let Some(s) = &d.dspark {
        ensure!(
            s.stages > 0 && s.block > 0 && !s.target_layers.is_empty(),
            "bad DSpark stages"
        );
        ensure!(
            s.target_layers.iter().all(|&i| i < c.n_layers),
            "a DSpark target layer is outside the stack"
        );
        ensure!(
            s.markov_rank > 0 && s.experts > 0 && s.top_k <= s.experts,
            "bad DSpark head"
        );
    }
    Ok(())
}

/// One line per fact `malleus inspect` reports about the geometry.
pub fn describe(c: &Config) -> Result<Vec<String>> {
    let d = c.deepseek.as_ref().context("no deepseek config")?;
    let mut out = vec![format!(
        "{} layers, hidden {} ({})",
        c.n_layers,
        c.hidden,
        if c.hidden == 5120 { "5120" } else { "see config" }
    )];
    out.push(format!(
        "  attention: {} heads x {} ({} nope + {} rope), q_lora {}, o_groups {} x {}",
        d.heads, d.head_dim, d.nope, d.rope, d.q_lora, d.o_groups, d.o_lora
    ));
    let mut ratios: Vec<(u8, usize)> = Vec::new();
    for r in &d.compress_ratios {
        match ratios.iter_mut().find(|(x, _)| x == r) {
            Some((_, n)) => *n += 1,
            None => ratios.push((*r, 1)),
        }
    }
    ratios.sort();
    let ratios: Vec<String> = ratios.iter().map(|(r, n)| format!("{n} layers at ratio {r}")).collect();
    out.push(format!(
        "  window {}, {}; kv sources {}, index sources {}, index {}x{} top {}",
        d.window,
        ratios.join(", "),
        d.kv_source_layers.len(),
        d.index_source_layers.len(),
        d.index.heads,
        d.index.dim,
        d.index.topk
    ));
    if let Some(cand) = &d.candidate {
        out.push(format!(
            "  candidates: layer {} keeps {} blocks of {}",
            cand.source_layer, cand.topk_blocks, cand.block_size
        ));
    }
    out.push(format!(
        "  moe: {} experts, {} active, width {}, swiglu limit {}, route scale {}",
        d.experts, d.top_k, d.moe_inter, d.swiglu_limit, d.route_scale
    ));
    out.push(format!(
        "  hyper-connections: {} streams, {} Sinkhorn iterations",
        d.hc_mult, d.hc_iters
    ));
    out.push(format!(
        "  rope: theta {}, compressed {}, YaRN x{} from {} (beta {}..{})",
        d.rope_params.theta,
        d.rope_params.compress_theta,
        d.rope_params.factor,
        d.rope_params.original_seq,
        d.rope_params.beta_slow,
        d.rope_params.beta_fast
    ));
    if let Some(e) = &d.engram {
        out.push(format!(
            "  engram: layers {:?}, {} orders x {} heads of {}, {} rows total",
            e.layers,
            e.max_ngram - 1,
            e.n_heads,
            e.head_dim,
            e.num_embeddings.iter().sum::<usize>()
        ));
    }
    if let Some(v) = &d.vision {
        out.push(format!(
            "  vision: {} blocks, hidden {} ({} inter, {} heads), patch {}, down {}x{}, max {} tokens",
            v.layers, v.hidden, v.inter, v.heads, v.patch, v.downsample, v.downsample, v.max_tokens
        ));
    }
    if let Some(s) = &d.dspark {
        out.push(format!(
            "  dspark: {} stages, block {}, {} experts ({} active), markov rank {}",
            s.stages, s.block, s.experts, s.top_k, s.markov_rank
        ));
    }
    Ok(out)
}

/// One tensor the checkpoint must carry.
pub struct Expect {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// What it belongs to, for the byte inventory.
    pub class: &'static str,
}

fn ceil32(n: usize) -> usize {
    n.div_ceil(FP8_BLOCK)
}

/// An fp8 linear: the e4m3 weight and its e8m0 scale, one per 32x32 block.
fn fp8(out: &mut Vec<Expect>, name: String, n: usize, k: usize, class: &'static str) {
    out.push(Expect {
        name: name.clone(),
        dtype: DType::F8E4M3,
        shape: vec![n, k],
        class,
    });
    out.push(Expect {
        name: scale_name(&name),
        dtype: DType::F8E8M0,
        shape: vec![ceil32(n), ceil32(k)],
        class: "scales",
    });
}

/// The block scale sits beside the weight, not inside it: `....wq_a.scale` next to `....wq_a.weight`.
fn scale_name(weight: &str) -> String {
    format!("{}.scale", weight.strip_suffix(".weight").unwrap_or(weight))
}

/// A routed expert projection: e2m1 pairs in a byte, one e8m0 scale per 32 values along K.
fn fp4(out: &mut Vec<Expect>, name: String, n: usize, k: usize, class: &'static str) -> Result<()> {
    ensure!(
        k.is_multiple_of(FP4_BLOCK * 2),
        "an fp4 projection needs a K multiple of 64"
    );
    out.push(Expect {
        name: name.clone(),
        dtype: DType::I8,
        shape: vec![n, k / 2],
        class,
    });
    out.push(Expect {
        name: scale_name(&name),
        dtype: DType::F8E8M0,
        shape: vec![n, k / FP4_BLOCK],
        class: "scales",
    });
    Ok(())
}

fn raw(out: &mut Vec<Expect>, name: String, dtype: DType, shape: Vec<usize>, class: &'static str) {
    out.push(Expect {
        name,
        dtype,
        shape,
        class,
    });
}

/// The tensor map of a DeepSeek-V4.1 checkpoint: every name, shape and storage dtype the forward will read.
pub fn expected(c: &Config) -> Result<Vec<Expect>> {
    let d = c.deepseek.as_ref().context("no deepseek config")?;
    let dim = c.hidden;
    let hc_mix = (2 + d.hc_mult) * d.hc_mult;
    let mut out = Vec::new();

    raw(
        &mut out,
        "embed.weight".into(),
        DType::BF16,
        vec![c.vocab, dim],
        "embed+head",
    );
    raw(
        &mut out,
        "head.weight".into(),
        DType::BF16,
        vec![c.vocab, dim],
        "embed+head",
    );
    raw(&mut out, "norm.weight".into(), DType::BF16, vec![dim], "attention");

    // The Engram tables are indexed by their position in `engram_layer_ids`, not by the layer.
    let engram_of =
        |i: usize| -> Option<usize> { d.engram.as_ref().and_then(|e| e.layers.iter().position(|&l| l == i)) };

    for i in 0..c.n_layers {
        let p = |s: &str| format!("layers.{i}.{s}");
        raw(&mut out, p("attn_norm.weight"), DType::BF16, vec![dim], "attention");
        raw(&mut out, p("ffn_norm.weight"), DType::BF16, vec![dim], "attention");
        for which in ["hc_attn", "hc_ffn"] {
            raw(
                &mut out,
                p(&format!("{which}_fn")),
                DType::F32,
                vec![hc_mix, d.hc_mult * dim],
                "attention",
            );
            raw(
                &mut out,
                p(&format!("{which}_base")),
                DType::F32,
                vec![hc_mix],
                "attention",
            );
            raw(&mut out, p(&format!("{which}_scale")), DType::F32, vec![3], "attention");
        }

        // Attention: one low-rank query, a direct 512-wide key/value latent, and a grouped output.
        fp8(&mut out, p("attn.wq_a.weight"), d.q_lora, dim, "attention");
        fp8(
            &mut out,
            p("attn.wq_b.weight"),
            d.heads * d.head_dim,
            d.q_lora,
            "attention",
        );
        fp8(&mut out, p("attn.wkv.weight"), d.head_dim, dim, "attention");
        fp8(
            &mut out,
            p("attn.wo_a.weight"),
            d.o_groups * d.o_lora,
            d.heads * d.head_dim / d.o_groups,
            "attention",
        );
        fp8(&mut out, p("attn.wo_b.weight"), dim, d.o_groups * d.o_lora, "attention");
        raw(&mut out, p("attn.attn_sink"), DType::F32, vec![d.heads], "attention");
        raw(
            &mut out,
            p("attn.q_norm.weight"),
            DType::BF16,
            vec![d.q_lora],
            "attention",
        );
        raw(
            &mut out,
            p("attn.kv_norm.weight"),
            DType::BF16,
            vec![d.head_dim],
            "attention",
        );

        let ratio = d.compress_ratios[i];
        if d.kv_source_layers.contains(&i) {
            ensure!(ratio > 0, "layer {i} is a KV source with ratio 0");
            raw(
                &mut out,
                p("attn.compressor.norm.weight"),
                DType::BF16,
                vec![d.head_dim],
                "attention",
            );
            raw(
                &mut out,
                p("attn.compressor.wkv.weight"),
                DType::BF16,
                vec![d.head_dim, dim],
                "attention",
            );
            if ratio > 1 {
                // A ratio of 1 is a plain projection; above it the pooling is a learned softmax gate.
                raw(
                    &mut out,
                    p("attn.compressor.wgate.weight"),
                    DType::BF16,
                    vec![d.head_dim, dim],
                    "attention",
                );
            }
        }
        if d.index_source_layers.contains(&i) {
            fp8(
                &mut out,
                p("attn.indexer.wq_b.weight"),
                d.index.heads * d.index.dim,
                d.q_lora,
                "attention",
            );
            raw(
                &mut out,
                p("attn.indexer.weights_proj.weight"),
                DType::BF16,
                vec![d.index.heads, dim],
                "attention",
            );
            if d.kv_source_layers.contains(&i) {
                // Only a layer that compresses its own KV can produce the index keys.
                raw(
                    &mut out,
                    p("attn.indexer.wk.weight"),
                    DType::BF16,
                    vec![d.index.dim, d.head_dim],
                    "attention",
                );
                raw(
                    &mut out,
                    p("attn.indexer.k_norm.weight"),
                    DType::BF16,
                    vec![d.index.dim],
                    "attention",
                );
            }
        }

        // MoE: a bf16 router with two biases (text and image spans), one shared expert, and the routed ones.
        raw(&mut out, p("ffn.gate.weight"), DType::BF16, vec![d.experts, dim], "moe");
        raw(&mut out, p("ffn.gate.bias"), DType::F32, vec![d.experts], "moe");
        if d.vision.is_some() {
            raw(&mut out, p("ffn.gate.bias_vl"), DType::F32, vec![d.experts], "moe");
        }
        fp8(&mut out, p("ffn.shared_experts.w1.weight"), d.moe_inter, dim, "moe");
        fp8(&mut out, p("ffn.shared_experts.w3.weight"), d.moe_inter, dim, "moe");
        fp8(&mut out, p("ffn.shared_experts.w2.weight"), dim, d.moe_inter, "moe");
        for e in 0..d.experts {
            let q = |s: &str| p(&format!("ffn.experts.{e}.{s}.weight"));
            fp4(&mut out, q("w1"), d.moe_inter, dim, "experts")?;
            fp4(&mut out, q("w3"), d.moe_inter, dim, "experts")?;
            fp4(&mut out, q("w2"), dim, d.moe_inter, "experts")?;
        }

        if let (Some(k), Some(e)) = (engram_of(i), d.engram.as_ref()) {
            let p = |s: &str| format!("layers.{i}.engram.{s}");
            let rows = e.num_embeddings[k];
            // The table stays fp8 with one e8m0 scale per 32-wide block of a row: it is a sparse lookup, read a
            // row at a time from the file, never a matmul weight.
            raw(
                &mut out,
                p("embed.weight"),
                DType::F8E4M3,
                vec![rows, e.head_dim],
                "engram",
            );
            raw(
                &mut out,
                p("embed.scale"),
                DType::F8E8M0,
                vec![rows, e.head_dim / FP8_BLOCK],
                "engram",
            );
            let cols = (e.max_ngram - 1) * e.n_heads * e.head_dim;
            fp8(&mut out, p("wkv.weight"), dim * (d.hc_mult + 1), cols, "engram");
            raw(&mut out, p("q_weight"), DType::BF16, vec![d.hc_mult, dim], "engram");
            raw(&mut out, p("k_weight"), DType::BF16, vec![d.hc_mult, dim], "engram");
        }
    }

    // DSpark: the draft stages live under `mtp.`, with their own attention, hyper-connections and 128-expert MoE.
    if let Some(s) = &d.dspark {
        for stage in 0..s.stages {
            let p = |x: &str| format!("mtp.{stage}.{x}");
            raw(&mut out, p("attn_norm.weight"), DType::BF16, vec![dim], "dspark");
            raw(&mut out, p("ffn_norm.weight"), DType::BF16, vec![dim], "dspark");
            for which in ["hc_attn", "hc_ffn"] {
                raw(
                    &mut out,
                    p(&format!("{which}_fn")),
                    DType::F32,
                    vec![hc_mix, d.hc_mult * dim],
                    "dspark",
                );
                raw(
                    &mut out,
                    p(&format!("{which}_base")),
                    DType::F32,
                    vec![hc_mix],
                    "dspark",
                );
                raw(&mut out, p(&format!("{which}_scale")), DType::F32, vec![3], "dspark");
            }
            fp8(&mut out, p("attn.wq_a.weight"), d.q_lora, dim, "dspark");
            fp8(
                &mut out,
                p("attn.wq_b.weight"),
                d.heads * d.head_dim,
                d.q_lora,
                "dspark",
            );
            fp8(&mut out, p("attn.wkv.weight"), d.head_dim, dim, "dspark");
            fp8(
                &mut out,
                p("attn.wo_a.weight"),
                d.o_groups * d.o_lora,
                d.heads * d.head_dim / d.o_groups,
                "dspark",
            );
            fp8(&mut out, p("attn.wo_b.weight"), dim, d.o_groups * d.o_lora, "dspark");
            raw(&mut out, p("attn.attn_sink"), DType::F32, vec![d.heads], "dspark");
            raw(&mut out, p("attn.q_norm.weight"), DType::BF16, vec![d.q_lora], "dspark");
            raw(
                &mut out,
                p("attn.kv_norm.weight"),
                DType::BF16,
                vec![d.head_dim],
                "dspark",
            );

            raw(
                &mut out,
                p("ffn.gate.weight"),
                DType::BF16,
                vec![s.experts, dim],
                "dspark",
            );
            raw(&mut out, p("ffn.gate.bias"), DType::F32, vec![s.experts], "dspark");
            if d.vision.is_some() {
                raw(&mut out, p("ffn.gate.bias_vl"), DType::F32, vec![s.experts], "dspark");
            }
            fp8(&mut out, p("ffn.shared_experts.w1.weight"), d.moe_inter, dim, "dspark");
            fp8(&mut out, p("ffn.shared_experts.w3.weight"), d.moe_inter, dim, "dspark");
            fp8(&mut out, p("ffn.shared_experts.w2.weight"), dim, d.moe_inter, "dspark");
            for e in 0..s.experts {
                let q = |x: &str| p(&format!("ffn.experts.{e}.{x}.weight"));
                fp4(&mut out, q("w1"), d.moe_inter, dim, "dspark")?;
                fp4(&mut out, q("w3"), d.moe_inter, dim, "dspark")?;
                fp4(&mut out, q("w2"), dim, d.moe_inter, "dspark")?;
            }

            if stage == 0 {
                // The stage reads the attention input of the target layers, concatenated.
                fp8(
                    &mut out,
                    p("main_proj.weight"),
                    dim,
                    dim * s.target_layers.len(),
                    "dspark",
                );
                raw(&mut out, p("main_norm.weight"), DType::BF16, vec![dim], "dspark");
            }
            if stage + 1 == s.stages {
                raw(&mut out, p("norm.weight"), DType::BF16, vec![dim], "dspark");
                raw(
                    &mut out,
                    p("markov_head.embed.weight"),
                    DType::BF16,
                    vec![c.vocab, s.markov_rank],
                    "dspark",
                );
                raw(
                    &mut out,
                    p("markov_head.head.weight"),
                    DType::BF16,
                    vec![c.vocab, s.markov_rank],
                    "dspark",
                );
                raw(
                    &mut out,
                    p("confidence_head.proj.weight"),
                    DType::BF16,
                    vec![1, dim + s.markov_rank],
                    "dspark",
                );
            }
        }
    }

    // Vision: a plain pre-LN ViT and a two-layer aligner, both bf16.
    if let Some(v) = &d.vision {
        let patch_in = v.patch * v.patch * 3;
        raw(
            &mut out,
            "vision.patch_embed.proj.weight".into(),
            DType::BF16,
            vec![v.hidden, patch_in],
            "vision",
        );
        raw(
            &mut out,
            "vision.patch_embed.proj.bias".into(),
            DType::BF16,
            vec![v.hidden],
            "vision",
        );
        for i in 0..v.layers {
            let p = |s: &str| format!("vision.blocks.{i}.{s}");
            raw(
                &mut out,
                p("attn.wqkv.weight"),
                DType::BF16,
                vec![3 * v.hidden, v.hidden],
                "vision",
            );
            raw(&mut out, p("attn.wqkv.bias"), DType::BF16, vec![3 * v.hidden], "vision");
            raw(
                &mut out,
                p("attn.wo.weight"),
                DType::BF16,
                vec![v.hidden, v.hidden],
                "vision",
            );
            raw(&mut out, p("attn.wo.bias"), DType::BF16, vec![v.hidden], "vision");
            raw(
                &mut out,
                p("mlp.w1.weight"),
                DType::BF16,
                vec![2 * v.inter, v.hidden],
                "vision",
            );
            raw(
                &mut out,
                p("mlp.w2.weight"),
                DType::BF16,
                vec![v.hidden, v.inter],
                "vision",
            );
            raw(&mut out, p("norm1.weight"), DType::BF16, vec![v.hidden], "vision");
            raw(&mut out, p("norm2.weight"), DType::BF16, vec![v.hidden], "vision");
        }
        raw(
            &mut out,
            "vision.norm.weight".into(),
            DType::BF16,
            vec![v.hidden],
            "vision",
        );
        let merged = v.hidden * v.downsample * v.downsample;
        raw(
            &mut out,
            "aligner.w1.weight".into(),
            DType::BF16,
            vec![dim, merged],
            "vision",
        );
        raw(&mut out, "aligner.w1.bias".into(), DType::BF16, vec![dim], "vision");
        raw(
            &mut out,
            "aligner.w2.weight".into(),
            DType::BF16,
            vec![dim, dim],
            "vision",
        );
        raw(&mut out, "aligner.w2.bias".into(), DType::BF16, vec![dim], "vision");
        for (name, what) in [
            ("image_start", "start"),
            ("image_end", "end"),
            ("image_newline", "newline"),
        ] {
            let _ = what;
            raw(&mut out, name.into(), DType::BF16, vec![dim], "vision");
        }
    }
    Ok(out)
}

/// Compare the checkpoint against the map: shape and dtype of every expected tensor, what is missing, what is
/// extra. Returns one line per fact, and the problems are the lines that start with `!` (`malleus inspect` exits
/// non-zero when there are any).
pub fn check(src: &dyn TensorSource, c: &Config) -> Result<Vec<String>> {
    let want = expected(c)?;
    let found = src.names();
    let mut lines = Vec::new();
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    let mut bytes: std::collections::BTreeMap<&'static str, (u64, usize)> = Default::default();
    for e in &want {
        match src.info(&e.name) {
            None => missing.push(e.name.clone()),
            Some(i) => {
                if i.dtype != e.dtype || i.shape != e.shape {
                    wrong.push(format!(
                        "{}: {:?} {:?}, expected {:?} {:?}",
                        e.name, i.dtype, i.shape, e.dtype, e.shape
                    ));
                }
                let b = bytes.entry(e.class).or_default();
                b.0 += i.size as u64;
                b.1 += 1;
            }
        }
    }
    let want_names: std::collections::HashSet<&str> = want.iter().map(|e| e.name.as_str()).collect();
    let extra: Vec<String> = found
        .iter()
        .copied()
        .filter(|n| !want_names.contains(n))
        .map(str::to_string)
        .collect();
    let total: u64 = bytes.values().map(|(b, _)| *b).sum();
    lines.push(format!(
        "deepseek_v41: {} tensors expected, {} in the checkpoint, {:.1} GB",
        want.len(),
        found.len(),
        total as f64 / 1e9
    ));
    for (class, (b, n)) in &bytes {
        lines.push(format!("  {class:10} {:8.1} GB  {n} tensors", *b as f64 / 1e9));
    }
    for (what, list) in [("missing", &missing), ("extra", &extra)] {
        if list.is_empty() {
            continue;
        }
        let head: Vec<&str> = list.iter().take(5).map(String::as_str).collect();
        lines.push(format!(
            "! {} {} tensors: {}{}",
            list.len(),
            what,
            head.join(", "),
            if list.len() > head.len() { ", ..." } else { "" }
        ));
    }
    if !wrong.is_empty() {
        for w in wrong.iter().take(5) {
            lines.push(format!("! {w}"));
        }
        if wrong.len() > 5 {
            lines.push(format!(
                "! {} more tensors with a wrong shape or dtype",
                wrong.len() - 5
            ));
        }
    }
    Ok(lines)
}

/// Bytes one session holds at `ctx`: the sliding-window ring of every layer, the compressed latent of the source
/// layers and the indexer's keys, all stored at 8 bits.
pub fn session_bytes(c: &Config, ctx: usize) -> u64 {
    let Some(d) = c.deepseek.as_ref() else {
        return 0;
    };
    let mut t = 0u64;
    t += (c.n_layers * d.window * d.head_dim) as u64;
    for &i in &d.kv_source_layers {
        if let Some(per_group) = (ctx as u64).checked_div(d.compress_ratios[i] as u64) {
            t += per_group * d.head_dim as u64;
        }
    }
    for &i in &d.index_source_layers {
        if let Some(per_group) = (ctx as u64).checked_div(d.compress_ratios[i] as u64) {
            t += per_group * d.index.dim as u64;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use malleus_gguf::TensorInfo;
    use std::collections::HashMap;

    /// A small DeepSeek-V4.1 config: 4 layers, one KV source, two index sources, one Engram layer, one DSpark
    /// stage. The geometry is the released one wherever it is scale-free.
    const SMALL: &str = r#"{
      "model_type": "deepseek_v41",
      "image_token_id": 99,
      "vision_config": {"model_type":"deepseek_v41_vision","num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,
        "intermediate_size":128,"patch_size":14,"rope_theta":10000,"downsample_ratio":3,"max_image_tokens":1024,
        "min_pixels":295936},
      "text_config": {
        "model_type":"deepseek_v41_text","vocab_size":1000,"hidden_size":256,"moe_intermediate_size":128,
        "num_hidden_layers":4,"num_attention_heads":8,"num_key_value_heads":1,"head_dim":256,"qk_rope_head_dim":64,
        "q_lora_rank":128,"o_lora_rank":256,"o_groups":8,"swiglu_limit":10.0,"rms_norm_eps":1e-20,
        "max_position_embeddings":1048576,"rope_theta":10000,
        "rope_scaling":{"rope_type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
          "original_max_position_embeddings":65536},
        "compress_rope_theta":160000,"n_routed_experts":16,"n_shared_experts":1,"num_experts_per_tok":4,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.5,
        "sliding_window":128,"compress_ratios":[0,0,2,2],"kv_source_layer_ids":[2],
        "index_source_layer_ids":[2,3],"index_n_heads":2,"index_head_dim":128,"index_topk":64,
        "candidate_source_layer_id":2,"candidate_topk_blocks":16,"candidate_block_size":8,
        "hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,"engram_layer_ids":[1],"engram_num_embeddings":[1000],
        "engram_max_ngram_size":4,"engram_vocab_size":16000000,"engram_n_heads":8,"engram_head_dim":256,
        "engram_pad_token_id":2,"engram_compressed_vocab_size":99092,"num_nextn_predict_layers":1,
        "dspark_block_size":5,"dspark_noise_token_id":998,"dspark_target_layer_ids":[0,3],
        "dspark_markov_rank":256,"dspark_n_routed_experts":8,"dspark_num_experts_per_tok":2,
        "eos_token_id":1
      }
    }"#;

    fn small() -> Config {
        Config::from_json(SMALL, None).unwrap()
    }

    #[test]
    fn deepseek_geometry() {
        let c = small();
        assert_eq!(c.family, Family::Deepseek41);
        assert_eq!((c.hidden, c.n_layers, c.vocab), (256, 4, 1000));
        let d = c.deepseek.as_ref().unwrap();
        assert_eq!((d.nope, d.rope, d.head_dim), (192, 64, 256));
        assert_eq!(d.compress_ratios, vec![0, 0, 2, 2]);
        assert_eq!(c.kind.iter().filter(|k| **k == MixerKind::Deepseek).count(), 4);
        assert_eq!(d.rope_params.factor, 16.0);
        assert_eq!(d.engram.as_ref().unwrap().layers, vec![1]);
        assert_eq!(d.dspark.as_ref().unwrap().target_layers, vec![0, 3]);
        // 4 layer rings + one ratio-2 compressed source + two index sources (one at ratio 2, one at ratio 2).
        assert_eq!(session_bytes(&c, 1024), 4 * 128 * 256 + 512 * 256 + 2 * 512 * 128);
    }

    #[test]
    fn deepseek_map_covers_the_layer() {
        let c = small();
        let want: HashMap<String, Expect> = expected(&c).unwrap().into_iter().map(|e| (e.name.clone(), e)).collect();
        // An fp8 linear carries its 32x32 e8m0 scale.
        let s = &want["layers.0.attn.wq_a.scale"];
        assert_eq!((s.dtype, s.shape.clone()), (DType::F8E8M0, vec![4, 8]));
        // The routed experts are e2m1 pairs in an I8 tensor with one scale per 32 values.
        let e = &want["layers.0.ffn.experts.0.w1.weight"];
        assert_eq!((e.dtype, e.shape.clone()), (DType::I8, vec![128, 128]));
        assert_eq!(want["layers.0.ffn.experts.0.w1.scale"].shape, vec![128, 8]);
        // The Engram table is fp8 with a scale per 32-wide block of the row.
        assert_eq!(want["layers.1.engram.embed.scale"].shape, vec![1000, 8]);
        // The compressor's gate exists only above ratio 1, on the source layer.
        assert!(want.contains_key("layers.2.attn.compressor.wgate.weight"));
        assert!(!want.contains_key("layers.3.attn.compressor.wgate.weight"));
        // The image bias is there because the checkpoint has a vision config.
        assert!(want.contains_key("layers.0.ffn.gate.bias_vl"));
        // The aligner merges 3x3 patches of the tower's width.
        assert_eq!(want["aligner.w1.weight"].shape, vec![256, 64 * 9]);
    }

    /// A source holding exactly what the map expects.
    struct Fake(HashMap<String, TensorInfo>);

    impl Fake {
        fn from(c: &Config) -> Self {
            let m = expected(c)
                .unwrap()
                .into_iter()
                .map(|e| {
                    let size = e.dtype.size_of_elems(e.shape.iter().product());
                    (
                        e.name.clone(),
                        TensorInfo {
                            name: e.name,
                            dtype: e.dtype,
                            shape: e.shape,
                            offset: 0,
                            size,
                        },
                    )
                })
                .collect();
            Fake(m)
        }
    }

    impl TensorSource for Fake {
        fn info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name)
        }
        fn names(&self) -> Vec<&str> {
            self.0.keys().map(|s| s.as_str()).collect()
        }
        fn read(&self, _: &str, _: usize, _: &mut [u8]) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_complete_checkpoint_has_no_problem_line() {
        let c = small();
        let src = Fake::from(&c);
        let lines = check(&src, &c).unwrap();
        assert!(lines.iter().all(|l| !l.starts_with('!')), "{lines:?}");
        assert!(lines[0].contains("tensors expected"));
    }

    #[test]
    fn a_missing_or_misshapen_tensor_is_reported() {
        let c = small();
        let mut src = Fake::from(&c);
        src.0.remove("layers.0.attn.wkv.weight");
        let lines = check(&src, &c).unwrap();
        assert!(
            lines
                .iter()
                .any(|l| l.contains("1 missing tensors") && l.contains("layers.0.attn.wkv.weight")),
            "{lines:?}"
        );

        let mut src = Fake::from(&c);
        src.0.get_mut("layers.0.attn.wkv.weight").unwrap().shape = vec![256, 128];
        let lines = check(&src, &c).unwrap();
        assert!(
            lines.iter().any(|l| l.starts_with("! layers.0.attn.wkv.weight")),
            "{lines:?}"
        );
    }
}
