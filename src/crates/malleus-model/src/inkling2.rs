//! Inkling-2 (`inkling2_loop`, issue #62): the config and the map of the checkpoint's text tensors.
//!
//! The graph is not the Qwen/Ling one. A block of `attn_res_block` layers feeds an *attention residual*: a
//! learned query attends over the running stack of block states, once for the attention input (three times in the
//! full layers, for q, k and v) and once for the MLP input. Every layer input and output passes a short
//! convolution, the full layers carry a relative-position bias with a per-head state, and the middle 12 layers run
//! `loop_passes` times, mixing cross-pass messages through a low-rank controller per stream.
//!
//! This module holds the geometry and the map; the loader and the forward are the next increment.

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::DeviceBuffer;
use serde_json::Value;

use crate::config::{Config, Family, Inkling2Config, Inkling2Mtp, Inkling2Vision, MixerKind, MoeConfig};
use crate::upload::Uploader;
use crate::weights::Linear;

/// The `text_config` of an Inkling-2 checkpoint, as `Config` plus the Inkling-2 geometry.
pub(crate) fn config(cfg: &Value, generation_config: Option<&str>) -> Result<Config> {
    let t = &cfg["text_config"];
    let u = |k: &str| -> Result<usize> {
        t[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: {k}"))
    };
    let f = |k: &str| -> Result<f32> { t[k].as_f64().map(|v| v as f32).with_context(|| format!("config: {k}")) };
    let n_layers = u("num_hidden_layers")?;
    let heads = u("num_attention_heads")?;
    let kv_heads = u("num_key_value_heads")?;
    let head_dim = u("head_dim")?;
    // The sliding layers are the ones `local_layer_ids` names; the full ones carry the relative-position bias.
    let local: Vec<usize> = t["local_layer_ids"]
        .as_array()
        .context("config: local_layer_ids")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).context("config: local_layer_ids"))
        .collect::<Result<_>>()?;
    let kind: Vec<MixerKind> = (0..n_layers)
        .map(|i| {
            if local.contains(&i) {
                MixerKind::Attn
            } else {
                MixerKind::AttnFull
            }
        })
        .collect();
    ensure!(local.len() <= n_layers, "config: local_layer_ids longer than the stack");
    let loop_cfg = &t["loop_config"];
    let lu = |k: &str| -> Result<usize> {
        loop_cfg[k]
            .as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: loop_config.{k}"))
    };
    let vision = match cfg["vision_config"].as_object() {
        Some(v) => {
            let vu = |k: &str| -> Result<usize> {
                v[k].as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: vision_config.{k}"))
            };
            Some(Inkling2Vision {
                hidden: vu("hidden_size")?,
                inter: vu("hidden_size")? * 4,
                heads: vu("num_attention_heads")?,
                layers: vu("num_hidden_layers")?,
                patch: vu("native_patch_size")?,
                temporal: vu("temporal_patch_size")?,
                merge: vu("adapter_merge_size")?,
                grid: vu("posemb_grid_size")?,
                proj_hidden: vu("projection_hidden_dim")?,
                eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
                // The image placeholder is the token the processor writes; the hub config does not carry it.
                image_token: cfg["image_token_id"].as_u64().unwrap_or(200054) as u32,
            })
        }
        None => None,
    };
    let mtp = match cfg["mtp_config"].as_object() {
        Some(v) => {
            let mu = |k: &str| -> Result<usize> {
                v[k].as_u64()
                    .map(|v| v as usize)
                    .with_context(|| format!("config: mtp_config.{k}"))
            };
            Some(Inkling2Mtp {
                layers: mu("num_hidden_layers")?,
                inter: mu("intermediate_size")?,
                heads: mu("num_attention_heads")?,
                kv_heads: mu("num_key_value_heads")?,
                head_dim: mu("head_dim")?,
                rope_theta: v["rope_theta"].as_f64().unwrap_or(1e6) as f32,
                window: mu("sliding_window_size")?,
                local: v["local_layer_ids"]
                    .as_array()
                    .context("config: mtp_config.local_layer_ids")?
                    .iter()
                    .map(|x| x.as_u64().map(|x| x as usize).context("local_layer_ids"))
                    .collect::<Result<_>>()?,
                block: mu("block_size")?,
                markov_rank: mu("markov_rank")?,
                target_layers: v["target_layer_ids"]
                    .as_array()
                    .context("config: mtp_config.target_layer_ids")?
                    .iter()
                    .map(|x| x.as_u64().map(|x| x as usize).context("target_layer_ids"))
                    .collect::<Result<_>>()?,
            })
        }
        None => None,
    };
    let hidden = u("hidden_size")?;
    let inter = u("intermediate_size")?;
    let experts = u("n_routed_experts")?;
    let top_k = u("num_experts_per_tok")?;
    let inkling2 = Inkling2Config {
        heads,
        kv_heads,
        head_dim,
        rel_extent: u("rel_extent")?,
        d_rel: u("d_rel")?,
        log_floor: u("log_scaling_n_floor")?,
        log_alpha: f("log_scaling_alpha")?,
        qk_clip: f("qk_l2_clip_threshold")?,
        rel_clip: f("rel_l2_clip_threshold")?,
        attn_res_block: u("n_layers_per_attn_res_block")?,
        sconv_k: u("sconv_kernel_size")?,
        window: u("sliding_window_size")?,
        experts,
        top_k,
        moe_inter: inter,
        route_scale: f("route_scale")?,
        gate_softcap: f("mlp_gate_softcap")?,
        up_softcap: f("mlp_up_softcap")?,
        logit_softcap: f("final_logit_softcapping")?,
        unpadded_vocab: u("unpadded_vocab_size")?,
        loop_start: lu("core_start")?,
        loop_end: lu("core_end")?,
        loop_passes: lu("num_passes")?,
        loop_rank: lu("gate_rank")?,
        vision,
        mtp,
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
    let c = Config {
        hidden,
        n_layers,
        inter,
        vocab: u("vocab_size")?,
        n_heads: heads,
        n_kv: kv_heads,
        head_dim,
        lin_nk: heads,
        lin_nv: heads,
        lin_dk: head_dim,
        lin_dv: head_dim,
        conv_k: u("sconv_kernel_size")?,
        eps: f("rms_norm_eps")?,
        // The sliding and full layers rotate nothing: the relative bias replaces RoPE inside the attention.
        rope_theta: 0.0,
        rot_dim: 0,
        mrope_section: [0, 0, 0],
        full_attn: kind.iter().map(|k| *k == MixerKind::AttnFull).collect(),
        eos,
        moe: Some(MoeConfig {
            experts,
            top_k,
            inter,
            sigmoid: t["scoring_func"].as_str() == Some("sigmoid"),
            n_group: 1,
            topk_group: 1,
            bias: t["use_gate_bias"].as_bool().unwrap_or(false),
            scaling: f("route_scale")?,
            shared_gate: false,
        }),
        hc: None,
        ple: None,
        indexer: None,
        gate_sigmoid: false,
        family: Family::Inkling2,
        kind,
        mla: None,
        kda_lower_bound: 0.0,
        norm_one_plus_w: false,
        mrope: false,
        dense_ffn: 0,
        swiglu: None,
        hy4: None,
        inkling2: Some(inkling2),
    };
    c.validate()?;
    Ok(c)
}

/// The geometry the kernels will require of an Inkling-2 checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    ensure!(
        c.n_layers == c.kind.len() && c.n_layers == c.full_attn.len(),
        "layer kind count"
    );
    ensure!(
        i.heads == c.n_heads && i.kv_heads == c.n_kv && i.head_dim == c.head_dim,
        "head geometry"
    );
    ensure!(c.n_heads.is_multiple_of(c.n_kv), "bad head grouping");
    ensure!(
        i.attn_res_block > 0 && c.n_layers.is_multiple_of(i.attn_res_block),
        "attention-residual blocks"
    );
    ensure!(i.sconv_k == 4, "the short convolution is kernel 4");
    ensure!(
        i.loop_start > 0 && i.loop_start < i.loop_end && i.loop_end <= c.n_layers,
        "the loop range must be a non-empty part of the stack"
    );
    ensure!(i.loop_passes >= 2, "a fixed loop runs at least twice");
    ensure!(
        c.hidden.is_multiple_of(128) && i.moe_inter.is_multiple_of(128),
        "the 4-bit rotation needs widths in 128s"
    );
    ensure!(i.top_k <= i.experts, "more experts per token than experts");
    Ok(())
}

/// One tensor the forward will read: its name in the file, what it holds, and its shape.
pub struct Expected {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// Every text tensor of the graph, in load order. The experts arrive as the fused `gate_up_proj`/`down_proj` the
/// converter writes, at 4 bits; everything else is what the checkpoint stores.
pub fn text_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let mut v = Vec::new();
    let mut push = |name: String, dtype: DType, shape: Vec<usize>| v.push(Expected { name, dtype, shape });
    let (h, qd, kvd) = (c.hidden, i.heads * i.head_dim, i.kv_heads * i.head_dim);
    push("model.llm.embed.weight".into(), DType::BF16, vec![c.vocab, h]);
    push("model.llm.embed_norm.weight".into(), DType::BF16, vec![h]);
    push("model.llm.norm.weight".into(), DType::BF16, vec![h]);
    push("model.llm.unembed.weight".into(), DType::Mls8, vec![c.vocab, h]);
    push("model.llm.unembed.weight.scales".into(), DType::F32, vec![c.vocab]);
    push("model.llm.attn_res.output_query".into(), DType::F32, vec![h]);
    // One attention query per layer after the first; the full layers take three (q, k, v).
    for l in 1..c.n_layers {
        let full = c.kind[l] == MixerKind::AttnFull;
        let shape = if full { vec![3, h] } else { vec![h] };
        push(format!("model.llm.attn_res.attn_queries.{}", l - 1), DType::F32, shape);
    }
    for l in 0..c.n_layers {
        push(format!("model.llm.attn_res.mlp_queries.{l}"), DType::F32, vec![h]);
    }
    let core = i.loop_end - i.loop_start;
    for name in ["middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"] {
        push(format!("model.llm.{name}"), DType::F32, vec![2 * core, h]);
    }
    for l in 0..c.n_layers {
        let p = |s: &str| format!("model.llm.layers.{l}.{s}");
        let full = c.kind[l] == MixerKind::AttnFull;
        push(p("mlp_norm.weight"), DType::BF16, vec![h]);
        push(p("attn_sconv.weight"), DType::BF16, vec![h, 1, i.sconv_k]);
        push(p("mlp_sconv.weight"), DType::BF16, vec![h, 1, i.sconv_k]);
        if full {
            for n in ["q", "k", "v"] {
                push(p(&format!("{n}_input_norm.weight")), DType::BF16, vec![h]);
            }
            for n in ["k", "v"] {
                push(
                    p(&format!("attn.{n}_sconv.static_w")),
                    DType::BF16,
                    vec![i.sconv_k, kvd],
                );
                push(
                    p(&format!("attn.{n}_sconv.weight_proj")),
                    DType::BF16,
                    vec![i.kv_heads * i.sconv_k, h],
                );
            }
        } else {
            push(p("attn_norm.weight"), DType::BF16, vec![h]);
            for n in ["k", "v"] {
                push(
                    p(&format!("attn.{n}_sconv.weight")),
                    DType::BF16,
                    vec![kvd, 1, i.sconv_k],
                );
            }
        }
        for (n, dtype, shape) in [
            ("attn.wq_du.weight", DType::Mls4, vec![qd, h]),
            ("attn.wk_dv.weight", DType::Mls4, vec![kvd, h]),
            ("attn.wv_dv.weight", DType::Mls4, vec![kvd, h]),
            ("attn.wo_ud.weight", DType::Mls4, vec![h, qd]),
            ("attn.wr_du.weight", DType::Mls4, vec![i.heads * i.d_rel, h]),
            ("attn.gate_attn.weight", DType::Mls4, vec![qd, h]),
        ] {
            let k = *shape.last().unwrap();
            let scales = vec![shape[0], k / 128];
            push(p(n), dtype, shape);
            push(p(&format!("{n}.scales")), DType::F16, scales);
        }
        push(p("attn.rel_logits_proj.proj"), DType::BF16, vec![i.d_rel, i.rel_extent]);
        push(p("attn.attn_sink_proj.weight"), DType::BF16, vec![i.heads, h]);
        push(p("attn.q_l2_clip.scale"), DType::BF16, vec![i.heads, i.head_dim]);
        push(p("attn.rel_l2_clip.scale"), DType::BF16, vec![i.heads, i.d_rel]);
        push(p("mlp.gate.weight"), DType::BF16, vec![i.experts, h]);
        push(p("mlp.gate.bias"), DType::F32, vec![i.experts]);
        push(
            p("mlp.experts.gate_up_proj"),
            DType::Mls4,
            vec![i.experts, 2 * i.moe_inter, h],
        );
        push(
            p("mlp.experts.gate_up_proj.scales"),
            DType::F16,
            vec![i.experts, 2 * i.moe_inter, h / 128],
        );
        push(p("mlp.experts.down_proj"), DType::Mls4, vec![i.experts, h, i.moe_inter]);
        push(
            p("mlp.experts.down_proj.scales"),
            DType::F16,
            vec![i.experts, h, i.moe_inter / 128],
        );
        // The loop's cross-pass controller: one low-rank gate per stream, in the layers it repeats.
        if (i.loop_start..i.loop_end).contains(&l) {
            for stream in ["attn", "k", "v", "mlp"] {
                push(
                    p(&format!("controllers.{stream}.down_weight")),
                    DType::BF16,
                    vec![i.loop_rank, 2 * h],
                );
                push(
                    p(&format!("controllers.{stream}.forget_weight")),
                    DType::BF16,
                    vec![h, i.loop_rank],
                );
                push(
                    p(&format!("controllers.{stream}.gain_weight")),
                    DType::BF16,
                    vec![h, i.loop_rank],
                );
            }
        }
    }
    Ok(v)
}

/// The whole file: the text graph, the vision tower and the MTP head. The order is the order the loader reads.
pub fn map(c: &Config) -> Result<Vec<Expected>> {
    let mut v = text_map(c)?;
    v.extend(vision_map(c)?);
    v.extend(mtp_map(c)?);
    Ok(v)
}

/// Checks the file against the map: every tensor the forward reads is there, with the shape and the dtype it
/// expects, and nothing in the file is left unclaimed.
pub fn check(src: &dyn TensorSource, c: &Config) -> Result<Vec<String>> {
    let expected = map(c)?;
    let mut lines = Vec::new();
    let mut claimed: Vec<&str> = Vec::with_capacity(expected.len());
    for e in &expected {
        let info = src
            .info(&e.name)
            .with_context(|| format!("{}: not in the file", e.name))?;
        ensure!(
            info.dtype == e.dtype && info.shape == e.shape,
            "{}: expected {:?} {:?}, the file has {:?} {:?}",
            e.name,
            e.dtype,
            e.shape,
            info.dtype,
            info.shape
        );
        claimed.push(&e.name);
    }
    let mut unclaimed: Vec<&str> = src
        .names()
        .into_iter()
        .filter(|n| !claimed.iter().any(|c| c == n))
        .collect();
    unclaimed.sort_unstable();
    lines.push(format!(
        "{} of {} tensors mapped ({} text, {} vision, {} mtp)",
        claimed.len(),
        claimed.len() + unclaimed.len(),
        text_map(c)?.len(),
        vision_map(c)?.len(),
        mtp_map(c)?.len()
    ));
    if !unclaimed.is_empty() {
        lines.push(format!(
            "{} unclaimed, first: {:?}",
            unclaimed.len(),
            &unclaimed[..unclaimed.len().min(8)]
        ));
    }
    // What the device will hold: the stored bytes of each part, which is what the plan has to fit.
    let bytes = |v: &[Expected]| -> f64 {
        v.iter()
            .map(|e| e.dtype.size_of_elems(e.shape.iter().product::<usize>()) as f64)
            .sum::<f64>()
            / 1e9
    };
    lines.push(format!(
        "footprint: {:.2} GB text, {:.2} GB vision, {:.2} GB mtp",
        bytes(&text_map(c)?),
        bytes(&vision_map(c)?),
        bytes(&mtp_map(c)?)
    ));
    lines.push(format!(
        "loop {}..{} x{}, attention-residual blocks of {}, window {}, {} experts of {} active",
        c.inkling2.as_ref().unwrap().loop_start,
        c.inkling2.as_ref().unwrap().loop_end,
        c.inkling2.as_ref().unwrap().loop_passes,
        c.inkling2.as_ref().unwrap().attn_res_block,
        c.inkling2.as_ref().unwrap().window,
        c.inkling2.as_ref().unwrap().top_k,
        c.inkling2.as_ref().unwrap().experts
    ));
    if let Some(m) = &c.inkling2.as_ref().unwrap().mtp {
        lines.push(format!(
            "mtp: {} layers, block {}, markov rank {}",
            m.layers, m.block, m.markov_rank
        ));
    }
    Ok(lines)
}

/// The text graph, the vision tower and the MTP head, as the buffers the forward will read.
/// The vision tower as the buffers its graph reads: 23 pre-norm blocks at 1024 wide, 16 heads of 64, a 4096-wide MLP
/// with exact GELU, and the 2x2 merger projecting 4096 lanes to the text width.
pub struct Vision {
    pub proj_in: DeviceBuffer,
    pub proj_out: DeviceBuffer,
    pub ln_pre: (DeviceBuffer, DeviceBuffer),
    pub class_embedding: DeviceBuffer,
    pub cls_position: DeviceBuffer,
    pub patch: DeviceBuffer,
    pub position: DeviceBuffer,
    pub layers: Vec<VisLayer>,
    /// Patch side (the patch embedding covers one frame), temporal patch, merge side and the side of the learned
    /// position grid.
    pub patch_side: usize,
    pub temporal: usize,
    pub merge: usize,
    pub grid: usize,
}

pub struct VisLayer {
    pub ln_1: (DeviceBuffer, DeviceBuffer),
    pub ln_2: (DeviceBuffer, DeviceBuffer),
    pub ls_1: DeviceBuffer,
    pub ls_2: DeviceBuffer,
    pub fc: (DeviceBuffer, DeviceBuffer),
    pub proj: (DeviceBuffer, DeviceBuffer),
    pub q: (DeviceBuffer, DeviceBuffer),
    pub k: (DeviceBuffer, DeviceBuffer),
    pub v: (DeviceBuffer, DeviceBuffer),
    pub out: (DeviceBuffer, DeviceBuffer),
}

impl Vision {
    fn load(src: &dyn TensorSource, c: &Config, up: &mut Uploader) -> Result<Option<Self>> {
        let v = match c.inkling2.as_ref().and_then(|i| i.vision.as_ref()) {
            Some(v) => v,
            None => return Ok(None),
        };
        let p = |s: &str| format!("model.visual.{s}");
        let b = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(s), shape);
        let mut layers = Vec::with_capacity(v.layers);
        for l in 0..v.layers {
            let q = |s: &str| format!("backbone.layers.{l}.{s}");
            let bb = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(&q(s)), shape);
            layers.push(VisLayer {
                ln_1: (bb(up, "ln_1.weight", &[v.hidden])?, bb(up, "ln_1.bias", &[v.hidden])?),
                ln_2: (bb(up, "ln_2.weight", &[v.hidden])?, bb(up, "ln_2.bias", &[v.hidden])?),
                ls_1: bb(up, "ls_1.gamma", &[v.hidden])?,
                ls_2: bb(up, "ls_2.gamma", &[v.hidden])?,
                fc: (
                    bb(up, "mlp.c_fc.weight", &[v.inter, v.hidden])?,
                    bb(up, "mlp.c_fc.bias", &[v.inter])?,
                ),
                proj: (
                    bb(up, "mlp.c_proj.weight", &[v.hidden, v.inter])?,
                    bb(up, "mlp.c_proj.bias", &[v.hidden])?,
                ),
                q: (
                    bb(up, "self_attn.q_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.q_proj.bias", &[v.hidden])?,
                ),
                k: (
                    bb(up, "self_attn.k_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.k_proj.bias", &[v.hidden])?,
                ),
                v: (
                    bb(up, "self_attn.v_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.v_proj.bias", &[v.hidden])?,
                ),
                out: (
                    bb(up, "self_attn.out_proj.weight", &[v.hidden, v.hidden])?,
                    bb(up, "self_attn.out_proj.bias", &[v.hidden])?,
                ),
            });
        }
        Ok(Some(Vision {
            proj_in: b(up, "proj_in.weight", &[v.proj_hidden, v.proj_hidden])?,
            proj_out: b(up, "proj_out.weight", &[c.hidden, v.proj_hidden])?,
            ln_pre: (
                b(up, "backbone.ln_pre.weight", &[v.hidden])?,
                b(up, "backbone.ln_pre.bias", &[v.hidden])?,
            ),
            class_embedding: b(up, "backbone.embeddings.class_embedding", &[v.hidden])?,
            cls_position: b(up, "backbone.embeddings.cls_position_embedding", &[v.hidden])?,
            patch: b(
                up,
                "backbone.embeddings.patch_embedding.weight",
                &[v.hidden, 3 * v.patch * v.patch],
            )?,
            position: b(
                up,
                "backbone.embeddings.position_embedding",
                &[v.hidden, v.grid * v.grid],
            )?,
            layers,
            patch_side: v.patch,
            temporal: v.temporal,
            merge: v.merge,
            grid: v.grid,
        }))
    }
}

/// The MTP head: three draft layers at the text width, the fuse of the four target hidden states, and the Markov
/// correction that shapes the proposed logits. The names are the file's; the semantics are the next increment.
pub struct Mtp {
    pub fuse: DeviceBuffer,
    pub mask_embed: DeviceBuffer,
    pub norm: DeviceBuffer,
    pub layers: Vec<MtpLayer>,
    pub markov: Markov,
}

pub struct MtpLayer {
    pub dflare_norm: DeviceBuffer,
    pub input_norm: DeviceBuffer,
    pub post_attention_norm: DeviceBuffer,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub k_ctx: DeviceBuffer,
    pub v_ctx: DeviceBuffer,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub k_norm_ctx: DeviceBuffer,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct Markov {
    pub dflare_norm: DeviceBuffer,
    pub hidden_out: (DeviceBuffer, DeviceBuffer),
    pub w1: DeviceBuffer,
    pub rnn_proj: (DeviceBuffer, DeviceBuffer),
}

impl Mtp {
    fn load(src: &dyn TensorSource, c: &Config, up: &mut Uploader) -> Result<Option<Self>> {
        let m = match c.inkling2.as_ref().and_then(|i| i.mtp.as_ref()) {
            Some(m) => m,
            None => return Ok(None),
        };
        let h = c.hidden;
        let mut layers = Vec::with_capacity(m.layers);
        for l in 0..m.layers {
            let p = |s: &str| format!("layers.{l}.{s}");
            let t = |up: &mut Uploader, s: &str, shape: &[usize]| up.tensor(src, &p(s), shape);
            layers.push(MtpLayer {
                dflare_norm: t(up, "dflare_norm.weight", &[h])?,
                input_norm: t(up, "input_layernorm.weight", &[h])?,
                post_attention_norm: t(up, "post_attention_layernorm.weight", &[h])?,
                q: up.linear(src, &p("self_attn.q_proj"), m.heads * m.head_dim, h)?,
                k: up.linear(src, &p("self_attn.k_proj"), m.kv_heads * m.head_dim, h)?,
                v: up.linear(src, &p("self_attn.v_proj"), m.kv_heads * m.head_dim, h)?,
                o: up.linear(src, &p("self_attn.o_proj"), h, m.heads * m.head_dim)?,
                k_ctx: t(up, "self_attn.k_proj_ctx.weight", &[m.kv_heads * m.head_dim, h])?,
                v_ctx: t(up, "self_attn.v_proj_ctx.weight", &[m.kv_heads * m.head_dim, h])?,
                q_norm: t(up, "self_attn.q_norm.weight", &[m.head_dim])?,
                k_norm: t(up, "self_attn.k_norm.weight", &[m.head_dim])?,
                k_norm_ctx: t(up, "self_attn.k_norm_ctx.weight", &[m.head_dim])?,
                gate: up.linear(src, &p("mlp.gate_proj"), m.inter, h)?,
                up: up.linear(src, &p("mlp.up_proj"), m.inter, h)?,
                down: up.linear(src, &p("mlp.down_proj"), h, m.inter)?,
            });
        }
        Ok(Some(Mtp {
            fuse: up.tensor(src, "fuse_weight", &[4, 4])?,
            mask_embed: up.tensor(src, "mask_embed", &[h])?,
            norm: up.tensor(src, "norm.weight", &[h])?,
            layers,
            markov: Markov {
                dflare_norm: up.tensor(src, "markov_head.dflare_norm.weight", &[h])?,
                hidden_out: (
                    up.tensor(src, "markov_head.hidden_out.weight", &[h, m.markov_rank])?,
                    up.tensor(src, "markov_head.hidden_out.bias", &[h])?,
                ),
                w1: up.tensor(src, "markov_head.markov_w1.weight", &[c.vocab, m.markov_rank])?,
                rnn_proj: (
                    up.tensor(src, "markov_head.rnn_proj.weight", &[3 * m.markov_rank, 5120])?,
                    up.tensor(src, "markov_head.rnn_proj.bias", &[3 * m.markov_rank])?,
                ),
            },
        }))
    }
}

pub struct Weights {
    pub embed: DeviceBuffer,
    pub embed_norm: DeviceBuffer,
    pub norm: DeviceBuffer,
    pub unembed: Linear,
    pub output_query: DeviceBuffer,
    /// One per layer after the first: `[hidden]`, or `[3, hidden]` in the full layers (q, k, v).
    pub attn_queries: Vec<DeviceBuffer>,
    pub mlp_queries: Vec<DeviceBuffer>,
    pub cross_pass_q: DeviceBuffer,
    pub cross_pass_kv: DeviceBuffer,
    pub layers: Vec<Layer>,
    pub vision: Option<Vision>,
    pub mtp: Option<Mtp>,
    pub bytes: usize,
}

pub struct Layer {
    pub full: bool,
    pub mlp_norm: DeviceBuffer,
    pub attn_norm: Option<DeviceBuffer>,
    pub attn_sconv: DeviceBuffer,
    pub mlp_sconv: DeviceBuffer,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub wr: Linear,
    pub gate: Linear,
    pub rel_logits: DeviceBuffer,
    pub sink: DeviceBuffer,
    pub q_clip: DeviceBuffer,
    pub rel_clip: DeviceBuffer,
    pub sconv: Sconv,
    pub moe: Moe,
    pub controller: Option<Controller>,
}

/// The short convolution of the keys and values. The sliding layers convolve with a fixed depthwise kernel; the full
/// ones add per-token taps projected from the hidden state to a fixed kernel, one tap set per key/value head.
pub enum Sconv {
    Local {
        k: DeviceBuffer,
        v: DeviceBuffer,
    },
    Dynamic {
        k_static: DeviceBuffer,
        k_proj: DeviceBuffer,
        v_static: DeviceBuffer,
        v_proj: DeviceBuffer,
        q_norm: DeviceBuffer,
        k_norm: DeviceBuffer,
        v_norm: DeviceBuffer,
    },
}

pub struct Moe {
    pub gate: DeviceBuffer,
    pub bias: DeviceBuffer,
    /// MLS4 `[experts, 2 * inter, hidden]` and its group scales.
    pub gate_up: (DeviceBuffer, DeviceBuffer),
    /// MLS4 `[experts, hidden, inter]` and its group scales.
    pub down: (DeviceBuffer, DeviceBuffer),
}

pub struct Gate {
    pub down: DeviceBuffer,
    pub forget: DeviceBuffer,
    pub gain: DeviceBuffer,
}

/// The loop's cross-pass controller: one low-rank gate per stream.
pub struct Controller {
    pub attn: Gate,
    pub k: Gate,
    pub v: Gate,
    pub mlp: Gate,
}

/// A small bf16 tensor read back as f32: the kernels that consume the clip scales and the relative-position
/// table read them as `const float*`.
fn bf16_as_f32(src: &dyn TensorSource, name: &str, n: usize) -> Result<DeviceBuffer> {
    let raw = src.read_all(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(raw.len() == n * 2, "{name}: expected {n} bf16 values");
    let v: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect();
    Ok(DeviceBuffer::from_slice(&v)?)
}

impl Weights {
    pub fn load(src: &dyn TensorSource, c: &Config, progress: &dyn Fn(usize, usize)) -> Result<Self> {
        let i = c.inkling2.as_ref().context("no inkling2 config")?;
        let mut up = Uploader::new()?;
        let p = |s: &str| format!("model.llm.{s}");
        let (h, qd, kvd) = (c.hidden, i.heads * i.head_dim, i.kv_heads * i.head_dim);
        progress(0, c.n_layers);
        let embed = up.tensor(src, &p("embed.weight"), &[c.vocab, h])?;
        let embed_norm = up.tensor(src, &p("embed_norm.weight"), &[h])?;
        let norm = up.tensor(src, &p("norm.weight"), &[h])?;
        let unembed = up.linear(src, &p("unembed"), c.vocab, h)?;
        let output_query = up.typed(src, &p("attn_res.output_query"), DType::F32, &[h])?;
        let mut attn_queries = Vec::with_capacity(c.n_layers - 1);
        for l in 1..c.n_layers {
            let full = c.kind[l] == MixerKind::AttnFull;
            let shape = if full { vec![3, h] } else { vec![h] };
            attn_queries.push(up.typed(src, &p(&format!("attn_res.attn_queries.{}", l - 1)), DType::F32, &shape)?);
        }
        let mut mlp_queries = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            mlp_queries.push(up.typed(src, &p(&format!("attn_res.mlp_queries.{l}")), DType::F32, &[h])?);
        }
        let cross = 2 * (i.loop_end - i.loop_start);
        let cross_pass_q = up.typed(src, &p("middle_cycle_cross_pass_queries"), DType::F32, &[cross, h])?;
        let cross_pass_kv = up.typed(src, &p("middle_cycle_cross_pass_kv_queries"), DType::F32, &[cross, h])?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            progress(l, c.n_layers);
            let p = |s: &str| format!("model.llm.layers.{l}.{s}");
            let full = c.kind[l] == MixerKind::AttnFull;
            let attn_norm = if full {
                None
            } else {
                Some(up.tensor(src, &p("attn_norm.weight"), &[h])?)
            };
            let sconv = if full {
                Sconv::Dynamic {
                    k_static: up.tensor(src, &p("attn.k_sconv.static_w"), &[i.sconv_k, kvd])?,
                    k_proj: up.tensor(src, &p("attn.k_sconv.weight_proj"), &[i.kv_heads * i.sconv_k, h])?,
                    v_static: up.tensor(src, &p("attn.v_sconv.static_w"), &[i.sconv_k, kvd])?,
                    v_proj: up.tensor(src, &p("attn.v_sconv.weight_proj"), &[i.kv_heads * i.sconv_k, h])?,
                    q_norm: up.tensor(src, &p("q_input_norm.weight"), &[h])?,
                    k_norm: up.tensor(src, &p("k_input_norm.weight"), &[h])?,
                    v_norm: up.tensor(src, &p("v_input_norm.weight"), &[h])?,
                }
            } else {
                Sconv::Local {
                    k: up.tensor(src, &p("attn.k_sconv.weight"), &[kvd, 1, i.sconv_k])?,
                    v: up.tensor(src, &p("attn.v_sconv.weight"), &[kvd, 1, i.sconv_k])?,
                }
            };
            let stack = |up: &mut Uploader, name: &str, n: usize, k: usize| -> Result<(DeviceBuffer, DeviceBuffer)> {
                let g = k / malleus_gguf::quant::BLOCK;
                let mut q = DeviceBuffer::alloc(i.experts * n * k / 2)?;
                let mut sc = DeviceBuffer::alloc(i.experts * n * g * 2)?;
                up.into(src, &p(name), DType::Mls4, &[i.experts, n, k], &mut q, 0)?;
                up.into(
                    src,
                    &p(&format!("{name}.scales")),
                    DType::F16,
                    &[i.experts, n, g],
                    &mut sc,
                    0,
                )?;
                Ok((q, sc))
            };
            let controller = if (i.loop_start..i.loop_end).contains(&l) {
                let gate = |up: &mut Uploader, stream: &str| -> Result<Gate> {
                    Ok(Gate {
                        down: up.tensor(
                            src,
                            &p(&format!("controllers.{stream}.down_weight")),
                            &[i.loop_rank, 2 * h],
                        )?,
                        forget: up.tensor(
                            src,
                            &p(&format!("controllers.{stream}.forget_weight")),
                            &[h, i.loop_rank],
                        )?,
                        gain: up.tensor(src, &p(&format!("controllers.{stream}.gain_weight")), &[h, i.loop_rank])?,
                    })
                };
                Some(Controller {
                    attn: gate(&mut up, "attn")?,
                    k: gate(&mut up, "k")?,
                    v: gate(&mut up, "v")?,
                    mlp: gate(&mut up, "mlp")?,
                })
            } else {
                None
            };
            layers.push(Layer {
                full,
                mlp_norm: up.tensor(src, &p("mlp_norm.weight"), &[h])?,
                attn_norm,
                attn_sconv: up.tensor(src, &p("attn_sconv.weight"), &[h, 1, i.sconv_k])?,
                mlp_sconv: up.tensor(src, &p("mlp_sconv.weight"), &[h, 1, i.sconv_k])?,
                wq: up.linear(src, &p("attn.wq_du"), qd, h)?,
                wk: up.linear(src, &p("attn.wk_dv"), kvd, h)?,
                wv: up.linear(src, &p("attn.wv_dv"), kvd, h)?,
                wo: up.linear(src, &p("attn.wo_ud"), h, qd)?,
                wr: up.linear(src, &p("attn.wr_du"), i.heads * i.d_rel, h)?,
                gate: up.linear(src, &p("attn.gate_attn"), qd, h)?,
                rel_logits: bf16_as_f32(src, &p("attn.rel_logits_proj.proj"), i.d_rel * i.rel_extent)?,
                sink: up.tensor(src, &p("attn.attn_sink_proj.weight"), &[i.heads, h])?,
                q_clip: bf16_as_f32(src, &p("attn.q_l2_clip.scale"), i.heads * i.head_dim)?,
                rel_clip: bf16_as_f32(src, &p("attn.rel_l2_clip.scale"), i.heads * i.d_rel)?,
                sconv,
                moe: Moe {
                    gate: up.tensor(src, &p("mlp.gate.weight"), &[i.experts, h])?,
                    bias: up.typed(src, &p("mlp.gate.bias"), DType::F32, &[i.experts])?,
                    gate_up: stack(&mut up, "mlp.experts.gate_up_proj", 2 * i.moe_inter, h)?,
                    down: stack(&mut up, "mlp.experts.down_proj", h, i.moe_inter)?,
                },
                controller,
            });
        }
        progress(c.n_layers, c.n_layers);
        let vision = Vision::load(src, c, &mut up)?;
        let mtp = Mtp::load(src, c, &mut up)?;
        Ok(Weights {
            embed,
            embed_norm,
            norm,
            unembed,
            output_query,
            attn_queries,
            mlp_queries,
            cross_pass_q,
            cross_pass_kv,
            layers,
            vision,
            mtp,
            bytes: up.bytes,
        })
    }
}

/// The vision tower: 23 pre-norm ViT blocks at 1024 wide, 16 heads, 4x MLP, native patches of 20x20 at two frames,
/// merged 2x2 by the adapter, then a 4096-wide projection in and out. Stored at 16 bits, like the checkpoint.
pub fn vision_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let v = match &i.vision {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        out.push(Expected {
            name,
            dtype: DType::BF16,
            shape,
        })
    };
    let h = v.hidden;
    push("model.visual.proj_in.weight".into(), vec![v.proj_hidden, v.proj_hidden]);
    push("model.visual.proj_out.weight".into(), vec![c.hidden, v.proj_hidden]);
    for n in ["weight", "bias"] {
        push(format!("model.visual.backbone.ln_pre.{n}"), vec![h]);
    }
    let e = "model.visual.backbone.embeddings";
    push(format!("{e}.class_embedding"), vec![h]);
    push(format!("{e}.cls_position_embedding"), vec![h]);
    push(format!("{e}.patch_embedding.weight"), vec![h, 3 * v.patch * v.patch]);
    push(format!("{e}.position_embedding"), vec![h, v.grid * v.grid]);
    for l in 0..v.layers {
        let p = |s: &str| format!("model.visual.backbone.layers.{l}.{s}");
        for n in ["ln_1", "ln_2"] {
            for t in ["weight", "bias"] {
                push(p(&format!("{n}.{t}")), vec![h]);
            }
        }
        for n in ["ls_1", "ls_2"] {
            push(p(&format!("{n}.gamma")), vec![h]);
        }
        push(p("mlp.c_fc.weight"), vec![v.inter, h]);
        push(p("mlp.c_fc.bias"), vec![v.inter]);
        push(p("mlp.c_proj.weight"), vec![h, v.inter]);
        push(p("mlp.c_proj.bias"), vec![h]);
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            push(p(&format!("self_attn.{n}.weight")), vec![h, h]);
            push(p(&format!("self_attn.{n}.bias")), vec![h]);
        }
    }
    Ok(out)
}

/// The MTP head: 3 layers at the text width, one sliding and two full, plus the Markov correction of the logits.
pub fn mtp_map(c: &Config) -> Result<Vec<Expected>> {
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let m = match &i.mtp {
        Some(m) => m,
        None => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    let mut push = |name: String, dtype: DType, shape: Vec<usize>| out.push(Expected { name, dtype, shape });
    let h = c.hidden;
    let qd = m.heads * m.head_dim;
    let kvd = m.kv_heads * m.head_dim;
    push("fuse_weight".into(), DType::BF16, vec![4, 4]);
    push("mask_embed".into(), DType::BF16, vec![h]);
    push("norm.weight".into(), DType::BF16, vec![h]);
    for l in 0..m.layers {
        let p = |s: &str| format!("layers.{l}.{s}");
        for n in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            push(p(&format!("{n}.weight")), DType::BF16, vec![h]);
        }
        for (n, shape) in [
            ("mlp.gate_proj.weight", vec![m.inter, h]),
            ("mlp.up_proj.weight", vec![m.inter, h]),
            ("mlp.down_proj.weight", vec![h, m.inter]),
            ("self_attn.q_proj.weight", vec![qd, h]),
            ("self_attn.k_proj.weight", vec![kvd, h]),
            ("self_attn.v_proj.weight", vec![kvd, h]),
            ("self_attn.o_proj.weight", vec![h, qd]),
        ] {
            let k = *shape.last().unwrap();
            let scales = vec![shape[0], k / 128];
            push(p(n), DType::Mls4, shape);
            push(p(&format!("{n}.scales")), DType::F16, scales);
        }
        for n in ["k_proj_ctx", "v_proj_ctx"] {
            push(p(&format!("self_attn.{n}.weight")), DType::BF16, vec![kvd, h]);
        }
        for n in ["q_norm", "k_norm", "k_norm_ctx"] {
            push(p(&format!("self_attn.{n}.weight")), DType::BF16, vec![m.head_dim]);
        }
    }
    let mk = "markov_head";
    push(format!("{mk}.dflare_norm.weight"), DType::BF16, vec![h]);
    push(format!("{mk}.hidden_out.weight"), DType::BF16, vec![h, m.markov_rank]);
    push(format!("{mk}.hidden_out.bias"), DType::BF16, vec![h]);
    push(
        format!("{mk}.markov_w1.weight"),
        DType::BF16,
        vec![c.vocab, m.markov_rank],
    );
    push(
        format!("{mk}.rnn_proj.weight"),
        DType::BF16,
        vec![3 * m.markov_rank, 5120],
    );
    push(format!("{mk}.rnn_proj.bias"), DType::BF16, vec![3 * m.markov_rank]);
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// The text forward. One pass over the 24 layers per call (the base graph: the loop's cross-pass messages are the
// next increment), on the whole token sequence. `generate` re-runs it for every new token, which is O(n^2) but is
// the correctness path: no KV cache, no session state.
// ---------------------------------------------------------------------------------------------------------------

use crate::engine::Model;
use malleus_hip::{Arg, Stream};

fn f32s(n: usize) -> Result<DeviceBuffer> {
    Ok(DeviceBuffer::alloc(n * 4)?)
}

fn argmax(v: &[f32], n: usize) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().take(n).enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

/// Runs `tokens` through the whole graph and returns the argmax of the last position's logits.
fn step(model: &Model, tokens: &[u32]) -> Result<u32> {
    let c = &model.cfg;
    let i = c.inkling2.as_ref().context("no inkling2 config")?;
    let w = model.inkling2.as_ref().context("no inkling2 weights")?;
    let st = &model.stream;
    let t = tokens.len();
    let h = c.hidden;
    let heads = i.heads;
    let kv = i.kv_heads;
    let hd = i.head_dim;
    let d_rel = i.d_rel;
    let qd = heads * hd;
    let kvd = kv * hd;
    let eps = c.eps;
    let inter = i.moe_inter;
    let experts = i.experts;
    let topk = i.top_k;
    let kmax = qd.max(h);
    let rd = heads * d_rel;

    let mut xr = f32s(t * kmax)?;
    let mut xh = DeviceBuffer::alloc(t * kmax * 2)?;
    let mut xsc = f32s(t * kmax.div_ceil(128))?;
    let x = f32s(t * h)?;
    let ids32: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
    let idbuf = DeviceBuffer::from_slice(&ids32)?;
    model.k.get("embed_bf16").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&w.embed).into(), (&idbuf).into(), h.into()],
    )?;
    rms(model, st, &x, &x, &w.embed_norm, t, h, eps)?;

    // The buffers the layers share.
    let normed = f32s(t * h)?;
    let states = f32s(8 * t * h)?;
    let mix_out = f32s(t * 3 * h)?;
    let q = f32s(t * qd)?;
    let k = f32s(t * kvd)?;
    let v = f32s(t * kvd)?;
    let r = f32s(t * rd)?;
    let gate = f32s(t * qd)?;
    let sink = f32s(t * heads)?;
    let bias = f32s(heads * t * t)?;
    let attn_out = f32s(t * qd)?;
    let y = f32s(t * h)?;
    let moe_out = f32s(t * h)?;
    let logits = f32s(t * experts)?;
    let route_ids = DeviceBuffer::zeroed(t * topk * 4)?;
    let route_w = f32s(t * topk)?;
    let gu = f32s(t * topk * 2 * inter)?;
    let act = f32s(t * topk * inter)?;
    let ones_k = DeviceBuffer::from_slice(&vec![1.0f32; kv * hd])?;
    // The log-scaled positions: 1 below the floor, so 1 for any sequence shorter than it.
    let mut tau: Vec<f32> = (0..t)
        .map(|p| 1.0 + i.log_alpha * (((p + 1) as f32) / i.log_floor as f32).max(1.0).ln())
        .collect();
    if tau.iter().all(|x| *x == 1.0) {
        tau.clear();
    }
    let taud = if tau.is_empty() {
        None
    } else {
        Some(DeviceBuffer::from_slice(&tau)?)
    };
    // The short-convolution history of a full-sequence pass: three zero tokens.
    let past = DeviceBuffer::zeroed(3 * h * 4)?;

    let mut blocks: Vec<DeviceBuffer> = Vec::with_capacity(8);
    // The attention inputs and the normalized forms, one buffer each: the projections read them by name.
    let mut qin = f32s(t * h)?;
    let mut kin = f32s(t * h)?;
    let mut vin = f32s(t * h)?;
    let qn = f32s(t * h)?;
    let kn = f32s(t * h)?;
    let vn = f32s(t * h)?;
    let kt = f32s(t * kvd)?;
    let vt = f32s(t * kvd)?;
    let yt = f32s(t * h)?;

    for l in 0..c.n_layers {
        let layer = &w.layers[l];
        let full = layer.full;
        let block_residual = f32s(t * h)?;
        block_residual.copy_from_shared(0, &x, 0, t * h * 4, st)?;

        // 1. The attention input: the raw state for layer 0, the attention residual above it.
        let nq = if full { 3 } else { 1 };
        if l == 0 {
            qin.copy_from_shared(0, &x, 0, t * h * 4, st)?;
            kin.copy_from_shared(0, &x, 0, t * h * 4, st)?;
            vin.copy_from_shared(0, &x, 0, t * h * 4, st)?;
        } else {
            mix(
                model,
                st,
                &mix_out,
                &states,
                &blocks,
                &x,
                &w.attn_queries[l - 1],
                nq,
                t,
                h,
                eps,
            )?;
            copy_nq(&mix_out, &mut qin, 0, t, h, nq, st)?;
            if full {
                copy_nq(&mix_out, &mut kin, 1, t, h, 3, st)?;
                copy_nq(&mix_out, &mut vin, 2, t, h, 3, st)?;
            } else {
                kin.copy_from_shared(0, &qin, 0, t * h * 4, st)?;
                vin.copy_from_shared(0, &qin, 0, t * h * 4, st)?;
            }
        }
        // 2. The input norms and the projections.
        let (dyn_k, dyn_v) = if full {
            let (q_norm, k_norm, v_norm) = match &layer.sconv {
                Sconv::Dynamic {
                    q_norm, k_norm, v_norm, ..
                } => (q_norm, k_norm, v_norm),
                _ => unreachable!(),
            };
            rms(model, st, &qn, &qin, q_norm, t, h, eps)?;
            rms(model, st, &kn, &kin, k_norm, t, h, eps)?;
            rms(model, st, &vn, &vin, v_norm, t, h, eps)?;
            let (dk, dv) = (f32s(t * kv * i.sconv_k)?, f32s(t * kv * i.sconv_k)?);
            let (k_proj, v_proj) = match &layer.sconv {
                Sconv::Dynamic { k_proj, v_proj, .. } => (k_proj, v_proj),
                _ => unreachable!(),
            };
            raw_gemm(model, st, &dk, &kn, k_proj, t, kv * i.sconv_k, h)?;
            raw_gemm(model, st, &dv, &vn, v_proj, t, kv * i.sconv_k, h)?;
            (Some(dk), Some(dv))
        } else {
            let an = layer.attn_norm.as_ref().context("sliding layer without attn_norm")?;
            rms(model, st, &qn, &qin, an, t, h, eps)?;
            kn.copy_from_shared(0, &qn, 0, t * h * 4, st)?;
            vn.copy_from_shared(0, &qn, 0, t * h * 4, st)?;
            (None, None)
        };
        lin(model, st, &mut xr, &mut xh, &mut xsc, &q, &qn, &layer.wq, t)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &k, &kn, &layer.wk, t)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &v, &vn, &layer.wv, t)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &r, &qn, &layer.wr, t)?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &gate, &qn, &layer.gate, t)?;
        raw_gemm(model, st, &sink, &qn, &layer.sink, t, heads, h)?;
        // 3. The short convolutions of the keys and the values, then the clips.
        let (k_w, v_w) = match &layer.sconv {
            Sconv::Local { k, v } => (k, v),
            Sconv::Dynamic { k_static, v_static, .. } => (k_static, v_static),
        };
        let div = if full { 3.0 } else { (i.sconv_k as f32 + 1.0).sqrt() };
        sconv2(
            model,
            st,
            &kt,
            &k,
            &vt,
            &v,
            k_w,
            v_w,
            dyn_k.as_ref(),
            dyn_v.as_ref(),
            &past,
            t,
            kvd,
            kv,
            div,
        )?;
        k.copy_from_shared(0, &kt, 0, t * kvd * 4, st)?;
        v.copy_from_shared(0, &vt, 0, t * kvd * 4, st)?;
        let tau = if full { taud.as_ref() } else { None };
        clip(model, st, &q, &layer.q_clip, tau, t, heads, hd, i.qk_clip)?;
        clip(model, st, &k, &ones_k, None, t, kv, hd, i.qk_clip)?;
        rms_unnorm(model, st, &v, t, kv, hd, eps)?;
        clip(model, st, &r, &layer.rel_clip, tau, t, heads, d_rel, i.rel_clip)?;
        rel_bias(model, st, &bias, &r, &layer.rel_logits, t, heads, d_rel, i.rel_extent)?;
        let window = if full { 0 } else { i.window };
        attn(
            model,
            st,
            &attn_out,
            &q,
            &k,
            &v,
            &bias,
            &sink,
            &gate,
            t,
            heads,
            kv,
            hd,
            window,
            1.0 / (hd as f32).sqrt(),
        )?;
        lin(model, st, &mut xr, &mut xh, &mut xsc, &y, &attn_out, &layer.wo, t)?;
        sconv_one(
            model,
            st,
            &yt,
            &y,
            &layer.attn_sconv,
            &past,
            t,
            h,
            (i.sconv_k as f32 + 1.0).sqrt(),
        )?;
        y.copy_from_shared(0, &yt, 0, t * h * 4, st)?;

        // 4. The block bookkeeping.
        let boundary = l % i.attn_res_block == 0;
        if boundary {
            let pushed = f32s(t * h)?;
            pushed.copy_from_shared(0, &block_residual, 0, t * h * 4, st)?;
            blocks.push(pushed);
            x.copy_from_shared(0, &y, 0, t * h * 4, st)?;
        } else {
            x.copy_from_shared(0, &block_residual, 0, t * h * 4, st)?;
            add(model, st, &x, &y, t * h)?;
        }
        // 5. The MLP input: the attention residual over the block stack.
        mix(
            model,
            st,
            &mix_out,
            &states,
            &blocks,
            &x,
            &w.mlp_queries[l],
            1,
            t,
            h,
            eps,
        )?;
        // 6. The MoE.
        rms(model, st, &normed, &mix_out, &layer.mlp_norm, t, h, eps)?;
        raw_gemm(model, st, &logits, &normed, &layer.moe.gate, t, experts, h)?;
        model.k.get("inkling2_moe_route").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&route_ids).into(),
                (&route_w).into(),
                (&logits).into(),
                (&layer.moe.bias).into(),
                experts.into(),
                topk.into(),
                i.route_scale.into(),
            ],
        )?;
        model.k.get("moe_gate_up").launch(
            ((2 * inter).div_ceil(32) as u32, (t * topk) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&gu).into(),
                (&normed).into(),
                (&layer.moe.gate_up.0).into(),
                (&layer.moe.gate_up.1).into(),
                (&route_ids).into(),
                topk.into(),
                (2 * inter).into(),
                h.into(),
                (&model.signs).into(),
            ],
        )?;
        model.k.get("inkling2_moe_act").launch(
            ((t * topk * inter / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&act).into(),
                (&gu).into(),
                (&model.signs).into(),
                inter.into(),
                i.gate_softcap.into(),
                i.up_softcap.into(),
            ],
        )?;
        model.k.get("moe_down").launch(
            (h.div_ceil(32) as u32, t as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&moe_out).into(),
                (&act).into(),
                (&layer.moe.down.0).into(),
                (&layer.moe.down.1).into(),
                (&route_ids).into(),
                (&route_w).into(),
                topk.into(),
                h.into(),
                inter.into(),
                0i32.into(),
            ],
        )?;
        sconv_one(
            model,
            st,
            &yt,
            &moe_out,
            &layer.mlp_sconv,
            &past,
            t,
            h,
            (i.sconv_k as f32 + 1.0).sqrt(),
        )?;
        moe_out.copy_from_shared(0, &yt, 0, t * h * 4, st)?;
        add(model, st, &x, &moe_out, t * h)?;
    }

    // The final attention residual, the norm and the head.
    mix(model, st, &mix_out, &states, &blocks, &x, &w.output_query, 1, t, h, eps)?;
    rms(model, st, &normed, &mix_out, &w.norm, t, h, eps)?;
    // The last row only.
    let last = f32s(h)?;
    last.copy_from_shared(0, &normed, (t - 1) * h * 4, h * 4, st)?;
    let logits_out = f32s(c.vocab)?;
    lin(model, st, &mut xr, &mut xh, &mut xsc, &logits_out, &last, &w.unembed, 1)?;
    softcap(model, st, &logits_out, c.vocab, i.logit_softcap)?;
    let raw: Vec<f32> = logits_out.download(c.vocab)?;
    Ok(argmax(&raw, i.unpadded_vocab))
}

fn rms(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("rmsnorm").launch(
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[out.into(), x.into(), w.into(), n.into(), eps.into(), 0i32.into()],
    )?;
    Ok(())
}

fn add(model: &Model, st: &Stream, y: &DeviceBuffer, x: &DeviceBuffer, n: usize) -> Result<()> {
    model.ew(st, "add_inplace", n, &[y.into(), x.into(), n.into()])?;
    Ok(())
}

fn lin(
    model: &Model,
    st: &Stream,
    xr: &mut DeviceBuffer,
    xh: &mut DeviceBuffer,
    xsc: &mut DeviceBuffer,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &Linear,
    t: usize,
) -> Result<()> {
    model.gemm(st, xr, xh, xsc, out, crate::engine::Act::X(x), w, t, false)?;
    Ok(())
}

fn raw_gemm(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    t: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    model.k.get("gemm_bf16").launch(
        (n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            x.into(),
            w.into(),
            t.into(),
            n.into(),
            k.into(),
            0i32.into(),
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mix(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    states: &DeviceBuffer,
    blocks: &[DeviceBuffer],
    partial: &DeviceBuffer,
    query: &DeviceBuffer,
    nq: usize,
    t: usize,
    h: usize,
    eps: f32,
) -> Result<()> {
    let n = blocks.len() + 1;
    for (j, b) in blocks.iter().enumerate() {
        states.copy_from_shared(j * t * h * 4, b, 0, t * h * 4, st)?;
    }
    states.copy_from_shared((n - 1) * t * h * 4, partial, 0, t * h * 4, st)?;
    model.k.get("inkling2_attn_res").launch(
        ((t * nq) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            query.into(),
            states.into(),
            t.into(),
            n.into(),
            h.into(),
            nq.into(),
            eps.into(),
        ],
    )?;
    Ok(())
}

/// Copies the `qi`-th of `nq` outputs of a mix (`[t, nq, h]`) into a `[t, h]` buffer.
fn copy_nq(
    src: &DeviceBuffer,
    dst: &mut DeviceBuffer,
    qi: usize,
    t: usize,
    h: usize,
    nq: usize,
    st: &Stream,
) -> Result<()> {
    for p in 0..t {
        dst.copy_from_shared(p * h * 4, src, (p * nq + qi) * h * 4, h * 4, st)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sconv2(
    model: &Model,
    st: &Stream,
    yk: &DeviceBuffer,
    xk: &DeviceBuffer,
    yv: &DeviceBuffer,
    xv: &DeviceBuffer,
    k_w: &DeviceBuffer,
    v_w: &DeviceBuffer,
    dyn_k: Option<&DeviceBuffer>,
    dyn_v: Option<&DeviceBuffer>,
    past: &DeviceBuffer,
    t: usize,
    c: usize,
    heads: usize,
    div: f32,
) -> Result<()> {
    for (y, x, w, dyn_w) in [(yk, xk, k_w, dyn_k), (yv, xv, v_w, dyn_v)] {
        let d: Arg = match dyn_w {
            Some(x) => x.into(),
            None => Arg::Ptr(0),
        };
        model.k.get("inkling2_sconv").launch(
            ((t * c).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                y.into(),
                x.into(),
                w.into(),
                d,
                past.into(),
                t.into(),
                c.into(),
                heads.into(),
                div.into(),
            ],
        )?;
    }
    Ok(())
}

fn sconv_one(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    past: &DeviceBuffer,
    t: usize,
    c: usize,
    div: f32,
) -> Result<()> {
    model.k.get("inkling2_sconv").launch(
        ((t * c).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            y.into(),
            x.into(),
            w.into(),
            Arg::Ptr(0),
            past.into(),
            t.into(),
            c.into(),
            1usize.into(),
            div.into(),
        ],
    )?;
    Ok(())
}

fn clip(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    scale: &DeviceBuffer,
    tau: Option<&DeviceBuffer>,
    t: usize,
    heads: usize,
    dim: usize,
    thr: f32,
) -> Result<()> {
    let ta: Arg = match tau {
        Some(x) => x.into(),
        None => Arg::Ptr(0),
    };
    model.k.get("inkling2_l2_clip").launch(
        ((t * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[
            y.into(),
            y.into(),
            scale.into(),
            ta,
            t.into(),
            heads.into(),
            dim.into(),
            thr.into(),
        ],
    )?;
    Ok(())
}

fn rms_unnorm(
    model: &Model,
    st: &Stream,
    y: &DeviceBuffer,
    t: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    model.k.get("inkling2_rms_unnorm").launch(
        ((t * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[y.into(), y.into(), (t * heads).into(), dim.into(), eps.into()],
    )?;
    Ok(())
}

fn rel_bias(
    model: &Model,
    st: &Stream,
    bias: &DeviceBuffer,
    rel: &DeviceBuffer,
    proj: &DeviceBuffer,
    t: usize,
    heads: usize,
    d_rel: usize,
    extent: usize,
) -> Result<()> {
    model.k.get("inkling2_rel_bias").launch(
        (((heads * t * t) as f32 / 256.0).ceil() as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            bias.into(),
            rel.into(),
            proj.into(),
            t.into(),
            heads.into(),
            d_rel.into(),
            extent.into(),
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attn(
    model: &Model,
    st: &Stream,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    bias: &DeviceBuffer,
    sink: &DeviceBuffer,
    gate: &DeviceBuffer,
    t: usize,
    heads: usize,
    nkv: usize,
    dim: usize,
    window: usize,
    scale: f32,
) -> Result<()> {
    model.k.get("inkling2_attn").launch(
        ((t * heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        st,
        &[
            out.into(),
            q.into(),
            k.into(),
            v.into(),
            bias.into(),
            sink.into(),
            gate.into(),
            t.into(),
            heads.into(),
            nkv.into(),
            dim.into(),
            window.into(),
            scale.into(),
        ],
    )?;
    Ok(())
}

fn softcap(model: &Model, st: &Stream, y: &DeviceBuffer, n: usize, cap: f32) -> Result<()> {
    model.k.get("inkling2_softcap").launch(
        (n.div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[y.into(), y.into(), n.into(), cap.into()],
    )?;
    Ok(())
}

/// The graph has no KV cache: every new token re-runs the whole sequence.
pub fn generate(model: &Model, tokens: &[u32], n: usize) -> Result<Vec<u32>> {
    let c = &model.cfg;
    let mut ids = tokens.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = step(model, &ids)?;
        ids.push(next);
        out.push(next);
        if c.eos.contains(&next) {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// The map against the real file: `MALLEUS_INKLING2_FILE=/path/to/Inkling-2-Mini-mls4.malleus
    /// cargo test -p malleus-model inkling2 -- --ignored`. The file is 15 GB, so it is not part of the suite.
    #[test]
    #[ignore = "needs MALLEUS_INKLING2_FILE"]
    fn the_map_covers_the_file() {
        let path = std::env::var("MALLEUS_INKLING2_FILE").expect("MALLEUS_INKLING2_FILE");
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let cfg_json = g.meta_str("openmalleus.hf_config").unwrap();
        let c = Config::from_json(cfg_json, None).unwrap();
        let lines = check(&g, &c).unwrap();
        for l in &lines {
            println!("{l}");
        }
        assert!(lines[0].starts_with("1384 of 1384"), "{}", lines[0]);
        assert_eq!(lines.len(), 4, "{lines:?}");
    }
}
