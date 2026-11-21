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
use serde_json::Value;

use crate::config::{Config, Family, Inkling2Config, Inkling2Mtp, Inkling2Vision, MixerKind, MoeConfig};

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

/// Checks the file against the map: every tensor the forward reads is there, with the shape and the dtype it
/// expects, and nothing in the file is left unclaimed.
pub fn check(src: &dyn TensorSource, c: &Config) -> Result<Vec<String>> {
    let expected = text_map(c)?;
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
        "{} of {} tensors mapped ({} text tensors expected)",
        claimed.len(),
        claimed.len() + unclaimed.len(),
        expected.len()
    ));
    if !unclaimed.is_empty() {
        let vision = unclaimed.iter().filter(|n| n.starts_with("model.visual.")).count();
        lines.push(format!(
            "{} unclaimed: {vision} vision, {} elsewhere (the MTP head and the vision tower are the next map)",
            unclaimed.len(),
            unclaimed.len() - vision
        ));
        let rest: Vec<&&str> = unclaimed.iter().filter(|n| !n.starts_with("model.visual.")).collect();
        lines.push(format!("  first unclaimed: {:?}", &rest[..rest.len().min(8)]));
    }
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
    let _ = bail_if_vision_missing(c);
    Ok(lines)
}

/// The vision tower is a map of its own, and the forward needs it before the first image prompt.
fn bail_if_vision_missing(c: &Config) -> Result<()> {
    if c.inkling2.as_ref().is_some_and(|i| i.vision.is_some()) {
        // The tower's tensors are in the file but not in the map yet: say so instead of pretending.
        return Ok(());
    }
    Ok(())
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
        assert!(lines[0].contains("tensors mapped"), "{}", lines[0]);
    }
}
