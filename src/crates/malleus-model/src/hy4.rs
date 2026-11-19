//! Hy4-flash (`hy_v4_flash`, issue #40): the config, the weights as the file stores them, and the Engram
//! memories read by row from the file.
//!
//! The graph is not the Qwen/Ling one: iHC streams instead of a residual, a gated MLA with a learned sink, two
//! attention shapes with two RoPE bases, a compressed indexer, and a latent MoE. So the weights live in the
//! structs below rather than in `Layer`, and `Model::forward` hands over to this module once it lands.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::DeviceBuffer;
use serde_json::Value;

use crate::config::{Config, Family, Hy4Config, Hy4Engram, MixerKind};
use crate::upload::Uploader;
use crate::weights::Linear;

/// The `text_config` of a Hy4-flash checkpoint, as `Config` plus the Hy4 geometry.
pub(crate) fn config(cfg: &Value, generation_config: Option<&str>) -> Result<Config> {
    let t = &cfg["text_config"];
    let u = |k: &str| -> Result<usize> {
        t[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config: {k}"))
    };
    let f = |k: &str| -> Result<f32> { t[k].as_f64().map(|v| v as f32).with_context(|| format!("config: {k}")) };
    let n_layers = u("num_hidden_layers")?;
    // Three sliding layers for every indexed one, starting with two indexed ones; the checkpoint spells the
    // global kind `deepseek_sparse_attention` and the canonical `indexed_attention` is the same layer.
    let kind: Vec<MixerKind> = t["layer_types"]
        .as_array()
        .context("config: layer_types")?
        .iter()
        .map(|v| match v.as_str() {
            Some("sliding_attention") => Ok(MixerKind::Hy4Sliding),
            Some("deepseek_sparse_attention") | Some("indexed_attention") => Ok(MixerKind::Hy4Indexed),
            other => bail!("config: unknown hy4 layer type {other:?}"),
        })
        .collect::<Result<_>>()?;
    ensure!(kind.len() == n_layers, "config: layer_types length");
    let sparse: Vec<bool> = t["mlp_layer_types"]
        .as_array()
        .context("config: mlp_layer_types")?
        .iter()
        .map(|v| v.as_str() == Some("sparse"))
        .collect();
    ensure!(sparse.len() == n_layers, "config: mlp_layer_types length");

    // The published checkpoint keys the rope parameters by its legacy layer-type names.
    let rope = &t["rope_parameters"];
    let rope_of = |name: &str| -> &Value {
        if rope[name].is_object() {
            &rope[name]
        } else {
            &Value::Null
        }
    };
    let indexed = rope_of("deepseek_sparse_attention");
    let sliding = rope_of("sliding_attention");
    let theta = |q: &Value, k: &str, d: f32| q[k].as_f64().map(|v| v as f32).unwrap_or(d);
    let sec = |q: &Value| -> [usize; 3] {
        let v: Vec<usize> = q["mrope_section"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        if v.len() == 3 { [v[0], v[1], v[2]] } else { [12, 10, 10] }
    };
    let mrope_section = sec(indexed);

    let engram = match t["engram_layer_ids"].as_array() {
        Some(ids) if !ids.is_empty() => {
            let layers: Vec<usize> = ids
                .iter()
                .map(|v| Ok(v.as_u64().context("config: engram_layer_ids")? as usize - 1))
                .collect::<Result<_>>()?;
            let sizes = &t["engram_layer_head_vocab_sizes"];
            let mut head_vocab_sizes = Vec::with_capacity(layers.len());
            for (i, &l) in layers.iter().enumerate() {
                let one_based = (l + 1).to_string();
                let orders = sizes[&one_based]
                    .as_array()
                    .with_context(|| format!("config: engram_layer_head_vocab_sizes[{one_based}]"))?;
                let mut per_order = Vec::with_capacity(orders.len());
                for order in orders {
                    per_order.push(
                        order
                            .as_array()
                            .context("config: engram head vocab sizes")?
                            .iter()
                            .map(|v| v.as_u64().unwrap_or(0) as usize)
                            .collect::<Vec<_>>(),
                    );
                }
                ensure!(
                    ids.len() > i && per_order.len() == u("engram_max_ngram_size")? - 1,
                    "config: engram orders for layer {one_based}"
                );
                head_vocab_sizes.push(per_order);
            }
            Some(Hy4Engram {
                layers,
                heads: u("engram_n_head_per_ngram")?,
                embed: u("engram_n_embed_per_ngram")?,
                max_ngram: u("engram_max_ngram_size")?,
                pad: u("engram_pad_id")? as u32,
                // The class defaults it to the tokenizer's vocabulary size, and the checkpoint leaves it out.
                lookup: t["engram_lookup_table_size"].as_u64().unwrap_or(120818) as usize,
                seed: t["engram_seed"].as_u64().unwrap_or(0),
                fp8: t["engram_fp8_embedding"].as_bool().unwrap_or(true),
                head_vocab_sizes,
            })
        }
        _ => None,
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

    let heads = u("num_attention_heads")?;
    let hy4 = Hy4Config {
        hc_mult: u("hc_mult")?,
        hc_magnitude: f("hc_magnitude")?,
        hc_eps: f("hc_eps")?,
        sink: t["learnable_sink"].as_bool().unwrap_or(false),
        q_lora: u("q_lora_rank")?,
        kv_lora: u("kv_lora_rank")?,
        heads,
        nope: u("qk_nope_head_dim")?,
        rope: u("qk_rope_head_dim")?,
        v: u("v_head_dim")?,
        swa_heads: u("swa_num_attention_heads")?,
        swa_nope: u("swa_qk_nope_head_dim")?,
        swa_rope: u("swa_qk_rope_head_dim")?,
        swa_v: u("swa_v_head_dim")?,
        window: u("sliding_window")?,
        theta_indexed: theta(indexed, "rope_theta", 1e7),
        theta_sliding: theta(sliding, "rope_theta", 1e4),
        mrope_section,
        index_pool: u("compressed_indexer_compress_ratio")?,
        index_topk: u("index_topk")?,
        index_heads: u("index_n_heads")?,
        index_dim: u("index_head_dim")?,
        latent: u("routed_expert_hidden_size")?,
        swiglu_limit: f("swiglu_limit")?,
        routed_scaling: f("routed_scaling_factor")?,
        sparse,
        engram,
    };
    let c = Config {
        hidden: u("hidden_size")?,
        n_layers,
        // The dense FFN of the first layer; the MoE width is `moe_intermediate_size` and the latent one.
        inter: u("intermediate_size")?,
        vocab: u("vocab_size")?,
        n_heads: heads,
        n_kv: heads,
        // Hy4 expands the latent to `nope + v` per head; there is no single per-head KV width.
        head_dim: hy4.nope + hy4.rope,
        lin_nk: 0,
        lin_nv: 0,
        lin_dk: 0,
        lin_dv: 0,
        conv_k: 0,
        eps: f("rms_norm_eps")?,
        rope_theta: hy4.theta_indexed,
        rot_dim: hy4.rope,
        mrope_section,
        full_attn: kind.iter().map(|k| *k == MixerKind::Hy4Indexed).collect(),
        eos,
        moe: None,
        hc: None,
        ple: None,
        indexer: None,
        gate_sigmoid: false,
        family: Family::Hy4,
        kind,
        mla: None,
        kda_lower_bound: 0.0,
        // `HYV4RMSNorm` multiplies by the weight, as Ling's does; the loader folds the one.
        norm_one_plus_w: false,
        mrope: true,
        dense_ffn: 1,
        swiglu: None,
        hy4: Some(hy4),
    };
    c.validate()?;
    Ok(c)
}

/// The geometry the kernels will require of a Hy4 checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let h = c.hy4.as_ref().context("config: hy4 geometry")?;
    ensure!(c.family == Family::Hy4 && c.n_layers == c.kind.len(), "bad hy4 layers");
    ensure!(h.hc_mult >= 2, "hy4 needs at least two streams");
    ensure!(h.rope == 64 && h.swa_rope == h.rope, "hy4 rotates 64 dims per head");
    ensure!(
        h.mrope_section.iter().sum::<usize>() * 2 == h.rope,
        "hy4 M-RoPE sections must cover half the rotary width"
    );
    ensure!(
        h.nope + h.rope == c.head_dim && h.swa_nope + h.swa_rope == c.head_dim,
        "hy4 head geometry: both layer types carry the same qk width"
    );
    ensure!(
        h.q_lora > 0 && h.kv_lora.is_multiple_of(64) && h.latent.is_multiple_of(128),
        "hy4 latent widths"
    );
    ensure!(
        h.index_pool > 0 && h.index_topk % h.index_pool == 0 && h.index_heads > 0,
        "hy4 compressed indexer geometry"
    );
    ensure!(h.sparse.len() == c.n_layers, "hy4 mlp_layer_types length");
    ensure!(
        c.kind.contains(&MixerKind::Hy4Indexed) && c.kind.contains(&MixerKind::Hy4Sliding),
        "hy4 needs both indexed and sliding layers"
    );
    if let Some(e) = &h.engram {
        ensure!(
            e.layers.iter().all(|&l| l < c.n_layers)
                && e.head_vocab_sizes.len() == e.layers.len()
                && e.max_ngram >= 2
                && e.embed.is_multiple_of(e.heads),
            "hy4 engram geometry"
        );
        ensure!(e.pad < e.lookup as u32, "hy4 engram pad id must be inside the map");
    }
    Ok(())
}

/// The iHC mixing of one block: `[2 * streams, streams * hidden]` fp32, its bias and its two scales.
#[allow(dead_code)] // read by the text forward, which lands after this loader (issue #40)
pub(crate) struct Ihc {
    pub(crate) f: DeviceBuffer,
    pub(crate) base: DeviceBuffer,
    pub(crate) scale: DeviceBuffer,
}

/// The compressed indexer of a global layer: the query projection reads the q_a latent, the token keys the
/// hidden state, and `head_weight` (16 rows) weights the heads into the score.
#[allow(dead_code)]
pub(crate) struct Indexer {
    pub(crate) query: Linear,
    pub(crate) token_key: Linear,
    pub(crate) token_key_norm: DeviceBuffer,
    /// fp32 `[index_dim, index_dim]`: the pooled key's value projection, applied on the host layout the kernel
    /// wants.
    pub(crate) value: DeviceBuffer,
    pub(crate) gate: DeviceBuffer,
    pub(crate) key_norm: DeviceBuffer,
    pub(crate) head_weight: Linear,
    /// fp32 `[index_dim, pool]`: the position embedding inside one pooled group.
    pub(crate) intra_ape: DeviceBuffer,
}

/// One gated-MLA layer. `heads`/`nope`/`rope`/`v` are the shape this layer type uses (the sliding layers are
/// twice as wide), so the loader resolves them from `MixerKind`.
#[allow(dead_code)]
pub(crate) struct Attn {
    pub(crate) heads: usize,
    pub(crate) nope: usize,
    pub(crate) rope: usize,
    pub(crate) v: usize,
    pub(crate) q_a: Linear,
    pub(crate) q_b: Linear,
    pub(crate) kv_a: Linear,
    pub(crate) kv_b: Linear,
    pub(crate) o: Linear,
    pub(crate) gate: Linear,
    pub(crate) q_a_norm: DeviceBuffer,
    pub(crate) kv_a_norm: DeviceBuffer,
    pub(crate) post_norm: DeviceBuffer,
    /// fp32 `[heads]`: the learned sink, an extra logit per head.
    pub(crate) sink: Option<DeviceBuffer>,
    pub(crate) indexer: Option<Indexer>,
}

/// The latent MoE: `pre` maps the hidden state into the latent, the routed experts run there, `up` maps back, and
/// the shared expert sees the full width. `combine_output_norm` closes the block.
#[allow(dead_code)]
pub(crate) struct Moe {
    /// bf16 `[experts, hidden]`: the router.
    pub(crate) router: DeviceBuffer,
    /// fp32 `[experts]`: the selection bias (`e_score_correction_bias`).
    pub(crate) bias: DeviceBuffer,
    pub(crate) pre: Linear,
    pub(crate) up: Linear,
    /// MLS4 `[experts, 2 * inter, latent]` and its group scales.
    pub(crate) gu: (DeviceBuffer, DeviceBuffer),
    /// MLS4 `[experts, latent, inter]` and its group scales.
    pub(crate) down: (DeviceBuffer, DeviceBuffer),
    pub(crate) shared: (Linear, Linear, Linear),
    pub(crate) norm: DeviceBuffer,
}

#[allow(dead_code, clippy::large_enum_variant)]
pub(crate) enum Ffn {
    /// The first layer's dense FFN, with its own sandwich norm (`mlp.post_fc2_norm`).
    Dense {
        gate: Linear,
        up: Linear,
        down: Linear,
        norm: DeviceBuffer,
    },
    Latent(Moe),
}

/// The Engram memory of one layer. The 32 GB table stays in the file and is read by row (`Table`); everything
/// else is a projection on the device.
#[allow(dead_code)]
pub(crate) struct Engram {
    pub(crate) key: Vec<Linear>,
    pub(crate) value: Linear,
    pub(crate) norm_key: Vec<DeviceBuffer>,
    pub(crate) norm_query: Vec<DeviceBuffer>,
    pub(crate) head_norms: Vec<DeviceBuffer>,
    pub(crate) pre_gamma: DeviceBuffer,
    pub(crate) scales: Vec<DeviceBuffer>,
    pub(crate) table: Table,
    /// int32 `[lookup]`: the frozen normalized-vocabulary map the hashes go through.
    pub(crate) lookup: Vec<i32>,
}

/// One Engram row table as the file stores it: `rows` rows of `k` 4-bit codes (bias 8) with an f16 scale each,
/// read on the host like the n-gram table of Qwen3.8-Flash-Next.
pub(crate) struct Table {
    file: std::sync::Arc<malleus_gguf::TensorFile>,
    pub(crate) rows: usize,
    pub(crate) k: usize,
}

#[allow(dead_code)]
impl Table {
    fn open(src: &dyn TensorSource, name: &str) -> Result<Self> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::RowI4 && info.shape.len() == 2,
            "{name}: expected RowI4 [rows, k]"
        );
        let file = src
            .open_tensor(name)
            .context("engram table: no file handle")?
            .reopened();
        if let Some((f, _, _)) = file.range(0, 0) {
            malleus_hip::file_hint::random(f);
        }
        Ok(Table {
            file: std::sync::Arc::new(file),
            rows: info.shape[0],
            k: info.shape[1],
        })
    }

    /// Dequantized rows `ids` as `[ids, k]` f32.
    pub(crate) fn rows(&self, ids: &[usize]) -> Result<Vec<f32>> {
        let rb = self.k / 2;
        let mut buf = vec![0u8; rb];
        let mut out = Vec::with_capacity(ids.len() * self.k);
        for &id in ids {
            ensure!(id < self.rows, "engram row {id} out of range");
            self.file.read_at(id * rb, &mut buf)?;
            for &b in &buf {
                out.push((b & 0xF) as f32 - 8.0);
                out.push((b >> 4) as f32 - 8.0);
            }
        }
        Ok(out)
    }

    /// The f16 scale of every row, as the file stores it next to the codes? No: the scales are a tensor of their
    /// own, loaded on the device by [`Weights::load`] when the forward needs them; here only the codes.
    fn bytes(&self) -> usize {
        self.rows * self.k / 2
    }
}

#[allow(dead_code)]
pub(crate) struct Layer {
    pub(crate) attn_hc: Ihc,
    pub(crate) ffn_hc: Ihc,
    pub(crate) in_norm: DeviceBuffer,
    pub(crate) post_norm: DeviceBuffer,
    pub(crate) attn: Attn,
    pub(crate) ffn: Ffn,
    pub(crate) engram: Option<Engram>,
}

/// Everything the text graph reads, loaded from the converted file. The final norm and the head (`hc_head`) are
/// loaded with the layers so that `forward` has one thing to take.
#[allow(dead_code)]
pub(crate) struct Weights {
    pub(crate) layers: Vec<Layer>,
    pub(crate) engram_scales: Vec<DeviceBuffer>,
    pub(crate) norm: DeviceBuffer,
    pub(crate) head: Ihc,
    pub(crate) bytes: usize,
}

impl Weights {
    pub(crate) fn load(
        src: &dyn TensorSource,
        c: &Config,
        up: &mut Uploader,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let h = c.hy4.as_ref().context("no hy4 config")?;
        let before = up.bytes;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            progress(i, c.n_layers);
            layers.push(load_layer(up, src, c, h, i)?);
        }
        let mut engram_scales = Vec::new();
        for (li, l) in layers.iter().enumerate() {
            if let Some(e) = &l.engram {
                // One f16 scale per row of the 4-bit table.
                engram_scales.push(up.typed(
                    src,
                    &format!("model.layers.{li}.engram_layer.multi_head_embedding.embedding.weight.scales"),
                    DType::F16,
                    &[e.table.rows],
                )?);
            }
        }
        let norm = up.norm(src, "model.norm.weight", c.hidden, false)?;
        let head = Ihc {
            f: up.typed(
                src,
                "model.hc_head.hc_head_fn",
                DType::F32,
                &[h.hc_mult, h.hc_mult * c.hidden],
            )?,
            base: up.typed(src, "model.hc_head.hc_head_base", DType::F32, &[h.hc_mult])?,
            scale: up.typed(src, "model.hc_head.hc_head_scale", DType::F32, &[1])?,
        };
        Ok(Weights {
            layers,
            engram_scales,
            norm,
            head,
            bytes: up.bytes - before,
        })
    }
}

fn load_layer(up: &mut Uploader, src: &dyn TensorSource, c: &Config, h: &Hy4Config, i: usize) -> Result<Layer> {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let hc = |up: &mut Uploader, which: &str| -> Result<Ihc> {
        Ok(Ihc {
            f: up.typed(
                src,
                &p(&format!("{which}.hc_fn")),
                DType::F32,
                &[2 * h.hc_mult, h.hc_mult * c.hidden],
            )?,
            base: up.typed(src, &p(&format!("{which}.hc_base")), DType::F32, &[2 * h.hc_mult])?,
            scale: up.typed(src, &p(&format!("{which}.hc_scale")), DType::F32, &[2])?,
        })
    };
    let (heads, nope, rope, v) = match c.kind[i] {
        MixerKind::Hy4Sliding => (h.swa_heads, h.swa_nope, h.swa_rope, h.swa_v),
        _ => (h.heads, h.nope, h.rope, h.v),
    };
    let qd = heads * (nope + rope);
    let kvd = heads * (nope + v);
    let od = heads * v;
    let indexer = match c.kind[i] {
        MixerKind::Hy4Indexed => {
            let ix = |s: &str| p(&format!("self_attn.compressed_indexer.{s}"));
            Some(Indexer {
                query: up.linear(src, &ix("query_projection"), h.index_heads * h.index_dim, h.q_lora)?,
                token_key: up.linear(src, &ix("token_key_projection"), h.index_dim, c.hidden)?,
                token_key_norm: up.tensor(src, &ix("token_key_norm.weight"), &[h.index_dim])?,
                value: up.typed(
                    src,
                    &ix("key_value_projection.weight"),
                    DType::F32,
                    &[h.index_dim, h.index_dim],
                )?,
                gate: up.typed(
                    src,
                    &ix("key_gate_projection.weight"),
                    DType::F32,
                    &[h.index_dim, h.index_dim],
                )?,
                key_norm: up.typed(src, &ix("key_norm.weight"), DType::F32, &[h.index_dim])?,
                head_weight: up.linear(src, &ix("head_weight_projection"), h.index_heads, c.hidden)?,
                intra_ape: up.typed(src, &ix("intra_group_ape"), DType::F32, &[h.index_pool, h.index_dim])?,
            })
        }
        _ => None,
    };
    let attn = Attn {
        heads,
        nope,
        rope,
        v,
        q_a: up.linear(src, &p("self_attn.q_a_proj"), h.q_lora, c.hidden)?,
        q_b: up.linear(src, &p("self_attn.q_b_proj"), qd, h.q_lora)?,
        kv_a: up.linear(src, &p("self_attn.kv_a_proj_with_mqa"), h.kv_lora + rope, c.hidden)?,
        kv_b: up.linear(src, &p("self_attn.kv_b_proj"), kvd, h.kv_lora)?,
        o: up.linear(src, &p("self_attn.o_proj"), c.hidden, od)?,
        gate: up.linear(src, &p("self_attn.linear_gate"), od, c.hidden)?,
        q_a_norm: up.norm(src, &p("self_attn.q_a_layernorm.weight"), h.q_lora, false)?,
        kv_a_norm: up.norm(src, &p("self_attn.kv_a_layernorm.weight"), h.kv_lora, false)?,
        post_norm: up.norm(src, &p("self_attn.post_proj_norm.weight"), c.hidden, false)?,
        sink: match h.sink {
            true => Some(up.typed(src, &p("self_attn.learnable_sink_param"), DType::F32, &[heads])?),
            false => None,
        },
        indexer,
    };
    let ffn = if h.sparse[i] {
        let e = h_experts(src)?;
        let inter = moe_inter_width(src, &p("mlp"))?;
        let (g, s) = stack_experts(src, up, &p("mlp"), e, h.latent, inter)?;
        Ffn::Latent(Moe {
            router: up.tensor(src, &p("mlp.gate.weight"), &[e, c.hidden])?,
            bias: up.typed(src, &p("mlp.gate.e_score_correction_bias"), DType::F32, &[e])?,
            pre: up.linear(src, &p("mlp.routed_expert_down_proj"), h.latent, c.hidden)?,
            up: up.linear(src, &p("mlp.routed_expert_up_proj"), c.hidden, h.latent)?,
            gu: g,
            down: s,
            shared: (
                up.linear(src, &p("mlp.shared_experts.gate_proj"), inter, c.hidden)?,
                up.linear(src, &p("mlp.shared_experts.up_proj"), inter, c.hidden)?,
                up.linear(src, &p("mlp.shared_experts.down_proj"), c.hidden, inter)?,
            ),
            norm: up.norm(src, &p("mlp.combine_output_norm.weight"), c.hidden, false)?,
        })
    } else {
        Ffn::Dense {
            gate: up.linear(src, &p("mlp.gate_proj"), c.inter, c.hidden)?,
            up: up.linear(src, &p("mlp.up_proj"), c.inter, c.hidden)?,
            down: up.linear(src, &p("mlp.down_proj"), c.hidden, c.inter)?,
            norm: up.norm(src, &p("mlp.post_fc2_norm.weight"), c.hidden, false)?,
        }
    };
    let engram = match &h.engram {
        Some(e) if e.layers.contains(&i) => {
            let k = e.layers.iter().position(|&l| l == i).unwrap();
            Some(load_engram(up, src, c, h, e, i, &e.head_vocab_sizes[k])?)
        }
        _ => None,
    };
    Ok(Layer {
        attn_hc: hc(up, "hc_attn_layer.hc_pre")?,
        ffn_hc: hc(up, "hc_mlp_layer.hc_pre")?,
        in_norm: up.norm(src, &p("input_layernorm.weight"), c.hidden, false)?,
        post_norm: up.norm(src, &p("post_attention_layernorm.weight"), c.hidden, false)?,
        attn,
        ffn,
        engram,
    })
}

fn h_experts(src: &dyn TensorSource) -> Result<usize> {
    let info = src
        .info("model.layers.1.mlp.experts.down_proj")
        .context("hy4 experts")?;
    Ok(info.shape[0])
}

fn moe_inter_width(src: &dyn TensorSource, base: &str) -> Result<usize> {
    let info = src
        .info(&format!("{base}.shared_experts.gate_proj.weight"))
        .context("hy4 shared expert")?;
    info.shape.first().copied().context("hy4 shared expert shape")
}

/// The routed experts: one MLS4 matrix of `[experts, 2 * inter, latent]` and one of `[experts, latent, inter]`.
fn stack_experts(
    src: &dyn TensorSource,
    up: &mut Uploader,
    base: &str,
    experts: usize,
    latent: usize,
    inter: usize,
) -> Result<((DeviceBuffer, DeviceBuffer), (DeviceBuffer, DeviceBuffer))> {
    let mut stack = |name: &str, n: usize, k: usize| -> Result<(DeviceBuffer, DeviceBuffer)> {
        let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            info.dtype == DType::Mls4,
            "{name}: hy4 routed experts must be MLS4, got {:?}",
            info.dtype
        );
        let g = k / malleus_gguf::quant::BLOCK;
        let mut q = DeviceBuffer::alloc(experts * n * k / 2)?;
        let mut sc = DeviceBuffer::alloc(experts * n * g * 2)?;
        up.into(src, name, DType::Mls4, &[experts, n, k], &mut q, 0)?;
        up.into(src, &format!("{name}.scales"), DType::F16, &[experts, n, g], &mut sc, 0)?;
        Ok((q, sc))
    };
    Ok((
        stack(&format!("{base}.experts.gate_up_proj"), 2 * inter, latent)?,
        stack(&format!("{base}.experts.down_proj"), latent, inter)?,
    ))
}

fn load_engram(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    h: &Hy4Config,
    e: &Hy4Engram,
    layer: usize,
    sizes: &[Vec<usize>],
) -> Result<Engram> {
    let p = |s: &str| format!("model.layers.{layer}.engram_layer.{s}");
    let width = (e.max_ngram - 1) * e.embed;
    ensure!(
        sizes.iter().flatten().count() == (e.max_ngram - 1) * e.heads,
        "engram heads for layer {layer}"
    );
    let mut key = Vec::with_capacity(h.hc_mult);
    let mut norm_key = Vec::with_capacity(h.hc_mult);
    let mut norm_query = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        key.push(up.linear(src, &p(&format!("key_projs.{s}")), c.hidden, width)?);
        norm_key.push(up.norm(src, &p(&format!("norm_key.{s}.weight")), c.hidden, false)?);
        norm_query.push(up.norm(src, &p(&format!("norm_query.{s}.weight")), c.hidden, false)?);
    }
    let mut head_norms = Vec::with_capacity(sizes.iter().flatten().count());
    for h in 0..sizes.iter().flatten().count() {
        head_norms.push(up.norm(
            src,
            &p(&format!("lookup_per_head_norms.{h}.weight")),
            e.embed / e.heads,
            false,
        )?);
    }
    let mut scales = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        scales.push(up.norm(src, &p(&format!("unified_per_hc_scales.{s}")), c.hidden, false)?);
    }
    let lookup: Vec<i32> = up
        .typed(src, &p("lookup_table"), DType::I32, &[e.lookup])?
        .download(e.lookup)?;
    Ok(Engram {
        key,
        value: up.linear(src, &p("value_proj"), c.hidden, width)?,
        norm_key,
        norm_query,
        head_norms,
        pre_gamma: up.norm(src, &p("unified_pre_gamma"), c.hidden, false)?,
        scales,
        table: Table::open(src, &p("multi_head_embedding.embedding.weight"))?,
        lookup,
    })
}

/// KV bytes of one session: the latent and the rotated dims per layer, fp16, plus the sliding window's own copy.
pub(crate) fn session_bytes(c: &Config, ctx: usize) -> u64 {
    let Some(h) = &c.hy4 else { return 0 };
    let per_token = h.kv_lora as u64 + h.rope as u64;
    let layers = c.n_layers as u64;
    (ctx as u64).div_ceil(32) * 32 * per_token * 2 * layers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the published checkpoint has, cut down to two layers of each kind and two n-gram heads.
    const HY4: &str = r#"{
      "model_type": "hy_v4_flash_text",
      "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 1024,
      "intermediate_size": 512, "rms_norm_eps": 1e-5,
      "num_attention_heads": 2, "q_lora_rank": 128, "kv_lora_rank": 512,
      "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
      "swa_num_attention_heads": 4, "swa_qk_nope_head_dim": 192,
      "swa_qk_rope_head_dim": 64, "swa_v_head_dim": 256, "sliding_window": 512,
      "layer_types": ["deepseek_sparse_attention", "sliding_attention",
                      "sliding_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
      "hc_mult": 4, "hc_magnitude": 2.0, "hc_eps": 1e-6, "learnable_sink": true,
      "routed_expert_hidden_size": 256, "n_routed_experts": 4, "num_experts_per_tok": 2,
      "swiglu_limit": 10.0, "routed_scaling_factor": 4.0,
      "compressed_indexer_compress_ratio": 4, "index_topk": 64,
      "index_n_heads": 2, "index_head_dim": 128,
      "engram_layer_ids": [2], "engram_max_ngram_size": 3, "engram_n_head_per_ngram": 2,
      "engram_n_embed_per_ngram": 64, "engram_pad_id": 7, "engram_lookup_table_size": 999,
      "engram_seed": 0, "engram_fp8_embedding": true,
      "engram_layer_head_vocab_sizes": {"2": [[101, 103], [107, 109]]},
      "rope_parameters": {
        "deepseek_sparse_attention": {"rope_theta": 10000000.0, "mrope_section": [12, 10, 10]},
        "sliding_attention": {"rope_theta": 10000.0, "mrope_section": [12, 10, 10]}
      }
    }"#;

    fn parse(json: &str) -> Result<Config> {
        let text: Value = serde_json::from_str(json)?;
        config(&serde_json::json!({ "text_config": text }), None)
    }

    #[test]
    fn geometry_and_engram() {
        let c = parse(HY4).unwrap();
        assert_eq!(c.family, Family::Hy4);
        assert_eq!((c.hidden, c.n_layers, c.vocab), (256, 4, 1024));
        assert_eq!(c.kind[0], MixerKind::Hy4Indexed);
        assert_eq!(c.kind[1], MixerKind::Hy4Sliding);
        assert_eq!(c.full_attn, vec![true, false, false, true]);
        // Both layer types carry the same qk width; the sliding one is twice as wide in v.
        assert_eq!(c.head_dim, 256);
        let h = c.hy4.as_ref().unwrap();
        assert_eq!((h.heads, h.swa_heads), (2, 4));
        assert_eq!((h.theta_indexed, h.theta_sliding), (1e7, 1e4));
        assert_eq!((h.hc_mult, h.hc_magnitude), (4, 2.0));
        assert!(h.sink);
        assert_eq!(h.sparse, vec![false, true, true, true]);
        let e = h.engram.as_ref().unwrap();
        assert_eq!(e.layers, vec![1]);
        assert_eq!(e.head_vocab_sizes[0], vec![vec![101, 103], vec![107, 109]]);
        assert_eq!((e.max_ngram, e.lookup, e.pad), (3, 999, 7));
    }

    #[test]
    fn a_bad_mrope_section_is_rejected() {
        let bad = HY4.replace(r#""mrope_section": [12, 10, 10]}"#, r#""mrope_section": [1, 1, 1]}"#);
        assert!(parse(&bad).is_err());
    }
}
