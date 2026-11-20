//! Hy4-flash (`hy_v4_flash`, issue #40): the config, the weights as the file stores them, and the Engram
//! memories read by row from the file.
//!
//! The graph is not the Qwen/Ling one: iHC streams instead of a residual, a gated MLA with a learned sink, two
//! attention shapes with two RoPE bases, a compressed indexer, and a latent MoE. So the weights live in the
//! structs below rather than in `Layer`, and `Model::forward` hands over to this module once it lands.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{Arg, DeviceBuffer};
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

    let vision = match cfg["vision_config"]["hidden_size"].as_u64() {
        Some(hidden) => {
            let v = &cfg["vision_config"];
            Some(crate::config::Hy4Vision {
                hidden: hidden as usize,
                inter: v["intermediate_size"]
                    .as_u64()
                    .context("vision_config: intermediate_size")? as usize,
                heads: v["num_attention_heads"].as_u64().context("vision_config: heads")? as usize,
                patch: v["patch_size"].as_u64().context("vision_config: patch_size")? as usize,
                edge: (v["max_image_size"].as_u64().unwrap_or(2048) / v["patch_size"].as_u64().unwrap_or(16)) as usize,
                merge: v["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
                out: v["text_hidden_size"]
                    .as_u64()
                    .context("vision_config: text_hidden_size")? as usize,
            })
        }
        None => None,
    };
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
        experts: u("n_routed_experts")?,
        top_k: u("num_experts_per_tok")?,
        moe_inter: u("moe_intermediate_size")?,
        swiglu_limit: f("swiglu_limit")?,
        routed_scaling: f("routed_scaling_factor")?,
        sparse,
        engram,
        vision,
        image_token: cfg["image_token_id"].as_u64().unwrap_or(0) as u32,
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
    /// fp32 `[heads * (nope + v), kv_lora]`, dequantized at load: the absorbed kernels read it as a matrix.
    pub(crate) kv_b: DeviceBuffer,
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
    /// bf16 `[experts, hidden]`: the router, as a `Linear` so the GEMM dispatch can read it.
    pub(crate) router: Linear,
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
    pub(crate) norm_key: Vec<Vec<f32>>,
    pub(crate) norm_query: Vec<Vec<f32>>,
    /// `[heads * dim]`: the per-hash-head norm weights, row-major by head.
    pub(crate) head_norms: Vec<f32>,
    pub(crate) pre_gamma: Vec<f32>,
    /// Per stream, the branch scale applied to the memory's value.
    pub(crate) scales: Vec<Vec<f32>>,
    /// The projections carry a bias, which `Linear` does not hold: added on the host after the GEMM.
    pub(crate) value_bias: Vec<f32>,
    pub(crate) key_bias: Vec<Vec<f32>>,
    pub(crate) head_vocab: Vec<usize>,
    /// Hash heads per n-gram order, the width of one head's row, the pad id and whether the rows round-trip
    /// through fp8.
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) pad: u32,
    pub(crate) fp8: bool,
    pub(crate) table: Table,
    /// int32 `[lookup]`: the frozen normalized-vocabulary map the hashes go through.
    pub(crate) lookup: Vec<i32>,
    /// The odd multipliers `hash_ngrams` mixes the n-gram with, one per order: drawn by the checkpoint's seed
    /// (`engram_seed + 10007 * layer_id`, numpy PCG64) from the loaded map, so they are a property of the file.
    pub(crate) hash_mult: [i64; 3],
}

/// One Engram row table as the file stores it: `rows` rows of `k` 4-bit codes (bias 8) with an f16 scale each,
/// read on the host like the n-gram table of Qwen3.8-Flash-Next.
pub(crate) struct Table {
    file: std::sync::Arc<malleus_gguf::TensorFile>,
    scales: std::sync::Arc<malleus_gguf::TensorFile>,
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
        // One f16 scale per row, in its own tensor.
        let scales = src
            .open_tensor(&format!("{name}.scales"))
            .context("engram table: no scale handle")?
            .reopened();
        if let Some((f, _, _)) = file.range(0, 0) {
            malleus_hip::file_hint::random(f);
        }
        Ok(Table {
            file: std::sync::Arc::new(file),
            scales: std::sync::Arc::new(scales),
            rows: info.shape[0],
            k: info.shape[1],
        })
    }

    /// Dequantized rows `ids` as `[ids, k]` f32: the 4-bit codes (bias 8) times the f16 scale of the row.
    pub(crate) fn rows(&self, ids: &[usize]) -> Result<Vec<f32>> {
        let rb = self.k / 2;
        let mut buf = vec![0u8; rb];
        let mut sb = [0u8; 2];
        let mut out = Vec::with_capacity(ids.len() * self.k);
        for &id in ids {
            ensure!(id < self.rows, "engram row {id} out of range");
            self.file.read_at(id * rb, &mut buf)?;
            self.scales.read_at(id * 2, &mut sb)?;
            let s = half::f16::from_le_bytes(sb).to_f32();
            for &b in &buf {
                out.push(((b & 0xF) as f32 - 8.0) * s);
                out.push(((b >> 4) as f32 - 8.0) * s);
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
        seed: u64,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Self> {
        let h = c.hy4.as_ref().context("no hy4 config")?;
        let before = up.bytes;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            progress(i, c.n_layers);
            layers.push(load_layer(up, src, c, h, i, seed)?);
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

fn load_layer(
    up: &mut Uploader,
    src: &dyn TensorSource,
    c: &Config,
    h: &Hy4Config,
    i: usize,
    seed: u64,
) -> Result<Layer> {
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
        kv_b: up.linear_f32(src, &p("self_attn.kv_b_proj"), kvd, h.kv_lora, seed)?,
        o: up.linear(src, &p("self_attn.o_proj"), c.hidden, od)?,
        gate: up.linear(src, &p("self_attn.linear_gate"), od, c.hidden)?,
        q_a_norm: up.norm(src, &p("self_attn.q_a_layernorm.weight"), h.q_lora, false)?,
        // `mla_cache_write` multiplies by this weight directly and reads it as f32, so it goes up raw in f32.
        kv_a_norm: up.norm_f32(src, &p("self_attn.kv_a_layernorm.weight"), h.kv_lora)?,
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
            router: Linear {
                w: crate::weights::Weights::Bf16(up.tensor(src, &p("mlp.gate.weight"), &[e, c.hidden])?),
                prefill: None,
                n: e,
                k: c.hidden,
            },
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
            // `combine_output_norm` is applied with the weight as stored (no fold): the kernel's one-plus is off.
            norm: up.norm(src, &p("mlp.combine_output_norm.weight"), c.hidden, true)?,
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

/// A bf16 tensor as a host vector (the Engram norms are applied on the host, where the memory is read).
fn bf16_vec(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>> {
    let raw = src.read_all(name).with_context(|| format!("missing tensor {name}"))?;
    let v: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect();
    ensure!(v.len() == n, "{name}: expected {n} values, got {}", v.len());
    Ok(v)
}

/// The hash multipliers `HYV4FlashEngramLayer.draw_hash_multipliers` draws for each Engram layer:
/// numpy's PCG64 seeded with `engram_seed + 10007 * layer_id` (0), `engram_max_ngram_size` odd draws bounded by
/// `i64::MAX / compressed_vocab / 2`. They are a property of the file (the memory is only readable with them), so
/// they are transcribed instead of depending on numpy: for the layers of this checkpoint, the compressed
/// vocabulary is 92620 and the drawn values are these.
fn hash_multipliers(layer_1based: usize) -> [i64; 3] {
    match layer_1based {
        2 => [81853575799317, 39690085502883, 24229915018165],
        12 => [19457948045719, 44729556510419, 84779352922315],
        22 => [72974230057307, 21530602121221, 55998057243097],
        32 => [29128509562787, 16745155128523, 22474588302733],
        // A checkpoint with Engram on other layers needs its seed drawn again (see the comment above).
        _ => unreachable!("hash multipliers for engram layer {layer_1based}"),
    }
}

/// fp8 (e4m3) round trip of one vector, as the memory was trained to be read: scale by the vector's amax, round to
/// the e4m3 grid, scale back.
fn fp8_round_trip(row: &mut [f32]) {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    let scale = amax / 448.0;
    for v in row.iter_mut() {
        *v = e4m3(*v / scale) * scale;
    }
}

/// Round to the nearest value of the e4m3 grid (1 sign, 4 exponent, 3 mantissa; subnormals below 2^-6).
fn e4m3(x: f32) -> f32 {
    const MAX: f32 = 448.0;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let a = x.abs().min(MAX);
    if !a.is_finite() {
        return sign * MAX;
    }
    if a < 0.015625 {
        // Subnormal: multiples of 2^-9.
        return sign * ((a / 0.001953125).round() * 0.001953125);
    }
    let e = a.log2().floor();
    let step = 2f32.powf(e - 3.0);
    sign * ((a / step).round() * step).min(MAX)
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
    let mut key_bias = Vec::with_capacity(h.hc_mult);
    let mut norm_key = Vec::with_capacity(h.hc_mult);
    let mut norm_query = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        key.push(up.linear(src, &p(&format!("key_projs.{s}")), c.hidden, width)?);
        key_bias.push(bf16_vec(src, &p(&format!("key_projs.{s}.bias")), c.hidden)?);
        norm_key.push(bf16_vec(src, &p(&format!("norm_key.{s}.weight")), c.hidden)?);
        norm_query.push(bf16_vec(src, &p(&format!("norm_query.{s}.weight")), c.hidden)?);
    }
    let nheads = sizes.iter().flatten().count();
    let mut head_norms = Vec::with_capacity(nheads * (e.embed / e.heads));
    for h in 0..nheads {
        head_norms.extend(bf16_vec(
            src,
            &p(&format!("lookup_per_head_norms.{h}.weight")),
            e.embed / e.heads,
        )?);
    }
    let mut scales = Vec::with_capacity(h.hc_mult);
    for s in 0..h.hc_mult {
        scales.push(bf16_vec(src, &p(&format!("unified_per_hc_scales.{s}")), c.hidden)?);
    }
    let lookup: Vec<i32> = up
        .typed(src, &p("lookup_table"), DType::I32, &[e.lookup])?
        .download(e.lookup)?;
    let head_vocab: Vec<usize> = sizes.iter().flatten().copied().collect();
    Ok(Engram {
        key,
        value: up.linear(src, &p("value_proj"), c.hidden, width)?,
        norm_key,
        norm_query,
        head_norms,
        pre_gamma: bf16_vec(src, &p("unified_pre_gamma"), c.hidden)?,
        scales,
        value_bias: bf16_vec(src, &p("value_proj.bias"), c.hidden)?,
        key_bias,
        head_vocab,
        heads: e.heads,
        dim: e.embed / e.heads,
        pad: e.pad,
        fp8: e.fp8,
        hash_mult: hash_multipliers(layer + 1),
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

/// An RMSNorm applied with the weight as stored (`w * x`, no `1 +`): what every Hy4 norm is.
fn rmsnorm_raw(
    model: &crate::engine::Model,
    st: &malleus_hip::Stream,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    rows: usize,
    n: usize,
) -> Result<()> {
    model.k.get("rmsnorm").launch(
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            out.into(),
            x.into(),
            w.into(),
            n.into(),
            model.cfg.eps.into(),
            0i32.into(),
        ],
    )?;
    Ok(())
}

/// The latent MoE: route on the full width, run the chosen experts in the latent space, add the shared expert and
/// close with `combine_output_norm`.
///
/// The routing and the per-expert GEMVs are exact but not fast: the top-k is host work and every (token, slot) pair
/// is two GEMV launches over its slice of the stacked matrix. The grouped GEMM path (the one the carried models use
/// for a prompt) is the next step; this one proves the math.
fn moe(
    model: &crate::engine::Model,
    m: &Moe,
    hn2: &DeviceBuffer,
    t: usize,
    sc: &mut crate::session::Scratch,
) -> Result<DeviceBuffer> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (hidden, latent, inter) = (c.hidden, h.latent, h.moe_inter);
    let (experts, top_k) = (h.experts, h.top_k);
    // The router sees the full-width input; the experts run on its projection into the latent space.
    let logits = crate::session::f32s(t * experts)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &logits, Act::X(hn2), &m.router, t, false)?;
    let rout_in = crate::session::f32s(t * latent)?;
    // `Act::X`, not `Same`: the router GEMM is bf16 and does not quantize its input, so there is nothing to reuse.
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &rout_in, Act::X(hn2), &m.pre, t, false)?;
    st.synchronize()?;
    let scores: Vec<f32> = logits.download(t * experts)?;
    let bias: Vec<f32> = m.bias.download(experts)?;
    let mut ids = vec![0i32; t * top_k];
    let mut ws = vec![0f32; t * top_k];
    for ti in 0..t {
        let row = &scores[ti * experts..][..experts];
        let sig: Vec<f32> = row.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        // Selection carries the bias, the weight does not; the top-k weights are renormalized and scaled.
        let mut order: Vec<usize> = (0..experts)
            .map(|e| (e, sig[e] + bias[e]))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(e, _)| e)
            .collect();
        order.sort_by(|&a, &b| (sig[b] + bias[b]).total_cmp(&(sig[a] + bias[a])));
        let sel = &order[..top_k];
        let sum: f32 = sel.iter().map(|&e| sig[e]).sum::<f32>().max(1e-20);
        for (slot, &e) in sel.iter().enumerate() {
            ids[ti * top_k + slot] = e as i32;
            ws[ti * top_k + slot] = sig[e] / sum * h.routed_scaling;
        }
    }
    let slots = top_k;
    let trace = std::env::var("MALLEUS_HY4_TRACE").is_ok();
    let dump = |name: &str, buf: &DeviceBuffer, n: usize| -> Result<()> {
        if !trace {
            return Ok(());
        }
        st.synchronize()?;
        let v: Vec<f32> = buf.download(n)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let dir = std::env::var("MALLEUS_DUMP_DIR")?;
        std::fs::write(std::path::Path::new(&dir).join(format!("M_{name}.f32")), bytes)?;
        Ok(())
    };
    dump("rout_in", &rout_in, t * latent)?;
    if trace {
        eprintln!(
            "routing ids {:?} w {:?}",
            &ids[..ids.len().min(4)],
            &ws[..ws.len().min(4)]
        );
    }
    let gu = crate::session::f32s(t * slots * 2 * inter)?;
    let act = crate::session::f32s(t * slots * inter)?;
    let y = crate::session::f32s(t * slots * latent)?;
    let routed = crate::session::f32s(t * latent)?;
    routed.memset(0, st)?;
    let (gq, gs_) = &m.gu;
    let (dq, ds) = &m.down;
    let (gu_row, gu_grp) = (2 * inter, 2 * inter * (latent / 128) * 2);
    let (dn_row, dn_grp) = (latent, latent * (inter / 128) * 2);
    let nexp = if std::env::var("MALLEUS_HY4_MOE_NOEXPERTS").is_ok() {
        0
    } else {
        slots
    };
    let xr = crate::session::f32s(inter.max(latent))?;
    for ti in 0..t {
        for slot in 0..nexp {
            let p = ti * slots + slot;
            let e = ids[p] as usize;
            model.k.get("hadamard128").launch(
                ((latent / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&xr).into(),
                    Arg::U64(rout_in.addr() + (ti * latent * 4) as u64),
                    (&model.signs).into(),
                    latent.into(),
                ],
            )?;
            model.k.get("hy4_expert_gemv").launch(
                ((2 * inter) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(gu.addr() + (p * 2 * inter * 4) as u64),
                    (&xr).into(),
                    Arg::U64(gq.addr() + (e * gu_row * latent / 2) as u64),
                    Arg::U64(gs_.addr() + (e * gu_grp) as u64),
                    (2 * inter).into(),
                    latent.into(),
                    0i32.into(),
                ],
            )?;
            model.k.get("hy4_glu").launch(
                (inter.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(act.addr() + (p * inter * 4) as u64),
                    Arg::U64(gu.addr() + (p * 2 * inter * 4) as u64),
                    inter.into(),
                    h.swiglu_limit.into(),
                ],
            )?;
            model.k.get("hadamard128").launch(
                ((inter / 128) as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &[
                    (&xr).into(),
                    Arg::U64(act.addr() + (p * inter * 4) as u64),
                    (&model.signs).into(),
                    inter.into(),
                ],
            )?;
            model.k.get("hy4_expert_gemv").launch(
                (latent as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(y.addr() + (p * latent * 4) as u64),
                    (&xr).into(),
                    Arg::U64(dq.addr() + (e * dn_row * inter / 2) as u64),
                    Arg::U64(ds.addr() + (e * dn_grp) as u64),
                    latent.into(),
                    inter.into(),
                    0i32.into(),
                ],
            )?;
            model.k.get("hy4_moe_acc").launch(
                (latent.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[
                    Arg::U64(routed.addr() + (ti * latent * 4) as u64),
                    Arg::U64(y.addr() + (p * latent * 4) as u64),
                    ws[p].into(),
                    latent.into(),
                ],
            )?;
        }
    }
    // Back to the hidden width, then the shared expert on the full-width input added in. Every projection is read
    // row by row through the GEMV the experts use: no GEMM state to carry across the pair loop.
    dump("routed", &routed, t * latent)?;
    let out = crate::session::f32s(t * hidden)?;
    let sg = crate::session::f32s(t * inter)?;
    let su = crate::session::f32s(t * inter)?;
    let xr = crate::session::f32s(inter.max(hidden).max(latent))?;
    let row = |w: &crate::weights::Linear, dst: &DeviceBuffer, x: &DeviceBuffer, xi: usize, acc: i32| -> Result<()> {
        let crate::weights::Weights::Mls4 { q, sc } = &w.w else {
            bail!("hy4 MoE projection is not MLS4")
        };
        model.k.get("hadamard128").launch(
            ((w.k / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            st,
            &[
                (&xr).into(),
                Arg::U64(x.addr() + (xi * w.k * 4) as u64),
                (&model.signs).into(),
                w.k.into(),
            ],
        )?;
        model.k.get("hy4_expert_gemv").launch(
            (w.n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                Arg::U64(dst.addr() + (xi * w.n * 4) as u64),
                (&xr).into(),
                q.into(),
                sc.into(),
                w.n.into(),
                w.k.into(),
                acc.into(),
            ],
        )?;
        Ok(())
    };
    let shared = std::env::var("MALLEUS_HY4_MOE_NOSHARED").is_err();
    for ti in 0..t {
        row(&m.up, &out, &routed, ti, 0)?;
        if !shared {
            continue;
        }
        row(&m.shared.0, &sg, hn2, ti, 0)?;
        row(&m.shared.1, &su, hn2, ti, 0)?;
        let off = (ti * inter * 4) as u64;
        model.ew(
            st,
            "silu_mul",
            inter,
            &[
                Arg::U64(sg.addr() + off),
                Arg::U64(sg.addr() + off),
                Arg::U64(su.addr() + off),
                inter.into(),
            ],
        )?;
        row(&m.shared.2, &out, &sg, ti, 1)?;
    }
    dump("out", &out, t * hidden)?;
    let normed = crate::session::f32s(t * hidden)?;
    rmsnorm_raw(model, st, &normed, &out, &m.norm, t, hidden)?;
    Ok(normed)
}

/// The processor's `smart_resize`: both sides a multiple of `patch * merge`, inside the pixel budget. It rounds
/// halves to even, like Python's `round`.
pub(crate) fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let (hf, wf) = (h as f64, w as f64);
    let f = factor as f64;
    let mut hb = (hf / f).round_ties_even() as usize * factor;
    let mut wb = (wf / f).round_ties_even() as usize * factor;
    if hb * wb > max_pixels {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = ((hf / beta / f).floor().max(1.0) as usize) * factor;
        wb = ((wf / beta / f).floor().max(1.0) as usize) * factor;
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = ((hf * beta / f).ceil() as usize) * factor;
        wb = ((wf * beta / f).ceil() as usize) * factor;
    }
    (hb.max(factor), wb.max(factor))
}

/// The weights of one PIL resample pass: for every output position, the first input index and the filter weights
/// around it (the triangle filter with its support widened by the reduction).
fn resample_weights(in_len: usize, out_len: usize) -> Vec<(usize, Vec<f64>, f64)> {
    let scale = in_len as f64 / out_len as f64;
    let fscale = scale.max(1.0);
    let support = fscale;
    (0..out_len)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let hi = (((center + support + 0.5).floor() as isize).max(0) as usize).min(in_len);
            let mut ws = Vec::with_capacity(hi.saturating_sub(lo));
            let mut total = 0.0f64;
            for i in lo..hi {
                let x = ((i as f64 - center + 0.5) / fscale).abs();
                let w = (1.0 - x).max(0.0);
                total += w;
                ws.push(w);
            }
            (lo, ws, total)
        })
        .collect()
}

/// PIL's bilinear resize of an `u8` image with `c` interleaved channels: horizontal then vertical, each pass
/// rounded back to `u8` the way PIL's 8-bit resampler is.
fn resize_u8(src: &[u8], h: usize, w: usize, nh: usize, nw: usize, c: usize) -> Vec<u8> {
    let wh = resample_weights(w, nw);
    let mut mid = vec![0u8; h * nw * c];
    for y in 0..h {
        for (o, (lo, ws, total)) in wh.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wt) in ws.iter().enumerate() {
                    acc += src[(y * w + lo + k) * c + ch] as f64 * wt;
                }
                let v = if *total > 0.0 { acc / total } else { 0.0 };
                mid[(y * nw + o) * c + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let wv = resample_weights(h, nh);
    let mut out = vec![0u8; nh * nw * c];
    for (o, (lo, ws, total)) in wv.iter().enumerate() {
        for x in 0..nw {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wt) in ws.iter().enumerate() {
                    acc += mid[((lo + k) * nw + x) * c + ch] as f64 * wt;
                }
                let v = if *total > 0.0 { acc / total } else { 0.0 };
                out[(o * nw + x) * c + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// The processor: smart resize, bilinear resize, rescale and normalise to `[-1, 1]` (mean and std 0.5), and the
/// row-major patches of `patch x patch` in `(channel, y, x)` order. Returns the flat patch tensor and the patch
/// grid.
pub(crate) fn preprocess(rgb: &[u8], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
    const PATCH: usize = 16;
    const MERGE: usize = 2;
    const MIN_PIXELS: usize = 4096;
    const MAX_PIXELS: usize = 4194304;
    ensure!(rgb.len() >= h * w * 3, "rgb buffer too small");
    let (rh, rw) = smart_resize(h, w, PATCH * MERGE, MIN_PIXELS, MAX_PIXELS);
    let img = resize_u8(rgb, h, w, rh, rw, 3);
    let (gh, gw) = (rh / PATCH, rw / PATCH);
    let mut out = vec![0f32; gh * gw * 3 * PATCH * PATCH];
    for py in 0..gh {
        for px in 0..gw {
            let p = py * gw + px;
            for c in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let v = img[((py * PATCH + y) * rw + px * PATCH + x) * 3 + c];
                        // (v / 255 - 0.5) / 0.5
                        out[p * 3 * PATCH * PATCH + c * PATCH * PATCH + y * PATCH + x] = v as f32 * (2.0 / 255.0) - 1.0;
                    }
                }
            }
        }
    }
    Ok((out, gh, gw))
}

/// The 3-D positions of a prompt that carries images, as `get_rope_index` builds them: text grows by one on all
/// three planes; an image block is `t` frames of `h/merge` rows of `w/merge + 1` tokens (every merged row ends
/// with a newline), all three planes offset by the block's start; the text after it continues at the block's
/// maximum plus one.
pub(crate) fn image_positions(
    tokens: &[u32],
    image_token: u32,
    merge: usize,
    grids: &[(usize, usize, usize)],
) -> Result<Vec<[i32; 3]>> {
    ensure!(merge > 0, "bad merge size");
    let mut out = Vec::with_capacity(tokens.len());
    let mut pos: i32 = 0;
    let mut img = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != image_token {
            out.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let start = i;
        while i < tokens.len() && tokens[i] == image_token {
            i += 1;
        }
        let (gt, gh, gw) = *grids.get(img).context("more image blocks than images")?;
        img += 1;
        let (hm, wm) = (gh / merge, gw / merge);
        ensure!(hm > 0 && wm > 0, "image grid smaller than the merge size");
        let rows = gt * hm * (wm + 1);
        ensure!(i - start == rows, "image block {} tokens, grid wants {rows}", i - start);
        let block = hm.max(wm + 1) as i32;
        let mut mx = pos;
        for t in 0..gt as i32 {
            let base = pos + t * block;
            for r in 0..hm as i32 {
                for c in 0..wm as i32 + 1 {
                    out.push([base, base + r, base + c]);
                    mx = mx.max(base + r).max(base + c);
                }
            }
        }
        pos = mx + 1;
    }
    ensure!(img == grids.len(), "fewer image blocks than images");
    Ok(out)
}

/// The precision `nn.RMSNorm` uses when its eps is left to the dtype (the Engram norms do that).
const BF16_EPS: f32 = 0.0;

/// The Engram memory of one layer, added to the streams before the attention.
///
/// The hashes and the 4-bit row reads are host work (the table stays in the file); the two projections that map the
/// memory to and from the hidden width are GEMMs on the device. The gates, the per-stream scales and the sum are
/// host arithmetic for this first pass: `t` is a prompt, so it is a few hundred KiB per Engram layer.
fn engram(
    model: &crate::engine::Model,
    e: &Engram,
    streams: &mut DeviceBuffer,
    tokens: &[u32],
    t: usize,
    sc: &mut crate::session::Scratch,
) -> Result<()> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let st = &model.stream;
    let (hidden, hc) = (c.hidden, h.hc_mult);
    let per_order = e.heads;
    let orders = e.head_vocab.len() / per_order;
    ensure!(orders >= 1 && orders + 1 == e.hash_mult.len(), "engram orders");
    let mut offsets = Vec::with_capacity(e.head_vocab.len());
    let mut off = 0usize;
    for &n in &e.head_vocab {
        offsets.push(off);
        off += n;
    }
    // One row of the shared table per (token, hash head).
    let mut ids = Vec::with_capacity(t * e.head_vocab.len());
    for ti in 0..t {
        let mut mixed = [0i64; 3];
        for (o, m) in mixed.iter_mut().enumerate().take(e.hash_mult.len()) {
            let id = if ti >= o { tokens[ti - o] } else { e.pad };
            let id = if (id as usize) < e.lookup.len() { id } else { e.pad };
            *m = (e.lookup[id as usize] as i64).wrapping_mul(e.hash_mult[o]);
        }
        for k in 0..orders {
            let n = k + 2;
            let mut mix = mixed[0];
            for &m in mixed.iter().take(n).skip(1) {
                mix ^= m;
            }
            for head in 0..per_order {
                let modulus = e.head_vocab[k * per_order + head].max(1) as i64;
                ids.push(offsets[k * per_order + head] + mix.rem_euclid(modulus) as usize);
            }
        }
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        eprintln!("engram ids {:?}", &ids[..ids.len().min(4)]);
    }
    let mut rows = e.table.rows(&ids)?;
    let heads = e.head_vocab.len();
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        eprintln!(
            "raw row0 {:?} row1 {:?} nw1 {:?} heads {heads} dim {}",
            &rows[..4],
            &rows[96..100],
            &e.head_norms[96..100],
            e.dim
        );
    }
    for (r, row) in rows.chunks_mut(e.dim).enumerate() {
        if false && e.fp8 {
            fp8_round_trip(row);
        }
        let w = &e.head_norms[(r % heads) * e.dim..][..e.dim];
        let ss = row.iter().map(|v| v * v).sum::<f32>() / e.dim as f32;
        let inv = 1.0 / (ss + BF16_EPS).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * inv * w[i];
        }
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok() {
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        eprintln!(
            "engram ids {:?} row0 {:?} gamma rms {:.5}",
            &ids[..ids.len().min(4)],
            rows[..rows.len().min(4)]
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
            rms(&e.pre_gamma)
        );
        let _ = rms;
        let _ = &rows;
    }
    if std::env::var("MALLEUS_HY4_TRACE").is_ok()
        && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
    {
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join("E_emb_norm.f32"), bytes)?;
    }
    let emb = DeviceBuffer::from_slice(&rows)?;
    let main = crate::session::f32s(t * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &main, Act::X(&emb), &e.value, t, false)?;
    let mut keys = Vec::with_capacity(hc);
    for k in &e.key {
        let buf = crate::session::f32s(t * hidden)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &buf, Act::X(&emb), k, t, false)?;
        keys.push(buf);
    }
    st.synchronize()?;
    let mut main: Vec<f32> = main.download(t * hidden)?;
    for ti in 0..t {
        for i in 0..hidden {
            main[ti * hidden + i] += e.value_bias[i];
        }
    }
    let mut keys_host = Vec::with_capacity(hc);
    for (s, k) in keys.iter().enumerate() {
        let mut v: Vec<f32> = k.download(t * hidden)?;
        for ti in 0..t {
            for i in 0..hidden {
                v[ti * hidden + i] += e.key_bias[s][i];
            }
        }
        keys_host.push(v);
    }
    let trace_engram = std::env::var("MALLEUS_HY4_TRACE").is_ok();
    if trace_engram && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR") {
        let bytes: Vec<u8> = main.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join("E_main.f32"), bytes)?;
    }
    let mut st_host: Vec<f32> = streams.download(t * hc * hidden)?;
    let nh = hidden as f32;
    let mut out = vec![0f32; t * hc * hidden];
    for ti in 0..t {
        let m = &main[ti * hidden..][..hidden];
        let ms = m.iter().map(|v| v * v).sum::<f32>() / nh;
        let inv = 1.0 / (ms + BF16_EPS).sqrt();
        let mut gamma = vec![0f32; hidden];
        for i in 0..hidden {
            gamma[i] = e.pre_gamma[i] * m[i] * inv;
        }
        for s in 0..hc {
            let q0 = &st_host[(ti * hc + s) * hidden..][..hidden];
            let qs = q0.iter().map(|v| v * v).sum::<f32>() / nh;
            let qinv = 1.0 / (qs + BF16_EPS).sqrt();
            let k0 = &keys_host[s][ti * hidden..][..hidden];
            let ks = k0.iter().map(|v| v * v).sum::<f32>() / nh;
            let kinv = 1.0 / (ks + BF16_EPS).sqrt();
            let mut dot = 0f32;
            for i in 0..hidden {
                let q = q0[i] * qinv * e.norm_query[s][i];
                let k = k0[i] * kinv * e.norm_key[s][i];
                dot += k * q;
            }
            let g = dot / nh.sqrt();
            let g = 1.0 / (1.0 + (-(g.abs().max(1e-6).sqrt() * g.signum())).exp());
            if tokens[ti] == h.image_token {
                // A visual placeholder has no token to hash: the memory contributes nothing there.
                continue;
            }
            for i in 0..hidden {
                out[(ti * hc + s) * hidden + i] = gamma[i] * (g + e.scales[s][i]);
            }
        }
    }
    if trace_engram && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR") {
        let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join("E_out.f32"), bytes)?;
    }
    if trace_engram {
        let r = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        eprintln!(
            "engram out rms {:.5} gamma0 {:.4} out_row0 {:.5}",
            r(&out),
            r(&out[..hidden.min(out.len())]),
            r(&out[..hidden.min(out.len())])
        );
    }
    for (a, b) in st_host.iter_mut().zip(&out) {
        *a += b;
    }
    streams.upload(&st_host)?;
    Ok(())
}

/// Hy4's vision tower (the `vit.*` tensors): HunYuan's ViT with Hy4's patch merger.
///
/// LayerNorm with bias, exact GELU, full per-image attention, no RoPE. The patch embed's convolution is a
/// `[1152, 3 * 16 * 16]` GEMM over the patches, and the merger's 2x2 convolution a `[4608, 4 * 1152]` one over the
/// packed 2x2 blocks; the position grid is interpolated on the host (bilinear, align_corners=False, as the module).
#[allow(dead_code)]
pub(crate) struct Vision {
    pub(crate) patch: Linear,
    pub(crate) patch_bias: DeviceBuffer,
    /// The 128x128 learned patch grid (the checkpoint's first row is a class token, never used).
    pub(crate) pos: Vec<f32>,
    pub(crate) blocks: Vec<VBlock>,
    pub(crate) norm: DeviceBuffer,
    pub(crate) proj0: Linear,
    pub(crate) proj0_bias: DeviceBuffer,
    pub(crate) proj2: Linear,
    pub(crate) proj2_bias: DeviceBuffer,
    pub(crate) newline: Vec<f32>,
    pub(crate) hidden: usize,
    pub(crate) inter: usize,
    pub(crate) heads: usize,
    pub(crate) edge: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

#[allow(dead_code)]
pub(crate) struct VBlock {
    pub(crate) ln1: (DeviceBuffer, DeviceBuffer),
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) o: Linear,
    pub(crate) qb: DeviceBuffer,
    pub(crate) kb: DeviceBuffer,
    pub(crate) vb: DeviceBuffer,
    pub(crate) ob: DeviceBuffer,
    pub(crate) ln2: (DeviceBuffer, DeviceBuffer),
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
    pub(crate) f1b: DeviceBuffer,
    pub(crate) f2b: DeviceBuffer,
}

impl Vision {
    pub(crate) fn load(src: &dyn TensorSource, up: &mut Uploader, cfg: &crate::config::Hy4Vision) -> Result<Self> {
        let (hidden, inter, heads, edge) = (cfg.hidden, cfg.inter, cfg.heads, cfg.edge);
        let out = cfg.out;
        let lin = |up: &mut Uploader, n: &str, rows: usize, cols: usize| -> Result<Linear> {
            up.linear(src, &format!("vit.{n}"), rows, cols)
        };
        let bias = |up: &mut Uploader, n: &str, len: usize| -> Result<DeviceBuffer> {
            up.tensor(src, &format!("vit.{n}.bias"), &[len])
        };
        // The convolutions are GEMMs: their weight flattening is the im2col order the kernels see.
        let conv = |up: &mut Uploader, n: &str, rows: usize, cols: usize, shape: &[usize]| -> Result<Linear> {
            Ok(Linear {
                w: crate::weights::Weights::Bf16(up.tensor(src, &format!("vit.{n}.weight"), shape)?),
                prefill: None,
                n: rows,
                k: cols,
            })
        };
        let patch = conv(
            up,
            "embeddings.patch_embedding",
            hidden,
            3 * 16 * 16,
            &[hidden, 3, 16, 16],
        )?;
        let patch_bias = bias(up, "embeddings.patch_embedding", hidden)?;
        // The learned grid, without the class token row and as f32 for the host interpolation.
        let pos = bf16_vec(
            src,
            "vit.embeddings.position_embedding.weight",
            (edge * edge + 1) * hidden,
        )?;
        let pos = pos[hidden..].to_vec();
        let n_layers = src
            .names()
            .iter()
            .filter(|n| n.starts_with("vit.layers.") && n.ends_with(".input_layernorm.weight"))
            .count();
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            blocks.push(VBlock {
                ln1: (
                    up.tensor(src, &format!("vit.{}.weight", p("input_layernorm")), &[hidden])?,
                    up.tensor(src, &format!("vit.{}.bias", p("input_layernorm")), &[hidden])?,
                ),
                q: lin(up, &p("self_attn.q_proj"), hidden, hidden)?,
                k: lin(up, &p("self_attn.k_proj"), hidden, hidden)?,
                v: lin(up, &p("self_attn.v_proj"), hidden, hidden)?,
                o: lin(up, &p("self_attn.o_proj"), hidden, hidden)?,
                qb: bias(up, &p("self_attn.q_proj"), hidden)?,
                kb: bias(up, &p("self_attn.k_proj"), hidden)?,
                vb: bias(up, &p("self_attn.v_proj"), hidden)?,
                ob: bias(up, &p("self_attn.o_proj"), hidden)?,
                ln2: (
                    up.tensor(src, &format!("vit.{}.weight", p("post_attention_layernorm")), &[hidden])?,
                    up.tensor(src, &format!("vit.{}.bias", p("post_attention_layernorm")), &[hidden])?,
                ),
                fc1: lin(up, &p("mlp.dense_h_to_4h"), inter, hidden)?,
                fc2: lin(up, &p("mlp.dense_4h_to_h"), hidden, inter)?,
                f1b: bias(up, &p("mlp.dense_h_to_4h"), inter)?,
                f2b: bias(up, &p("mlp.dense_4h_to_h"), hidden)?,
            });
        }
        Ok(Vision {
            patch,
            patch_bias,
            pos,
            blocks,
            // `before_rms`: RMSNorm without bias, applied raw (one-plus off).
            norm: up.norm(src, "vit.perceive.norm.weight", hidden, true)?,
            proj0: conv(
                up,
                "perceive.proj.0",
                4 * hidden,
                4 * hidden,
                &[4 * hidden, hidden, 2, 2],
            )?,
            proj0_bias: bias(up, "perceive.proj.0", 4 * hidden)?,
            proj2: conv(up, "perceive.proj.2", out, 4 * hidden, &[out, 4 * hidden, 1, 1])?,
            proj2_bias: bias(up, "perceive.proj.2", out)?,
            newline: bf16_vec(src, "vit.perceive.image_newline", out)?,
            hidden,
            inter,
            heads,
            edge,
            rows: cfg.merge,
            cols: cfg.merge,
        })
    }
}

/// Bilinear interpolation of the learned patch grid to `(gh, gw)`, as `torch.nn.functional.interpolate` with
/// `align_corners=False` does it.
fn interp_pos(pos: &[f32], edge: usize, gh: usize, gw: usize, hidden: usize) -> Vec<f32> {
    let (sy, sx) = (edge as f32 / gh as f32, edge as f32 / gw as f32);
    let mut out = vec![0f32; gh * gw * hidden];
    for oy in 0..gh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(edge - 1);
        let y1 = (y0 + 1).min(edge - 1);
        let wy = fy - y0 as f32;
        for ox in 0..gw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(edge - 1);
            let x1 = (x0 + 1).min(edge - 1);
            let wx = fx - x0 as f32;
            let dst = (oy * gw + ox) * hidden;
            for c in 0..hidden {
                let p = |y: usize, x: usize| pos[(y * edge + x) * hidden + c];
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[dst + c] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    out
}

/// The vision tower on one image: `pixels` are `[n, 3 * 16 * 16]` (the processor's flat patches), the grid is
/// `gh x gw`. Returns the tower's last hidden state `[n, hidden]` and the merged features
/// `[gh/2 * (gw/2 + 1), out_hidden]`, the newline column appended per row.
pub(crate) fn vision_forward(
    model: &crate::engine::Model,
    v: &Vision,
    pixels: &[f32],
    gh: usize,
    gw: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::engine::Act;
    let st = &model.stream;
    let c = &model.cfg;
    let n = gh * gw;
    let hidden = v.hidden;
    ensure!(pixels.len() >= n * 3 * 16 * 16, "pixel values too small for the grid");
    let sc = model.scratch(n)?;
    let pix = DeviceBuffer::from_slice(&pixels[..n * 3 * 16 * 16])?;
    let mut x = crate::session::f32s(n * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &x, Act::X(&pix), &v.patch, n, false)?;
    model.k.get("hy4_bias").launch(
        ((n * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&x).into(), (&v.patch_bias).into(), hidden.into(), (n * hidden).into()],
    )?;
    // The position grid, interpolated and added on the host.
    let pos = interp_pos(&v.pos, v.edge, gh, gw, hidden);
    let mut xh: Vec<f32> = x.download(n * hidden)?;
    for i in 0..n * hidden {
        xh[i] += pos[i];
    }
    x.upload(&xh)?;
    let ln = crate::session::f32s(n * hidden)?;
    let qq = crate::session::f32s(n * hidden)?;
    let kk = crate::session::f32s(n * hidden)?;
    let vv = crate::session::f32s(n * hidden)?;
    let attn = crate::session::f32s(n * hidden)?;
    let tmp = crate::session::f32s(n * hidden.max(v.inter))?;
    let dim = hidden / v.heads;
    let scale = 1.0 / (dim as f32).sqrt();
    for b in &v.blocks {
        model.k.get("hy4_ln").launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&ln).into(),
                (&x).into(),
                (&b.ln1.0).into(),
                (&b.ln1.1).into(),
                hidden.into(),
            ],
        )?;
        for (dst, w, bias) in [(&qq, &b.q, &b.qb), (&kk, &b.k, &b.kb), (&vv, &b.v, &b.vb)] {
            model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, dst, Act::X(&ln), w, n, false)?;
            model.k.get("hy4_bias").launch(
                ((n * hidden).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &[dst.into(), bias.into(), hidden.into(), (n * hidden).into()],
            )?;
        }
        model.k.get("hy4_vit_attn").launch(
            (v.heads as u32, n as u32, 1),
            (256, 1, 1),
            (4096 * 4 + 256 * 4) as u32,
            st,
            &[
                (&attn).into(),
                (&qq).into(),
                (&kk).into(),
                (&vv).into(),
                n.into(),
                v.heads.into(),
                dim.into(),
                scale.into(),
            ],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &tmp, Act::X(&attn), &b.o, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (&b.ob).into(), hidden.into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_add").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&tmp).into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_ln").launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&ln).into(),
                (&x).into(),
                (&b.ln2.0).into(),
                (&b.ln2.1).into(),
                hidden.into(),
            ],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &tmp, Act::X(&ln), &b.fc1, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * v.inter).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (&b.f1b).into(), v.inter.into(), (n * v.inter).into()],
        )?;
        model.k.get("hy4_gelu").launch(
            ((n * v.inter).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&tmp).into(), (n * v.inter).into()],
        )?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &vv, Act::X(&tmp), &b.fc2, n, false)?;
        model.k.get("hy4_bias").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&vv).into(), (&b.f2b).into(), hidden.into(), (n * hidden).into()],
        )?;
        model.k.get("hy4_add").launch(
            ((n * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&x).into(), (&vv).into(), (n * hidden).into()],
        )?;
    }
    let tower_out: Vec<f32> = {
        st.synchronize()?;
        x.download(n * hidden)?
    };
    // The merger: before_rms, the 2x2 pack, the two convolutions and the newline per merged row.
    rmsnorm_raw(model, st, &ln, &x, &v.norm, n, hidden)?;
    let (mh, mw) = (gh / 2, gw / 2);
    ensure!(mh > 0 && mw > 0, "the grid must be at least 2x2");
    let nm = mh * mw;
    let packed = crate::session::f32s(nm * 4 * hidden)?;
    model.k.get("hy4_merge_pack").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&packed).into(), (&ln).into(), hidden.into(), mh.into(), mw.into()],
    )?;
    let big = crate::session::f32s(nm * 4 * hidden)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &big, Act::X(&packed), &v.proj0, nm, false)?;
    model.k.get("hy4_bias").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[
            (&big).into(),
            (&v.proj0_bias).into(),
            (4 * hidden).into(),
            (nm * 4 * hidden).into(),
        ],
    )?;
    model.k.get("hy4_gelu").launch(
        ((nm * 4 * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&big).into(), (nm * 4 * hidden).into()],
    )?;
    let out_w = c.hidden;
    let o = crate::session::f32s(nm * out_w)?;
    model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &o, Act::X(&big), &v.proj2, nm, false)?;
    model.k.get("hy4_bias").launch(
        ((nm * out_w).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&o).into(), (&v.proj2_bias).into(), out_w.into(), (nm * out_w).into()],
    )?;
    st.synchronize()?;
    let merged_raw: Vec<f32> = o.download(nm * out_w)?;
    // The newline column goes at the end of every merged row.
    let mut merged = Vec::with_capacity(mh * (mw + 1) * out_w);
    for r in 0..mh {
        merged.extend_from_slice(&merged_raw[r * mw * out_w..(r + 1) * mw * out_w]);
        merged.extend_from_slice(&v.newline);
    }
    *model.scratch_cache.lock().unwrap() = Some(sc);
    Ok((tower_out, merged))
}

/// Hy4's session: the latent and the rotated dims per layer (fp16), no conv or recurrent state.
pub(crate) fn new_session(model: &crate::engine::Model, max_ctx: usize) -> Result<crate::session::Session> {
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let mut s = crate::session::Session {
        max_ctx,
        len: 0,
        next_pos: 0,
        kv: Vec::with_capacity(c.n_layers),
        conv: Vec::new(),
        rec: Vec::new(),
        attn_part: crate::session::f32s(1)?,
        mla_part: None,
        conv_snap: Vec::new(),
        dn_rec: Vec::new(),
        dn_np: 0,
        dn_par: 0,
        snap_t: 0,
        mtp_kv: Vec::new(),
        mtp_len: 0,
        mtp_h: None,
        mtp_pos: Vec::new(),
        mtp_out: None,
        mtp_out_pos: [0; 3],
        mtp_chain: 0,
        ple_state: None,
        ngram_ctx: Vec::new(),
        idx_kb: Vec::new(),
        idx_tk: Vec::new(),
        idx_tp: Vec::new(),
        idx_scores: None,
        idx_sel: None,
        idx_sel_cache: Vec::new(),
        idx_cache_len: 0,
        mtp_idx: None,
        idx_list: None,
        ple_snap: None,
        ngram_prev: Vec::new(),
    };
    let n = crate::weights::kv_ctx(max_ctx);
    for _ in 0..c.n_layers {
        s.kv.push((
            DeviceBuffer::zeroed(n * h.kv_lora * 2)?,
            DeviceBuffer::zeroed(n * h.rope * 2)?,
        ));
    }
    Ok(s)
}

/// One forward of the text graph. The first slice: iHC, the gated MLA with its sink (dense, the indexer comes
/// later), the dense FFN of layer 0 and the head. The latent MoE and the Engram memories are not wired yet.
pub(crate) fn forward(
    model: &crate::engine::Model,
    sess: &mut crate::session::Session,
    tokens: &[u32],
    positions: Option<&[[i32; 3]]>,
    n_logits: usize,
) -> Result<Vec<f32>> {
    use crate::engine::Act;
    let c = &model.cfg;
    let h = c.hy4.as_ref().context("no hy4 config")?;
    let w = model.hy4.as_ref().context("no hy4 weights")?;
    // Debug: run the vision tower alone on the processor's pixels (`MALLEUS_HY4_VISION_PIX` + `..._GRID`), dump
    // the two stages and stop. It is how the tower is compared against `--vision` in the reference tool.
    if let Ok(pix) = std::env::var("MALLEUS_HY4_VISION_PIX") {
        let v = model.hy4_vision.as_ref().context("the file has no vision tower")?;
        let bytes = std::fs::read(&pix)?;
        let pixels: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let grid = std::env::var("MALLEUS_HY4_VISION_GRID").context("MALLEUS_HY4_VISION_GRID")?;
        let (gh, gw) = grid.split_once('x').context("grid as HxW")?;
        let (gh, gw): (usize, usize) = (gh.parse()?, gw.parse()?);
        let (tower_back, merged) = vision_forward(model, v, &pixels, gh, gw)?;
        let dir = std::env::var("MALLEUS_DUMP_DIR").context("MALLEUS_DUMP_DIR")?;
        let put = |name: &str, v: &[f32]| -> Result<()> {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok(std::fs::write(std::path::Path::new(&dir).join(name), bytes)?)
        };
        put("V_tower.f32", &tower_back)?;
        put("V_merged.f32", &merged)?;
        bail!(
            "vision tower: {} patches, {} merged tokens of {} (dumped to MALLEUS_DUMP_DIR)",
            gh * gw,
            merged.len() / c.hidden,
            c.hidden
        );
    }
    let t = tokens.len();
    ensure!(t > 0 && n_logits >= 1 && n_logits <= t, "bad hy4 step");
    ensure!(
        sess.len + t <= sess.max_ctx,
        "context full ({} + {} > {})",
        sess.len,
        t,
        sess.max_ctx
    );
    let st = &model.stream;
    // The kv_b expansion GEMM runs over the whole cache (`nk = len + t`), so the GEMM scratch is sized for the
    // larger of the step and the cache: with no history the two coincide, which is why this only showed up once
    // the bench decoded into a prompt.
    let sc_cap = (sess.len + t).max(t);
    let mut sc = model.take_scratch(sc_cap)?;
    let ids: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
    // An image prompt: the merged features replace the placeholder rows and the positions are the 3-D M-RoPE
    // ones (an image block is a rectangle of patches with a newline column, not a run of positions).
    let merge = h.vision.as_ref().map_or(2, |v| v.merge);
    let mut image: Option<(usize, usize, Vec<f32>)> = None;
    let mut grid: Option<(usize, usize, usize)> = None;
    // The whole image path from a PNG: the processor, the tower, then the prompt with the features in place.
    if let Ok(png) = std::env::var("MALLEUS_HY4_IMAGE_PNG") {
        let v = model.hy4_vision.as_ref().context("the file has no vision tower")?;
        let bytes = std::fs::read(&png)?;
        let (rgb, ih, iw) = crate::image_pre::decode(&bytes, &crate::image_pre::PreConfig::default())?;
        let (pixels, gh, gw) = preprocess(&rgb, ih, iw)?;
        let (tower_back, merged) = vision_forward(model, v, &pixels, gh, gw)?;
        if std::env::var("MALLEUS_HY4_TRACE").is_ok()
            && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
        {
            let put = |name: &str, v: &[f32]| -> Result<()> {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                Ok(std::fs::write(std::path::Path::new(&dir).join(name), bytes)?)
            };
            put("P_pixels.f32", &pixels)?;
            put("P_tower.f32", &tower_back)?;
            put("P_merged.f32", &merged)?;
            std::fs::write(std::path::Path::new(&dir).join("P_grid.txt"), format!("1x{gh}x{gw}"))?;
        }
        eprintln!(
            "image {}x{} -> {}x{} patches, {} merged",
            iw,
            ih,
            gh,
            gw,
            merged.len() / c.hidden
        );
        let start = tokens
            .iter()
            .position(|&x| x == h.image_token)
            .context("no image placeholder in the prompt")?;
        image = Some((start, merged.len() / c.hidden, merged));
        grid = Some((1, gh, gw));
    } else if let Ok(f) = std::env::var("MALLEUS_HY4_IMAGE_FEATURES") {
        let g = std::env::var("MALLEUS_HY4_IMAGE_GRID").context("MALLEUS_HY4_IMAGE_GRID")?;
        let d: Vec<usize> = g
            .split('x')
            .map(|v| v.parse::<usize>().context("grid as TxHxW"))
            .collect::<Result<_>>()?;
        ensure!(d.len() == 3, "grid as TxHxW");
        let start = tokens
            .iter()
            .position(|&x| x == h.image_token)
            .context("no image placeholder in the prompt")?;
        let rows = d[0] * (d[1] / merge) * (d[2] / merge + 1);
        let bytes = std::fs::read(&f)?;
        let v: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(v.len() >= rows * c.hidden, "image features too small");
        image = Some((start, rows, v));
    }
    let pos: Vec<i32> = match (&image, positions) {
        (Some((_, _, _)), _) => {
            let g = match grid {
                Some(g) => g,
                None => {
                    let g = std::env::var("MALLEUS_HY4_IMAGE_GRID").context("MALLEUS_HY4_IMAGE_GRID")?;
                    let d: Vec<usize> = g.split('x').filter_map(|v| v.parse().ok()).collect();
                    (d[0], d[1], d[2])
                }
            };
            let p3 = image_positions(tokens, h.image_token, merge, &[g])?;
            if std::env::var("MALLEUS_HY4_TRACE").is_ok()
                && let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR")
            {
                let mut text = String::new();
                for plane in 0..3 {
                    let line: Vec<String> = p3.iter().map(|p| p[plane].to_string()).collect();
                    text.push_str(&line.join(" "));
                    text.push('\n');
                }
                std::fs::write(std::path::Path::new(&dir).join("V_pos.txt"), text)?;
            }
            p3.iter().flatten().copied().collect()
        }
        (None, Some(p)) => p.iter().flatten().copied().collect(),
        (None, None) => (0..t).flat_map(|i| [(sess.next_pos + i as i64) as i32; 3]).collect(),
    };
    let max_pos = pos.iter().copied().max().unwrap_or(0) as i64;
    sc.upload_step(sess.len as i32, &ids, &pos)?;
    model.k.get("embed_bf16").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        0,
        st,
        &[(&sc.x).into(), (&model.embed).into(), (&sc.ids).into(), c.hidden.into()],
    )?;
    if let Some((start, rows, feat)) = &image {
        let buf = DeviceBuffer::from_slice(feat)?;
        sc.x.copy_from(start * c.hidden * 4, &buf, 0, rows * c.hidden * 4, st)?;
    }
    // The streams are the embedding repeated `hc_mult` times.
    let sd = h.hc_mult * c.hidden;
    let mut streams = crate::session::f32s(t * sd)?;
    model.ew(
        st,
        "hc_expand",
        t * sd,
        &[
            (&streams).into(),
            (&sc.x).into(),
            t.into(),
            h.hc_mult.into(),
            c.hidden.into(),
        ],
    )?;
    let (hidden, hc, eps) = (c.hidden, h.hc_mult, c.hy4.as_ref().unwrap().hc_eps);
    let hc_shared = (2 * hc * 4 + 256 * 4) as u32;
    let layers = std::env::var("MALLEUS_HY4_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(w.layers.len())
        .min(w.layers.len());
    let trace = std::env::var("MALLEUS_HY4_TRACE").is_ok();
    let dump = |name: &str, li: usize, buf: &DeviceBuffer, n: usize| -> Result<()> {
        if !trace {
            return Ok(());
        }
        st.synchronize()?;
        let v: Vec<f32> = buf.download(n)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let dir = std::env::var("MALLEUS_DUMP_DIR").context("MALLEUS_HY4_TRACE needs MALLEUS_DUMP_DIR")?;
        std::fs::write(std::path::Path::new(&dir).join(format!("T{li}_{name}.f32")), bytes)?;
        Ok(())
    };
    let post = crate::session::f32s(t * hc)?;
    let mut hn = crate::session::f32s(t * hidden)?;
    let mut hn2 = crate::session::f32s(t * hidden)?;
    // The sliding layers are twice as wide: the per-step buffers are sized for the wider of the two shapes.
    let nh_max = h.heads.max(h.swa_heads);
    let base = h.nope.max(h.swa_nope) + h.rope.max(h.swa_rope);
    let vmax = h.v.max(h.swa_v);
    let qd = nh_max * base;
    let kvd = nh_max * (base + vmax - h.rope.max(h.swa_rope));
    let vd = nh_max * vmax;
    let q = crate::session::f32s(t * qd)?;
    let qa = crate::session::f32s(t * h.q_lora)?;
    let kva = crate::session::f32s(t * (h.kv_lora + h.rope))?;
    let gate = crate::session::f32s(t * vd)?;
    let attn_out = crate::session::f32s(t * vd)?;
    let o = crate::session::f32s(t * hidden)?;
    let _ = (sess.len + t, kvd);
    for (li, layer) in w.layers.iter().take(layers).enumerate() {
        let a = &layer.attn;
        // Debug: take the streams from a file (`MALLEUS_HY4_STREAMS_IN`), so the Engram can be compared against
        // the reference on the reference's own input instead of the layer-0 output the engine happens to have.
        if li == 1
            && let Ok(f) = std::env::var("MALLEUS_HY4_STREAMS_IN")
        {
            let bytes = std::fs::read(&f)?;
            let v: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(v.len() >= t * sd, "streams input file too small");
            streams.upload(&v[..t * sd])?;
        }
        if let Some(e) = &layer.engram {
            engram(model, e, &mut streams, tokens, t, &mut sc)?;
            dump("engram_out", li, &streams, t * sd)?;
        }
        let (ckv, ckrot) = &sess.kv[li];
        // iHC pre-mix of the mixer, its norm, and the MLA projections.
        model.k.get("ihc_pre").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            hc_shared,
            st,
            &[
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (&layer.attn_hc.f).into(),
                (&layer.attn_hc.base).into(),
                (&layer.attn_hc.scale).into(),
                t.into(),
                hc.into(),
                hidden.into(),
                h.hc_magnitude.into(),
                eps.into(),
            ],
        )?;
        dump("ihc_pre", li, &hn, t * hidden)?;
        dump("ihc_post", li, &post, t * hc)?;
        dump("ihc_out", li, &hn, t * hidden)?;
        model.rmsnorm(st, &hn2, &hn, &layer.in_norm, t, hidden)?;
        dump("in_norm", li, &hn2, t * hidden)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &qa, Act::X(&hn2), &a.q_a, t, false)?;
        model.rmsnorm(st, &hn, &qa, &a.q_a_norm, t, h.q_lora)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &q, Act::X(&hn), &a.q_b, t, false)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &kva, Act::X(&hn2), &a.kv_a, t, false)?;
        dump("q", li, &q, t * qd)?;
        model.k.get("hy4_rope").launch(
            ((t * (h.heads + 1) * (h.rope / 2)).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&q).into(),
                (&kva).into(),
                (&sc.pos).into(),
                t.into(),
                h.heads.into(),
                h.nope.into(),
                h.kv_lora.into(),
                h.rope.into(),
                h.mrope_section[1].into(),
                h.mrope_section[2].into(),
                // The sliding layers rotate with the short base, the global ones with the long one.
                match c.kind[li] {
                    crate::config::MixerKind::Hy4Sliding => h.theta_sliding,
                    _ => h.theta_indexed,
                }
                .into(),
            ],
        )?;
        model.k.get("mla_cache_write").launch(
            ((t * (h.kv_lora + h.rope)).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                ckv.into(),
                ckrot.into(),
                (&kva).into(),
                (&a.kv_a_norm).into(),
                eps.into(),
                sess.len.into(),
                t.into(),
                h.kv_lora.into(),
                h.rope.into(),
            ],
        )?;
        // Absorbed MLA: `kv_b`'s QK half is folded into the query once per token and head, the split walks the cached
        // latents (the sliding layers only the last `window` of them), and the combine applies its V half and the gate.
        let nsplit = 8usize;
        let qt = crate::session::f32s(t * a.heads * h.kv_lora)?;
        model.k.get("hy4_absorb").launch(
            ((t * a.heads).div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&qt).into(),
                (&q).into(),
                (&a.kv_b).into(),
                t.into(),
                a.heads.into(),
                a.nope.into(),
                h.kv_lora.into(),
                a.rope.into(),
                (a.nope + a.v).into(),
            ],
        )?;
        dump("qt", li, &qt, t * a.heads * h.kv_lora)?;
        if let Ok(dir) = std::env::var("MALLEUS_HY4_DUMP_W") {
            st.synchronize()?;
            let v: Vec<f32> = a.kv_b.download(a.kv_b.len() / 4)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join(format!("W{li}_kvb.f32")), bytes)?;
            // The gate weight, dequantized like linear_f32 does (MLS4 -> f32), to check the row order.
            if let crate::weights::Weights::Mls4 { q, sc } = &a.gate.w {
                use malleus_gguf::quant::{self, BLOCK};
                let (n, k) = (a.gate.n, a.gate.k);
                let g = k / BLOCK;
                let signs = quant::hadamard_signs(malleus_gguf::quant::HADAMARD_SEED);
                let qw = q.download::<u8>(n * k / 2)?;
                let sw = sc.download::<u8>(n * g * 2)?;
                let mut out = vec![0f32; n * k];
                for r in 0..n {
                    for j in 0..g {
                        let at = (r * g + j) * 2;
                        let s = half::f16::from_le_bytes([sw[at], sw[at + 1]]).to_f32();
                        let mut blk = [0f32; BLOCK];
                        quant::dequant_int4(&qw[r * k / 2 + j * BLOCK / 2..][..BLOCK / 2], s, &mut blk);
                        quant::unrotate_block(&mut blk, &signs);
                        out[r * k + j * BLOCK..][..BLOCK].copy_from_slice(&blk);
                    }
                }
                let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("W{li}_gate.f32")), bytes)?;
            }
        }
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &gate, Act::X(&hn2), &a.gate, t, false)?;
        dump("gate", li, &gate, t * vd)?;
        let part = crate::session::f32s(t * a.heads * nsplit * (h.kv_lora + 2))?;
        let sink = a.sink.as_ref().context("hy4 gated MLA needs the learned sink")?;
        let window = match c.kind[li] {
            crate::config::MixerKind::Hy4Sliding => h.window,
            _ => 0,
        };
        model.k.get("hy4_mla_split").launch(
            ((t * a.heads) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&part).into(),
                (&qt).into(),
                (&q).into(),
                ckv.into(),
                ckrot.into(),
                sink.into(),
                sess.len.into(),
                t.into(),
                a.heads.into(),
                a.nope.into(),
                h.kv_lora.into(),
                a.rope.into(),
                (1.0f32 / ((a.nope + a.rope) as f32).sqrt()).into(),
                nsplit.into(),
                window.into(),
            ],
        )?;
        // The merge into a global buffer, then the V half of kv_b_proj and the gate: two kernels so that the
        // latent the V projection reads is the one written, not a block of shared memory.
        let lat_merged = crate::session::f32s(t * a.heads * h.kv_lora)?;
        model.k.get("hy4_mla_merge").launch(
            ((t * a.heads) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[(&lat_merged).into(), (&part).into(), h.kv_lora.into(), nsplit.into()],
        )?;
        if std::env::var("MALLEUS_HY4_DUMP_PART").is_ok() && li == 0 {
            st.synchronize()?;
            let dir = std::env::var("MALLEUS_DUMP_DIR")?;
            let n_lat = crate::weights::kv_ctx(sess.max_ctx) * h.kv_lora;
            let cache: Vec<u16> = ckv.download(n_lat)?;
            let bytes: Vec<u8> = cache.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_ckv.f16"), bytes)?;
            let v: Vec<f32> = part.download(t * a.heads * nsplit * (h.kv_lora + 2))?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_part.f32"), bytes)?;
            let lm: Vec<f32> = lat_merged.download(t * a.heads * h.kv_lora)?;
            let bytes: Vec<u8> = lm.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_latmerged.f32"), bytes)?;
            let gl: Vec<f32> = gate.download(t * vd)?;
            let bytes: Vec<u8> = gl.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("T0_gate2.f32"), bytes)?;
        }
        model.k.get("hy4_mla_v").launch(
            ((t * a.heads) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&attn_out).into(),
                (&lat_merged).into(),
                (&a.kv_b).into(),
                (&gate).into(),
                a.heads.into(),
                a.nope.into(),
                a.v.into(),
                h.kv_lora.into(),
            ],
        )?;
        dump("attn_raw2", li, &attn_out, t * vd)?;
        model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &o, Act::X(&attn_out), &a.o, t, false)?;
        model.rmsnorm(st, &hn, &o, &a.post_norm, t, hidden)?;
        dump("attn", li, &hn, t * hidden)?;
        model.k.get("ihc_res").launch(
            ((t * sd).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&streams).into(),
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (t * sd).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        dump("streams_attn", li, &streams, t * sd)?;
        // FFN: iHC pre-mix, pre-norm, the dense SwiGLU and its sandwich norm, then the residual.
        model.k.get("ihc_pre").launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            hc_shared,
            st,
            &[
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (&layer.ffn_hc.f).into(),
                (&layer.ffn_hc.base).into(),
                (&layer.ffn_hc.scale).into(),
                t.into(),
                hc.into(),
                hidden.into(),
                h.hc_magnitude.into(),
                eps.into(),
            ],
        )?;
        dump("ffn_pre", li, &hn, t * hidden)?;
        dump("ffn_post", li, &post, t * hc)?;
        model.rmsnorm(st, &hn2, &hn, &layer.post_norm, t, hidden)?;
        dump("ffn_in", li, &hn2, t * hidden)?;
        // Debug: take the FFN's input from a file, to check the MoE on the reference's own input.
        if li == 1
            && let Ok(f) = std::env::var("MALLEUS_HY4_FFN_IN")
        {
            let bytes = std::fs::read(&f)?;
            let v: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(v.len() >= t * hidden, "ffn input file too small");
            hn2.upload(&v[..t * hidden])?;
        }
        match &layer.ffn {
            Ffn::Dense {
                gate: g,
                up,
                down,
                norm,
            } => {
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big0, Act::X(&hn2), g, t, false)?;
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &sc.big1, Act::Same(&hn2), up, t, false)?;
                let n = t * c.inter;
                model.ew(
                    st,
                    "silu_mul",
                    n,
                    &[(&sc.big0).into(), (&sc.big0).into(), (&sc.big1).into(), n.into()],
                )?;
                model.gemm(st, &sc.xr, &sc.xh, &sc.xsc, &o, Act::X(&sc.big0), down, t, false)?;
                model.rmsnorm(st, &hn, &o, norm, t, hidden)?;
            }
            Ffn::Latent(m) => {
                let y = moe(model, m, &hn2, t, &mut sc)?;
                hn.copy_from(0, &y, 0, t * hidden * 4, st)?;
            }
        }
        dump("ffn_out", li, &hn, t * hidden)?;
        model.k.get("ihc_res").launch(
            ((t * sd).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                (&streams).into(),
                (&hn).into(),
                (&post).into(),
                (&streams).into(),
                (t * sd).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        // The layer's state in the reference's layout (`layers/<id>/L<i>.f32`, [t, streams * hidden] f32).
        if let Ok(dir) = std::env::var("MALLEUS_DUMP_DIR") {
            st.synchronize()?;
            let v: Vec<f32> = streams.download(t * sd)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join(format!("L{li}_t{t}.f32")), bytes)?;
        }
    }
    sess.len += t;
    sess.next_pos = max_pos + 1;
    // The head: the streams mixed down, the final norm and the logits.
    let normed = crate::session::f32s(t * hidden)?;
    model.k.get("ihc_head").launch(
        (t as u32, 1, 1),
        (256, 1, 1),
        (256 * 4 + 64 * 4) as u32,
        st,
        &[
            (&normed).into(),
            (&streams).into(),
            (&w.head.f).into(),
            (&w.head.base).into(),
            (&w.head.scale).into(),
            t.into(),
            hc.into(),
            hidden.into(),
            eps.into(),
        ],
    )?;
    model.rmsnorm(st, &hn, &normed, &w.norm, t, hidden)?;
    let first = t - n_logits;
    let mut last = crate::session::f32s(n_logits * hidden)?;
    last.copy_from(0, &hn, first * hidden * 4, n_logits * hidden * 4, st)?;
    let logits = crate::session::f32s(n_logits * c.vocab)?;
    model.gemm(
        st,
        &sc.xr,
        &sc.xh,
        &sc.xsc,
        &logits,
        Act::X(&last),
        &model.lm_head,
        n_logits,
        false,
    )?;
    st.synchronize()?;
    *model.scratch_cache.lock().unwrap() = Some(sc);
    Ok(logits.download(n_logits * c.vocab)?)
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
