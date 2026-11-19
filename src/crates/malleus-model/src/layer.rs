//! One transformer layer as the loader builds it: the mixer, the FFN (dense or MoE) and the optional
//! hyper-connections/PLE around them. The per-family mixers come from `qwen35` and `ling3`.

use anyhow::{Context, Result, bail};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{DeviceBuffer, Stream};

use crate::config::{Config, Family, MixerKind, MoeConfig};
use crate::qwen4exp::{Hc, Indexer, Ple, load_hc};
use crate::upload::Uploader;
use crate::weights::{Fused, Linear, Merged, Weights, fuse, int8_rows, merge_mls4};

pub(crate) enum Mixer {
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
pub(crate) struct Layer {
    /// Pre-norms (absent with hyper-connections, whose mix normalizes the streams).
    pub(crate) in_norm: Option<DeviceBuffer>,
    pub(crate) post_norm: Option<DeviceBuffer>,
    /// Hyper-connections around the mixer and the FFN.
    pub(crate) hc: Option<[Hc; 2]>,
    pub(crate) ple: Option<Ple>,
    pub(crate) mixer: Mixer,
    pub(crate) ffn: Ffn,
    /// Prefill int8 weights of the projections reading the layer input (q|k|v or qkv|z), fused into one GEMM.
    pub(crate) fin: Option<Fused>,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Ffn {
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
pub(crate) struct Moe {
    /// bf16 [experts + 1, hidden]: router rows, then the shared expert gate.
    pub(crate) router: DeviceBuffer,
    /// [experts + 1, 2 * inter, hidden]: per expert gate rows then up rows. Absent with `MALLEUS_EXPERT_CACHE`:
    /// the experts stay in the file and `cache` decides where the kernels read them from.
    pub(crate) gu: Option<(DeviceBuffer, DeviceBuffer)>,
    /// [experts + 1, hidden, inter].
    pub(crate) down: Option<(DeviceBuffer, DeviceBuffer)>,
    /// The experts are not resident: their slots (decode) and the layer streamed into the staging (prefill).
    pub(crate) cache: Option<Box<crate::expert_cache::Layer>>,
    /// MLS8 copies of `gu` and `down` (int8 + fp32 row scales) for the grouped prefill GEMMs.
    pub(crate) pre: Option<[(DeviceBuffer, DeviceBuffer); 2]>,
    /// `expert_bias` of the group router (Ling, `moe_router_enable_expert_bias`): added to the sigmoid scores
    /// for the selection only, never for the weights. ADR: docs/model-ling3-flash.md, § MoE.
    #[allow(dead_code)] // read by `moe_route_g`, which lands with the Ling FFN wiring (I6)
    pub(crate) bias: Option<DeviceBuffer>,
    /// The shared expert of Ling: no gate, and possibly another weight format from the routed experts, so it
    /// lives in its own Linears and is added to the routed sum with weight 1.
    pub(crate) shared: Option<(Linear, Linear, Linear)>,
    /// Per-layer SwiGLU clamp of ADR 0004: `silu(gate).clamp(max = L) * up.clamp(-L, L)`, 0 = off. Routed and
    /// shared experts have their own list, and the dense FFN of the first layers is never clamped.
    pub(crate) swiglu: (f32, f32),
    /// The stacked experts are MLS3 (Ling) rather than MLS4 (Qwen): different kernels, because the 3-bit codes
    /// are packed 8 per 3 bytes and the core does not fold the input rotation. One flag per projection: a file can
    /// mix them (`ling3-flash-mls3.malleus` has MLS3 gate/up and MLS4 down, and reading the MLS4 rows with the MLS3
    /// stride is silent garbage).
    pub(crate) gu_mls3: bool,
    pub(crate) down_mls3: bool,
    /// The same two projections at 2 bits. A file never mixes MLS2 with MLS3 on one projection; one flag per
    /// projection for the same reason as above.
    pub(crate) gu_mls2: bool,
    pub(crate) down_mls2: bool,
}

/// Loads a MoE FFN: the router, the stacked routed experts and (Qwen) the shared expert as their last slot,
/// or (Ling) the shared expert as its own dense MLP.
pub(crate) fn load_moe(
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
    // A family that stacks its shared expert (Qwen's `shared_expert_gate`) writes the same packing for it: the
    // converter keeps it on the routed grid, and one stacked buffer has one stride.
    let shared_matrix = m.shared_gate;
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
        let shared = if shared_matrix {
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
    let pad = if shared_matrix { 0 } else { 1 };
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
            DType::Mls2 => (k / 4, g * 2),
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
    if shared_matrix {
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
    let is_mls2 = |name: &str| src.info(name).is_some_and(|i| i.dtype == DType::Mls2);
    let (gu_mls3, down_mls3) = (is_mls3(&l("experts.gate_up_proj")), is_mls3(&l("experts.down_proj")));
    let (gu_mls2, down_mls2) = (is_mls2(&l("experts.gate_up_proj")), is_mls2(&l("experts.down_proj")));
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
        gu_mls2,
        down_mls2,
    })
}

/// The geometry the shared staging buffer has to hold: the largest cached MoE layer of the model.
pub(crate) fn max_cache_geom(layers: &[Layer], mtp: Option<&Mtp>) -> Option<crate::expert_cache::Geom> {
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

/// Multi-token prediction head: fc([norm(embed(next)); norm(h)]), one full-attention layer, norm, shared lm_head.
/// Qwen3.8-Flash-Next: fc_embedding(norm(embed(next))) added to fc_hidden of every stream of norm(streams), one
/// layer with hyper-connections, the MTP stream mixer, shared lm_head; the streams feed the next chained draft.
pub(crate) struct Mtp {
    pub(crate) fc: Option<Linear>,
    pub(crate) fc_e: Option<Linear>,
    pub(crate) fc_h: Option<Linear>,
    pub(crate) pre_e: DeviceBuffer,
    pub(crate) pre_h: DeviceBuffer,
    pub(crate) norm: Option<DeviceBuffer>,
    pub(crate) mixer: Option<Hc>,
    pub(crate) layer: Layer,
}

pub(crate) fn load_layer(
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
