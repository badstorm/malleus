//! Ling-3.0-flash: the Kimi Delta Attention and MLA mixers.

use anyhow::{Context, Result, bail, ensure};
use malleus_hip::{Arg, DeviceBuffer, Stream};
use serde_json::Value;

use crate::config::{Config, MixerKind, MlaConfig, Swiglu};
use crate::engine::{Act, Model};
use crate::layer::{Layer, Mixer};
use crate::session::{ATTN_SPLIT, LayerStates, Scratch};
use crate::weights::{GEMV_MAX_T, MLA_PREFILL_TILE};

/// Ling spells the rope settings as plain keys and rotates 64 of the 128 dims: no M-RoPE sections.
pub(crate) fn rope() -> ([usize; 3], bool) {
    ([0; 3], false)
}

/// Ling has no `layer_types`: the rule is in the code of the checkpoint, MLA every `layer_group_size` (plus a
/// final partial group) and KDA elsewhere.
pub(crate) fn mixer_kinds(n_layers: usize, t: &Value) -> Vec<MixerKind> {
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
}

pub(crate) fn mla(u: &dyn Fn(&str) -> Result<usize>) -> Result<MlaConfig> {
    Ok(MlaConfig {
        kv_lora: u("kv_lora_rank")?,
        nope: u("qk_nope_head_dim")?,
        rope: u("qk_rope_head_dim")?,
        v: u("v_head_dim")?,
    })
}

/// Per-layer SwiGLU clamps, routed and shared (ADR 0004).
pub(crate) fn swiglu(t: &Value) -> Option<Swiglu> {
    match (
        t["expert_swiglu_limit_list"].as_array(),
        t["share_expert_swiglu_limit_list"].as_array(),
    ) {
        (Some(r), Some(s)) => Some(Swiglu {
            routed: r.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect(),
            shared: s.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect(),
        }),
        _ => None,
    }
}

/// The geometry the KDA and MLA kernels require of the checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    let m = c.mla.context("config: MLA geometry")?;
    ensure!(
        c.kind.contains(&MixerKind::Mla) && c.kind.contains(&MixerKind::Kda),
        "Ling needs both KDA and MLA layers"
    );
    ensure!(
        m.nope == c.head_dim && m.v == c.head_dim,
        "MLA nope and value dims must agree with head_dim"
    );
    ensure!(
        m.kv_lora.is_multiple_of(64) && m.rope == 64,
        "MLA kernels need a latent multiple of 64 and 64 rotated dims"
    );
    ensure!(c.lin_dk == 128 && c.lin_dv == 128, "KDA kernels require head dim 128");
    ensure!(c.kda_lower_bound < 0.0, "the KDA gate floor must be negative");
    let s = c.swiglu.as_ref().context("config: swiglu limits")?;
    ensure!(
        s.routed.len() == c.n_layers && s.shared.len() == c.n_layers,
        "swiglu limit lists must cover every layer"
    );
    ensure!(c.dense_ffn <= c.n_layers, "first_k_dense_replace too large");
    Ok(())
}

impl Model {
    /// One layer of this family: the mixer only, added to the residual (`res`) or left in the layer's
    /// output buffer for the hyper-connection injection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer_kda(
        &self,
        layer: &Layer,
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        res: &DeviceBuffer,
        res_acc: bool,
        di: &mut usize,
    ) -> Result<()> {
        let c = &self.cfg;
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
                let (conv_state, rec_state) = (&ls.conv[*di], &ls.rec[*di]);
                *di += 1;
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
            _ => bail!("mixer_kda called on a layer without KDA"),
        }
        Ok(())
    }
    /// One layer of this family: the mixer only, added to the residual (`res`) or left in the layer's
    /// output buffer for the hyper-connection injection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer_mla(
        &self,
        layer: &Layer,
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        res: &DeviceBuffer,
        res_acc: bool,
        ai: &mut usize,
    ) -> Result<()> {
        let c = &self.cfg;
        match &layer.mixer {
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
                let (ckv, krot) = &ls.kv[*ai];
                let li = *ai;
                *ai += 1;
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
            _ => bail!("mixer_mla called on a layer without MLA"),
        }
        Ok(())
    }
}
