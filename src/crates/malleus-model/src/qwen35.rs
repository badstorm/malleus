//! Qwen3.5 and Qwen3.5-MoE: the gated attention (with M-RoPE) and the gated DeltaNet mixers, and the
//! YaRN rotation table. Qwen3.8-Flash-Next builds on these (`crate::qwen4exp`).

use anyhow::{Context, Result, bail, ensure};
use malleus_hip::{Arg, DeviceBuffer, Stream};
use serde_json::Value;

use crate::config::{Config, MixerKind};
use crate::engine::{Act, Model};
use crate::layer::{Layer, Mixer};
use crate::qwen4exp::Indexer;
use crate::session::{ATTN_SPLIT, ATTN_WMMA_MIN, Batch, DN_CHUNK, LayerStates, Scratch};
use crate::weights::{GEMV_MAX_T, kv_ctx};

/// The rope settings of a Qwen checkpoint: three interleaved M-RoPE sections, no plain rope.
pub(crate) fn rope(t: &Value) -> Result<([usize; 3], bool)> {
    let rope = &t["rope_parameters"];
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
    Ok(([sec[0], sec[1], sec[2]], true))
}

/// The mixer of every layer, from `layer_types`: `full_attention` is gated attention, everything else DeltaNet.
pub(crate) fn mixer_kinds(t: &Value) -> Result<Vec<MixerKind>> {
    Ok(t["layer_types"]
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
        .collect())
}

/// The geometry the Qwen kernels require of the checkpoint.
pub(crate) fn validate(c: &Config) -> Result<()> {
    ensure!(c.head_dim == 256, "reference kernels require head_dim 256");
    ensure!(
        c.lin_dk == 128 && c.lin_dv == 128,
        "reference kernels require DeltaNet head dim 128"
    );
    ensure!(c.mrope, "only interleaved M-RoPE supported");
    Ok(())
}

/// `MALLEUS_ROPE_YARN` (factor, default 1 = native rotation) and `MALLEUS_ROPE_ORIG` (native positions, default
/// 262144, the checkpoint's limit). With a factor, the long-wavelength frequencies are divided by it and the short
/// ones stay native, with a linear ramp in between — YaRN (Peng et al. 2023) in the form Hugging Face also uses. It
/// is needed because past `max_position_embeddings` the native rope extrapolates and the output is wrong: at 262144
/// the model is at home, at 524288 it is not.
pub(crate) fn yarn_factor() -> f32 {
    std::env::var("MALLEUS_ROPE_YARN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|f| *f > 1.0)
        .unwrap_or(1.0)
}

/// The attention scale multiplier YaRN asks for, 1 when the rotation is native.
/// The attention scale YaRN asks for at `factor`: 1 when the rotation is native.
pub(crate) fn mscale_for(factor: f32) -> f32 {
    if factor > 1.0 { 0.1 * factor.ln() + 1.0 } else { 1.0 }
}

/// The factor the process is forced to (`MALLEUS_ROPE_YARN`), if any: it wins over the per-session choice, for the
/// gates and the A/B.
pub(crate) fn forced_factor() -> Option<f32> {
    let f = yarn_factor();
    (f > 1.0).then_some(f)
}

/// The YaRN inverse-frequency table for `factor` (and the scale that goes with it), or `None` for the native
/// rotation. `factor` 1 is the checkpoint's own rotation, bit for bit.
pub(crate) fn rope_table(c: &Config, factor: f32) -> Result<Option<(Vec<f32>, f32)>> {
    if factor <= 1.0 {
        return Ok(None);
    }
    let orig: f32 = std::env::var("MALLEUS_ROPE_ORIG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(c.max_positions as f32);
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
    let mscale = mscale_for(factor);
    eprintln!("rope: YaRN x{factor} over {orig} positions, attention scale {mscale:.6}, {half} frequencies");
    Ok(Some((t, mscale)))
}

/// Largest union of the four block lists `attention_gather_u` keeps in LDS; must match UMAX in kernels/attn.hip.
pub(crate) const GATHER_UNION_MAX: usize = 2080;

/// `MALLEUS_HIDDEN_DUMP=<dir>`: for every attention layer, writes the layer input and the K of the chunk being
/// processed. Diagnostics for #10: it is the material with which one tests whether a block's sensitivity can be
/// learnt. It synchronizes at every layer, so a run with this variable set is very slow: not a path to leave on.
static ENV_HIDDEN_DUMP: std::sync::LazyLock<Option<String>> =
    std::sync::LazyLock::new(|| std::env::var("MALLEUS_HIDDEN_DUMP").ok());

impl Model {
    /// Runs `layers`, the first of which is layer `base` of the model: the index only names the layers in the
    /// routing trace (`MALLEUS_TRACE_ROUTE`), so that the MTP head (passed as layer `cfg.layers`) is not confused
    /// with layer 0.
    /// The attention of one run of rows: the run's own KV cache, written at the run's own length, and its own
    /// state. `q_norm`/`k_norm`/`indexer` belong to the layer, `proj_ready` says the projections already wrote the
    /// indexer's query into the scratch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend(
        &self,
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        li: usize,
        q_norm: &DeviceBuffer,
        k_norm: &DeviceBuffer,
        indexer: Option<&Indexer>,
        proj_ready: bool,
        // Row of the batch this run starts at. Every buffer here packs its rows at the tensor's own width, so the
        // offset is `row * width` of the tensor the layer writes, not of the scratch allocation.
        r0: usize,
        // The KV position the kernels read (the run's own, on the device).
        len_dev: Arg,
        // Row widths in elements of this layer's tensors: the q projection (wider than the head layout, it carries
        // the gate), the k/v projections, and the normed q / attention output.
        qin: usize,
        kvw: usize,
        qout: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let (kc, vc) = &ls.kv[li];
        let qw = qout;
        let kw = kvw;
        // `w` is the width in elements of the tensor this layer writes: rows are packed at that width, and the
        // address arithmetic is in bytes.
        let rowof = |b: &DeviceBuffer, w: usize| -> Arg { Arg::U64(b.addr() + (r0 * w * 4) as u64) };
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
        let qin_arg = rowof(&sc.big0, qin);
        self.k.get("attn_prep").launch(
            (t as u32, (c.n_heads + c.n_kv) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &[
                rowof(&sc.small2, qw),
                kc.into(),
                vc.into(),
                qin_arg,
                rowof(&sc.small0, kw),
                rowof(&sc.small1, kw),
                q_norm.into(),
                k_norm.into(),
                rowof(&sc.pos, 3),
                len_dev,
                c.n_heads.into(),
                c.n_kv.into(),
                c.eps.into(),
                c.rope_theta.into(),
                c.rot_dim.into(),
                sh.into(),
                sw.into(),
                ctx.into(),
                vr_arg,
                ls.rope.map_or(Arg::U64(0), |b| Arg::U64(b.addr())),
            ],
        )?;
        // Diagnostics `MALLEUS_HIDDEN_DUMP=<dir>` (#10): for every layer, the layer input (`sc.hn`) and the
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
        let scale = 1.0 / (c.head_dim as f32).sqrt() * ls.mscale;
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
                        rowof(&sc.small3, qw),
                        rowof(&sc.small2, qw),
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
                let q1 = std::env::var("MALLEUS_GATHER_Q1").map_or(!per && self.k.arch == "gfx1151", |v| v == "1");
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
                            Arg::U64(sc.small3.addr() + (nd * row + r0 * qw) as u64),
                            Arg::U64(sc.small2.addr() + (nd * row + r0 * qw) as u64),
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
                    rowof(&sc.small3, qw),
                    rowof(&sc.small2, qw),
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
                    rowof(&sc.small2, qw),
                    kc.into(),
                    vc.into(),
                    list.into(),
                    cnt,
                    len_dev,
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
                &[rowof(&sc.small3, qw), ls.attn_part.into(), nsplit.into()],
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
                    rowof(&sc.small2, qw),
                    kc.into(),
                    vc.into(),
                    len_dev,
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
                &[rowof(&sc.small3, qw), ls.attn_part.into(), nsplit.into()],
            )?;
        }
        Ok(())
    }

    /// The DeltaNet recurrence of one run of rows: the run's conv and recurrent state, its own length.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn delta_state(
        &self,
        ls: &LayerStates,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        di: usize,
        conv: &DeviceBuffer,
        a_log: &DeviceBuffer,
        dt_bias: &DeviceBuffer,
        fused: bool,
        ab_merged: bool,
        // Row of the batch this run starts at, at the width of the tensor the layer writes (see `attend`).
        r0: usize,
        // Row widths in elements: the qkv projection (which the conv reads), the z / value layout (what the scan
        // writes) and one gate head.
        wqkv: usize,
        wval: usize,
        wn: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let wq = wqkv;
        let wab = if ab_merged { 2 * wn } else { wn };
        // `w` is the width in elements of the tensor this layer writes: rows are packed at that width, and the
        // address arithmetic is in bytes.
        let rowof = |b: &DeviceBuffer, w: usize| -> Arg { Arg::U64(b.addr() + (r0 * w * 4) as u64) };
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
        let cd = c.lin_qkv_dim();
        self.k.get("conv_qk_norm").launch(
            ((t * cd / 128).div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &[
                rowof(&sc.big1, wq),
                rowof(&sc.big0, wq),
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
            &[rowof(&sc.big0, wq), conv_state.into(), t.into(), cd.into(), conv_snap],
        )?;
        let (beta, g) = (rowof(&sc.big2, wn), rowof(&sc.small3, wn));
        self.ew(
            st,
            "deltanet_gates",
            t * c.lin_nv,
            &[
                beta,
                g,
                rowof(&sc.small1, wab),
                if fused || ab_merged {
                    rowof(&sc.small1, wab)
                } else {
                    rowof(&sc.small2, wn)
                },
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
                    // The scan writes one row of `nv * 128` per token, not the qkv width.
                    rowof(&sc.big0, wval),
                    rowof(&sc.big1, wq),
                    beta,
                    g,
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
                    rowof(&sc.big1, wq),
                    beta,
                    g,
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
                    rowof(&sc.big0, wq),
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
        Ok(())
    }
}

impl Model {
    /// One layer of this family: the mixer only, added to the residual (`res`) or left in the layer's
    /// output buffer for the hyper-connection injection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer_attn(
        &self,
        layer: &Layer,
        ls: &LayerStates,
        batch: Option<(&Batch, &[Scratch])>,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        x_in: Act,
        res: &DeviceBuffer,
        res_acc: bool,
        ai: &mut usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let fused = self.fused_ok(t) && layer.fin.is_some();
        match &layer.mixer {
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
                let li = *ai;
                *ai += 1;
                let mut proj_ready = false;
                match (&layer.fin, dec) {
                    (Some(f), _) if fused => self.gemm_fused(st, sc, [&sc.big0, &sc.small0, &sc.small1], x_in, f, t)?,
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
                match batch {
                    None => self.attend(
                        ls,
                        sc,
                        t,
                        st,
                        li,
                        q_norm,
                        k_norm,
                        indexer.as_ref(),
                        proj_ready,
                        0,
                        (&sc.len_dev).into(),
                        q.n,
                        k.n,
                        o.k,
                    )?,
                    Some((b, scs)) => {
                        let _ = scs;
                        for run in b.runs.iter() {
                            self.attend(
                                run.ls,
                                sc,
                                run.rows,
                                st,
                                li,
                                q_norm,
                                k_norm,
                                indexer.as_ref(),
                                proj_ready,
                                run.off,
                                Arg::U64(run.len_dev),
                                q.n,
                                k.n,
                                o.k,
                            )?;
                        }
                    }
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
            _ => bail!("mixer_attn called on a layer without gated attention"),
        }
        Ok(())
    }
    /// One layer of this family: the mixer only, added to the residual (`res`) or left in the layer's
    /// output buffer for the hyper-connection injection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer_delta(
        &self,
        layer: &Layer,
        ls: &LayerStates,
        batch: Option<(&Batch, &[Scratch])>,
        sc: &Scratch,
        t: usize,
        st: &Stream,
        x_in: Act,
        res: &DeviceBuffer,
        res_acc: bool,
        di: &mut usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let fused = self.fused_ok(t) && layer.fin.is_some();
        match &layer.mixer {
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
                let di_now = *di;
                *di += 1;
                let ab_merged = !fused && t <= GEMV_MAX_T && dec.is_some() && ab.is_some();
                match (&layer.fin, dec) {
                    (Some(f), _) if fused => self.gemm_fused(st, sc, [&sc.big0, &sc.small0, &sc.small1], x_in, f, t)?,
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
                match batch {
                    None => self.delta_state(
                        ls, sc, t, st, di_now, conv, a_log, dt_bias, fused, ab_merged, 0, qkv.n, z.n, c.lin_nv,
                    )?,
                    Some((b, scs)) => {
                        let _ = scs;
                        for run in b.runs.iter() {
                            self.delta_state(
                                run.ls, sc, run.rows, st, di_now, conv, a_log, dt_bias, fused, ab_merged, run.off,
                                qkv.n, z.n, c.lin_nv,
                            )?;
                        }
                    }
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
            _ => bail!("mixer_delta called on a layer without DeltaNet"),
        }
        Ok(())
    }
}
