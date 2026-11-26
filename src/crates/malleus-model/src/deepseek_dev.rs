//! DeepSeek-V4.1 on the device (issue #128): one block of the stack, and the harness that compares it against the
//! host graph stage by stage.
//!
//! The host graph (`crate::deepseek_text`) is the specification, gated against the release's own implementation on
//! real weights (issues #83, #121). A kernel is written against it, one stage at a time, one position at a time —
//! which is what a decode step runs. Three increments live here, each with its gate:
//!
//! - `WindowAttn`: the attention of a *window* layer (layer 0: no compressor, no indexer). The shapes are
//!   `deepseek_text::Attn`: 64 heads of 512, `q_lora` 1280, 8 output groups of 1024, a 128-row ring of raw latents and
//!   a learned sink per head.
//! - `BlockDev::step_attention`: the attention half of the block around it — the sub-layer's hyper-connections (the
//!   mixes, the collapse of the streams, the write back, Sinkhorn included).
//! - `BlockDev::step`: the whole block — the FFN half on top: its own hyper-connections, the router (sqrt-softplus,
//!   the steering bias, the top-k, the renormalisation), the shared expert and the six routed experts of the position,
//!   each read from the file at 2 bits and 9.4 MB. Layer 0 is the block that can run alone: the indexed layers read a
//!   compressed KV their source layers publish, and layer 1 carries an Engram table.
//!
//! The attention reads 127 MB of weights a position for one layer (5.1 GB over the stack, the number the issue starts
//! from) and runs at 1.2 ms against a 0.55 ms memory wall: it is launch-bound at twenty launches a step, the shape the
//! other families hit before their decode was fused. A block is ~60 launches and reads 6 x 9.4 MB of experts per
//! position; the router's selection comes back to the host because the weights are on disk, which is the one round trip
//! a served path would overlap with the previous position.
//!
//! What is *not* here yet: the Engram gather, the compressor and the two-level indexer of the indexed layers, the MTP
//! stages (kept at the release's FP8, so they need a format of their own), the prefill (this path takes one position at
//! a time) and the loops around the block. The gate is the host graph itself: the same quantized weights, the same
//! input, a cosine per stage.

use anyhow::{Context, Result, ensure};
use malleus_gguf::TensorSource;
use malleus_hip::{DeviceBuffer, Stream};

use crate::config::Config;
use crate::engine::{Act, Model};
use crate::upload::Uploader;
use crate::weights::{Linear, Weights};

/// One layer's window attention on the device: the five projections, the four norms, the learned sink and the ring
/// of raw latents a position reads. `step` runs one position and leaves its result in `out`.
pub struct WindowAttn {
    pub heads: usize,
    pub hd: usize,
    pub rope: usize,
    pub groups: usize,
    pub o_lora: usize,
    pub window: usize,
    pub hidden: usize,
    pub q_lora: usize,
    /// The stages a step fills, in order: the normed input, the low-rank query after its norm, the queries after
    /// RoPE, the shared key after RoPE, the sparse attention's output, the same rotated back, the grouped
    /// projection's input and the layer's output. The harness reads them; the hot path only writes them.
    pub xn: DeviceBuffer,
    pub qr: DeviceBuffer,
    pub q: DeviceBuffer,
    pub kv: DeviceBuffer,
    pub att: DeviceBuffer,
    pub or: DeviceBuffer,
    pub g: DeviceBuffer,
    pub out: DeviceBuffer,
    attn_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    kv_norm: DeviceBuffer,
    sink: DeviceBuffer,
    inv: DeviceBuffer,
    wq_a: Linear,
    wq_b: Linear,
    wkv: Linear,
    /// `wo_a` is block-diagonal over `groups`: one `Linear` per block, each `o_lora` rows wide.
    wo_a: Vec<Linear>,
    wo_b: Linear,
    /// The slices a block of the output projection reads (`or`) and writes (`g`): a GEMV takes whole buffers.
    or_g: Vec<DeviceBuffer>,
    g_g: Vec<DeviceBuffer>,
    /// The ring: `window` rows of `hd` raw latents, key and value at once, at slot `pos % window`.
    ring: DeviceBuffer,
    /// GEMM scratch of the widest input a projection takes (`wo_b`, 8192 wide) for one token. A block reuses it for
    /// its FFN half: the widest input there is the same.
    pub(crate) xr: DeviceBuffer,
    pub(crate) xh: DeviceBuffer,
    pub(crate) xsc: DeviceBuffer,
    eps: f32,
}

/// A buffer and disjoint windows into it. A GEMV takes a whole buffer, so a projection that reads or writes a slice
/// of one needs the slice as a buffer of its own; the windows keep the allocation alive (`into_views`).
fn windows(b: DeviceBuffer, parts: &[(usize, usize)]) -> (DeviceBuffer, Vec<DeviceBuffer>) {
    let mut all = Vec::with_capacity(parts.len() + 1);
    all.push((0, b.len()));
    all.extend_from_slice(parts);
    let mut v = b.into_views(&all);
    let whole = v.remove(0);
    (whole, v)
}

impl WindowAttn {
    /// Loads one layer's attention from a `.malleus` file. Only the attention is read: the shared expert, the router
    /// and the 384 routed experts of the block stay on disk.
    pub fn load(src: &dyn TensorSource, c: &Config, layer: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let (heads, hd, rope, ql) = (d.heads, d.head_dim, d.rope, d.q_lora);
        let (groups, o_lora, window) = (d.o_groups, d.o_lora, d.window);
        ensure!(
            hd % 256 == 0 && hd <= 4 * 256,
            "the window kernel reads a {hd}-wide head with 256 threads"
        );
        ensure!(window <= 128, "the window kernel keeps at most 128 scores a head");
        ensure!(
            heads.is_multiple_of(groups),
            "{heads} heads do not split into {groups} output groups"
        );
        let p = |s: &str| format!("layers.{layer}.{s}");
        let vector = |s: &str, n: usize| -> Result<DeviceBuffer> {
            Ok(DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(
                src,
                &p(s),
                n,
            )?)?)
        };
        let mut up = Uploader::new()?;
        // A decode step reads the decode weights: no MLS8 prefill copy.
        up.use_prefill = false;
        let wq_a = up.linear(src, &p("attn.wq_a"), ql, c.hidden)?;
        let wq_b = up.linear(src, &p("attn.wq_b"), heads * hd, ql)?;
        let wkv = up.linear(src, &p("attn.wkv"), hd, c.hidden)?;
        let wo_b = up.linear(src, &p("attn.wo_b"), c.hidden, groups * o_lora)?;
        let per_group = heads * hd / groups;
        let mut wo_a = Vec::with_capacity(groups);
        for g in 0..groups {
            wo_a.push(up.linear_rows(src, &p("attn.wo_a"), groups * o_lora, per_group, g * o_lora, o_lora)?);
        }
        // The window layers rotate at the plain base; an indexed layer would use the longer theta with YaRN.
        let r = &d.rope_params;
        let inv = if d.compress_ratios[layer] > 0 {
            crate::deepseek_text::rope_inv(
                rope,
                r.compress_theta,
                r.original_seq,
                r.factor,
                r.beta_fast,
                r.beta_slow,
            )
        } else {
            crate::deepseek_text::rope_inv(rope, r.theta, 0, 1.0, 0.0, 0.0)
        };
        let (or, or_g) = windows(
            DeviceBuffer::zeroed(heads * hd * 4)?,
            &(0..groups)
                .map(|g| (g * per_group * 4, per_group * 4))
                .collect::<Vec<_>>(),
        );
        let (g, g_g) = windows(
            DeviceBuffer::zeroed(groups * o_lora * 4)?,
            &(0..groups).map(|i| (i * o_lora * 4, o_lora * 4)).collect::<Vec<_>>(),
        );
        let widest = c.hidden.max(ql).max(groups * o_lora);
        Ok(WindowAttn {
            heads,
            hd,
            rope,
            groups,
            o_lora,
            window,
            hidden: c.hidden,
            q_lora: ql,
            xn: DeviceBuffer::zeroed(c.hidden * 4)?,
            qr: DeviceBuffer::zeroed(ql * 4)?,
            q: DeviceBuffer::zeroed(heads * hd * 4)?,
            kv: DeviceBuffer::zeroed(hd * 4)?,
            att: DeviceBuffer::zeroed(heads * hd * 4)?,
            or,
            g,
            out: DeviceBuffer::zeroed(c.hidden * 4)?,
            attn_norm: vector("attn_norm.weight", c.hidden)?,
            q_norm: vector("attn.q_norm.weight", ql)?,
            kv_norm: vector("attn.kv_norm.weight", hd)?,
            sink: vector("attn.attn_sink", heads)?,
            inv: DeviceBuffer::from_slice(&inv)?,
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            or_g,
            g_g,
            ring: DeviceBuffer::zeroed(window * hd * 4)?,
            xr: DeviceBuffer::alloc(widest * 4)?,
            xh: DeviceBuffer::alloc(widest * 2)?,
            xsc: DeviceBuffer::alloc(widest.div_ceil(128) * 4)?,
            eps: d.eps,
        })
    }

    /// Bytes the device holds for this layer.
    pub fn bytes(&self) -> usize {
        fn lin(l: &Linear) -> usize {
            match &l.w {
                Weights::Bf16(b) => b.len(),
                Weights::Mls2 { q, sc }
                | Weights::Mls3 { q, sc }
                | Weights::Mls4 { q, sc }
                | Weights::Mls8 { q, sc } => q.len() + sc.len(),
            }
        }
        lin(&self.wq_a)
            + lin(&self.wq_b)
            + lin(&self.wkv)
            + lin(&self.wo_b)
            + self.wo_a.iter().map(lin).sum::<usize>()
            + self.ring.len()
            + self.q.len()
            + self.att.len()
            + self.g.len()
            + self.out.len()
    }

    /// Forgets the ring: a new sequence starts with an empty window.
    pub fn reset(&mut self) -> Result<()> {
        self.ring.memset(0, &Stream::null())?;
        Ok(Stream::null().synchronize()?)
    }

    /// One position: `x` is the block's input (`hidden` values, before `attn_norm`), `pos` its absolute position in
    /// the sequence. The window and the RoPE phase both read `pos`, so a cached sequence continues where it was.
    pub fn step(&mut self, m: &Model, x: &DeviceBuffer, pos: usize) -> Result<()> {
        let s = &m.stream;
        let k = &m.k;
        let norm = |out: &DeviceBuffer, x: &DeviceBuffer, w: &DeviceBuffer, n: usize| -> Result<()> {
            k.get("dsv_rmsnorm").launch(
                (1, 1, 1),
                (256, 1, 1),
                0,
                s,
                &[out.into(), x.into(), w.into(), n.into(), self.eps.into()],
            )?;
            Ok(())
        };
        norm(&self.xn, x, &self.attn_norm, self.hidden)?;
        m.gemm(
            s,
            &self.xr,
            &self.xh,
            &self.xsc,
            &self.qr,
            Act::X(&self.xn),
            &self.wq_a,
            1,
            false,
        )?;
        norm(&self.qr, &self.qr, &self.q_norm, self.q_lora)?;
        m.gemm(
            s,
            &self.xr,
            &self.xh,
            &self.xsc,
            &self.q,
            Act::X(&self.qr),
            &self.wq_b,
            1,
            false,
        )?;
        m.gemm(
            s,
            &self.xr,
            &self.xh,
            &self.xsc,
            &self.kv,
            Act::X(&self.xn),
            &self.wkv,
            1,
            false,
        )?;
        norm(&self.kv, &self.kv, &self.kv_norm, self.hd)?;
        // One rotation for the query heads and the shared key: the same phase, the same table.
        let rotary = ((self.heads + 1) * self.rope / 2).div_ceil(256) as u32;
        k.get("dsv_rope").launch(
            (rotary, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.q).into(),
                (&self.kv).into(),
                (&self.inv).into(),
                self.heads.into(),
                self.hd.into(),
                self.rope.into(),
                1i32.into(),
                pos.into(),
                1i32.into(),
            ],
        )?;
        self.ring
            .copy_from((pos % self.window) * self.hd * 4, &self.kv, 0, self.hd * 4, s)?;
        let scale = 1.0 / (self.hd as f32).sqrt();
        k.get("dsv_win_attn").launch(
            (self.heads as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.att).into(),
                (&self.q).into(),
                (&self.ring).into(),
                (&self.sink).into(),
                self.heads.into(),
                self.hd.into(),
                self.window.into(),
                pos.into(),
                scale.into(),
            ],
        )?;
        // The output projection rotates the heads back before it groups them. `dsv_rope` rotates in place, so the
        // raw attention output is copied out first and stays readable for the gate.
        self.or.copy_from(0, &self.att, 0, self.att.len(), s)?;
        k.get("dsv_rope").launch(
            (((self.heads * self.rope / 2).div_ceil(256)) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.or).into(),
                (&self.kv).into(),
                (&self.inv).into(),
                self.heads.into(),
                self.hd.into(),
                self.rope.into(),
                0i32.into(),
                pos.into(),
                (-1i32).into(),
            ],
        )?;
        for g in 0..self.groups {
            m.gemm(
                s,
                &self.xr,
                &self.xh,
                &self.xsc,
                &self.g_g[g],
                Act::X(&self.or_g[g]),
                &self.wo_a[g],
                1,
                false,
            )?;
        }
        m.gemm(
            s,
            &self.xr,
            &self.xh,
            &self.xsc,
            &self.out,
            Act::X(&self.g),
            &self.wo_b,
            1,
            false,
        )?;
        Ok(())
    }
}

/// The hyper-connections of one sub-layer on the device: one projection of the flattened streams into the three
/// coefficient sets, plus the Sinkhorn rounds that make `comb` doubly stochastic. A block has two of these.
struct HcDev {
    hc: usize,
    hidden: usize,
    /// The projection of the streams, and the coefficient sets it splits into: `pre` (the collapse), `post` (the
    /// write back) and `comb` (the old streams' share).
    pub mix: DeviceBuffer,
    pub pre: DeviceBuffer,
    pub post: DeviceBuffer,
    pub comb: DeviceBuffer,
    f: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    iters: usize,
    eps: f32,
}

impl HcDev {
    /// `which` is `hc_attn` or `hc_ffn`: the two sub-layers of a block have their own weights.
    fn load(src: &dyn TensorSource, c: &Config, layer: usize, which: &str) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let (hc, hidden) = (d.hc_mult, c.hidden);
        ensure!(
            hc * hc <= 64,
            "the hyper-connection kernel keeps a {hc}x{hc} matrix in shared memory"
        );
        let p = |s: &str| format!("layers.{layer}.{which}_{s}");
        let mixes = (2 + hc) * hc;
        Ok(HcDev {
            hc,
            hidden,
            mix: DeviceBuffer::zeroed(mixes * 4)?,
            pre: DeviceBuffer::zeroed(hc * 4)?,
            post: DeviceBuffer::zeroed(hc * 4)?,
            comb: DeviceBuffer::zeroed(hc * hc * 4)?,
            f: DeviceBuffer::from_slice(&crate::deepseek_text::load_mat(src, &p("fn"), mixes, hc * hidden)?)?,
            base: DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(src, &p("base"), mixes)?)?,
            scale: DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(src, &p("scale"), 3)?)?,
            iters: d.hc_iters,
            eps: d.hc_eps,
        })
    }

    /// The mixes and the coefficient sets of the streams `h`, one position.
    fn mixes(&mut self, m: &Model, h: &DeviceBuffer) -> Result<()> {
        let (s, k) = (&m.stream, &m.k);
        let (hc, hidden) = (self.hc, self.hidden);
        let mixes = (2 + hc) * hc;
        k.get("dsv_hc_mixes").launch(
            (mixes as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.mix).into(),
                h.into(),
                (&self.f).into(),
                mixes.into(),
                (hc * hidden).into(),
                self.eps.into(),
            ],
        )?;
        k.get("dsv_hc_coef").launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            s,
            &[
                (&self.pre).into(),
                (&self.post).into(),
                (&self.comb).into(),
                (&self.mix).into(),
                (&self.base).into(),
                (&self.scale).into(),
                hc.into(),
                self.eps.into(),
                self.iters.into(),
            ],
        )?;
        Ok(())
    }
}

/// The routed experts of one layer, read from the file a few at a time: the 384 of a block do not fit beside the rest
/// of the backbone, and a position takes six, which it uses in a row. `hold` is how many stay on the device.
struct ExpertCache {
    up: Uploader,
    hold: usize,
    map: std::collections::HashMap<usize, Box<(Linear, Linear, Linear)>>,
}

impl ExpertCache {
    fn new(hold: usize) -> Result<Self> {
        Ok(ExpertCache {
            up: Uploader::new()?,
            hold: hold.max(1),
            map: std::collections::HashMap::new(),
        })
    }

    /// The three projections of expert `id`, read from the file when they are not held.
    fn get(
        &mut self,
        src: &dyn TensorSource,
        c: &Config,
        layer: usize,
        id: usize,
    ) -> Result<&(Linear, Linear, Linear)> {
        if !self.map.contains_key(&id) {
            if self.map.len() >= self.hold {
                self.map.clear();
            }
            let d = c.deepseek.as_ref().context("no deepseek config")?;
            let p = |s: &str| format!("layers.{layer}.ffn.experts.{id}.{s}");
            self.up.use_prefill = false;
            let w = Box::new((
                self.up.linear(src, &p("w1"), d.moe_inter, c.hidden)?,
                self.up.linear(src, &p("w2"), c.hidden, d.moe_inter)?,
                self.up.linear(src, &p("w3"), d.moe_inter, c.hidden)?,
            ));
            self.map.insert(id, w);
        }
        Ok(&self.map[&id])
    }
}

/// One block on the device: the attention half (the sub-layer's hyper-connections, the collapse, `WindowAttn`, the
/// write back over the streams) and the FFN half (its own hyper-connections, the collapse, the router, the shared
/// expert and the six routed experts the position takes, read from the file at 2 bits).
pub struct BlockDev {
    layer: usize,
    pub attn: WindowAttn,
    pub hc: usize,
    pub hidden: usize,
    pub inter: usize,
    pub experts: usize,
    pub top_k: usize,
    /// The attention sub-layer's input, and the FFN's: the collapse over the streams.
    pub x: DeviceBuffer,
    pub xf: DeviceBuffer,
    pub xn: DeviceBuffer,
    /// The router's scores and the selection it made.
    pub scores: DeviceBuffer,
    pub ids: DeviceBuffer,
    pub wts: DeviceBuffer,
    /// The gate and up products of one expert, the SwiGLU between them, and the FFN's output.
    pub gate: DeviceBuffer,
    pub up: DeviceBuffer,
    pub act: DeviceBuffer,
    pub ffn: DeviceBuffer,
    /// The streams after the attention and after the FFN.
    pub h1: DeviceBuffer,
    pub h_out: DeviceBuffer,
    a_hc: HcDev,
    f_hc: HcDev,
    ffn_norm: DeviceBuffer,
    router: Linear,
    bias: DeviceBuffer,
    shared: (Linear, Linear, Linear),
    cache: ExpertCache,
    /// The collapse weights of the previous sub-layer, one per stream.
    pre_mix: DeviceBuffer,
    route_scale: f32,
    renorm: bool,
    limit: f32,
    eps: f32,
}

impl BlockDev {
    pub fn load(src: &dyn TensorSource, c: &Config, layer: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let (hc, hidden, inter) = (d.hc_mult, c.hidden, d.moe_inter);
        let p = |s: &str| format!("layers.{layer}.{s}");
        let attn = WindowAttn::load(src, c, layer)?;
        let a_hc = HcDev::load(src, c, layer, "hc_attn")?;
        let f_hc = HcDev::load(src, c, layer, "hc_ffn")?;
        let mut up = Uploader::new()?;
        up.use_prefill = false;
        let shared = (
            up.linear(src, &p("ffn.shared_experts.w1"), inter, hidden)?,
            up.linear(src, &p("ffn.shared_experts.w2"), hidden, inter)?,
            up.linear(src, &p("ffn.shared_experts.w3"), inter, hidden)?,
        );
        let router = up.linear(src, &p("ffn.gate"), d.experts, hidden)?;
        let vector = |s: &str, n: usize| -> Result<DeviceBuffer> {
            Ok(DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(
                src,
                &p(s),
                n,
            )?)?)
        };
        Ok(BlockDev {
            layer,
            hc,
            hidden,
            inter,
            experts: d.experts,
            top_k: d.top_k,
            x: DeviceBuffer::zeroed(hidden * 4)?,
            xf: DeviceBuffer::zeroed(hidden * 4)?,
            xn: DeviceBuffer::zeroed(hidden * 4)?,
            scores: DeviceBuffer::zeroed(d.experts * 4)?,
            ids: DeviceBuffer::zeroed(d.top_k * 4)?,
            wts: DeviceBuffer::zeroed(d.top_k * 4)?,
            gate: DeviceBuffer::zeroed(inter * 4)?,
            up: DeviceBuffer::zeroed(inter * 4)?,
            act: DeviceBuffer::zeroed(inter * 4)?,
            ffn: DeviceBuffer::zeroed(hidden * 4)?,
            h1: DeviceBuffer::zeroed(hc * hidden * 4)?,
            h_out: DeviceBuffer::zeroed(hc * hidden * 4)?,
            ffn_norm: vector("ffn_norm.weight", hidden)?,
            bias: vector("ffn.gate.bias", d.experts)?,
            router,
            shared,
            cache: ExpertCache::new(8)?,
            pre_mix: DeviceBuffer::zeroed(hc * 4)?,
            route_scale: d.route_scale,
            renorm: d.norm_topk_prob,
            limit: d.swiglu_limit,
            eps: d.eps,
            a_hc,
            f_hc,
            attn,
        })
    }

    /// Bytes the device holds: the attention, the shared expert, the router and the experts currently cached.
    pub fn bytes(&self) -> usize {
        fn lin(l: &Linear) -> usize {
            match &l.w {
                Weights::Bf16(b) => b.len(),
                Weights::Mls2 { q, sc }
                | Weights::Mls3 { q, sc }
                | Weights::Mls4 { q, sc }
                | Weights::Mls8 { q, sc } => q.len() + sc.len(),
            }
        }
        let experts: usize = self.cache.map.values().map(|w| lin(&w.0) + lin(&w.1) + lin(&w.2)).sum();
        self.attn.bytes()
            + lin(&self.shared.0)
            + lin(&self.shared.1)
            + lin(&self.shared.2)
            + lin(&self.router)
            + experts
    }

    /// The collapse weights of the previous sub-layer (`identity` before the first one).
    pub fn set_pre_mix(&mut self, pre: &[f32]) -> Result<()> {
        ensure!(
            pre.len() == self.hc,
            "{} pre-mix weights for {} streams",
            pre.len(),
            self.hc
        );
        self.pre_mix.upload(pre)?;
        Ok(())
    }

    /// Forgets the window: a new sequence starts with an empty ring.
    pub fn reset(&mut self) -> Result<()> {
        self.attn.reset()
    }

    /// The attention half of the block: `h` are the `hc_mult * hidden` streams of the input, `pos` the position.
    /// Leaves the streams after the attention sub-layer in `h1`.
    pub fn step_attention(&mut self, m: &Model, h: &DeviceBuffer, pos: usize) -> Result<()> {
        let (s, k) = (&m.stream, &m.k);
        let (hc, hidden) = (self.hc, self.hidden);
        self.a_hc.mixes(m, h)?;
        k.get("dsv_hc_pre").launch(
            (hidden.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.x).into(),
                h.into(),
                (&self.pre_mix).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        self.attn.step(m, &self.x, pos)?;
        k.get("dsv_hc_post").launch(
            ((hc * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.h1).into(),
                (&self.attn.out).into(),
                h.into(),
                (&self.a_hc.post).into(),
                (&self.a_hc.comb).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        Ok(())
    }

    /// One position of the whole block: the attention half, then the FFN half (the router, the six routed experts the
    /// position takes and the shared one), written back over the streams into `h_out`.
    pub fn step(&mut self, m: &Model, h: &DeviceBuffer, src: &dyn TensorSource, pos: usize) -> Result<()> {
        self.step_attention(m, h, pos)?;
        let (s, k) = (&m.stream, &m.k);
        let (hc, hidden, inter) = (self.hc, self.hidden, self.inter);
        let (xr, xh, xsc) = (&self.attn.xr, &self.attn.xh, &self.attn.xsc);
        // The FFN's own mixes, and the collapse with the weights the attention left.
        self.f_hc.mixes(m, &self.h1)?;
        k.get("dsv_hc_pre").launch(
            (hidden.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.xf).into(),
                (&self.h1).into(),
                (&self.a_hc.pre).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        k.get("dsv_rmsnorm").launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.xn).into(),
                (&self.xf).into(),
                (&self.ffn_norm).into(),
                hidden.into(),
                self.eps.into(),
            ],
        )?;
        m.gemm(s, xr, xh, xsc, &self.scores, Act::X(&self.xn), &self.router, 1, false)?;
        k.get("dsv_route").launch(
            (1, 1, 1),
            (32, 1, 1),
            (2 * self.experts * 4) as u32,
            s,
            &[
                (&self.ids).into(),
                (&self.wts).into(),
                (&self.scores).into(),
                (&self.bias).into(),
                self.experts.into(),
                self.top_k.into(),
                i32::from(self.renorm).into(),
                self.route_scale.into(),
                self.experts.into(),
            ],
        )?;
        // The experts' weights are read from the file, so the selection comes back to the host: the one round trip of
        // a step, and the only reason this path is not yet a served one.
        let ids = self.ids.download::<i32>(self.top_k)?;
        let wts = self.wts.download::<f32>(self.top_k)?;
        let limit = self.limit;
        for (slot, &id) in ids.iter().enumerate() {
            let w = self.cache.get(src, &m.cfg, self.layer, id as usize)?;
            m.gemm(s, xr, xh, xsc, &self.gate, Act::X(&self.xn), &w.0, 1, false)?;
            m.gemm(s, xr, xh, xsc, &self.up, Act::X(&self.xn), &w.2, 1, false)?;
            k.get("dsv_swiglu").launch(
                (inter.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                s,
                &[
                    (&self.act).into(),
                    (&self.gate).into(),
                    (&self.up).into(),
                    inter.into(),
                    limit.into(),
                    wts[slot].into(),
                ],
            )?;
            m.gemm(s, xr, xh, xsc, &self.ffn, Act::X(&self.act), &w.1, 1, slot > 0)?;
        }
        // The shared expert, on top of the routed ones.
        m.gemm(s, xr, xh, xsc, &self.gate, Act::X(&self.xn), &self.shared.0, 1, false)?;
        m.gemm(s, xr, xh, xsc, &self.up, Act::X(&self.xn), &self.shared.2, 1, false)?;
        k.get("dsv_swiglu").launch(
            (inter.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.act).into(),
                (&self.gate).into(),
                (&self.up).into(),
                inter.into(),
                limit.into(),
                1f32.into(),
            ],
        )?;
        m.gemm(s, xr, xh, xsc, &self.ffn, Act::X(&self.act), &self.shared.1, 1, true)?;
        k.get("dsv_hc_post").launch(
            ((hc * hidden).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                (&self.h_out).into(),
                (&self.ffn).into(),
                (&self.h1).into(),
                (&self.f_hc.post).into(),
                (&self.f_hc.comb).into(),
                hc.into(),
                hidden.into(),
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Kernels, select_device};

    /// Cosine similarity of two vectors.
    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        d / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
    }

    /// The RMS of a vector and the largest absolute difference between two: a cosine alone does not say whether the
    /// values are a real signal.
    fn rms_and_worst(a: &[f32], b: &[f32]) -> (f64, f64) {
        let rms = (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0f64, f64::max);
        (rms, worst)
    }

    /// The device's window attention against the host graph's, stage by stage, on the same quantized weights and the
    /// same input. Layer 0 (a window layer: no compressor, no indexer) is the first increment of issue #128.
    ///
    ///   DEEPSEEK_FILE=/models/deepseek-v41-flash-mls2.malleus \
    ///     cargo test -p malleus-model --release -- the_device_window_attention_matches_the_host_graph --ignored \
    ///     --nocapture
    ///
    /// `DEEPSEEK_X` names a raw f32 `[t, hidden]` file to feed both sides (the `x0.f32` of a layer dump); without it
    /// the input is synthetic, because the arithmetic is what this gate is about. The whole model's own logits, and
    /// the end-to-end gate the issue describes, come when the whole layer runs on the device.
    #[test]
    #[ignore = "reads a real layer from the converted file and needs the GPU"]
    fn the_device_window_attention_matches_the_host_graph() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let layer: usize = std::env::var("DEEPSEEK_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&file)).unwrap();
        let c = Config::from_json(
            g.meta_str("openmalleus.hf_config").unwrap(),
            g.meta_str("openmalleus.generation_config").ok(),
        )
        .unwrap();
        let t: usize = std::env::var("DEEPSEEK_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let x: Vec<f32> = match std::env::var("DEEPSEEK_X") {
            Ok(path) => {
                let raw = std::fs::read(&path).unwrap();
                raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
            }
            Err(_) => {
                // A deterministic input of unit variance: enough to touch every weight of the layer.
                let mut state = 0x2545_f491_4f6c_dd1du64;
                (0..t * c.hidden)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 40) as f32 / 8388608.0) - 1.0
                    })
                    .collect()
            }
        };
        let t = x.len() / c.hidden;
        let eps = c.deepseek.as_ref().unwrap().eps;

        // The host side: one block's attention on the converted file's own weights.
        let t0 = std::time::Instant::now();
        let host = crate::deepseek_text::Weights::block_only(&g, &c, layer).unwrap();
        eprintln!("host: one block loaded in {:.0}s", t0.elapsed().as_secs_f64());
        let attn_norm =
            crate::deepseek_text::load_vector(&g, &format!("layers.{layer}.attn_norm.weight"), c.hidden).unwrap();
        let mut xn = x.clone();
        crate::hostmath::rmsnorm(&mut xn, c.hidden, &attn_norm, eps);
        let mut stages: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        // `block_only` loads one block, at index 0: its hooks are named after that index, not after the layer id.
        let host_out = host
            .attention_stage_with(0, &xn, t, |name, values| {
                stages.insert(name.to_string(), values.to_vec());
                Ok(())
            })
            .unwrap();

        // The device side: the same input, one position at a time.
        let kernels = Kernels::load(&select_device().unwrap()).unwrap();
        let m = Model::host_only(c.clone(), kernels, 0, malleus_gguf::quant::HADAMARD_SEED).unwrap();
        let mut dev = WindowAttn::load(&g, &c, layer).unwrap();
        eprintln!("device: {:.2} GB for the layer's attention", dev.bytes() as f64 / 1e9);
        let rows: Vec<DeviceBuffer> = (0..t)
            .map(|i| DeviceBuffer::from_slice(&x[i * c.hidden..(i + 1) * c.hidden]).unwrap())
            .collect();
        m.stream.synchronize().unwrap();
        // The steps alone, as a served decode runs them, then the same steps reading the layer's output back.
        let t0 = std::time::Instant::now();
        for (pos, row) in rows.iter().enumerate() {
            dev.step(&m, row, pos).unwrap();
        }
        m.stream.synchronize().unwrap();
        eprintln!(
            "device: {:.2} ms a position",
            t0.elapsed().as_secs_f64() / t as f64 * 1e3
        );
        dev.reset().unwrap();
        let mut dev_out = vec![0f32; t * c.hidden];
        let t0 = std::time::Instant::now();
        for (pos, row) in rows.iter().enumerate() {
            dev.step(&m, row, pos).unwrap();
            dev_out[pos * c.hidden..(pos + 1) * c.hidden].copy_from_slice(&dev.out.download::<f32>(c.hidden).unwrap());
        }
        m.stream.synchronize().unwrap();
        eprintln!(
            "device: {:.2} ms a position with the readback",
            t0.elapsed().as_secs_f64() / t as f64 * 1e3
        );

        // The last position is where the stages are compared: it reads a full window.
        let at = t - 1;
        for (stage, buf, width) in [
            ("q_a", &dev.qr, dev.q_lora),
            ("q_rope", &dev.q, dev.heads * dev.hd),
            ("kv_rope", &dev.kv, dev.hd),
            ("att_raw", &dev.att, dev.heads * dev.hd),
        ] {
            let name = format!("{stage}0");
            let got = buf.download::<f32>(width).unwrap();
            let want = stages
                .get(&name)
                .unwrap_or_else(|| panic!("the host graph has no {name}"));
            let want = &want[at * width..(at + 1) * width];
            let (rms, worst) = rms_and_worst(want, &got);
            let cos = cos(&got, want);
            eprintln!("{name}: cosine {cos:.8}, rms {rms:.4}, max |diff| {worst:.2e}");
            assert!(cos > 0.9999, "{name}: {cos}");
        }
        let mut worst = 1.0f64;
        for pos in 0..t {
            let cos = cos(
                &dev_out[pos * c.hidden..(pos + 1) * c.hidden],
                &host_out[pos * c.hidden..][..c.hidden],
            );
            worst = worst.min(cos);
        }
        let last = cos(&dev_out[at * c.hidden..], &host_out[at * c.hidden..]);
        let (rms, worst_abs) = rms_and_worst(&host_out, &dev_out);
        eprintln!("out: rms {rms:.4}, max |diff| {worst_abs:.2e}, worst position {worst:.8}, last {last:.8}");
        assert!(worst > 0.9999, "a position's attention diverges: {worst}");
    }

    /// The attention half of a block — the hyper-connections, the collapse, the attention and the write back over the
    /// streams — against the host graph, stage by stage. This is what a device block is built on; the FFN half comes
    /// next. The input is the block's streams (`hc_mult` copies of `hidden`, one position each), synthetic unless
    /// `DEEPSEEK_H` names a raw f32 file of them.
    ///
    ///   DEEPSEEK_FILE=/models/deepseek-v41-flash-mls2.malleus \
    ///     cargo test -p malleus-model --release -- --ignored --nocapture the_device_attention_half_block
    #[test]
    #[ignore = "reads a real layer from the converted file and needs the GPU"]
    fn the_device_attention_half_block_matches_the_host_graph() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let layer: usize = std::env::var("DEEPSEEK_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&file)).unwrap();
        let c = Config::from_json(
            g.meta_str("openmalleus.hf_config").unwrap(),
            g.meta_str("openmalleus.generation_config").ok(),
        )
        .unwrap();
        let d = c.deepseek.as_ref().unwrap();
        let (hc, hidden) = (d.hc_mult, c.hidden);
        let t: usize = std::env::var("DEEPSEEK_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let h: Vec<f32> = match std::env::var("DEEPSEEK_H") {
            Ok(path) => {
                let raw = std::fs::read(&path).unwrap();
                raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
            }
            Err(_) => {
                // Distinct streams, not the copies of a first block: the write-back is what this gate is for.
                let mut state = 0x9e37_79b9_7f4a_7c15u64;
                (0..t * hc * hidden)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 40) as f32 / 4194304.0) - 2.0
                    })
                    .collect()
            }
        };
        let t = h.len() / (hc * hidden);
        // The collapse weights of the previous sub-layer. The first one is the identity (stream 0 only), which is
        // what a block starts from; here they are a sigmoid's worth each, so the collapse is not a copy.
        let pre_mix: Vec<f32> = if std::env::var("DEEPSEEK_H").is_ok() {
            crate::deepseek_text::identity_mix(t, hc)
        } else {
            let mut state = 0x0123_4567_89ab_cdefu64;
            (0..t * hc)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    0.2 + 0.8 * ((state >> 40) as f32 / 8388608.0)
                })
                .collect()
        };

        let host = crate::deepseek_text::Weights::block_only(&g, &c, layer).unwrap();
        let mut stages: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        let host_out = host
            .attention_block_stage_with(0, &h, &pre_mix, t, |name, values| {
                stages.insert(name.to_string(), values.to_vec());
                Ok(())
            })
            .unwrap();

        let kernels = Kernels::load(&select_device().unwrap()).unwrap();
        let m = Model::host_only(c.clone(), kernels, 0, malleus_gguf::quant::HADAMARD_SEED).unwrap();
        let mut dev = BlockDev::load(&g, &c, layer).unwrap();
        let rows: Vec<DeviceBuffer> = (0..t)
            .map(|i| DeviceBuffer::from_slice(&h[i * hc * hidden..(i + 1) * hc * hidden]).unwrap())
            .collect();
        m.stream.synchronize().unwrap();
        let steps = |dev: &mut BlockDev, read_back: bool| -> Vec<f32> {
            let mut out = vec![0f32; t * hc * hidden];
            for (pos, row) in rows.iter().enumerate() {
                dev.set_pre_mix(&pre_mix[pos * hc..(pos + 1) * hc]).unwrap();
                dev.step_attention(&m, row, pos).unwrap();
                if read_back {
                    out[pos * hc * hidden..(pos + 1) * hc * hidden]
                        .copy_from_slice(&dev.h1.download::<f32>(hc * hidden).unwrap());
                }
            }
            out
        };
        // The steps alone, as a served decode runs them, then the same with the readback of the streams.
        let t0 = std::time::Instant::now();
        steps(&mut dev, false);
        m.stream.synchronize().unwrap();
        eprintln!(
            "device: {:.2} ms a position for the whole attention half",
            t0.elapsed().as_secs_f64() / t as f64 * 1e3
        );
        dev.attn.reset().unwrap();
        let dev_out = steps(&mut dev, true);
        m.stream.synchronize().unwrap();

        let at = t - 1;
        let mut worst = 1.0f64;
        for (stage, buf, width) in [
            ("hc_attn_pre", &dev.a_hc.pre, hc),
            ("hc_attn_post", &dev.a_hc.post, hc),
            ("hc_attn_comb", &dev.a_hc.comb, hc * hc),
            ("hc_attn_x", &dev.x, hidden),
            ("attn", &dev.attn.out, hidden),
        ] {
            let name = format!("{stage}0");
            let got = buf.download::<f32>(width).unwrap();
            let want = stages
                .get(&name)
                .unwrap_or_else(|| panic!("the host graph has no {name}"));
            let want = &want[at * width..(at + 1) * width];
            let (rms, worst_abs) = rms_and_worst(want, &got);
            let cos = cos(&got, want);
            eprintln!("{name}: cosine {cos:.8}, rms {rms:.4}, max |diff| {worst_abs:.2e}");
            worst = worst.min(cos);
            assert!(cos > 0.9999, "{name}: {cos}");
        }
        let mut stream_worst = 1.0f64;
        for pos in 0..t {
            let w = hc * hidden;
            stream_worst = stream_worst.min(cos(&dev_out[pos * w..(pos + 1) * w], &host_out[pos * w..][..w]));
        }
        let (rms, worst_abs) = rms_and_worst(&host_out, &dev_out);
        eprintln!(
            "streams: rms {rms:.4}, max |diff| {worst_abs:.2e}, worst position {stream_worst:.8}, stages {worst:.8}"
        );
        assert!(stream_worst > 0.9999, "a position's streams diverge: {stream_worst}");
    }

    /// The whole block on the device — the attention half and the FFN half (router, the six routed experts of each
    /// position and the shared one) — against the host graph. Layer 0 is the only block this can run alone: the
    /// indexed layers read a compressed KV their source layers publish, and layer 1 carries an Engram table.
    ///
    ///   DEEPSEEK_FILE=/models/deepseek-v41-flash-mls2.malleus \
    ///     cargo test -p malleus-model --release -- --ignored --nocapture the_device_block
    #[test]
    #[ignore = "reads a real block from the converted file and needs the GPU"]
    fn the_device_block_matches_the_host_graph() {
        let file = std::env::var("DEEPSEEK_FILE").expect("DEEPSEEK_FILE");
        let g = malleus_gguf::gguf::Gguf::open(std::path::Path::new(&file)).unwrap();
        let c = Config::from_json(
            g.meta_str("openmalleus.hf_config").unwrap(),
            g.meta_str("openmalleus.generation_config").ok(),
        )
        .unwrap();
        let d = c.deepseek.as_ref().unwrap();
        let (hc, hidden) = (d.hc_mult, c.hidden);
        let t: usize = std::env::var("DEEPSEEK_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let h: Vec<f32> = {
            let mut state = 0x2545_f491_4f6c_dd1du64;
            (0..t * hc * hidden)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / 4194304.0) - 2.0
                })
                .collect()
        };
        let t = h.len() / (hc * hidden);
        // The block starts from the identity collapse, as the first sub-layer of the stack does.
        let pre_mix = crate::deepseek_text::identity_mix(t, hc);

        let host = crate::deepseek_text::Weights::block_only(&g, &c, 0).unwrap();
        let mut bank = crate::deepseek_text::StreamedExperts::new(&g, &c, 0, 8);
        let mut stages: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        let t0 = std::time::Instant::now();
        let host_out = host
            .block_stage_with(0, &h, &pre_mix, t, &mut bank, |name, values| {
                stages.insert(name.to_string(), values.to_vec());
                Ok(())
            })
            .unwrap();
        eprintln!(
            "host: one block and {} experts read in {:.1}s",
            bank.read,
            t0.elapsed().as_secs_f64()
        );

        let kernels = Kernels::load(&select_device().unwrap()).unwrap();
        let m = Model::host_only(c.clone(), kernels, 0, malleus_gguf::quant::HADAMARD_SEED).unwrap();
        let mut dev = BlockDev::load(&g, &c, 0).unwrap();
        eprintln!("device: {:.2} GB loaded", dev.bytes() as f64 / 1e9);
        let rows: Vec<DeviceBuffer> = (0..t)
            .map(|i| DeviceBuffer::from_slice(&h[i * hc * hidden..(i + 1) * hc * hidden]).unwrap())
            .collect();
        m.stream.synchronize().unwrap();
        let mut dev_out = vec![0f32; t * hc * hidden];
        let mut dev_ffn = vec![0f32; t * hidden];
        let mut dev_ids = Vec::with_capacity(t * d.top_k);
        let mut dev_wts = Vec::with_capacity(t * d.top_k);
        let t0 = std::time::Instant::now();
        for (pos, row) in rows.iter().enumerate() {
            dev.set_pre_mix(&pre_mix[pos * hc..(pos + 1) * hc]).unwrap();
            dev.step(&m, row, &g, pos).unwrap();
            dev_out[pos * hc * hidden..(pos + 1) * hc * hidden]
                .copy_from_slice(&dev.h_out.download::<f32>(hc * hidden).unwrap());
            dev_ffn[pos * hidden..(pos + 1) * hidden].copy_from_slice(&dev.ffn.download::<f32>(hidden).unwrap());
            dev_ids.extend(dev.ids.download::<i32>(d.top_k).unwrap());
            dev_wts.extend(dev.wts.download::<f32>(d.top_k).unwrap());
        }
        m.stream.synchronize().unwrap();
        eprintln!(
            "device: {:.0} ms a position for the whole block, {:.2} GB held",
            t0.elapsed().as_secs_f64() / t as f64 * 1e3,
            dev.bytes() as f64 / 1e9
        );

        // Every position's selection, then the FFN's output and the streams, over the whole chunk.
        let ids = stages.get("ffn_ids0").unwrap();
        let wts = stages.get("ffn_w0").unwrap();
        let top_k = d.top_k;
        for pos in 0..t {
            let got: Vec<usize> = dev_ids[pos * top_k..(pos + 1) * top_k]
                .iter()
                .map(|v| *v as usize)
                .collect();
            let want: Vec<usize> = ids[pos * top_k..(pos + 1) * top_k]
                .iter()
                .map(|v| *v as usize)
                .collect();
            let (mut a, mut b) = (want.clone(), got.clone());
            a.sort_unstable();
            b.sort_unstable();
            eprintln!("router position {pos}: host {want:?}, device {got:?}");
            assert_eq!(a, b, "position {pos}: a different set of experts");
            for (slot, id) in want.iter().enumerate() {
                let k = got.iter().position(|v| v == id).unwrap();
                let want_w = wts[pos * top_k + slot];
                let w = dev_wts[pos * top_k + k];
                assert!(
                    (w - want_w).abs() <= 1e-5 * want_w.abs().max(1.0),
                    "position {pos}, expert {id}: weight {w} against {want_w}"
                );
            }
        }
        let ffn_want = stages.get("ffn0").unwrap();
        assert_eq!(
            ffn_want.len(),
            t * hidden,
            "the host's FFN output has a row per position"
        );
        let mut ffn_worst = 1.0f64;
        for pos in 0..t {
            let (want, got) = (
                &ffn_want[pos * hidden..(pos + 1) * hidden],
                &dev_ffn[pos * hidden..(pos + 1) * hidden],
            );
            ffn_worst = ffn_worst.min(cos(got, want));
        }
        let (rms, worst) = rms_and_worst(ffn_want, &dev_ffn);
        eprintln!("ffn0: worst position {ffn_worst:.8}, rms {rms:.4}, max |diff| {worst:.2e}");
        assert!(ffn_worst > 0.999, "the FFN diverges: {ffn_worst}");
        let (rms, worst) = rms_and_worst(&host_out, &dev_out);
        let cos = cos(&dev_out, &host_out);
        eprintln!("block0: cosine {cos:.8}, rms {rms:.4}, max |diff| {worst:.2e}");
        assert!(cos > 0.999, "the block diverges: {cos}");
    }
}
