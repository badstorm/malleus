//! DeepSeek-V4.1 on the device (issue #128): the sliding-window attention of one layer, and the harness that
//! compares it against the host graph stage by stage.
//!
//! The host graph (`crate::deepseek_text`) is the specification, gated against the release's own implementation on
//! real weights (issues #83, #121). A kernel is written against it, one stage at a time, one position at a time —
//! which is what a decode step runs. Two increments live here:
//!
//! - `WindowAttn`: the attention of a *window* layer (layer 0: no compressor, no indexer). The shapes are
//!   `deepseek_text::Attn`: 64 heads of 512, `q_lora` 1280, 8 output groups of 1024, a 128-row ring of raw latents and
//!   a learned sink per head.
//! - `BlockAttn`: the attention half of a block around it — the sub-layer's hyper-connections (the mixes, the
//!   collapse of the streams and the write back over them, Sinkhorn included).
//!
//! The attention reads 127 MB of weights a position for one layer (5.1 GB over the stack, the number the issue
//! starts from) and runs at 1.2 ms against a 0.55 ms memory wall: it is launch-bound at twenty launches a step, the
//! shape the other families hit before their decode was fused.
//!
//! What is *not* here yet: the FFN half (the router, the shared expert and the 384 routed experts at 2 bits), the
//! Engram gather, the compressor and the two-level indexer of the indexed layers, and the prefill. The gate is the
//! host graph itself: the same quantized weights, the same input, a cosine per stage.

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
    /// GEMM scratch of the widest input a projection takes (`wo_b`, 8192 wide) for one token.
    xr: DeviceBuffer,
    xh: DeviceBuffer,
    xsc: DeviceBuffer,
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

/// One block's attention half on the device: the hyper-connections of the sub-layer (the mixes, the collapse and
/// the write back over the streams) and then `WindowAttn`. `step` takes the `hc_mult` streams of one position and
/// leaves the new streams in `h_out`.
pub struct BlockAttn {
    pub attn: WindowAttn,
    pub hc: usize,
    pub hidden: usize,
    /// The mixes of the sub-layer and the three coefficient sets they split into: `pre` (the collapse), `post` (the
    /// write back) and `comb` (the old streams' share, a doubly stochastic `hc x hc` matrix).
    pub mix: DeviceBuffer,
    pub pre: DeviceBuffer,
    pub post: DeviceBuffer,
    pub comb: DeviceBuffer,
    /// The sub-layer's input, collapsed from the streams.
    pub x: DeviceBuffer,
    pub h_out: DeviceBuffer,
    f: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    /// The collapse weights of the previous sub-layer (one value per stream), set by the caller each step.
    pre_mix: DeviceBuffer,
    iters: usize,
    eps: f32,
}

impl BlockAttn {
    pub fn load(src: &dyn TensorSource, c: &Config, layer: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let (hc, hidden) = (d.hc_mult, c.hidden);
        ensure!(
            hc * hc <= 64,
            "the hyper-connection kernel keeps a {hc}x{hc} matrix in shared memory"
        );
        let p = |s: &str| format!("layers.{layer}.{s}");
        let mixes = (2 + hc) * hc;
        let attn = WindowAttn::load(src, c, layer)?;
        Ok(BlockAttn {
            hc,
            hidden,
            mix: DeviceBuffer::zeroed(mixes * 4)?,
            pre: DeviceBuffer::zeroed(hc * 4)?,
            post: DeviceBuffer::zeroed(hc * 4)?,
            comb: DeviceBuffer::zeroed(hc * hc * 4)?,
            x: DeviceBuffer::zeroed(hidden * 4)?,
            h_out: DeviceBuffer::zeroed(hc * hidden * 4)?,
            f: DeviceBuffer::from_slice(&crate::deepseek_text::load_mat(
                src,
                &p("hc_attn_fn"),
                mixes,
                hc * hidden,
            )?)?,
            base: DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(src, &p("hc_attn_base"), mixes)?)?,
            scale: DeviceBuffer::from_slice(&crate::deepseek_text::load_vector(src, &p("hc_attn_scale"), 3)?)?,
            attn,
            pre_mix: DeviceBuffer::zeroed(hc * 4)?,
            iters: d.hc_iters,
            eps: d.hc_eps,
        })
    }

    /// The collapse weights of the previous sub-layer, one per stream (`identity` at the first one).
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

    /// One position: `h` is the `hc_mult * hidden` streams of the block's input, `pos` the position in the sequence.
    pub fn step(&mut self, m: &Model, h: &DeviceBuffer, pos: usize) -> Result<()> {
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
                (&self.h_out).into(),
                (&self.attn.out).into(),
                h.into(),
                (&self.post).into(),
                (&self.comb).into(),
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
        let mut dev = BlockAttn::load(&g, &c, layer).unwrap();
        let rows: Vec<DeviceBuffer> = (0..t)
            .map(|i| DeviceBuffer::from_slice(&h[i * hc * hidden..(i + 1) * hc * hidden]).unwrap())
            .collect();
        m.stream.synchronize().unwrap();
        let steps = |dev: &mut BlockAttn, read_back: bool| -> Vec<f32> {
            let mut out = vec![0f32; t * hc * hidden];
            for (pos, row) in rows.iter().enumerate() {
                dev.set_pre_mix(&pre_mix[pos * hc..(pos + 1) * hc]).unwrap();
                dev.step(&m, row, pos).unwrap();
                if read_back {
                    out[pos * hc * hidden..(pos + 1) * hc * hidden]
                        .copy_from_slice(&dev.h_out.download::<f32>(hc * hidden).unwrap());
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
            ("hc_attn_pre", &dev.pre, hc),
            ("hc_attn_post", &dev.post, hc),
            ("hc_attn_comb", &dev.comb, hc * hc),
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
}
