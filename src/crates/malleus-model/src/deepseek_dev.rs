//! DeepSeek-V4.1 on the device (issue #128): the sliding-window attention of one layer, and the harness that
//! compares it against the host graph stage by stage.
//!
//! The host graph (`crate::deepseek_text`) is the specification, gated against the release's own implementation on
//! real weights (issues #83, #121). A kernel is written against it, one stage at a time: this increment is the
//! attention of a *window* layer (layer 0: no compressor, no indexer), one position at a time, which is what a decode
//! step runs. The shapes are `deepseek_text::Attn`: 64 heads of 512, `q_lora` 1280, 8 output groups of 1024, a
//! 128-row ring of raw latents and a learned sink per head.
//!
//! What is *not* here yet: the hyper-connections that produce the attention's input, the MoE, the Engram gather, the
//! compressor and the two-level indexer of the indexed layers, and the prefill (this path takes one position, the
//! way the decode does). The gate is the host graph itself: the same quantized weights, the same input, a cosine per
//! stage.

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
        let mut dev_out = vec![0f32; t * c.hidden];
        let t0 = std::time::Instant::now();
        for (pos, row) in rows.iter().enumerate() {
            dev.step(&m, row, pos).unwrap();
            dev_out[pos * c.hidden..(pos + 1) * c.hidden].copy_from_slice(&dev.out.download::<f32>(c.hidden).unwrap());
        }
        m.stream.synchronize().unwrap();
        eprintln!(
            "device: {:.2} ms a position",
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
}
