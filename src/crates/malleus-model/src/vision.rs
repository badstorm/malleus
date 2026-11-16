use crate::image_pre::Preprocessed;
use crate::kernels::Kernels;
use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};
use malleus_hip::{Arg, DeviceBuffer, Stream};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub out_hidden: usize,
    pub num_pos: usize,
    pub rope_theta: f32,
}

impl VisionConfig {
    pub fn from_json(config: &str) -> Result<Option<Self>> {
        let cfg: Value = serde_json::from_str(config)?;
        let v = &cfg["vision_config"];
        if v.is_null() {
            return Ok(None);
        }
        let u = |k: &str| {
            v[k].as_u64()
                .map(|x| x as usize)
                .with_context(|| format!("vision_config.{k}"))
        };
        Ok(Some(VisionConfig {
            depth: u("depth")?,
            hidden: u("hidden_size")?,
            inter: u("intermediate_size")?,
            heads: u("num_heads")?,
            patch: u("patch_size")?,
            temporal: u("temporal_patch_size")?,
            merge: u("spatial_merge_size")?,
            out_hidden: u("out_hidden_size")?,
            num_pos: u("num_position_embeddings")?,
            rope_theta: v["rope_parameters"]["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        }))
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    pub fn patch_dim(&self) -> usize {
        3 * self.temporal * self.patch * self.patch
    }
}

struct Lin {
    w: DeviceBuffer,
    b: DeviceBuffer,
    n: usize,
    k: usize,
}

struct Block {
    norm1: (DeviceBuffer, DeviceBuffer),
    norm2: (DeviceBuffer, DeviceBuffer),
    qkv: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

pub struct VisionModel {
    pub cfg: VisionConfig,
    patch_embed: Lin,
    pos_table: Vec<f32>,
    blocks: Vec<Block>,
    merger_norm: (DeviceBuffer, DeviceBuffer),
    merger_fc1: Lin,
    merger_fc2: Lin,
    pub weight_bytes: usize,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|x| f32::from_bits((u16::from_le_bytes(*x) as u32) << 16))
        .collect()
}

fn f32s(n: usize) -> Result<DeviceBuffer> {
    Ok(DeviceBuffer::alloc(n * 4)?)
}

impl VisionModel {
    pub fn load(cfg: VisionConfig, src: &dyn TensorSource, prefix: &str) -> Result<Self> {
        let bytes = std::cell::Cell::new(0usize);
        let raw = |name: &str, numel: usize| -> Result<DeviceBuffer> {
            let full = format!("{prefix}{name}");
            let info = src.info(&full).with_context(|| format!("missing tensor {full}"))?;
            ensure!(
                info.dtype == DType::BF16 && info.numel() == numel,
                "{full}: unexpected dtype/shape {:?} {:?}",
                info.dtype,
                info.shape
            );
            let data = src.read_all(&full)?;
            bytes.set(bytes.get() + data.len());
            Ok(DeviceBuffer::from_slice(&data)?)
        };
        let lin = |name: &str, n: usize, k: usize| -> Result<Lin> {
            Ok(Lin {
                w: raw(&format!("{name}.weight"), n * k)?,
                b: raw(&format!("{name}.bias"), n)?,
                n,
                k,
            })
        };
        let (h, i) = (cfg.hidden, cfg.inter);
        let patch_embed = lin("patch_embed.proj", h, cfg.patch_dim())?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for d in 0..cfg.depth {
            let p = format!("blocks.{d}");
            blocks.push(Block {
                norm1: (
                    raw(&format!("{p}.norm1.weight"), h)?,
                    raw(&format!("{p}.norm1.bias"), h)?,
                ),
                norm2: (
                    raw(&format!("{p}.norm2.weight"), h)?,
                    raw(&format!("{p}.norm2.bias"), h)?,
                ),
                qkv: lin(&format!("{p}.attn.qkv"), 3 * h, h)?,
                proj: lin(&format!("{p}.attn.proj"), h, h)?,
                fc1: lin(&format!("{p}.mlp.linear_fc1"), i, h)?,
                fc2: lin(&format!("{p}.mlp.linear_fc2"), h, i)?,
            });
        }
        let mh = h * cfg.merge * cfg.merge;
        let merger_norm = (raw("merger.norm.weight", h)?, raw("merger.norm.bias", h)?);
        let merger_fc1 = lin("merger.linear_fc1", mh, mh)?;
        let merger_fc2 = lin("merger.linear_fc2", cfg.out_hidden, mh)?;
        let pos_name = format!("{prefix}pos_embed.weight");
        let pos_table = bf16_to_f32(&src.read_all(&pos_name)?);
        ensure!(pos_table.len() == cfg.num_pos * h, "pos_embed shape");
        Ok(VisionModel {
            cfg,
            patch_embed,
            pos_table,
            blocks,
            merger_norm,
            merger_fc1,
            merger_fc2,
            weight_bytes: bytes.get(),
        })
    }

    /// Bilinear (align_corners) resampling of the learned position grid, in merge-block patch order.
    fn pos_embeds(&self, gh: usize, gw: usize) -> Vec<f32> {
        let side = (self.cfg.num_pos as f64).sqrt() as usize;
        let h = self.cfg.hidden;
        let m = self.cfg.merge;
        let taps = |idx: usize, size: usize| -> [(usize, f32); 2] {
            let src = if size > 1 {
                idx as f32 * (side - 1) as f32 / (size - 1) as f32
            } else {
                0.0
            };
            let f = src.floor();
            let lo = (f as usize).min(side - 1);
            let hi = (lo + 1).min(side - 1);
            let fr = src - f;
            [(lo, 1.0 - fr), (hi, fr)]
        };
        let mut out = vec![0f32; gh * gw * h];
        let mut n = 0;
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for ih in 0..m {
                    for iw in 0..m {
                        let (r, c) = (bh * m + ih, bw * m + iw);
                        let row = &mut out[n * h..(n + 1) * h];
                        for (ty, wy) in taps(r, gh) {
                            for (tx, wx) in taps(c, gw) {
                                let wgt = wy * wx;
                                if wgt == 0.0 {
                                    continue;
                                }
                                let src = &self.pos_table[(ty * side + tx) * h..(ty * side + tx + 1) * h];
                                for (o, s) in row.iter_mut().zip(src) {
                                    *o += wgt * s;
                                }
                            }
                        }
                        n += 1;
                    }
                }
            }
        }
        out
    }

    fn linear(
        &self,
        k: &Kernels,
        s: &Stream,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        l: &Lin,
        t: usize,
        act: i32,
    ) -> Result<()> {
        let grid = (l.n.div_ceil(8) as u32, t.div_ceil(8) as u32, 1);
        let args: [Arg; 7] = [
            out.into(),
            x.into(),
            (&l.w).into(),
            t.into(),
            l.n.into(),
            l.k.into(),
            0i32.into(),
        ];
        k.get("gemm_bf16").launch(grid, (256, 1, 1), 0, s, &args)?;
        let n = t * l.n;
        k.get("bias_act").launch(
            (n.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[out.into(), (&l.b).into(), t.into(), l.n.into(), act.into()],
        )?;
        Ok(())
    }

    fn layernorm(
        &self,
        k: &Kernels,
        s: &Stream,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        w: &(DeviceBuffer, DeviceBuffer),
        rows: usize,
        n: usize,
    ) -> Result<()> {
        k.get("layernorm").launch(
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            s,
            &[
                out.into(),
                x.into(),
                (&w.0).into(),
                (&w.1).into(),
                n.into(),
                1e-6f32.into(),
            ],
        )?;
        Ok(())
    }

    /// Returns (vit_out [P, hidden], merged [P / merge², out_hidden]).
    pub fn encode(&self, k: &Kernels, pre: &Preprocessed) -> Result<(DeviceBuffer, DeviceBuffer)> {
        let c = &self.cfg;
        let (h, p) = (c.hidden, pre.n_patches());
        ensure!(pre.grid_t == 1, "video input not supported");
        ensure!(pre.pixels.len() == p * c.patch_dim(), "pixel buffer size mismatch");
        let s = Stream::null();
        let px = DeviceBuffer::from_slice(&pre.pixels)?;
        let x = f32s(p * h)?;
        self.linear(k, &s, &x, &px, &self.patch_embed, p, 0)?;
        let pos = DeviceBuffer::from_slice(&self.pos_embeds(pre.grid_h, pre.grid_w))?;
        k.get("add_inplace").launch(
            ((p * h).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &s,
            &[(&x).into(), (&pos).into(), (p * h).into()],
        )?;
        let m = c.merge;
        let mut hw = Vec::with_capacity(p * 2);
        for bh in 0..pre.grid_h / m {
            for bw in 0..pre.grid_w / m {
                for ih in 0..m {
                    for iw in 0..m {
                        hw.push((bh * m + ih) as i32);
                        hw.push((bw * m + iw) as i32);
                    }
                }
            }
        }
        let hw = DeviceBuffer::from_slice(&hw)?;
        let hn = f32s(p * h)?;
        let qkv = f32s(p * 3 * h)?;
        let att = f32s(p * h)?;
        let mlp = f32s(p * c.inter)?;
        let hd = c.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        for b in &self.blocks {
            self.layernorm(k, &s, &hn, &x, &b.norm1, p, h)?;
            self.linear(k, &s, &qkv, &hn, &b.qkv, p, 0)?;
            k.get("vision_rope").launch(
                (p as u32, c.heads as u32, 2),
                (128, 1, 1),
                0,
                &s,
                &[
                    (&qkv).into(),
                    (&hw).into(),
                    c.heads.into(),
                    hd.into(),
                    c.rope_theta.into(),
                ],
            )?;
            k.get("vision_attention").launch(
                (p as u32, c.heads as u32, 1),
                (256, 1, 1),
                0,
                &s,
                &[
                    (&att).into(),
                    (&qkv).into(),
                    p.into(),
                    c.heads.into(),
                    hd.into(),
                    scale.into(),
                ],
            )?;
            self.linear(k, &s, &hn, &att, &b.proj, p, 0)?;
            k.get("add_inplace").launch(
                ((p * h).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &[(&x).into(), (&hn).into(), (p * h).into()],
            )?;
            self.layernorm(k, &s, &hn, &x, &b.norm2, p, h)?;
            self.linear(k, &s, &mlp, &hn, &b.fc1, p, 1)?;
            self.linear(k, &s, &hn, &mlp, &b.fc2, p, 0)?;
            k.get("add_inplace").launch(
                ((p * h).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                &[(&x).into(), (&hn).into(), (p * h).into()],
            )?;
        }
        let pm = p / (m * m);
        self.layernorm(k, &s, &hn, &x, &self.merger_norm, p, h)?;
        let mid = f32s(pm * h * m * m)?;
        self.linear(k, &s, &mid, &hn, &self.merger_fc1, pm, 2)?;
        let out = f32s(pm * c.out_hidden)?;
        self.linear(k, &s, &out, &mid, &self.merger_fc2, pm, 0)?;
        s.synchronize()?;
        Ok((x, out))
    }
}
