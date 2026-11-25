//! The vision tower of DeepSeek-V4.1: the processor, the 32-block ViT and the aligner (issue #83).
//!
//! The tower is 1.0 GB and runs on one machine, so it is the part of this model that can be checked today, before
//! the 510 GB text graph runs: `scripts/deepseek_vision_ref.py` dumps what the reference implementation produces for
//! an image, and the tests here compare against it.
//!
//! The weights stay on the host (bf16 read once at load, f32 math): an image is one call of a few seconds, and the
//! tower is not on the decode path. The kernels move to the device when the text forward needs them there.

use anyhow::{Context, Result, ensure};
use malleus_gguf::{DType, TensorSource};

use crate::config::{Config, DeepseekVision};
use crate::hostmath::{dot, gelu, linear, parallel_chunks, rmsnorm, silu};

/// 2-D RoPE frequencies of a patch grid: row and column positions over the first and second half of the rotated width.
fn rope_table(gh: usize, gw: usize, rope: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rope / 2;
    let n = gh * gw;
    let inv: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f32 / rope as f32))
        .collect();
    let mut cos = vec![0f32; n * rope];
    let mut sin = vec![0f32; n * rope];
    for p in 0..n {
        let (row, col) = ((p / gw) as f32, (p % gw) as f32);
        for j in 0..half {
            let (a, b) = (row * inv[j], col * inv[j]);
            cos[p * rope + j] = a.cos();
            cos[p * rope + half + j] = b.cos();
            sin[p * rope + j] = a.sin();
            sin[p * rope + half + j] = b.sin();
        }
    }
    (cos, sin)
}

/// Bidirectional attention of every head over the whole image, with the 2-D RoPE already applied to q and k.
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    const BAND: usize = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; n * heads * head_dim];
    let per_band = BAND * heads * head_dim;
    parallel_chunks(&mut out, per_band, |band, o| {
        let row0 = band * BAND;
        let rows = o.len() / (heads * head_dim);
        let mut scores = vec![0f32; rows * n];
        for h in 0..heads {
            for r in 0..rows {
                let qr = &q[(row0 + r) * heads * head_dim + h * head_dim..][..head_dim];
                let sc = &mut scores[r * n..(r + 1) * n];
                for (j, sj) in sc.iter_mut().enumerate() {
                    let kj = &k[j * heads * head_dim + h * head_dim..][..head_dim];
                    *sj = dot(qr, kj, head_dim) * scale;
                }
                let max = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in sc.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let orow = &mut o[r * heads * head_dim + h * head_dim..][..head_dim];
                for (j, &p) in sc.iter().enumerate() {
                    let w = p * inv;
                    let vj = &v[j * heads * head_dim + h * head_dim..][..head_dim];
                    for t in 0..head_dim {
                        orow[t] += w * vj[t];
                    }
                }
            }
        }
    });
    out
}

use crate::image_pre::{PreConfig, decode, resize_bicubic};

/// One pre-LN ViT block: a fused qkv with biases, the output projection with its bias, and a gated MLP.
pub struct VBlock {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Vec<f32>,
    qkv_b: Vec<f32>,
    o: Vec<f32>,
    o_b: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

/// The tower and the aligner, f32 on the host.
pub struct VisionWeights {
    pub cfg: DeepseekVision,
    /// Text hidden width the aligner projects to.
    pub dim: usize,
    pub head_dim: usize,
    /// Rotated width of a head: `head_dim / 2`, the number of 2-D RoPE frequencies.
    pub rope: usize,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    blocks: Vec<VBlock>,
    norm: Vec<f32>,
    a1: Vec<f32>,
    a1_b: Vec<f32>,
    a2: Vec<f32>,
    a2_b: Vec<f32>,
}

fn bf16_tensor(src: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = src.info(name).with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        info.dtype == DType::BF16 && info.shape == shape,
        "{name}: {:?} {:?}, expected BF16 {shape:?}",
        info.dtype,
        info.shape
    );
    let bytes = src.read_all(name)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
        .collect())
}

impl VisionWeights {
    /// Reads `vision.*` and `aligner.*` from the checkpoint. `progress` is told the block count.
    pub fn load(src: &dyn TensorSource, c: &Config, progress: &dyn Fn(usize, usize)) -> Result<Self> {
        let v = c
            .deepseek
            .as_ref()
            .and_then(|d| d.vision.as_ref())
            .context("no vision config")?;
        ensure!(v.hidden == v.heads * (v.hidden / v.heads), "bad vision heads");
        let head_dim = v.hidden / v.heads;
        let patch_in = 3 * v.patch * v.patch;
        let patch_w = bf16_tensor(src, "vision.patch_embed.proj.weight", &[v.hidden, patch_in])?;
        let patch_b = bf16_tensor(src, "vision.patch_embed.proj.bias", &[v.hidden])?;
        let mut blocks = Vec::with_capacity(v.layers);
        for i in 0..v.layers {
            let p = |s: &str| format!("vision.blocks.{i}.{s}");
            blocks.push(VBlock {
                norm1: bf16_tensor(src, &p("norm1.weight"), &[v.hidden])?,
                norm2: bf16_tensor(src, &p("norm2.weight"), &[v.hidden])?,
                qkv: bf16_tensor(src, &p("attn.wqkv.weight"), &[3 * v.hidden, v.hidden])?,
                qkv_b: bf16_tensor(src, &p("attn.wqkv.bias"), &[3 * v.hidden])?,
                o: bf16_tensor(src, &p("attn.wo.weight"), &[v.hidden, v.hidden])?,
                o_b: bf16_tensor(src, &p("attn.wo.bias"), &[v.hidden])?,
                w1: bf16_tensor(src, &p("mlp.w1.weight"), &[2 * v.inter, v.hidden])?,
                w2: bf16_tensor(src, &p("mlp.w2.weight"), &[v.hidden, v.inter])?,
            });
            progress(i + 1, v.layers);
        }
        let merged = v.hidden * v.downsample * v.downsample;
        Ok(VisionWeights {
            cfg: v.clone(),
            dim: c.hidden,
            head_dim,
            rope: head_dim / 2,
            patch_w,
            patch_b,
            blocks,
            norm: bf16_tensor(src, "vision.norm.weight", &[v.hidden])?,
            a1: bf16_tensor(src, "aligner.w1.weight", &[c.hidden, merged])?,
            a1_b: bf16_tensor(src, "aligner.w1.bias", &[c.hidden])?,
            a2: bf16_tensor(src, "aligner.w2.weight", &[c.hidden, c.hidden])?,
            a2_b: bf16_tensor(src, "aligner.w2.bias", &[c.hidden])?,
        })
    }

    /// Visitor seen by the memory plan: the tower is host memory, reported apart.
    pub fn bytes(&self) -> usize {
        let b = &self.blocks;
        let per = 2 * self.cfg.hidden
            + 3 * self.cfg.hidden * self.cfg.hidden
            + 3 * self.cfg.hidden
            + self.cfg.hidden * self.cfg.hidden
            + self.cfg.hidden
            + 2 * self.cfg.inter * self.cfg.hidden
            + self.cfg.hidden * self.cfg.inter;
        (std::mem::size_of::<f32>()
            * (b.len() * per
                + self.patch_w.len()
                + self.patch_b.len()
                + self.norm.len()
                + self.a1.len()
                + self.a2.len()
                + self.a1_b.len()
                + self.a2_b.len())) as usize
    }
}

/// Rows of the token grid the aligner produces from a patch grid, and the pad it needs.
fn llm_grid(gh: usize, gw: usize, downsample: usize) -> (usize, usize) {
    (gh.div_ceil(downsample), gw.div_ceil(downsample))
}

/// `num_image_tokens`: the aligner rows plus one newline per row and the two delimiters.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// Largest aspect-preserving pixel size whose token grid still fits in `max_tokens`, from the reference processor.
fn solve_resize_ratio(h: usize, w: usize, patch: usize, downsample: usize, max_tokens: usize) -> (usize, usize) {
    let r = h as f64 / w as f64;
    let max_w = ((max_tokens as f64 - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h = max_w * r;
    let cell = patch * downsample;
    if max_w < 1.0 {
        return ((max_tokens - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (max_tokens - 3) * cell);
    }
    let beta = (max_w.floor() * cell as f64 / w as f64).min(max_h.floor() * cell as f64 / h as f64);
    (
        ((h as f64 * beta / patch as f64).floor() as usize) * patch,
        ((w as f64 * beta / patch as f64).floor() as usize) * patch,
    )
}

/// The resize plan of one image: the patch grid and the token grid.
pub fn plan_grid(v: &DeepseekVision, h: usize, w: usize) -> Result<(usize, usize, usize, usize)> {
    ensure!(
        h > 0 && w > 0 && v.patch > 0 && v.downsample > 0,
        "empty image or bad tower"
    );
    let p = v.patch;
    let (mut width, mut height) = (w, h);
    if v.min_pixels > 0 && width * height > 0 && width * height < v.min_pixels {
        let ratio = (v.min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let mut best_w = width.div_ceil(p) * p;
    let mut best_h = height.div_ceil(p) * p;
    let (mut n_llm_h, mut n_llm_w) = llm_grid(best_h / p, best_w / p, v.downsample);
    if v.max_tokens > 0 && num_image_tokens(n_llm_h, n_llm_w) > v.max_tokens {
        (best_h, best_w) = solve_resize_ratio(height, width, p, v.downsample, v.max_tokens);
        (n_llm_h, n_llm_w) = llm_grid(best_h / p, best_w / p, v.downsample);
        ensure!(
            num_image_tokens(n_llm_h, n_llm_w) <= v.max_tokens,
            "the token budget cannot hold {}x{} image tokens",
            n_llm_h,
            n_llm_w
        );
    }
    Ok((n_llm_h, n_llm_w, best_h, best_w))
}

/// `ImageOps.pad` with the gray fill: scale to contain, center, fill the rest with 127.
fn pad_to(rgb: &[u8], h: usize, w: usize, best_h: usize, best_w: usize) -> Vec<u8> {
    let im_ratio = w as f64 / h as f64;
    let dest_ratio = best_w as f64 / best_h as f64;
    let (mut rw, mut rh) = (best_w, best_h);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            rh = (h as f64 / w as f64 * best_w as f64).round_ties_even() as usize;
        } else {
            rw = (w as f64 / h as f64 * best_h as f64).round_ties_even() as usize;
        }
    }
    let resized = if (rh, rw) == (h, w) {
        rgb.to_vec()
    } else {
        resize_bicubic(rgb, h, w, rh, rw)
    };
    if (rh, rw) == (best_h, best_w) {
        return resized;
    }
    let mut canvas = vec![127u8; best_h * best_w * 3];
    let (x0, y0) = if rw != best_w {
        (((best_w - rw) as f64 * 0.5).round_ties_even() as usize, 0)
    } else {
        (0, ((best_h - rh) as f64 * 0.5).round_ties_even() as usize)
    };
    for y in 0..rh {
        let src = y * rw * 3;
        let dst = ((y0 + y) * best_w + x0) * 3;
        canvas[dst..dst + rw * 3].copy_from_slice(&resized[src..src + rw * 3]);
    }
    canvas
}

/// The processor: decode, plan, pad, normalise to `[-1, 1]` and cut the patches in `(channel, y, x)` order.
pub fn preprocess(
    v: &DeepseekVision,
    bytes: &[u8],
    max_pixels: usize,
) -> Result<(Vec<f32>, usize, usize, usize, usize)> {
    // The decoder only needs the size limits here: the patch grid is this processor's own.
    let cfg = PreConfig {
        patch: v.patch,
        merge: 1,
        temporal: 1,
        max_pixels,
        ..Default::default()
    };
    let (rgb, h, w) = decode(bytes, &cfg)?;
    let (n_llm_h, n_llm_w, best_h, best_w) = plan_grid(v, h, w)?;
    let img = pad_to(&rgb, h, w, best_h, best_w);
    let (gh, gw) = (best_h / v.patch, best_w / v.patch);
    let p = v.patch;
    let mut out = vec![0f32; gh * gw * 3 * p * p];
    for py in 0..gh {
        for px in 0..gw {
            let patch = py * gw + px;
            for c in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        let at = ((py * p + y) * best_w + px * p + x) * 3 + c;
                        out[patch * 3 * p * p + c * p * p + y * p + x] = img[at] as f32 * (2.0 / 255.0) - 1.0;
                    }
                }
            }
        }
    }
    Ok((out, gh, gw, n_llm_h, n_llm_w))
}

/// The token layout of one image: `[START] ([IMAGE] * w + [NEW_LINE]) * h [END]`.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<u8> {
    const START: u8 = 0;
    const IMAGE: u8 = 1;
    const NEW_LINE: u8 = 2;
    const END: u8 = 3;
    let mut t = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    t.push(START);
    for _ in 0..n_llm_h {
        t.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        t.push(NEW_LINE);
    }
    t.push(END);
    t
}

/// The three learned embeddings that frame an image span.
pub struct ImageTokens {
    start: Vec<f32>,
    end: Vec<f32>,
    newline: Vec<f32>,
}

impl ImageTokens {
    pub fn load(src: &dyn TensorSource, c: &Config) -> Result<Self> {
        Ok(ImageTokens {
            start: bf16_tensor(src, "image_start", &[c.hidden])?,
            end: bf16_tensor(src, "image_end", &[c.hidden])?,
            newline: bf16_tensor(src, "image_newline", &[c.hidden])?,
        })
    }
}

/// The embeddings that replace one image span in the prompt, in the order `image_token_types` lays the span out:
/// the start delimiter, then every aligner row followed by a newline, then the end delimiter. `merged` is the
/// tower's output, `n_llm_h * n_llm_w` rows of `dim`.
pub fn span_embeddings(
    t: &ImageTokens,
    merged: &[f32],
    n_llm_h: usize,
    n_llm_w: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    ensure!(
        merged.len() == n_llm_h * n_llm_w * dim,
        "the aligner gave {} rows, the grid wants {}",
        merged.len() / dim,
        n_llm_h * n_llm_w
    );
    let span = num_image_tokens(n_llm_h, n_llm_w);
    let mut out = Vec::with_capacity(span * dim);
    out.extend_from_slice(&t.start);
    for row in 0..n_llm_h {
        for col in 0..n_llm_w {
            let at = (row * n_llm_w + col) * dim;
            out.extend_from_slice(&merged[at..at + dim]);
        }
        out.extend_from_slice(&t.newline);
    }
    out.extend_from_slice(&t.end);
    Ok(out)
}

// ------------------------------------------------------------------------------------------------------------------
// Host math: a few seconds of f32 for one image, which is what the tower costs until it moves to the device.
// ------------------------------------------------------------------------------------------------------------------

/// The tower's forward and the aligner's merge. Returns `(vit [gh*gw, hidden], merged [n_llm_h*n_llm_w, dim])`.
pub fn forward(w: &VisionWeights, patches: &[f32], gh: usize, gw: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    forward_with(w, patches, gh, gw, |_, _| Ok(()))
}

/// The same, calling `hook(step, x)` after the patch embedding (step 0) and after each block (1..=layers).
pub fn forward_with(
    w: &VisionWeights,
    patches: &[f32],
    gh: usize,
    gw: usize,
    mut hook: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let v = &w.cfg;
    let (n, d) = (gh * gw, v.hidden);
    ensure!(
        patches.len() == n * 3 * v.patch * v.patch,
        "patches do not match the grid"
    );
    let eps = 1e-6;
    let heads = v.heads;
    let hd = w.head_dim;
    let (cos, sin) = rope_table(gh, gw, w.rope, v.rope_theta);
    let mut x = linear(patches, n, 3 * v.patch * v.patch, &w.patch_w, d, Some(&w.patch_b));
    hook(0, &x)?;
    for (step, b) in w.blocks.iter().enumerate() {
        let mut norm = x.clone();
        rmsnorm(&mut norm, d, &b.norm1, eps);
        let mut qkv = linear(&norm, n, d, &b.qkv, 3 * d, Some(&b.qkv_b));
        // `chunk(3)`: the first third is q, then k, then v; each is [n, heads, head_dim]. The reference rotates
        // q and k with the same 2-D frequencies: the first half of a head turns by the row, the second by the
        // column (so the head's two halves are the rotated pair, not adjacent elements).
        let r = w.rope;
        for p in 0..n {
            for h in 0..heads {
                for part in 0..2 {
                    let base = p * 3 * d + part * d + h * hd;
                    for t in 0..r {
                        let (i, j) = (base + t, base + hd - r + t);
                        let (a, bb) = (qkv[i], qkv[j]);
                        let (c, s) = (cos[p * r + t], sin[p * r + t]);
                        qkv[i] = a * c - bb * s;
                        qkv[j] = bb * c + a * s;
                    }
                }
            }
        }
        // q in [0, d), k in [d, 2d), v in [2d, 3d) of the output: gather them contiguous for the attention.
        let mut q = vec![0f32; n * d];
        let mut k = vec![0f32; n * d];
        let mut vv = vec![0f32; n * d];
        for p in 0..n {
            q[p * d..(p + 1) * d].copy_from_slice(&qkv[p * 3 * d..p * 3 * d + d]);
            k[p * d..(p + 1) * d].copy_from_slice(&qkv[p * 3 * d + d..p * 3 * d + 2 * d]);
            vv[p * d..(p + 1) * d].copy_from_slice(&qkv[p * 3 * d + 2 * d..p * 3 * d + 3 * d]);
        }
        let o = attention(&q, &k, &vv, n, heads, hd);
        let o = linear(&o, n, d, &b.o, d, Some(&b.o_b));
        for (xi, oi) in x.iter_mut().zip(&o) {
            *xi += oi;
        }
        let mut norm = x.clone();
        rmsnorm(&mut norm, d, &b.norm2, eps);
        let gu = linear(&norm, n, d, &b.w1, 2 * v.inter, None);
        let mut act = vec![0f32; n * v.inter];
        for p in 0..n {
            for j in 0..v.inter {
                act[p * v.inter + j] = silu(gu[p * 2 * v.inter + j]) * gu[p * 2 * v.inter + v.inter + j];
            }
        }
        let mlp = linear(&act, n, v.inter, &b.w2, d, None);
        for (xi, mi) in x.iter_mut().zip(&mlp) {
            *xi += mi;
        }
        hook(step + 1, &x)?;
    }
    rmsnorm(&mut x, d, &w.norm, eps);

    // Aligner: `[gh*gw, d]` viewed as `[gh, gw, d]` -> `[d, gh, gw]`, zero-padded to the downsample grid, and
    // each `r x r` cell flattened in `(channel, y, x)` order (what `F.unfold` produces).
    let r = v.downsample;
    let (n_llm_h, n_llm_w) = llm_grid(gh, gw, r);
    let cell = d * r * r;
    let mut grid = vec![0f32; cell * n_llm_h * n_llm_w];
    for oy in 0..n_llm_h {
        for ox in 0..n_llm_w {
            let out = &mut grid[(oy * n_llm_w + ox) * cell..][..cell];
            for c in 0..d {
                for ky in 0..r {
                    for kx in 0..r {
                        let (y, xx) = (oy * r + ky, ox * r + kx);
                        if y < gh && xx < gw {
                            out[c * r * r + ky * r + kx] = x[(y * gw + xx) * d + c];
                        }
                    }
                }
            }
        }
    }
    let hidden = linear(&grid, n_llm_h * n_llm_w, cell, &w.a1, w.dim, Some(&w.a1_b));
    let act: Vec<f32> = hidden.iter().map(|v| gelu(*v)).collect();
    let merged = linear(&act, n_llm_h * n_llm_w, w.dim, &w.a2, w.dim, Some(&w.a2_b));
    Ok((x, merged))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_layout_has_a_newline_per_row() {
        let t = image_token_types(2, 3);
        assert_eq!(t, vec![0, 1, 1, 1, 2, 1, 1, 1, 2, 3]);
        assert_eq!(t.len(), num_image_tokens(2, 3));
    }

    #[test]
    fn the_span_puts_the_delimiters_around_the_rows() {
        let dim = 2;
        let t = ImageTokens {
            start: vec![10.0, 10.0],
            end: vec![30.0, 30.0],
            newline: vec![20.0, 20.0],
        };
        let merged: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let span = span_embeddings(&t, &merged, 2, 3, dim).unwrap();
        assert_eq!(span.len(), num_image_tokens(2, 3) * dim);
        let rows: Vec<f32> = span.chunks(dim).map(|c| c[0]).collect();
        assert_eq!(rows, vec![10.0, 0.0, 2.0, 4.0, 20.0, 6.0, 8.0, 10.0, 20.0, 30.0]);
        assert!(span_embeddings(&t, &merged, 3, 3, dim).is_err());
    }

    #[test]
    fn the_grid_fits_the_token_budget() {
        // A 450x308 photo, the example in the checkpoint's inference directory.
        let v = DeepseekVision {
            layers: 32,
            hidden: 1024,
            inter: 2816,
            heads: 16,
            patch: 14,
            downsample: 3,
            rope_theta: 1e4,
            max_tokens: 1024,
            min_pixels: 295936,
        };
        let (n_llm_h, n_llm_w, bh, bw) = plan_grid(&v, 308, 450).unwrap();
        assert_eq!((bh, bw), (462, 658));
        assert_eq!((n_llm_h, n_llm_w), (11, 16));
        assert!(num_image_tokens(n_llm_h, n_llm_w) <= 1024);
        // A square image at the minimum pixel count: the grid is the whole budget.
        let (n_llm_h, n_llm_w, _, _) = plan_grid(&v, 544, 544).unwrap();
        assert!(num_image_tokens(n_llm_h, n_llm_w) <= 1024);
        // A 4:1 panorama is capped instead of blowing the budget.
        let (n_llm_h, n_llm_w, _, _) = plan_grid(&v, 200, 800).unwrap();
        assert!(num_image_tokens(n_llm_h, n_llm_w) <= 1024, "{n_llm_h}x{n_llm_w}");
    }

    /// Reads a little-endian f32 file.
    fn f32_file(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Compares two vectors the way the gate does: cosine, and the worst and mean absolute difference.
    fn compare(what: &str, got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
        let (mut max, mut sum) = (0f64, 0f64);
        for (a, b) in got.iter().zip(want) {
            dot += (*a as f64) * (*b as f64);
            ng += (*a as f64) * (*a as f64);
            nw += (*b as f64) * (*b as f64);
            let d = (*a as f64 - *b as f64).abs();
            max = max.max(d);
            sum += d;
        }
        if std::env::var("DEEPSEEK_VISION_DEBUG").is_ok() {
            let mut worst: Vec<(f64, usize, f32, f32)> = got
                .iter()
                .zip(want)
                .enumerate()
                .map(|(i, (a, b))| ((*a as f64 - *b as f64).abs(), i, *a, *b))
                .collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (d, i, a, b) in worst.iter().take(6) {
                eprintln!("  {what}[{i}] got {a} want {b} ({d:.4})");
            }
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        eprintln!(
            "{what}: cosine {cos:.7}, max |diff| {max:.5}, mean |diff| {:.5}",
            sum / got.len() as f64
        );
        cos
    }

    /// The tower against the reference, from the patches the reference processor made: the weights, the RoPE, the
    /// attention and the aligner all at once.
    ///
    /// `DEEPSEEK_VISION_REF` is the prefix `scripts/deepseek_vision_ref.py --out` wrote; the test also needs
    /// `DEEPSEEK_CKPT` to read the weights.
    #[test]
    #[ignore = "needs the 1 GB vision weights and a reference dump"]
    fn the_tower_matches_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let reference = std::env::var("DEEPSEEK_VISION_REF").expect("DEEPSEEK_VISION_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{reference}.json")).unwrap()).unwrap();
        let (gh, gw) = (
            manifest["n_vit_h"].as_u64().unwrap() as usize,
            manifest["n_vit_w"].as_u64().unwrap() as usize,
        );
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let src = malleus_gguf::safetensors::SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let w = VisionWeights::load(&src, &c, &|_, _| {}).unwrap();
        let patches = f32_file(&format!("{reference}-patches.f32"));
        // With `--blocks` the reference also wrote the embedding and every block's output: compare them, so a
        // divergence names the stage it starts at.
        let mut worst_step = 0.0f64;
        let (vit, merged) = forward_with(&w, &patches, gh, gw, |step, x| {
            let path = format!("{reference}-step{step:02}.f32");
            if std::path::Path::new(&path).exists() {
                let cos = compare(&format!("step {step}"), x, &f32_file(&path));
                if step > 0 {
                    worst_step = if step == 1 { cos } else { worst_step.max(cos) };
                }
            }
            Ok(())
        })
        .unwrap();
        let cos_vit = compare("vit", &vit, &f32_file(&format!("{reference}-vit.f32")));
        let cos_merged = compare("merged", &merged, &f32_file(&format!("{reference}-merged.f32")));
        assert!(cos_vit > 0.9999, "the tower diverges from the reference: {cos_vit}");
        assert!(
            cos_merged > 0.9999,
            "the aligner diverges from the reference: {cos_merged}"
        );
    }

    /// The span the LLM sees, against the release's own layout: the three learned delimiters and the aligner rows
    /// in reading order, `[START] ([IMAGE] * w + [NEW_LINE]) * h [END]`. The rows are the reference's, so this pins
    /// the layout and the delimiters — the tower has its own gate.
    ///
    ///   DEEPSEEK_CKPT=... DEEPSEEK_VISION_REF=/golden/deepseek-vision/corn
    #[test]
    #[ignore = "needs the 1 GB vision weights and a reference dump"]
    fn the_image_span_matches_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let reference = std::env::var("DEEPSEEK_VISION_REF").expect("DEEPSEEK_VISION_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{reference}.json")).unwrap()).unwrap();
        let (nh, nw) = (
            manifest["n_llm_h"].as_u64().unwrap() as usize,
            manifest["n_llm_w"].as_u64().unwrap() as usize,
        );
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let src = malleus_gguf::safetensors::SafeTensors::open_dir(std::path::Path::new(&ckpt)).unwrap();
        let delimiters = ImageTokens::load(&src, &c).unwrap();
        let merged = f32_file(&format!("{reference}-merged.f32"));
        let span = span_embeddings(&delimiters, &merged, nh, nw, c.hidden).unwrap();
        let want = f32_file(&format!("{reference}-span.f32"));
        assert_eq!(span.len(), want.len(), "the span's length");
        let differing = span.iter().zip(&want).filter(|(a, b)| a != b).count();
        let cos = compare("span", &span, &want);
        // Both sides take the same rows and the same bf16 delimiters widened to f32: the layout is exact or wrong.
        assert_eq!(differing, 0, "{differing} of {} values differ", span.len());
        assert!(cos > 0.99999, "the span diverges: {cos}");
    }

    /// The processor against the reference: same grid, same patches (a JPEG decoder can differ by a level or two).
    #[test]
    #[ignore = "needs the reference dump and the same image"]
    fn the_processor_matches_the_reference() {
        let ckpt = std::env::var("DEEPSEEK_CKPT").expect("DEEPSEEK_CKPT");
        let reference = std::env::var("DEEPSEEK_VISION_REF").expect("DEEPSEEK_VISION_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{reference}.json")).unwrap()).unwrap();
        // The dump and the test can see the same image under different mount points.
        let image = std::env::var("DEEPSEEK_IMAGE").unwrap_or_else(|_| manifest["image"].as_str().unwrap().to_string());
        let c = Config::from_hf(std::path::Path::new(&ckpt)).unwrap();
        let v = c.deepseek.unwrap().vision.unwrap();
        let (got, gh, gw, nh, nw) = preprocess(&v, &std::fs::read(&image).unwrap(), 4 << 20).unwrap();
        assert_eq!(gh, manifest["n_vit_h"].as_u64().unwrap() as usize);
        assert_eq!(gw, manifest["n_vit_w"].as_u64().unwrap() as usize);
        assert_eq!(nh, manifest["n_llm_h"].as_u64().unwrap() as usize);
        assert_eq!(nw, manifest["n_llm_w"].as_u64().unwrap() as usize);
        compare("patches", &got, &f32_file(&format!("{reference}-patches.f32")));
        let want = f32_file(&format!("{reference}-patches.f32"));
        let mean: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>()
            / got.len() as f64;
        assert!(mean < 0.02, "the processor diverges from the reference: {mean}");
    }
}
