//! Image preprocessing for the Qwen3.x vision tower (Qwen2-VL style processor).

use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug)]
pub struct PreConfig {
    pub patch: usize,
    pub merge: usize,
    pub temporal: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Maximum encoded size accepted before decoding.
    pub max_bytes: usize,
    /// Maximum decoded dimensions accepted (checked from the header before decoding).
    pub max_decode_pixels: usize,
}

impl Default for PreConfig {
    fn default() -> Self {
        PreConfig {
            patch: 16,
            merge: 2,
            temporal: 2,
            min_pixels: 65536,
            max_pixels: 1 << 21,
            mean: [0.5; 3],
            std: [0.5; 3],
            max_bytes: 32 << 20,
            max_decode_pixels: 64 << 20,
        }
    }
}

pub struct Preprocessed {
    /// [n_patches, 3 * temporal * patch * patch], merge-block order.
    pub pixels: Vec<f32>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Preprocessed {
    pub fn n_patches(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }
}

pub fn smart_resize(h: usize, w: usize, factor: usize, min_px: usize, max_px: usize) -> Result<(usize, usize)> {
    ensure!(h > 0 && w > 0, "empty image");
    if h.max(w) as f64 / h.min(w) as f64 > 200.0 {
        bail!("aspect ratio larger than 200");
    }
    let f = factor as f64;
    let round = |x: f64| (x / f).round_ties_even() * f;
    let (mut hb, mut wb) = (round(h as f64).max(f), round(w as f64).max(f));
    let (hf, wf) = (h as f64, w as f64);
    if hb * wb > max_px as f64 {
        let beta = (hf * wf / max_px as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min_px as f64 {
        let beta = (min_px as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// PIL-style separable bicubic resampling weights (antialiased when downscaling).
fn weights(src: usize, dst: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = src as f64 / dst as f64;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(src);
            let mut w: Vec<f64> = (lo..hi).map(|j| cubic((j as f64 - center + 0.5) / fscale)).collect();
            let s: f64 = w.iter().sum();
            if s != 0.0 {
                w.iter_mut().for_each(|v| *v /= s);
            }
            (lo, w)
        })
        .collect()
}

/// Resizes an RGB u8 image ([h][w][3]) to (nh, nw) with values in [0, 255] (f32, no clamping of overshoot
/// beyond rounding like PIL: results are rounded and clamped to u8).
pub fn resize_bicubic(img: &[u8], h: usize, w: usize, nh: usize, nw: usize) -> Vec<u8> {
    let wx = weights(w, nw);
    let wy = weights(h, nh);
    let mut tmp = vec![0f32; h * nw * 3];
    for y in 0..h {
        for (x, (lo, ws)) in wx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0f64;
                for (k, wv) in ws.iter().enumerate() {
                    acc += img[(y * w + lo + k) * 3 + c] as f64 * wv;
                }
                tmp[(y * nw + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as f32;
            }
        }
    }
    let mut out = vec![0u8; nh * nw * 3];
    for (y, (lo, ws)) in wy.iter().enumerate() {
        for x in 0..nw {
            for c in 0..3 {
                let mut acc = 0f64;
                for (k, wv) in ws.iter().enumerate() {
                    acc += tmp[((lo + k) * nw + x) * 3 + c] as f64 * wv;
                }
                out[(y * nw + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

pub fn decode(bytes: &[u8], cfg: &PreConfig) -> Result<(Vec<u8>, usize, usize)> {
    ensure!(
        bytes.len() <= cfg.max_bytes,
        "image larger than {} bytes",
        cfg.max_bytes
    );
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let fmt = reader.format().ok_or_else(|| anyhow::anyhow!("unknown image format"))?;
    ensure!(
        matches!(
            fmt,
            image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP | image::ImageFormat::Gif
        ),
        "unsupported image format {fmt:?}"
    );
    let (w, h) = reader.into_dimensions()?;
    ensure!(
        (w as usize) * (h as usize) <= cfg.max_decode_pixels,
        "image of {w}x{h} pixels is too large"
    );
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(w);
    limits.max_image_height = Some(h);
    limits.max_alloc = Some(512 << 20);
    reader.limits(limits);
    let img = reader.decode()?.to_rgb8();
    Ok((img.into_raw(), h as usize, w as usize))
}

pub fn preprocess_rgb(rgb: &[u8], h: usize, w: usize, cfg: &PreConfig) -> Result<Preprocessed> {
    let factor = cfg.patch * cfg.merge;
    let (nh, nw) = smart_resize(h, w, factor, cfg.min_pixels, cfg.max_pixels)?;
    let img = if (nh, nw) == (h, w) {
        rgb.to_vec()
    } else {
        resize_bicubic(rgb, h, w, nh, nw)
    };
    let (gh, gw) = (nh / cfg.patch, nw / cfg.patch);
    let (p, m, tp) = (cfg.patch, cfg.merge, cfg.temporal);
    let dim = 3 * tp * p * p;
    let mut pixels = Vec::with_capacity(gh * gw * dim);
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for ih in 0..m {
                for iw in 0..m {
                    let (py, px) = ((bh * m + ih) * p, (bw * m + iw) * p);
                    for c in 0..3 {
                        for _t in 0..tp {
                            for y in 0..p {
                                for x in 0..p {
                                    let v = img[((py + y) * nw + px + x) * 3 + c] as f32 / 255.0;
                                    pixels.push((v - cfg.mean[c]) / cfg.std[c]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Preprocessed {
        pixels,
        grid_t: 1,
        grid_h: gh,
        grid_w: gw,
    })
}

/// Inkling-2's processor (issue #74): **no resize**. The image is padded on the right and the bottom to a multiple
/// of the patch side (40) with `-1.0` *on the 0..255 scale* — which after the normalisation is not a neutral grey but
/// -1.0078 — normalised by mean/std 0.5, and cut into one patch per vision token, row-major over the padded grid of
/// `ceil(h / 40) x ceil(w / 40)`.
///
/// The layout of a patch is the checkpoint's, channel last (`[y][x][c]`), not the Qwen tower's `[c][y][x]`: the
/// reference hands the tower `(patches, 40, 40, 3)` straight from the processor's reshape.
pub fn preprocess_inkling2(rgb: &[u8], h: usize, w: usize, cfg: &PreConfig) -> Result<Preprocessed> {
    ensure!(h > 0 && w > 0, "empty image");
    let p = cfg.patch;
    ensure!(p > 0, "patch side");
    let (rows, cols) = (h.div_ceil(p), w.div_ceil(p));
    // `max_pixels` bounds the padded image the way the other family bounds the resized one.
    ensure!(
        rows * p * cols * p <= cfg.max_pixels,
        "image {w}x{h} would be {} patches of {p}x{p}, over max_pixels={}",
        rows * cols,
        cfg.max_pixels
    );
    let pad: [f32; 3] = [0, 1, 2].map(|c| ((-1.0f32 / 255.0) - cfg.mean[c]) / cfg.std[c]);
    let mut pixels = Vec::with_capacity(rows * cols * p * p * 3);
    for r in 0..rows {
        for c in 0..cols {
            for y in 0..p {
                let sy = r * p + y;
                for x in 0..p {
                    let sx = c * p + x;
                    for ch in 0..3 {
                        pixels.push(if sy < h && sx < w {
                            let v = rgb[(sy * w + sx) * 3 + ch] as f32 / 255.0;
                            (v - cfg.mean[ch]) / cfg.std[ch]
                        } else {
                            pad[ch]
                        });
                    }
                }
            }
        }
    }
    Ok(Preprocessed {
        pixels,
        grid_t: 1,
        grid_h: rows,
        grid_w: cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_reference_cases() {
        assert_eq!(smart_resize(512, 512, 32, 65536, 1 << 30).unwrap(), (512, 512));
        assert_eq!(smart_resize(96, 96, 32, 65536, 1 << 30).unwrap(), (256, 256));
        assert_eq!(smart_resize(517, 333, 32, 65536, 1 << 30).unwrap(), (512, 320));
    }

    #[test]
    fn identity_resize_keeps_pixels() {
        let img: Vec<u8> = (0..4 * 4 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_bicubic(&img, 4, 4, 4, 4), img);
    }
}
