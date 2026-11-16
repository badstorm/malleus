//! Multimodal rope positions (T, H, W) for a token sequence containing image placeholders.

/// `images`: for each image in order, the number of merged rows and columns (grid_h / merge, grid_w / merge).
/// Each run of consecutive `image_token` ids is matched with the next image. Returns positions and the next
/// free text position.
pub fn positions(
    tokens: &[u32],
    image_token: u32,
    images: &[(usize, usize)],
    start: i64,
) -> anyhow::Result<(Vec<[i32; 3]>, i64)> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut pos = start;
    let mut img = 0;
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] != image_token {
            out.push([pos as i32; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let &(gh, gw) = images
            .get(img)
            .ok_or_else(|| anyhow::anyhow!("more image placeholders than images"))?;
        anyhow::ensure!(
            i + gh * gw <= tokens.len() && tokens[i..i + gh * gw].iter().all(|&t| t == image_token),
            "image placeholder count mismatch"
        );
        for r in 0..gh {
            for c in 0..gw {
                out.push([pos as i32, (pos + r as i64) as i32, (pos + c as i64) as i32]);
            }
        }
        pos += gh.max(gw) as i64;
        i += gh * gw;
        img += 1;
    }
    anyhow::ensure!(img == images.len(), "fewer image placeholders than images");
    Ok((out, pos))
}

#[cfg(test)]
mod tests {
    #[test]
    fn image_positions() {
        let (p, next) = super::positions(&[1, 9, 9, 9, 9, 9, 9, 2], 9, &[(2, 3)], 0).unwrap();
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[1], [1, 1, 1]);
        assert_eq!(p[3], [1, 1, 3]);
        assert_eq!(p[4], [1, 2, 1]);
        assert_eq!(p[7], [4, 4, 4]);
        assert_eq!(next, 5);
    }
}
