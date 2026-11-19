use malleus_gguf::quant::{self, BLOCK, HADAMARD_SEED, INT2_GROUP_BYTES, INT3_GROUP_BYTES, Int2Grid};
use malleus_gguf::safetensors::SafeTensors;
use malleus_gguf::{DType, TensorSource};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("model dir");
    let layers: Vec<usize> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0,3,24,47".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let st = SafeTensors::open_dir(std::path::Path::new(&dir))?;
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    for name in names {
        let Some(li) = name
            .split("layers.")
            .nth(1)
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        if !layers.contains(&li) || !name.contains(".mlp.experts.") {
            continue;
        }
        let info = st.info(&name).unwrap();
        if info.dtype != DType::BF16 {
            continue;
        }
        let k = *info.shape.last().unwrap();
        let rows: usize = info.shape.iter().rev().skip(1).product();
        let mut e2u = 0f64;
        let mut e2e = 0f64;
        let mut e3 = 0f64;
        let mut e4 = 0f64;
        let mut norm = 0f64;
        let max_rows: usize = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let rows = rows.min(max_rows);
        let step = (64 << 20) / (k * 2);
        for r0 in (0..rows).step_by(step) {
            let nrows = step.min(rows - r0);
            let mut raw = vec![0u8; nrows * k * 2];
            st.read(&name, r0 * k * 2, &mut raw)?;
            for r in 0..nrows {
                let mut row: Vec<f32> = raw[r * k * 2..(r + 1) * k * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                    .collect();
                quant::rotate_row(&mut row, &signs);
                for g in row.as_chunks::<BLOCK>().0 {
                    let gn: f64 = g.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
                    if gn == 0.0 {
                        continue;
                    }
                    let mut d2u = [0f32; BLOCK];
                    let mut d2e = [0f32; BLOCK];
                    let mut d3 = [0f32; BLOCK];
                    let mut d4 = [0f32; BLOCK];
                    let (bu, su) = quant::quantize_group_int2(g, Int2Grid::Uniform);
                    let (be, se) = quant::quantize_group_int2(g, Int2Grid::E2m1);
                    let (b3, s3) = quant::quantize_group_int3(g);
                    let (b4, s4) = quant::quantize_group_int4(g);
                    assert_eq!(bu.len(), INT2_GROUP_BYTES);
                    assert_eq!(be.len(), INT2_GROUP_BYTES);
                    assert_eq!(b3.len(), INT3_GROUP_BYTES);
                    quant::dequant_int2(&bu, su, Int2Grid::Uniform, &mut d2u);
                    quant::dequant_int2(&be, se, Int2Grid::E2m1, &mut d2e);
                    quant::dequant_int3(&b3, s3, &mut d3);
                    quant::dequant_int4(&b4, s4, &mut d4);
                    let rel = |dq: &[f32; BLOCK]| {
                        g.iter()
                            .zip(dq)
                            .map(|(a, b)| ((a - b) as f64).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            / gn
                    };
                    e2u += rel(&d2u);
                    e2e += rel(&d2e);
                    e3 += rel(&d3);
                    e4 += rel(&d4);
                    norm += 1.0;
                }
            }
        }
        println!(
            "{name}\n  MLS2 uniform {:.4}  MLS2 e2m1 {:.4}  MLS3 {:.4}  MLS4 {:.4}",
            e2u / norm,
            e2e / norm,
            e3 / norm,
            e4 / norm
        );
    }
    Ok(())
}
