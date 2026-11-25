//! Host f32 math shared by the DeepSeek-V4.1 text graph and its vision tower (issue #83): the parts that run on
//! the CPU until the kernels take them. Everything here is plain f32 with disjoint-chunk threading, so a small
//! model can be compared stage by stage against the reference on one machine.

/// Runs `f(index, chunk)` on disjoint chunks of `data`, one thread per available core. A chunk of zero is a
/// degenerate split (a linear with no rows, an empty matrix), not a reason to panic: nothing is left to do.
pub(crate) fn parallel_chunks<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let chunks: Vec<(usize, &mut [T])> = data.chunks_mut(chunk.max(1)).enumerate().collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, |v| v.get())
        .min(chunks.len().max(1));
    if threads <= 1 {
        for (i, c) in chunks {
            f(i, c);
        }
        return;
    }
    let queue = std::sync::Mutex::new(chunks.into_iter());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let next = queue.lock().unwrap().next();
                    match next {
                        Some((i, c)) => f(i, c),
                        None => break,
                    }
                }
            });
        }
    });
}

/// A dot product with eight independent accumulators: Rust cannot reassociate a float reduction, so the lanes
/// are what lets the compiler vectorise it.
pub(crate) fn dot(x: &[f32], w: &[f32], k: usize) -> f32 {
    let (xb, xt) = x[..k].as_chunks::<8>();
    let (wb, wt) = w[..k].as_chunks::<8>();
    let mut acc = [0f32; 8];
    for (a, b) in xb.iter().zip(wb) {
        for u in 0..8 {
            acc[u] += a[u] * b[u];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (a, b) in xt.iter().zip(wt) {
        s += a * b;
    }
    s
}

/// Two dots against the same weight row: the row is read once for two output rows.
pub(crate) fn dots2(x0: &[f32], x1: &[f32], w: &[f32], k: usize) -> [f32; 2] {
    let (w8, wt) = w[..k].as_chunks::<8>();
    let (x08, x0t) = x0[..k].as_chunks::<8>();
    let (x18, x1t) = x1[..k].as_chunks::<8>();
    let mut a0 = [0f32; 8];
    let mut a1 = [0f32; 8];
    for ((c0, c1), cw) in x08.iter().zip(x18).zip(w8) {
        for u in 0..8 {
            a0[u] += c0[u] * cw[u];
            a1[u] += c1[u] * cw[u];
        }
    }
    let (mut s0, mut s1): (f32, f32) = (a0.iter().sum(), a1.iter().sum());
    for ((t0, t1), tw) in x0t.iter().zip(x1t).zip(wt) {
        s0 += t0 * tw;
        s1 += t1 * tw;
    }
    [s0, s1]
}

/// `x [n, k] · wᵀ`, with `w [m, k]`, plus an optional bias: the reference's `nn.Linear`.
///
/// The output is built transposed (`[m, n]`, one contiguous run per output column) so that a task owns a range of
/// columns and reads the whole input a few times instead of once per column; the rows are walked two at a time so
/// each weight row serves two dots. The transpose at the end is blocked.
pub(crate) fn linear(x: &[f32], n: usize, k: usize, w: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
    const JB: usize = 32;
    const IB: usize = 64;
    let mut t = vec![0f32; m * n];
    parallel_chunks(&mut t, JB * n, |b, cols| {
        let j0 = b * JB;
        let jn = cols.len() / n;
        for i0 in (0..n).step_by(IB) {
            let inner = IB.min(n - i0);
            let pairs = inner / 2;
            for jj in 0..jn {
                let wj = &w[(j0 + jj) * k..][..k];
                let bj = bias.map_or(0.0, |z| z[j0 + jj]);
                let out = &mut cols[jj * n + i0..jj * n + i0 + inner];
                for p in 0..pairs {
                    let (r0, r1) = (i0 + 2 * p, i0 + 2 * p + 1);
                    let [a, c] = dots2(&x[r0 * k..(r0 + 1) * k], &x[r1 * k..(r1 + 1) * k], wj, k);
                    out[2 * p] = a + bj;
                    out[2 * p + 1] = c + bj;
                }
                if inner % 2 == 1 {
                    let r = i0 + inner - 1;
                    out[inner - 1] = dot(&x[r * k..(r + 1) * k], wj, k) + bj;
                }
            }
        }
    });
    // Blocked transpose of `[m, n]` into `[n, m]`.
    const TB: usize = 32;
    let mut out = vec![0f32; n * m];
    for j0 in (0..m).step_by(TB) {
        for i0 in (0..n).step_by(TB) {
            let jn = TB.min(m - j0);
            let inn = TB.min(n - i0);
            for j in 0..jn {
                for i in 0..inn {
                    out[(i0 + i) * m + j0 + j] = t[(j0 + j) * n + i0 + i];
                }
            }
        }
    }
    out
}

/// In-place RMSNorm over the last dimension of an `[n, dim]` matrix, multiplying by `weight`.
pub(crate) fn rmsnorm(x: &mut [f32], dim: usize, weight: &[f32], eps: f32) {
    parallel_chunks(x, dim, |_, row| {
        let sq: f32 = row.iter().map(|v| v * v).sum();
        let s = 1.0 / (sq / dim as f32 + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * s * w;
        }
    });
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The exact `erf` of `F.gelu` (Abramowitz-Stegun 7.1.26, absolute error below 1.5e-7).
pub(crate) fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2) as f32)
}
