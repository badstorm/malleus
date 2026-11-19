//! What a run is about to ask the machine for, before it asks for it.
//!
//! On these APUs the GPU memory *is* system RAM. A model that does not fit does not fail cleanly: the
//! allocator evicts the page cache, the model streams from disk while the GPU waits, and a host that
//! also runs a desktop can spend minutes at full CPU with no output. The three terms that dominate are
//! the weights (exact, from the file), the per-session state (exact, from the config) and the working
//! set of a chunk (an estimate: it is printed measured after loading, see `malleus_hip::mem_get_info`).
//!
//! So `serve`, `run` and `bench` compute this first and refuse with a number instead of dying inside
//! `hipMalloc`. `MALLEUS_MEM_IGNORE=1` skips the refusal.
//!
//! The working set grows with the prefill chunk, and it used to be a fixed 8 GiB: at a 32768-token chunk
//! Flash-Next takes 28 GiB, the process reached 116.6 GiB of 122.8 and left 0.6 GiB of page cache to a 49 GiB
//! lookup table, which cost 7-8 % of decode. It is now estimated from the chunk (`working_set_bytes`), and a
//! model that reads a table from disk keeps a share of RAM free for that table's page cache
//! (`MALLEUS_HOST_RESERVE_GIB`, default 8). `MALLEUS_MEM_RESERVE_GIB` still overrides the working set.

use anyhow::{Result, bail};
use malleus_gguf::TensorSource;

use crate::config::{Config, MixerKind};
use crate::engine::MOE_BM;
use crate::session::DN_CHUNK;
use crate::weights::{GEMV_MAX_T, kv_ctx, pad128};

const GIB: u64 = 1 << 30;
/// The part of the working set that does not depend on the chunk (see `working_set_bytes`).
const WORKING_SET_FIXED: u64 = 2 * GIB;

/// `MemTotal` / `MemAvailable` from `/proc/meminfo`, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
}

impl MemInfo {
    /// Parses the `MemTotal:`/`MemAvailable:` lines. `MemAvailable` is what matters: it is the kernel's
    /// own estimate of what can still be handed out without swapping, page cache included.
    pub fn parse(text: &str) -> Option<MemInfo> {
        fn field(text: &str, key: &str) -> Option<u64> {
            text.lines().find_map(|l| {
                let mut it = l.split_whitespace();
                if it.next()? != key {
                    return None;
                }
                let kib: u64 = it.next()?.parse().ok()?;
                Some(kib * 1024)
            })
        }
        Some(MemInfo {
            total: field(text, "MemTotal:")?,
            available: field(text, "MemAvailable:")?,
        })
    }

    pub fn read() -> Option<MemInfo> {
        Self::parse(&std::fs::read_to_string("/proc/meminfo").ok()?)
    }
}

/// Bytes the loader will upload for this checkpoint.
///
/// Not the file size: the n-gram lookup table is read from the file on demand and never resident (it is
/// 49 GiB of the 116 GiB of the Flash-Next file), and with `MALLEUS_PREFILL_WEIGHTS=none` the int8
/// prefill copies (`*.prefill` and their scales) are not loaded either.
pub fn weights_on_device(src: &dyn TensorSource, prefill_int8: bool) -> u64 {
    let cache = expert_cache_bytes(src);
    let resident: u64 = src
        .names()
        .iter()
        .filter_map(|n| src.info(n).map(|i| (n, i)))
        .filter(|(n, _)| uploaded(n, prefill_int8) && !(cache.is_some() && is_routed_expert(n)))
        .map(|(_, i)| i.size as u64)
        .sum();
    resident + cache.unwrap_or(0)
}

/// Whether the tensor is a routed expert, which `MALLEUS_EXPERT_CACHE` leaves in the file. The shared expert of a
/// family lives elsewhere (`.mlp.shared_expert...`), so it stays resident.
fn is_routed_expert(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

/// Bytes `MALLEUS_EXPERT_CACHE` puts on the device instead of the routed experts: the slot pools (the requested
/// GiB, rounded down to whole experts per layer) plus the staging of one full layer, which a prefill needs.
fn expert_cache_bytes(src: &dyn TensorSource) -> Option<u64> {
    let gib = crate::expert_cache::gib();
    if gib <= 0.0 {
        return None;
    }
    let mut per_layer: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
    for name in src.names() {
        if !is_routed_expert(name) {
            continue;
        }
        if let Some(i) = src.info(name) {
            let layer = name.split(".mlp.experts.").next().unwrap_or(name);
            *per_layer.entry(layer).or_default() += i.size as u64;
        }
    }
    let stage = per_layer.values().copied().max().unwrap_or(0);
    Some((gib * GIB as f64) as u64 + stage)
}

/// Bytes of the tables the engine reads from the file while it runs (the n-gram lookup table): never on the
/// device, but every decode step reads rows of them through the page cache.
pub fn tables_on_disk(src: &dyn TensorSource) -> u64 {
    src.names()
        .iter()
        .filter(|n| n.ends_with("ngram_table"))
        .filter_map(|n| src.info(n))
        .map(|i| i.size as u64)
        .sum()
}

/// Prompt tokens per forward call: `MALLEUS_PREFILL_CHUNK`, default 16384 with hyper-connections and 2048
/// otherwise. Larger chunks read the MoE expert weights fewer times per token, at the cost of the working set.
pub fn prefill_chunk(cfg: &Config) -> usize {
    let default = if cfg.hc.is_some() { 16384 } else { 2048 };
    std::env::var("MALLEUS_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
        .max(64)
}

/// Bytes of the activations and scratch of one forward call of `chunk` tokens.
///
/// This mirrors `Model::scratch`, buffer for buffer: if that allocation changes, this has to change with it.
/// What the loader derives from the weights and what a forward allocates besides (the draft head's rows, the
/// logits, the rows kept for the MTP head, the attention partials) does not grow with the chunk: 1.9 GiB at
/// 8192 tokens and 2.4 at 32768 on Flash-Next, counted here as `WORKING_SET_FIXED`.
pub fn working_set_bytes(c: &Config, chunk: usize, moe_y_half: bool) -> u64 {
    let cap = chunk.max(16);
    let f32 = 4usize;
    let big = c.inter.max(c.n_heads * c.head_dim * 2).max(c.lin_qkv_dim());
    let small = (c.n_heads * c.head_dim).max(c.lin_nv * c.lin_dv);
    let kmax = c
        .inter
        .max(c.hidden)
        .max(c.n_heads * c.head_dim)
        .max(c.stream_dim())
        .max(c.ple.as_ref().map_or(0, |p| p.dim));
    // x, hn; big0..2; small0..3; xr, xh, xsc; ids, pos; mh, cat.
    let mut b = 2 * cap * c.hidden * f32 + 3 * cap * big * f32 + 4 * cap * small * f32;
    b += cap * kmax * f32 + cap * kmax * 2 + cap * kmax / 128 * 4;
    b += cap * 16 + cap * c.stream_dim() * f32 + cap * 2 * c.hidden * f32;
    if c.kind.contains(&MixerKind::Kda) {
        b += 2 * cap * c.lin_nv * c.lin_dv * f32;
    }
    let blocks = cap.div_ceil(DN_CHUNK) * c.lin_nv;
    let (cc, d) = (DN_CHUNK, c.lin_dv);
    b += blocks * (cc * d * 2 + cc * d * 4 + cc * d * 2 + cc * cc * 2 + d * cc * 2 + 4);
    if let Some(m) = &c.moe {
        let (p, e1) = (cap * m.slots(), m.experts + 1);
        b += cap * e1 * f32 + 4 * p * 4 + p * 2 * m.inter * f32 + p * m.inter * f32;
        b += (e1 + 1) * 4 + (p.div_ceil(MOE_BM) + 3 * e1 + 1) * 8 + 8;
        b += p * c.hidden + p * c.hidden.div_ceil(128) * f32 + p * m.inter + p * m.inter.div_ceil(128) * f32;
        b += p * c.hidden * if moe_y_half { 2 } else { 4 };
    }
    if let Some(ic) = &c.indexer {
        b += cap * ((ic.heads + 1) * ic.dim + ic.heads * ic.dim + ic.dim) * f32;
    }
    if let Some(h) = &c.hc {
        let (sd, rp) = (c.stream_dim(), pad128(h.rank + h.count));
        b += cap * (3 * sd + 2 * rp + 3 * h.count + c.hidden + c.ple.as_ref().map_or(1, |p| p.dim)) * f32;
    }
    b as u64 + WORKING_SET_FIXED
}

/// Whether the loader puts this tensor on the device. Kept in one place so the check and the loader
/// cannot disagree about the n-gram table or the prefill copies.
fn uploaded(name: &str, prefill_int8: bool) -> bool {
    if name.ends_with("ngram_table") {
        return false;
    }
    if !prefill_int8 && (name.ends_with(".prefill") || name.ends_with(".prefill.scales")) {
        return false;
    }
    true
}

/// Bytes one session holds for the whole of `ctx`: the KV of the attention layers (plus the indexer's
/// pooled keys), the conv and recurrent state of the linear layers, the PLE conv state, and the MTP row.
///
/// This mirrors `Model::new_session`: if that allocation changes, this has to change with it.
pub fn session_bytes(cfg: &Config, ctx: usize, mtp: bool) -> u64 {
    let kv = (kv_ctx(ctx) * cfg.n_kv * cfg.head_dim * 2 + 64) as u64;
    // With the indexer the V buffer also holds V row-major after V^T.
    let v = if cfg.indexer.is_some() { 2 * kv } else { kv };
    let mut t = 0u64;
    for kind in &cfg.kind {
        t += match kind {
            MixerKind::Attn => kv + v,
            MixerKind::Mla => cfg
                .mla
                .as_ref()
                .map_or(0, |m| (kv_ctx(ctx) * (m.kv_lora + m.rope) * 2) as u64),
            _ => (cfg.lin_qkv_dim() * (cfg.conv_k - 1) * 4 + cfg.lin_nv * cfg.lin_dk * cfg.lin_dv * 4) as u64,
        };
    }
    if let Some(ic) = &cfg.indexer {
        // Per attention layer: pooled block keys, the raw keys of the last incomplete block, positions.
        let per = (ctx / ic.ratio.max(1) + 1) * ic.dim * 2 + ic.ratio * ic.dim * 4 + ic.ratio * 3 * 4;
        let n = cfg.kind.iter().filter(|k| **k == MixerKind::Attn).count() as u64;
        t += per as u64 * n;
    }
    if let Some(p) = &cfg.ple {
        t += (p.conv_state() * cfg.stream_dim() * 4) as u64;
    }
    // Ling keeps one split buffer for the MLA decode, sized for GEMV_MAX_T tokens.
    if let Some(m) = &cfg.mla {
        t += (GEMV_MAX_T * cfg.n_heads * kv_ctx(ctx).div_ceil(256) * (m.kv_lora + 2) * 4) as u64;
    }
    if mtp {
        t += kv + v + (GEMV_MAX_T * cfg.stream_dim() * 4) as u64;
    }
    t
}

/// Weights, one session per concurrent request, and the working set of a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub weights: u64,
    pub session: u64,
    pub sessions: usize,
    /// What the check keeps free for the chunk's activations and scratch; `malleus_hip::mem_get_info`
    /// prints what the run actually took.
    pub reserve: u64,
    /// RAM left to the host for the page cache of the tables read from disk (0 for a model without one).
    pub host: u64,
}

impl Usage {
    pub fn state(&self) -> u64 {
        self.session * self.sessions as u64
    }

    pub fn total(&self) -> u64 {
        self.weights + self.state() + self.reserve + self.host
    }

    /// `None` when it fits with the reserve intact, otherwise how many bytes are missing.
    pub fn short_by(&self, available: u64) -> Option<u64> {
        let want = self.total();
        (want > available).then(|| want - available)
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

/// Prints one line with the three terms and refuses when they do not fit.
///
/// `path_label` is what to show in the messages (the model path as the user gave it).
pub fn check(label: &str, ctx: usize, usage: &Usage, info: Option<MemInfo>) -> Result<()> {
    let state = if usage.sessions > 1 {
        format!(
            "{:.1} GiB of state ({} sessions of {} tokens)",
            gib(usage.state()),
            usage.sessions,
            ctx
        )
    } else {
        format!("{:.1} GiB of state", gib(usage.state()))
    };
    let host = if usage.host > 0 {
        format!(" + {:.1} GiB left to the table's page cache", gib(usage.host))
    } else {
        String::new()
    };
    eprintln!(
        "memory: {:.1} GiB of weights + {} + {:.1} GiB of working set{host} = {:.1} GiB asked{}",
        gib(usage.weights),
        state,
        gib(usage.reserve),
        gib(usage.total()),
        info.map_or(String::new(), |i| format!(
            ", {:.1} GiB available of {:.1}",
            gib(i.available),
            gib(i.total)
        ))
    );
    let Some(info) = info else {
        eprintln!("memory: /proc/meminfo is not readable, skipping the check");
        return Ok(());
    };
    if std::env::var("MALLEUS_MEM_IGNORE").is_ok_and(|v| v != "0" && v != "false") {
        if usage.short_by(info.available).is_some() {
            eprintln!("memory: this does not fit, but MALLEUS_MEM_IGNORE is set: the allocator decides");
        }
        return Ok(());
    }
    let Some(short) = usage.short_by(info.available) else {
        return Ok(());
    };
    // What to lower: the context is the only knob that moves the state term on its own.
    let per_position = usage.session / ctx.max(1) as u64;
    let per_32k = per_position * 32_768;
    bail!(
        "memory: `{label}` does not fit on this host.\n\
         Need {:.1} GiB (weights {:.1} + state {:.1} x {} sessions + working set {:.1} + page cache {:.1}), \
         MemAvailable is {:.1}: {:.1} GiB missing.\n\
         Each session at this context costs {:.1} GiB (~{} KiB per position, {:.1} GiB per 32k positions).\n\
         The working set follows the prefill chunk: lower MALLEUS_PREFILL_CHUNK first (it costs no prefill speed \
         down to 8192), then MALLEUS_CTX (or the session count, MALLEUS_SLOTS), or use a checkpoint with fewer \
         prefill copies. MALLEUS_MEM_IGNORE=1 lets the allocator try anyway.",
        gib(usage.total()),
        gib(usage.weights),
        gib(usage.session),
        usage.sessions,
        gib(usage.reserve),
        gib(usage.host),
        gib(info.available),
        gib(short),
        gib(usage.session),
        per_position / 1024,
        gib(per_32k),
    );
}

fn env_gib(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g >= 0.0)
        .map(|g| (g * GIB as f64) as u64)
}

/// The working set the check leaves free: `MALLEUS_MEM_RESERVE_GIB` when set, otherwise the estimate for a
/// chunk of `chunk` tokens.
pub fn reserve_bytes(cfg: &Config, chunk: usize, moe_y_half: bool) -> u64 {
    env_gib("MALLEUS_MEM_RESERVE_GIB").unwrap_or_else(|| working_set_bytes(cfg, chunk, moe_y_half))
}

/// RAM the check leaves to the host when `tables` bytes are read from disk while the engine runs:
/// `MALLEUS_HOST_RESERVE_GIB`, default 8 GiB, never more than the tables themselves.
pub fn host_reserve_bytes(tables: u64) -> u64 {
    env_gib("MALLEUS_HOST_RESERVE_GIB").unwrap_or(8 * GIB).min(tables)
}

/// Sessions the scheduler keeps resident: `MALLEUS_SLOTS`, default 1. One in-flight request holds one session and
/// the context is split between them (`ctx / slots` each), so this is also the depth of the concurrency: with one
/// slot the server serves one request at a time, exactly as it always did.
pub fn sessions() -> usize {
    std::env::var("MALLEUS_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str = "MemTotal:       128785860 kB\n\
                           MemFree:         67274332 kB\n\
                           MemAvailable:   122971191 kB\n\
                           Buffers:            1024 kB\n";

    #[test]
    fn the_ngram_table_and_the_prefill_copies_are_not_counted() {
        let table = "model.language_model.layers.1.ple.ple_embedding.ngram_table";
        assert!(!uploaded(table, true));
        assert!(!uploaded(table, false));
        let prefill = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.prefill";
        let scales = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.prefill.scales";
        assert!(uploaded(prefill, true));
        assert!(!uploaded(prefill, false));
        assert!(!uploaded(scales, false));
        let decode = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        assert!(uploaded(decode, true) && uploaded(decode, false));
    }

    #[test]
    fn parses_meminfo() {
        let m = MemInfo::parse(MEMINFO).unwrap();
        assert_eq!(m.total, 128_785_860 * 1024);
        assert_eq!(m.available, 122_971_191 * 1024);
    }

    #[test]
    fn missing_available_is_not_a_number() {
        assert!(MemInfo::parse("MemTotal: 1 kB\n").is_none());
        assert!(MemInfo::parse("").is_none());
    }

    #[test]
    fn a_single_session_is_the_weights_plus_state() {
        let u = Usage {
            weights: 12 * GIB,
            session: 3 * GIB,
            sessions: 1,
            reserve: 2 * GIB,
            host: 0,
        };
        assert_eq!(u.state(), 3 * GIB);
        assert_eq!(u.total(), 17 * GIB);
        assert_eq!(u.short_by(17 * GIB), None);
        assert_eq!(u.short_by(16 * GIB), Some(GIB));
    }

    #[test]
    fn the_host_reserve_counts_and_never_exceeds_the_tables() {
        let u = Usage {
            weights: 10 * GIB,
            session: GIB,
            sessions: 1,
            reserve: GIB,
            host: 8 * GIB,
        };
        assert_eq!(u.total(), 20 * GIB);
        assert_eq!(host_reserve_bytes(0), 0);
        assert_eq!(host_reserve_bytes(3 * GIB), 3 * GIB);
    }

    #[test]
    fn sessions_multiply_and_report_the_shortfall() {
        let u = Usage {
            weights: 10 * GIB,
            session: 4 * GIB,
            sessions: 3,
            reserve: GIB,
            host: 0,
        };
        assert_eq!(u.total(), 23 * GIB);
        assert_eq!(u.short_by(20 * GIB), Some(3 * GIB));
    }
}
