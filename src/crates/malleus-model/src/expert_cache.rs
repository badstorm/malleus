//! Routed experts on disk, the hot ones in a slot cache (issue #14).
//!
//! `MALLEUS_EXPERT_CACHE=<GiB>` keeps the routed experts of a MoE out of the device memory. Every MoE layer gets a
//! pool of slots: an expert is a **contiguous** run of the four tensors the decode kernels read (`gate_up`, its
//! scales, `down`, its scales), so bringing one in is four positional reads and four copies, and the kernels are
//! handed the *slot* index in place of the expert id (`scripts/expert_cache.py` documents the layout and measures
//! what it costs).
//!
//! Two paths, because they need different amounts of the layer at once:
//!
//! Families that stack the shared expert into the expert matrix (everything but Ling: Qwen's `shared_expert_gate`)
//! keep it in three tensors of its own, but the router addresses it as expert number `experts`: the cache assembles
//! that block once at load, into a slot no eviction can touch.
//!
//! - **decode** (`t` below the grouped-GEMM threshold): the router's ids come back to the host, the missing experts
//!   are read into their slot, the ids become slot ids. The cache is per layer, LRU.
//! - **prefill** (grouped GEMM): that needs *every* expert of the layer (all 512 of Ling) at once, so the layer is
//!   streamed into one staging buffer shared by all layers, once per chunk. Expensive by construction.
//!
//! The weights are the same bytes either way, so the output is bit-identical to the resident path.

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::{TensorFile, TensorSource};
use malleus_hip::DeviceBuffer;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// Host chunk of the staging reads: bounds the buffer the process holds, not the device memory.
const CHUNK: usize = 256 << 20;

/// `MALLEUS_EXPERT_CACHE=<GiB>`; 0 or unset means off.
pub fn gib() -> f64 {
    std::env::var("MALLEUS_EXPERT_CACHE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .unwrap_or(0.0)
}

pub fn enabled() -> bool {
    gib() > 0.0
}

/// Pairs served from a slot, pairs that had to be read, and bytes read: the numbers to compare a cache size with the
/// byte count of the issue. Process-wide, printed by the CLI when the cache is on.
static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (HITS.load(Relaxed), MISSES.load(Relaxed), BYTES.load(Relaxed))
}

/// A poisoned lock means a previous holder panicked; the buffers are still valid, so take them.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes of one expert in the four tensors, and how many experts the layer has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    pub experts: usize,
    pub gu_q: usize,
    pub gu_s: usize,
    pub down_q: usize,
    pub down_s: usize,
}

impl Geom {
    pub fn per_slot(&self) -> usize {
        self.gu_q + self.gu_s + self.down_q + self.down_s
    }

    /// The geometry from the file, not from the config: `experts` is the outermost dim of the four expert tensors,
    /// so `size / experts` is the contiguous run of one expert whatever the format (MLS3/MLS4/MLS8).
    pub fn from_source(src: &dyn TensorSource, names: &[String; 4], experts: usize) -> Result<Self> {
        ensure!(experts > 0, "a MoE layer with no experts");
        let per = |name: &String| -> Result<usize> {
            let info = src.info(name).with_context(|| format!("missing {name}"))?;
            ensure!(
                info.size.is_multiple_of(experts),
                "{name}: {} bytes are not a whole number of {experts} experts",
                info.size
            );
            ensure!(
                info.shape.len() != 3 || info.shape[0] == experts,
                "{name}: shape {:?} has no {experts}-expert outermost dim",
                info.shape
            );
            Ok(info.size / experts)
        };
        Ok(Geom {
            experts,
            gu_q: per(&names[0])?,
            gu_s: per(&names[1])?,
            down_q: per(&names[2])?,
            down_s: per(&names[3])?,
        })
    }
}

struct State {
    /// expert -> slot
    map: HashMap<usize, usize>,
    /// slot -> expert, for the eviction
    slot: Vec<Option<usize>>,
    /// per slot, the logical time of the last use (LRU)
    used: Vec<u64>,
    clock: u64,
    /// Slots are handed out in order until the pool is full, then the LRU is evicted.
    next_free: usize,
}

/// The four expert buffers of one layer, in the layout the kernels read (`q + e * N * K / 2`).
struct Pool {
    gu_q: DeviceBuffer,
    gu_s: DeviceBuffer,
    down_q: DeviceBuffer,
    down_s: DeviceBuffer,
}

/// The shared expert of a family that stacks it into the expert matrix (Qwen: `shared_expert_gate`). Its block in
/// the stacked matrix is `[gate rows | up rows]` and `[down rows]`, assembled from three tensors of its own.
struct SharedBlock {
    /// gate, gate.scales, up, up.scales, down, down.scales
    files: [TensorFile; 6],
    sizes: [usize; 6],
}

impl SharedBlock {
    fn build(src: &dyn TensorSource, names: &[String; 6], geom: &Geom) -> Result<Self> {
        let mut files = Vec::with_capacity(6);
        let mut sizes = [0usize; 6];
        for (i, name) in names.iter().enumerate() {
            let info = src.info(name).with_context(|| format!("missing {name}"))?;
            sizes[i] = info.size;
            files.push(
                src.open_tensor(name)
                    .with_context(|| format!("{name}: the source cannot be read after loading"))?,
            );
        }
        // The three tensors have to fill exactly one expert block of the stacked matrix, or the kernels would read
        // past it.
        ensure!(
            sizes[0] + sizes[2] == geom.gu_q
                && sizes[1] + sizes[3] == geom.gu_s
                && sizes[4] == geom.down_q
                && sizes[5] == geom.down_s,
            "the shared expert does not fill one expert block ({} + {} != {} bytes)",
            sizes[0],
            sizes[2],
            geom.gu_q
        );
        Ok(SharedBlock {
            files: files
                .try_into()
                .map_err(|_| anyhow::anyhow!("the shared expert needs six tensors"))?,
            sizes,
        })
    }

    /// Writes the shared block into the four buffers at `at` (the four have different slot strides).
    fn write_into(
        &self,
        gu_q: &mut DeviceBuffer,
        gu_s: &mut DeviceBuffer,
        down_q: &mut DeviceBuffer,
        down_s: &mut DeviceBuffer,
        at: [usize; 4],
    ) -> Result<()> {
        read_all(&self.files[0], self.sizes[0], gu_q, at[0])?;
        read_all(&self.files[2], self.sizes[2], gu_q, at[0] + self.sizes[0])?;
        read_all(&self.files[1], self.sizes[1], gu_s, at[1])?;
        read_all(&self.files[3], self.sizes[3], gu_s, at[1] + self.sizes[1])?;
        read_all(&self.files[4], self.sizes[4], down_q, at[2])?;
        read_all(&self.files[5], self.sizes[5], down_s, at[3])?;
        Ok(())
    }
}

/// One MoE layer's pool: expert slots plus a last slot that stays zero, which is where the router's shared-expert
/// slot (`id == experts`, Ling) is mapped.
pub struct Layer {
    pub geom: Geom,
    files: [TensorFile; 4],
    /// Expert slots `0..expert_slots` are the LRU pool; the slots above it are the shared expert (when the layer
    /// stacks it, Qwen) and the zero slot (the router's shared slot of a layer that has none, Ling).
    expert_slots: usize,
    slots: usize,
    /// The shared expert lives in the same stacked matrix as expert number `experts` for every non-Ling family.
    /// It comes from three tensors of its own, so it is assembled once at load and never evicted.
    shared: Option<SharedBlock>,
    shared_slot: Option<usize>,
    /// Addresses of the four pool buffers: fixed for the life of the pool, so the kernels take them without a lock.
    addr: [u64; 4],
    pool: Mutex<Pool>,
    scratch: Mutex<Vec<u8>>,
    state: Mutex<State>,
}

impl Layer {
    /// A pool of `budget` bytes (at least one expert), reading the four tensors `names` of the layer.
    pub fn build(
        src: &dyn TensorSource,
        names: &[String; 4],
        experts: usize,
        budget: usize,
        shared_names: Option<&[String; 6]>,
    ) -> Result<Self> {
        let geom = Geom::from_source(src, names, experts)?;
        let mut files = Vec::with_capacity(4);
        for name in names {
            files.push(
                src.open_tensor(name)
                    .with_context(|| format!("{name}: the source cannot be read after loading"))?,
            );
        }
        let shared = shared_names.map(|n| SharedBlock::build(src, n, &geom)).transpose()?;
        // At least one expert slot, at most every expert of the layer; then the shared slot (if the layer stacks
        // its shared expert) and the zero slot.
        let want = (budget / geom.per_slot().max(1)).clamp(1, experts);
        let slots = want + 1 + usize::from(shared.is_some());
        let pool = Pool {
            gu_q: DeviceBuffer::zeroed(slots * geom.gu_q)?,
            gu_s: DeviceBuffer::zeroed(slots * geom.gu_s)?,
            down_q: DeviceBuffer::zeroed(slots * geom.down_q)?,
            down_s: DeviceBuffer::zeroed(slots * geom.down_s)?,
        };
        let addr = [
            pool.gu_q.addr(),
            pool.gu_s.addr(),
            pool.down_q.addr(),
            pool.down_s.addr(),
        ];
        let layer = Layer {
            geom,
            files: files
                .try_into()
                .map_err(|_| anyhow::anyhow!("a MoE layer needs four expert tensors"))?,
            expert_slots: want,
            slots,
            shared,
            shared_slot: None,
            addr,
            pool: Mutex::new(pool),
            scratch: Mutex::new(vec![0u8; geom.gu_q.max(geom.gu_s).max(geom.down_q).max(geom.down_s)]),
            state: Mutex::new(State {
                map: HashMap::new(),
                slot: vec![None; slots],
                used: vec![0; slots],
                clock: 0,
                next_free: 0,
            }),
        };
        let mut layer = layer;
        if let Some(shared) = &layer.shared {
            // The shared expert is constant and always used: read it once, into a slot no eviction can touch.
            let slot = want;
            layer.shared_slot = Some(slot);
            let (gq, gs) = (layer.geom.gu_q, layer.geom.gu_s);
            let (dq, ds) = (layer.geom.down_q, layer.geom.down_s);
            let at = [slot * gq, slot * gs, slot * dq, slot * ds];
            let mut pool = lock(&layer.pool);
            let Pool {
                gu_q,
                gu_s,
                down_q,
                down_s,
            } = &mut *pool;
            shared.write_into(gu_q, gu_s, down_q, down_s, at)?;
        }
        Ok(layer)
    }

    /// Device bytes the pool holds.
    pub fn bytes(&self) -> usize {
        self.slots * self.geom.per_slot()
    }

    pub fn expert_slots(&self) -> usize {
        self.expert_slots
    }

    /// (quant, scales) addresses of `gate|up`.
    pub fn gu(&self) -> (u64, u64) {
        (self.addr[0], self.addr[1])
    }

    /// (quant, scales) addresses of `down`.
    pub fn down(&self) -> (u64, u64) {
        (self.addr[2], self.addr[3])
    }

    /// Makes the experts of `ids` resident and writes their slots into `out`. Ids past the last expert (the shared
    /// slot of Ling) map to the zero slot.
    pub fn prepare(&self, ids: &[i32], out: &mut DeviceBuffer) -> Result<()> {
        ensure!(out.len() >= ids.len() * 4, "slot-id buffer too small");
        let mut st = lock(&self.state);
        let mut slots = Vec::with_capacity(ids.len());
        for &id in ids {
            let e = id as usize;
            if e >= self.geom.experts {
                // The stacked matrix has one slot past the experts: the shared expert (Qwen) or zeros (Ling).
                slots.push(self.shared_slot.unwrap_or(self.slots - 1));
                continue;
            }
            let slot = match st.map.get(&e).copied() {
                Some(s) => {
                    st.clock += 1;
                    st.used[s] = st.clock;
                    HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    s
                }
                None => {
                    MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    fetch(
                        &mut st,
                        &self.geom,
                        self.expert_slots,
                        &self.files,
                        &self.pool,
                        &self.scratch,
                        e,
                    )?
                }
            };
            slots.push(slot);
        }
        drop(st);
        let v: Vec<i32> = slots.iter().map(|s| *s as i32).collect();
        out.upload(&v)?;
        Ok(())
    }

    /// Fills the pool with the experts the chunk routed most, copying them **from the staging buffer**: the
    /// prefill has just read the whole layer from disk, so warming the cache costs a device-to-device copy instead
    /// of a second read. Without it the decode starts with empty pools and pays the working set from the disk again
    /// (measured on Ling: 0,59 tok/s against 16,37 resident).
    pub fn warm_from_stage(&self, ids: &[i32], stage: &Stage, st: &malleus_hip::Stream) -> Result<()> {
        ensure!(stage.fits(&self.geom), "staging too small to warm the pool");
        let mut freq: HashMap<usize, u32> = HashMap::new();
        for &id in ids {
            let e = id as usize;
            if e < self.geom.experts {
                *freq.entry(e).or_default() += 1;
            }
        }
        if freq.is_empty() {
            return Ok(());
        }
        // Most routed first: the LRU will evict the tail, which is the part of the prompt that mattered least.
        let mut ranked: Vec<(usize, u32)> = freq.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let take = ranked.len().min(self.expert_slots);
        {
            let mut state = lock(&self.state);
            state.map.clear();
            for s in state.slot.iter_mut() {
                *s = None;
            }
            // `used` is the LRU key: the most routed expert must be the *most* recently used, so that an eviction
            // takes the tail of the ranking and not the head.
            state.clock = take as u64;
            for (rank, &(e, _)) in ranked.iter().take(take).enumerate() {
                state.slot[rank] = Some(e);
                state.map.insert(e, rank);
                state.used[rank] = take as u64 - rank as u64;
            }
            state.next_free = take;
        }
        let g = self.geom;
        let mut pool = lock(&self.pool);
        for (slot, &(e, _)) in ranked.iter().take(take).enumerate() {
            pool.gu_q
                .copy_from(slot * g.gu_q, &stage.gu_q, e * g.gu_q, g.gu_q, st)?;
            pool.gu_s
                .copy_from(slot * g.gu_s, &stage.gu_s, e * g.gu_s, g.gu_s, st)?;
            pool.down_q
                .copy_from(slot * g.down_q, &stage.down_q, e * g.down_q, g.down_q, st)?;
            pool.down_s
                .copy_from(slot * g.down_s, &stage.down_s, e * g.down_s, g.down_s, st)?;
        }
        st.synchronize()?;
        Ok(())
    }

    /// Streams every expert of the layer into `stage` (the prefill needs them all at once).
    pub fn stage_into(&self, stage: &mut Stage) -> Result<()> {
        stage.ensure(&self.geom)?;
        let (e, g) = (self.geom.experts, &self.geom);
        read_all(&self.files[0], e * g.gu_q, &mut stage.gu_q, 0)?;
        read_all(&self.files[1], e * g.gu_s, &mut stage.gu_s, 0)?;
        read_all(&self.files[2], e * g.down_q, &mut stage.down_q, 0)?;
        read_all(&self.files[3], e * g.down_s, &mut stage.down_s, 0)?;
        if let Some(shared) = &self.shared {
            // The grouped GEMM indexes this buffer by expert id, and the router gives the shared expert the id
            // `experts`: its block has to be there like any other.
            let at = [e * g.gu_q, e * g.gu_s, e * g.down_q, e * g.down_s];
            let Stage {
                gu_q,
                gu_s,
                down_q,
                down_s,
                ..
            } = stage;
            shared.write_into(gu_q, gu_s, down_q, down_s, at)?;
        }
        Ok(())
    }
}

/// Reads expert `e` into a slot (the least recently used one when the pool is full) and returns the slot.
#[allow(clippy::too_many_arguments)]
fn fetch(
    st: &mut State,
    geom: &Geom,
    slots: usize, // expert slots, not the pool size: the shared and zero slots are never evicted
    files: &[TensorFile; 4],
    pool: &Mutex<Pool>,
    scratch: &Mutex<Vec<u8>>,
    e: usize,
) -> Result<usize> {
    let slot = if st.map.len() < slots - 1 {
        let s = st.next_free;
        st.next_free += 1;
        s
    } else {
        let (mut victim, mut oldest) = (0, u64::MAX);
        for s in 0..slots - 1 {
            if st.used[s] < oldest {
                oldest = st.used[s];
                victim = s;
            }
        }
        if let Some(old) = st.slot[victim].take() {
            st.map.remove(&old);
        }
        victim
    };
    st.slot[slot] = Some(e);
    st.map.insert(e, slot);
    st.clock += 1;
    st.used[slot] = st.clock;
    BYTES.fetch_add(geom.per_slot() as u64, std::sync::atomic::Ordering::Relaxed);
    let per = [geom.gu_q, geom.gu_s, geom.down_q, geom.down_s];
    let mut buf = lock(scratch);
    let mut dst = lock(pool);
    for i in 0..4 {
        let b = &mut buf[..per[i]];
        files[i].read_at(e * per[i], b)?;
        let target = match i {
            0 => &mut dst.gu_q,
            1 => &mut dst.gu_s,
            2 => &mut dst.down_q,
            _ => &mut dst.down_s,
        };
        target.upload_at(slot * per[i], b)?;
    }
    // Debug: the pool must hold exactly the bytes of the file (the kernels read `slot` where the resident path
    // reads the expert id, so a mismatch here is the difference between the two paths).
    if std::env::var_os("MALLEUS_EXPERT_CACHE_VERIFY").is_some() {
        for i in 0..4 {
            let target = match i {
                0 => &dst.gu_q,
                1 => &dst.gu_s,
                2 => &dst.down_q,
                _ => &dst.down_s,
            };
            let want = &mut buf[..per[i]];
            files[i].read_at(e * per[i], want)?;
            let got: Vec<u8> = target.download_at::<u8>(slot * per[i], per[i])?;
            if got != want {
                let at = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(0);
                bail!("expert cache: slot {slot} of tensor {i} differs from expert {e} at byte {at}");
            }
        }
    }
    Ok(slot)
}

/// The full layer a prefill needs, shared by every MoE layer and reused at each layer visit.
pub struct Stage {
    pub gu_q: DeviceBuffer,
    pub gu_s: DeviceBuffer,
    pub down_q: DeviceBuffer,
    pub down_s: DeviceBuffer,
    geom: Geom,
}

impl Stage {
    pub fn alloc(geom: Geom) -> Result<Self> {
        let slots = geom.experts + 1;
        Ok(Stage {
            gu_q: DeviceBuffer::zeroed(slots * geom.gu_q)?,
            gu_s: DeviceBuffer::zeroed(slots * geom.gu_s)?,
            down_q: DeviceBuffer::zeroed(slots * geom.down_q)?,
            down_s: DeviceBuffer::zeroed(slots * geom.down_s)?,
            geom,
        })
    }

    pub fn bytes(&self) -> usize {
        (self.geom.experts + 1) * self.geom.per_slot()
    }

    pub fn fits(&self, g: &Geom) -> bool {
        let slots = g.experts + 1;
        self.gu_q.len() >= slots * g.gu_q
            && self.gu_s.len() >= slots * g.gu_s
            && self.down_q.len() >= slots * g.down_q
            && self.down_s.len() >= slots * g.down_s
    }

    /// The stage is allocated once for the largest layer of the model; a bigger one would be a loader bug.
    fn ensure(&mut self, g: &Geom) -> Result<()> {
        if self.fits(g) {
            return Ok(());
        }
        bail!(
            "expert staging too small for this layer ({} experts x {} B, stage {} B): the loader sized it for the \
             largest layer",
            g.experts,
            g.per_slot(),
            self.bytes()
        )
    }
}

/// Reads `size` bytes of `f` from offset 0 into `dst` at `at`, in host chunks.
fn read_all(f: &TensorFile, size: usize, dst: &mut DeviceBuffer, at: usize) -> Result<()> {
    ensure!(at + size <= dst.len(), "staging write out of bounds");
    let mut buf = vec![0u8; CHUNK.min(size)];
    let mut off = 0;
    while off < size {
        let n = buf.len().min(size - off);
        f.read_at(off, &mut buf[..n])?;
        dst.upload_at(at + off, &buf[..n])?;
        off += n;
    }
    Ok(())
}
