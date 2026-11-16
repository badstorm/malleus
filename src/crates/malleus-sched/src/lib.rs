#![forbid(unsafe_code)]

//! Single-slot scheduler: one GPU thread owns the model and serves requests in FIFO order.

use anyhow::{Context, Result};
use malleus_hip::DeviceBuffer;
use malleus_model::image_pre::{PreConfig, decode, preprocess_rgb};
use malleus_model::mrope;
use malleus_model::qwen35::{Model, Session, Snapshot};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc as tmpsc;

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    /// Repetition control, llama.cpp semantics: with `repeat_penalty` different from 1.0 a token that already appears
    /// in the last `repeat_last_n` of the sequence has its logit divided by the penalty (or multiplied, when the logit
    /// is negative). `frequency_penalty`/`presence_penalty` are the OpenAI ones, applied in logit space before the
    /// temperature. All off by default: the sampling stays bit-identical to what it was before they existed.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl Sampling {
    fn penalizes(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0)
    }
}

/// The logits row to sample from: `None` when no penalty is active, so the caller keeps using the row it has.
///
/// `history` is the sequence so far and `tail` the tokens accepted in a speculative round after it; the window is the
/// last `repeat_last_n` of `history ++ tail`, which is what makes the serial step and the speculative loop see the
/// same history (a draft that gets committed is a token like any other).
fn penalized(logits: &[f32], s: &Sampling, history: &[u32], tail: &[u32]) -> Option<Vec<f32>> {
    if !s.penalizes() {
        return None;
    }
    let mut counts: Vec<(u32, u32)> = Vec::with_capacity(s.repeat_last_n.min(64));
    for &id in history.iter().rev().chain(tail.iter().rev()).take(s.repeat_last_n) {
        match counts.iter_mut().find(|(t, _)| *t == id) {
            Some((_, c)) => *c += 1,
            None => counts.push((id, 1)),
        }
    }
    if counts.is_empty() {
        return None;
    }
    let mut row = logits.to_vec();
    for (id, count) in counts {
        let Some(v) = row.get_mut(id as usize) else {
            continue;
        };
        let mut l = *v;
        if s.repeat_penalty != 1.0 {
            l = if l > 0.0 {
                l / s.repeat_penalty
            } else {
                l * s.repeat_penalty
            };
        }
        l -= s.presence_penalty;
        l -= s.frequency_penalty * count as f32;
        *v = l;
    }
    Some(row)
}

pub struct Request {
    pub prompt: Vec<u32>,
    /// Token sequences a prompt-lookup draft may come from besides the prompt itself: the original text of the recent
    /// messages (line numbers and JSON escapes removed by the API layer). A tool result reaches the model as a
    /// line-numbered rendering and is written back as plain text, so without these the tokens the model is about to
    /// copy are never in the prompt's own token stream.
    pub drafts: Vec<Vec<u32>>,
    /// Raw encoded images (PNG/JPEG/WebP/GIF), in the order their `<|image_pad|>` placeholders appear.
    pub images: Vec<Vec<u8>>,
    /// Tokenizer id of `<|image_pad|>`; required when `images` is not empty.
    pub image_token: Option<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop_tokens: Vec<u32>,
    /// Per-emitted-token temperature-1 log-probabilities in the `Token` events (OpenAI `logprobs`). Zero-cost when
    /// false: the sampling path is untouched, only the events grow a field.
    pub logprobs: bool,
    /// How many top alternatives each `Token` event also carries; 0 = none (capped by the API layer, 20 like OpenAI).
    pub top_logprobs: usize,
    /// Also keep the states after this many prompt tokens (the start of the last chat turn: the next request of
    /// the conversation shares it even when the previous answer comes back rewritten).
    pub snapshot_at: Option<usize>,
    pub cancel: Arc<AtomicBool>,
    pub events: tmpsc::UnboundedSender<Event>,
}

#[derive(Debug)]
pub enum Event {
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    /// A generated token. `top` carries its log-probability data only when the request asked for logprobs;
    /// otherwise it is None and emitting is exactly as cheap as a bare integer event.
    Token {
        id: u32,
        top: Option<TokenProbs>,
    },
    Done(Finish),
    Error(String),
}

/// Temperature-1 log-probabilities of one emitted token, from the logits row that produced it.
#[derive(Debug, Clone)]
pub struct TokenProbs {
    pub id: u32,
    /// The token's own log-probability (log-softmax over the whole vocabulary).
    pub logprob: f32,
    /// `(id, logprob)` of the alternatives, most likely first; up to the request's `top_logprobs`.
    pub top: Vec<(u32, f32)>,
}

/// The top-N of a logits row plus the token's own probability, as temperature-1 log-probabilities (log-softmax over
/// the full vocabulary). The top-N is kept as a partial queue the size of the request (an insert into a Vec of at
/// most `top` entries, never a vocabulary sort); when `top` is 0 only the token's own log-probability is returned,
/// for `logprobs: true` without `top_logprobs`.
pub fn token_probs(logits: &[f32], id: u32, top: usize) -> TokenProbs {
    let mut max = f32::NEG_INFINITY;
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(top.min(64));
    for (i, &l) in logits.iter().enumerate() {
        max = max.max(l);
        if top == 0 {
            continue;
        }
        // Position of the new entry in the descending-by-logit queue; equal logits order by id so the result is
        // deterministic.
        let pos = best.iter().position(|&(b, j)| l > b || (l == b && i < j as usize));
        match pos {
            Some(p) => {
                best.insert(p, (l, i as u32));
                if best.len() > top {
                    best.pop();
                }
            }
            None if best.len() < top => best.push((l, i as u32)),
            None => {}
        }
    }
    let mut norm = 0.0f64;
    for &l in logits {
        norm += ((l - max) as f64).exp();
    }
    let lse = max as f64 + norm.ln();
    let top = best.into_iter().map(|(l, i)| (i, (l as f64 - lse) as f32)).collect();
    TokenProbs {
        id,
        logprob: logits
            .get(id as usize)
            .map_or(f32::NEG_INFINITY, |&l| (l as f64 - lse) as f32),
        top,
    }
}

#[derive(Debug, Clone)]
pub struct Finish {
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

pub struct Engine {
    tx: mpsc::Sender<Request>,
    pub ctx: usize,
    pub queued: Arc<AtomicUsize>,
    /// Wall clock (ms since the epoch) at which the request in flight started, 0 when the worker is idle. Read by
    /// the watchdog: a GPU kernel cannot be cancelled from the host, so an overrun has to end the process.
    busy_since: Arc<AtomicU64>,
}

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0 as u32
}

/// Prompt lookup: the tokens that followed the latest earlier occurrence of the last `n` tokens of `hist` (up to
/// `max`), or none.
pub fn lookup_drafts(hist: &[u32], sources: &[Vec<u32>], n: usize, max: usize) -> Vec<u32> {
    if max == 0 || hist.len() < n || n == 0 {
        return vec![];
    }
    let key = &hist[hist.len() - n..];
    // The longest continuation wins, not the most recent match. A tool result reaches the prompt with its line
    // numbers, so a match inside the prompt's own tokens stops at the next line break (the numbered prefix is not what
    // the model writes) while the same text in `sources` continues across lines. Taking the first match instead would
    // mean the short one always shadows the long one, which is what the sources are here for.
    let mut best: Vec<u32> = vec![];
    // The match may not overlap the key itself.
    for i in (0..hist.len() - n).rev() {
        if &hist[i..i + n] == key {
            let end = (i + n + max).min(hist.len());
            if i + n < end && end - (i + n) > best.len() {
                best = hist[i + n..end].to_vec();
            }
            break;
        }
    }
    // Then the original text of the recent messages, newest first.
    for s in sources.iter().rev() {
        if s.len() <= n || best.len() >= max {
            break;
        }
        for i in (0..=s.len() - n).rev() {
            if &s[i..i + n] == key {
                let end = (i + n + max).min(s.len());
                if i + n < end && end - (i + n) > best.len() {
                    best = s[i + n..end].to_vec();
                }
                break;
            }
        }
    }
    best
}

/// Greedy MTP drafts: up to `max` tokens, chaining while the product of the draft probabilities stays >= `pmin`.
/// With `hist` (the tokens so far, ending with the pending ones), a prompt-lookup continuation that starts with the
/// MTP draft and is longer than one token replaces the chained drafts.
pub fn draft_tokens(
    model: &Model,
    sess: &mut Session,
    pend: &[u32],
    max: usize,
    pmin: f32,
    hist: Option<(&[u32], &[Vec<u32>])>,
) -> Result<Vec<u32>> {
    let (d, mut p) = model.mtp_draft(sess, pend)?;
    if let Some((h, sources)) = hist {
        let lk = lookup_drafts(h, sources, env_or("MALLEUS_LOOKUP_NGRAM", LOOKUP_NGRAM), max);
        if lk.len() >= 2 && lk[0] == d {
            return Ok(lk);
        }
    }
    let mut ds = vec![d];
    while ds.len() < max && p >= pmin {
        let (d, q) = model.mtp_chain(sess, *ds.last().unwrap())?;
        ds.push(d);
        p *= q;
    }
    Ok(ds)
}

/// Candidate tokens and their probabilities after temperature, top-k, min-p and top-p (normalized).
fn distribution(logits: &[f32], s: &Sampling) -> (Vec<u32>, Vec<f32>) {
    let k = if s.top_k == 0 { 256 } else { s.top_k.min(1024) };
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let m = logits[idx[0] as usize];
    let mut p: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i as usize] - m) / s.temperature).exp())
        .collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= sum);
    let mut keep = p.len();
    if s.min_p > 0.0 {
        keep = keep.min(p.iter().take_while(|&&v| v >= s.min_p * p[0]).count().max(1));
    }
    if s.top_p < 1.0 {
        let mut c = 0.0;
        for (i, v) in p.iter().enumerate() {
            c += v;
            if c >= s.top_p {
                keep = keep.min(i + 1);
                break;
            }
        }
    }
    idx.truncate(keep);
    p.truncate(keep);
    let total: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= total);
    (idx, p)
}

fn pick(idx: &[u32], p: &[f32], mut r: f32) -> u32 {
    for (i, &v) in p.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return idx[i];
        }
    }
    *idx.iter()
        .zip(p)
        .rev()
        .find(|(_, v)| **v > 0.0)
        .map_or(&idx[idx.len() - 1], |(i, _)| i)
}

pub fn sample(logits: &[f32], s: &Sampling, rng: &mut impl FnMut() -> f32) -> u32 {
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    let (idx, p) = distribution(logits, s);
    pick(&idx, &p, rng())
}

/// Speculative sampling with a deterministic draft: keeps `draft` with probability p(draft), otherwise samples
/// from p without it, so the output follows p exactly. Returns (token, draft accepted).
pub fn sample_spec(logits: &[f32], s: &Sampling, draft: u32, rng: &mut impl FnMut() -> f32) -> (u32, bool) {
    if s.temperature <= 0.0 {
        let t = argmax(logits);
        return (t, t == draft);
    }
    let (idx, mut p) = distribution(logits, s);
    let pd = idx.iter().position(|&t| t == draft).map_or(0.0, |i| p[i]);
    if rng() < pd {
        return (draft, true);
    }
    if let Some(i) = idx.iter().position(|&t| t == draft) {
        p[i] = 0.0;
    }
    let rest = 1.0 - pd;
    if rest <= 0.0 {
        return (draft, true);
    }
    (pick(&idx, &p, rng() * rest), false)
}

/// Prompt tokens per forward (MALLEUS_PREFILL_CHUNK overrides): larger chunks read the MoE expert weights fewer
/// times per token, at the cost of scratch memory.
fn prefill_chunk(model: &Model) -> usize {
    malleus_model::memplan::prefill_chunk(&model.cfg)
}

/// The `(start, end)` blocks an image occupies in the expanded prompt.
fn image_blocks(inserts: &[(usize, DeviceBuffer)], hidden: usize) -> Vec<(usize, usize)> {
    inserts.iter().map(|(s, b)| (*s, *s + b.len() / (hidden * 4))).collect()
}

/// Ranges of one image prompt's prefill: at most `chunk` tokens each, and never cut across an image block.
///
/// An image prompt used to be a single forward, with `positions` (T,3) and the embedding inserts indexed against
/// the whole prompt. Both are per token, so they can be sliced per piece; what cannot be sliced is an insert,
/// because `forward_mm` copies it whole (`copy_from(start, buf, 0, rows)`, source offset 0). Hence the cuts.
///
/// The cost of not chunking is the reason this exists: `Model::forward` sizes its scratch on the token count, so
/// one forward over a 30k-token agent prompt with a screenshot asks ~13 GiB of GTT (= system RAM on these APUs)
/// in one go, and leaves it cached afterwards (issue #87).
fn image_pieces(len: usize, blocks: &[(usize, usize)], chunk: usize) -> Vec<(usize, usize)> {
    let chunk = chunk.max(1);
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < len {
        let mut end = (start + chunk).min(len);
        for &(bs, be) in blocks {
            if bs < end && be > end {
                // This block would be cut by `end`: stop before it, or take it whole when the piece starts there
                // (an image block wider than the chunk gets a piece of its own).
                end = if bs > start { bs } else { be.min(len) };
                break;
            }
        }
        pieces.push((start, end));
        start = end;
    }
    pieces
}

const DEFAULT_DRAFTS: usize = 5;
/// A chain of drafts goes on while the product of their probabilities stays above this. At 0.5 the tail of the
/// chain was mostly rejected: each such draft costs a pass of the draft head and a row of the verify. Measured on
/// Flash-Next, eight prompts: 56.5 tok/s at 0.5, 58.3 at 0.65, 59.5 at 0.8, and a plateau up to 0.95.
pub const DEFAULT_DRAFT_PMIN: f32 = 0.8;

/// Patches (16 px each) an image may be reduced to. The vision tower's attention is **quadratic** in the patches:
/// measured on gfx1150 with this model, 512x512 (1024 patches) encodes in ~3.8 s and a 12 Mpx phone photo
/// (47 000 patches) needs hours, which blocked the only GPU worker with the service still answering `/v1/models`.
/// 2048 patches (~15 s by that scaling) is the default; `MALLEUS_VISION_MAX_PATCHES` raises it for models whose
/// tower can afford more.
const DEFAULT_VISION_MAX_PATCHES: usize = 2048;

/// Seconds a single request may run before the watchdog ends the process (the unit restarts it). 0 disables it.
/// Without this, any long or pathological request hangs the service until someone restarts it by hand.
const DEFAULT_REQUEST_TIMEOUT: u64 = 600;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
/// Tokens matched by prompt lookup: `MALLEUS_LOOKUP_NGRAM`, default 3. A longer context matches less often but its
/// continuation is more likely to be right, which is what pays on code and structured text.
const LOOKUP_NGRAM: usize = 3;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Worker {
    model: Model,
    sess: Session,
    cached: Vec<u32>,
    /// MTP drafts per decode step (0: plain decode).
    drafts: usize,
    draft_pmin: f32,
    /// Prompt-lookup drafts from the context when they agree with the MTP head.
    lookup: bool,
    /// Prompt snapshots (tokens, states), least recently used first; at most `max_snaps`.
    snaps: Vec<(Vec<u32>, Snapshot)>,
    /// Snapshots whose GPU buffers are already allocated, waiting to be refilled: a prompt is snapshotted on every
    /// request, and allocating the ~100 state buffers costs far more than copying them (`Model::new_snapshot`).
    snap_pool: Vec<Snapshot>,
    max_snaps: usize,
}

impl Worker {
    /// Hands a retired snapshot's buffers back to the pool; freeing them would give the pages right back to the
    /// allocator we are trying not to call on the request path.
    fn recycle(&mut self, snap: Snapshot) {
        if self.snap_pool.len() < 2 {
            self.snap_pool.push(snap);
        }
    }

    /// A snapshot to fill: one from the pool, else the buffers of the least recently used snapshot when all
    /// `max_snaps` are full (it is the one about to be dropped anyway), else a fresh allocation.
    fn take_snap(&mut self) -> Option<Snapshot> {
        self.snap_pool.pop().or_else(|| {
            if !self.snaps.is_empty() && self.snaps.len() >= self.max_snaps {
                return Some(self.snaps.remove(0).1);
            }
            self.model.new_snapshot(&self.sess).ok()
        })
    }

    /// Keeps the session states for the tokens in `cached` (least recently used snapshots dropped first).
    fn push_snapshot(&mut self) {
        if self.max_snaps == 0 {
            return;
        }
        let mut snap = match self.take_snap() {
            Some(s) => s,
            None => return,
        };
        if self.model.snapshot_into(&self.sess, &mut snap).is_err() {
            self.recycle(snap);
            return;
        }
        if let Some(i) = self.snaps.iter().position(|(t, _)| *t == self.cached) {
            let old = self.snaps.remove(i).1;
            self.recycle(old);
        }
        self.snaps.push((self.cached.clone(), snap));
        while self.snaps.len() > self.max_snaps {
            let old = self.snaps.remove(0).1;
            self.recycle(old);
        }
    }

    fn serve(&mut self, mut req: Request) -> Result<Finish> {
        // Images are expanded and encoded before anything else: the prompt the rest of this function sees is the
        // one with the `<|image_pad|>` placeholder replaced by the grid's tokens, and its mrope positions are the
        // ones the vision tower's merge size implies.
        let image_req = !req.images.is_empty();
        let mut image_pos: Option<Vec<[i32; 3]>> = None;
        let mut image_inserts: Vec<(usize, DeviceBuffer)> = Vec::new();
        if image_req {
            let token = req.image_token.context("images without an <|image_pad|> token id")?;
            let vision = self.model.vision.as_ref().context("model has no vision tower")?;
            let merge = vision.cfg.merge;
            let mut expanded = Vec::with_capacity(req.prompt.len() + 256);
            let mut grids = Vec::new();
            let mut starts = Vec::new();
            let mut merged = Vec::new();
            let mut images = req.images.iter();
            for &tok in &req.prompt {
                if tok != token {
                    expanded.push(tok);
                    continue;
                }
                let bytes = images.next().context("more image placeholders than images")?;
                let (rgb, h, w) = decode(bytes, &PreConfig::default())?;
                // The tower's attention is quadratic in the patches, so the image is reduced to a patch budget
                // instead of the 16.7 Mpx the processor would accept (~65 000 patches, hours of GPU).
                let max_patches = env_or("MALLEUS_VISION_MAX_PATCHES", DEFAULT_VISION_MAX_PATCHES).max(1);
                let patch = vision.cfg.patch;
                let pre = preprocess_rgb(
                    &rgb,
                    h,
                    w,
                    &PreConfig {
                        max_pixels: max_patches * patch * patch,
                        ..Default::default()
                    },
                )?;
                anyhow::ensure!(
                    pre.n_patches() <= max_patches,
                    "image {w}x{h} needs {} vision patches, over MALLEUS_VISION_MAX_PATCHES={max_patches}",
                    pre.n_patches()
                );
                if pre.grid_h * patch < h || pre.grid_w * patch < w {
                    eprintln!(
                        "image {w}x{h} -> {}x{} ({} patches, cap {max_patches}): the vision tower is quadratic in \
                         the patches, MALLEUS_VISION_MAX_PATCHES raises the cap",
                        pre.grid_w * patch,
                        pre.grid_h * patch,
                        pre.n_patches()
                    );
                }
                anyhow::ensure!(pre.grid_t == 1, "video input not supported");
                anyhow::ensure!(
                    pre.grid_h % merge == 0 && pre.grid_w % merge == 0,
                    "image grid {}x{} is not a multiple of merge {merge}",
                    pre.grid_h,
                    pre.grid_w
                );
                let (gh, gw) = (pre.grid_h / merge, pre.grid_w / merge);
                let start = expanded.len();
                expanded.extend(std::iter::repeat_n(token, gh * gw));
                grids.push((gh, gw));
                starts.push(start);
                let (_, m) = vision.encode(&self.model.k, &pre)?;
                merged.push(m);
            }
            anyhow::ensure!(images.next().is_none(), "more images than image placeholders");
            req.prompt = expanded;
            let (pos, _) = mrope::positions(&req.prompt, token, &grids, 0)?;
            image_pos = Some(pos);
            image_inserts = starts.into_iter().zip(merged).collect();
        }
        let ctx = self.sess.max_ctx;
        anyhow::ensure!(
            req.prompt.len() < ctx,
            "prompt of {} tokens exceeds the context of {ctx}",
            req.prompt.len()
        );
        let max_tokens = req.max_tokens.min(ctx - req.prompt.len());
        // A prompt with images cannot reuse a token-prefix snapshot: two different images of the same grid have
        // the same tokens but different embeddings. Image requests restart from an empty session and drop the
        // snapshots, so a later text request cannot restore a state that contains an image either.
        let reuse = !image_req
            && self.cached.len() < req.prompt.len()
            && req.prompt.starts_with(&self.cached)
            && !self.cached.is_empty();
        if !reuse {
            // The longest snapshot that is a prefix of the prompt and whose tokens the session's KV rows still hold
            // (they were not overwritten since): restart there instead of from the beginning.
            let lcp = self.cached.iter().zip(&req.prompt).take_while(|(a, b)| a == b).count();
            let best = if image_req {
                None
            } else {
                self.snaps
                    .iter()
                    .enumerate()
                    .filter(|(_, (toks, snap))| {
                        snap.len == toks.len()
                            && toks.len() <= lcp
                            && toks.len() < req.prompt.len()
                            && req.prompt.starts_with(toks)
                    })
                    .max_by_key(|(_, (toks, _))| toks.len())
                    .map(|(i, _)| i)
            };
            match best {
                Some(i) => {
                    let (toks, snap) = &self.snaps[i];
                    self.model.restore(&mut self.sess, snap)?;
                    self.cached.truncate(toks.len());
                    // Most recently used last.
                    let e = self.snaps.remove(i);
                    self.snaps.push(e);
                }
                None => {
                    self.model.reset_session(&mut self.sess)?;
                    self.cached.clear();
                }
            }
        }
        if image_req {
            while let Some((_, snap)) = self.snaps.pop() {
                self.recycle(snap);
            }
        }
        let cached_tokens = self.cached.len();
        let _ = req.events.send(Event::Started {
            prompt_tokens: req.prompt.len(),
            cached_tokens,
        });
        let t0 = Instant::now();
        let todo = &req.prompt[cached_tokens..];
        let mut logits = Vec::new();
        let dbg = std::env::var_os("MALLEUS_DEBUG_SNAPS").is_some();
        // Prefill pieces: the chunking alone. Splitting at the snapshot point costs a whole extra forward, and a
        // forward costs what it costs whatever the number of tokens (on the 35B-A3B: 502 ms for 7 tokens against
        // 576 ms for 306), so the mid-prompt snapshot is taken at the last piece boundary that still precedes it
        // and skipped when there is none. The end-of-prompt snapshot covers the common case on its own: the prompt
        // ends with the assistant marker, so it is already the point a rewritten answer still shares.
        // An image prompt is chunked like a text one: its `positions` and its embedding inserts are per token, so
        // they are sliced per piece. What made it a single forward, and what made a long prompt with a screenshot
        // ask for tens of GiB of scratch in one allocation, is in `image_pieces`.
        let chunk_tokens = prefill_chunk(&self.model);
        let pieces: Vec<&[u32]> = if image_req && !todo.is_empty() {
            let blocks = image_blocks(&image_inserts, self.model.cfg.hidden);
            let pieces = image_pieces(todo.len(), &blocks, chunk_tokens);
            if pieces.len() > 1 {
                eprintln!("image prompt: {} tokens in {} prefill pieces", todo.len(), pieces.len());
            }
            pieces.into_iter().map(|(s, e)| &todo[s..e]).collect()
        } else {
            todo.chunks(chunk_tokens).collect()
        };
        let snap_at = req
            .snapshot_at
            .filter(|&c| self.max_snaps > 0 && c > cached_tokens && c < req.prompt.len());
        let mid = if image_req {
            None
        } else {
            snap_at.and_then(|at| {
                let mut acc = cached_tokens;
                let mut best = None;
                for (i, pc) in pieces.iter().enumerate() {
                    acc += pc.len();
                    if acc <= at && acc < req.prompt.len() {
                        best = Some(i);
                    }
                }
                best
            })
        };
        // Offset of the piece being prefilled inside `todo`: the image positions and inserts are indexed against
        // the whole prompt (an image request always restarts from an empty session, so `cached_tokens` is 0, but
        // the sum is what the indices mean).
        let mut off = 0usize;
        for (i, chunk) in pieces.iter().enumerate() {
            if req.cancel.load(Ordering::Relaxed) {
                self.cached.clear();
                self.model.reset_session(&mut self.sess)?;
                return Ok(Finish {
                    reason: FinishReason::Cancelled,
                    prompt_tokens: req.prompt.len(),
                    completion_tokens: 0,
                    prefill_s: t0.elapsed().as_secs_f64(),
                    decode_s: 0.0,
                });
            }
            let tp = Instant::now();
            logits = if image_req {
                let base = cached_tokens + off;
                let pos = image_pos.as_deref().map(|p| &p[base..base + chunk.len()]);
                let ins: Vec<(usize, &DeviceBuffer)> = image_inserts
                    .iter()
                    .filter_map(|(s, b)| {
                        let rows = b.len() / (self.model.cfg.hidden * 4);
                        (*s >= base && *s + rows <= base + chunk.len()).then(|| (*s - base, b))
                    })
                    .collect();
                self.model.forward_mm(&mut self.sess, chunk, pos, 1, &ins)?
            } else {
                self.model.forward(&mut self.sess, chunk, None, 1)?
            };
            off += chunk.len();
            if dbg {
                eprintln!("piece {} tok: {:.1} ms", chunk.len(), tp.elapsed().as_secs_f64() * 1e3);
            }
            self.cached.extend_from_slice(chunk);
            if mid == Some(i) {
                let ts = Instant::now();
                self.push_snapshot();
                if dbg {
                    eprintln!("snap mid: {:.1} ms", ts.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        // Snapshot at the end of the prompt: the next turn of a conversation usually extends it (the previous
        // answer may come back rewritten, e.g. without its reasoning, so the session itself would not match).
        if !todo.is_empty() && self.cached.len() == req.prompt.len() {
            let tsn = Instant::now();
            self.push_snapshot();
            if std::env::var_os("MALLEUS_DEBUG_SNAPS").is_some() {
                eprintln!("snap end: {:.1} ms", tsn.elapsed().as_secs_f64() * 1e3);
            }
        }
        let t1 = Instant::now();
        let mut rng = Rng(req.sampling.seed ^ 0x5EED);
        let mut n = 0;
        let mut reason = FinishReason::Length;
        let vocab = self.model.cfg.vocab;
        let mut tok = {
            let pen = penalized(&logits, &req.sampling, &self.cached, &[]);
            sample(pen.as_deref().unwrap_or(&logits), &req.sampling, &mut || rng.next_f32())
        };
        let mut pend = vec![tok];
        // Emits a token with its log-probability data (None when logprobs were not asked); false when generation
        // must stop.
        let emit = |t: u32, top: Option<TokenProbs>, n: &mut usize, reason: &mut FinishReason| -> bool {
            *n += 1;
            if req.stop_tokens.contains(&t) {
                *reason = FinishReason::Stop;
                return false;
            }
            if req.events.send(Event::Token { id: t, top }).is_err() || req.cancel.load(Ordering::Relaxed) {
                *reason = FinishReason::Cancelled;
                return false;
            }
            *n < max_tokens
        };
        // The data of the token about to be emitted, captured from the row it was sampled from. In the raw path the
        // row lives in `logits` until the next forward; after a verify step the next token's row is the slice saved
        // before `l` goes out of scope, so `top` is carried from one iteration to the next.
        let mut top = if req.logprobs {
            Some(token_probs(&logits, tok, req.top_logprobs))
        } else {
            None
        };
        while n < max_tokens {
            if !emit(tok, top.take(), &mut n, &mut reason) || self.sess.len + 1 > ctx {
                break;
            }
            let spec = self.drafts > 0 && self.sess.len + 2 <= ctx && self.model.mtp_pending(&self.sess) == pend.len();
            if !spec {
                logits = self.model.forward(&mut self.sess, &[tok], None, 1)?;
                self.cached.push(tok);
                let pen = penalized(&logits, &req.sampling, &self.cached, &[]);
                let row = pen.as_deref().unwrap_or(&logits);
                tok = sample(row, &req.sampling, &mut || rng.next_f32());
                pend = vec![tok];
                top = if req.logprobs {
                    Some(token_probs(row, tok, req.top_logprobs))
                } else {
                    None
                };
                continue;
            }
            let max = self.drafts.min(ctx - self.sess.len - 1);
            self.cached.push(tok);
            let ds = draft_tokens(
                &self.model,
                &mut self.sess,
                &pend,
                max,
                self.draft_pmin,
                self.lookup.then_some((&self.cached[..], &req.drafts[..])),
            )?;
            self.cached.pop();
            let nd = ds.len();
            let toks: Vec<u32> = std::iter::once(tok).chain(ds.iter().copied()).collect();
            let l = self.model.forward_verify(&mut self.sess, &toks)?;
            self.cached.push(tok);
            let mut acc = 0;
            let mut y = 0;
            while acc <= nd {
                let row = &l[acc * vocab..(acc + 1) * vocab];
                // A draft accepted in this round is a token of the sequence like any other: the row that samples
                // after it must see it, or the penalty would differ from the serial step.
                let pen = penalized(row, &req.sampling, &self.cached, &ds[..acc]);
                let row = pen.as_deref().unwrap_or(row);
                if acc == nd {
                    y = sample(row, &req.sampling, &mut || rng.next_f32());
                    break;
                }
                let (t, ok) = sample_spec(row, &req.sampling, ds[acc], &mut || rng.next_f32());
                if !ok {
                    y = t;
                    break;
                }
                acc += 1;
            }
            self.model.rollback(&mut self.sess, nd - acc)?;
            self.cached.extend_from_slice(&ds[..acc]);
            // One event per accepted draft, from the verify row that produced each of them: row `i` of the verify
            // predicts `toks[i + 1]`, so draft `i` carries the log-probabilities of slice `i`. Iterated by index so
            // the no-logprobs path stays a plain slice over `ds` with no extra copy.
            let mut done = false;
            for i in 0..acc {
                let d = ds[i];
                let t = if req.logprobs {
                    Some(token_probs(&l[i * vocab..(i + 1) * vocab], d, req.top_logprobs))
                } else {
                    None
                };
                if !emit(d, t, &mut n, &mut reason) {
                    done = true;
                    break;
                }
            }
            if done {
                break;
            }
            // The next token's data, from the row it was sampled off: slice `acc` is the fallback row when draft
            // `acc` was rejected, the last verify row when all `n_drafts` were accepted.
            top = if req.logprobs {
                Some(token_probs(&l[acc * vocab..(acc + 1) * vocab], y, req.top_logprobs))
            } else {
                None
            };
            pend = ds[..acc].to_vec();
            pend.push(y);
            tok = y;
        }
        Ok(Finish {
            reason,
            prompt_tokens: req.prompt.len(),
            completion_tokens: n,
            prefill_s,
            decode_s: t1.elapsed().as_secs_f64(),
        })
    }
}

impl Engine {
    /// Starts the GPU thread. `init` runs on that thread and must build the model (HIP context is per thread).
    pub fn start(ctx: usize, init: impl FnOnce() -> Result<Model> + Send + 'static) -> Result<Engine> {
        let (tx, rx) = mpsc::channel::<Request>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let queued = Arc::new(AtomicUsize::new(0));
        let q2 = queued.clone();
        let busy_since = Arc::new(AtomicU64::new(0));
        let b2 = busy_since.clone();
        std::thread::Builder::new().name("malleus-gpu".into()).spawn(move || {
            let worker = init().and_then(|model| {
                let mut sess = model.new_session(ctx)?;
                // One prefill chunk at startup: its scratch buffers are allocated and their pages mapped before the
                // first request (a longer request than any before otherwise pays for both, up to ~25 %).
                if std::env::var("MALLEUS_WARMUP").map_or(true, |v| v != "0") {
                    let n = prefill_chunk(&model).min(ctx.saturating_sub(1));
                    let toks: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
                    let t0 = Instant::now();
                    model.forward(&mut sess, &toks, None, 1)?;
                    model.reset_session(&mut sess)?;
                    eprintln!("warm-up: {n} tokens in {:.1}s", t0.elapsed().as_secs_f64());
                    // The number the pre-flight's estimate is checked against: the weights, one session and
                    // the working set of a full chunk are all allocated now.
                    eprintln!(
                        "memory: {:.1} GiB live on the device after the warm-up",
                        malleus_hip::device_bytes_live() as f64 / (1u64 << 30) as f64
                    );
                }
                Ok(Worker {
                    drafts: if model.has_mtp() {
                        env_or("MALLEUS_SPEC_DRAFTS", DEFAULT_DRAFTS).min(malleus_model::qwen35::GEMV_MAX_T - 1)
                    } else {
                        0
                    },
                    draft_pmin: env_or("MALLEUS_DRAFT_PMIN", DEFAULT_DRAFT_PMIN),
                    lookup: std::env::var("MALLEUS_LOOKUP").map_or(true, |v| v != "0"),
                    snaps: Vec::new(),
                    snap_pool: Vec::new(),
                    max_snaps: env_or("MALLEUS_PROMPT_SNAPSHOTS", 8),
                    model,
                    sess,
                    cached: Vec::new(),
                })
            });
            let mut worker = match worker {
                Ok(w) => {
                    let _ = ready_tx.send(Ok(()));
                    w
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            while let Ok(req) = rx.recv() {
                q2.fetch_sub(1, Ordering::Relaxed);
                b2.store(now_ms(), Ordering::Relaxed);
                let events = req.events.clone();
                let served = worker.serve(req);
                b2.store(0, Ordering::Relaxed);
                match served {
                    Ok(f) => {
                        let _ = events.send(Event::Done(f));
                    }
                    Err(e) => {
                        worker.cached.clear();
                        let _ = events.send(Event::Error(format!("{e:#}")));
                    }
                }
            }
        })?;
        ready_rx.recv()??;
        start_watchdog(busy_since.clone())?;
        Ok(Engine {
            tx,
            ctx,
            queued,
            busy_since,
        })
    }

    pub fn submit(&self, req: Request) -> Result<()> {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.tx.send(req).map_err(|_| anyhow::anyhow!("engine thread stopped"))
    }

    /// How long the request in flight has been running, or `None` when the worker is idle.
    pub fn busy_for(&self) -> Option<Duration> {
        match self.busy_since.load(Ordering::Relaxed) {
            0 => None,
            t => Some(Duration::from_millis(now_ms().saturating_sub(t))),
        }
    }
}

/// Ends the process when a request overruns `MALLEUS_REQUEST_TIMEOUT` seconds. HIP has no way to cancel a kernel
/// already running, so a request that overruns cannot be abandoned in place: the alternative to restarting is
/// leaving the only GPU worker busy forever, with the service still answering `/v1/models` and every later request
/// queued behind it. Exiting is what makes the unit's `Restart=on-failure` recover the service on its own.
fn start_watchdog(busy_since: Arc<AtomicU64>) -> Result<()> {
    let timeout = env_or("MALLEUS_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT);
    if timeout == 0 {
        return Ok(());
    }
    std::thread::Builder::new()
        .name("malleus-watchdog".into())
        .spawn(move || {
            let limit = Duration::from_secs(timeout);
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let started = busy_since.load(Ordering::Relaxed);
                if started == 0 {
                    continue;
                }
                let running = Duration::from_millis(now_ms().saturating_sub(started));
                if running > limit {
                    eprintln!(
                        "watchdog: a request has been running for {:.0}s, over MALLEUS_REQUEST_TIMEOUT={timeout}s; \
                         a GPU kernel cannot be cancelled, so the process exits and the service restarts",
                        running.as_secs_f64()
                    );
                    std::process::exit(70);
                }
            }
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_pieces_cover_the_prompt_and_never_cut_a_block() {
        // Two image blocks (10..18 and 33..45) inside a 50-token prompt, chunk 16.
        let blocks = [(10, 18), (33, 45)];
        let p = image_pieces(50, &blocks, 16);
        assert_eq!(p.first().unwrap().0, 0, "the pieces start at the first token");
        assert_eq!(p.last().unwrap().1, 50, "the pieces reach the last token");
        for w in p.windows(2) {
            assert_eq!(w[0].1, w[1].0, "contiguous: no gaps and no overlaps");
        }
        for (s, e) in &p {
            assert!(e > s && e - s <= 16, "piece {s}..{e} over the chunk");
            for (bs, be) in blocks {
                // A piece may contain a whole block, and may begin or end exactly on an edge; what it must never do
                // is put a boundary strictly inside one, because `forward_mm` copies the insert whole.
                assert!(
                    !(*s > bs && *s < be) && !(*e > bs && *e < be),
                    "piece {s}..{e} cuts the image block {bs}..{be}"
                );
            }
        }
        // Greedy: 0-10 stops before the first block, 10-26 takes it whole, 26-33 stops before the second, 33-49
        // takes it whole, 49-50 is the tail.
        assert_eq!(p, vec![(0, 10), (10, 26), (26, 33), (33, 49), (49, 50)]);
    }

    #[test]
    fn image_pieces_without_images_are_the_plain_chunks() {
        let p = image_pieces(70, &[], 16);
        assert_eq!(p, vec![(0, 16), (16, 32), (32, 48), (48, 64), (64, 70)]);
        assert!(image_pieces(0, &[], 16).is_empty(), "an empty prompt has no piece");
        // A block ending on the chunk boundary adds no cut; a block wider than the chunk still fits a piece of its
        // own, and the tokens after it go back to the chunk size.
        assert_eq!(image_pieces(32, &[(16, 32)], 16), vec![(0, 16), (16, 32)]);
        assert_eq!(image_pieces(40, &[(4, 34)], 16), vec![(0, 4), (4, 34), (34, 40)]);
        // A block that already fits inside a piece does not cut it.
        assert_eq!(image_pieces(10, &[(2, 4)], 16), vec![(0, 10)]);
    }

    #[test]
    fn repeat_penalty_pushes_a_repeated_token_down() {
        // Token 3 keeps winning; after it has been generated it must lose to token 5.
        let mut logits = vec![0.0f32; 8];
        logits[3] = 4.0;
        logits[5] = 3.0;
        let greedy = Sampling::default();
        assert_eq!(
            sample(&logits, &greedy, &mut || 0.5),
            3,
            "without a penalty the argmax wins"
        );
        let s = Sampling {
            repeat_penalty: 1.5,
            repeat_last_n: 64,
            ..Default::default()
        };
        let row = penalized(&logits, &s, &[3, 7, 3], &[]).expect("penalty active");
        assert_eq!(sample(&row, &s, &mut || 0.5), 5, "4.0 / 1.5 = 2.67 < 3.0");
        // Only the last `repeat_last_n` tokens count: with a window of 1 the window is the 7 only, the 3 is out and
        // the argmax stays where it was.
        let narrow = Sampling {
            repeat_penalty: 1.5,
            repeat_last_n: 1,
            ..Default::default()
        };
        let row = penalized(&logits, &narrow, &[3, 7], &[]).expect("the window still holds the 7");
        assert_eq!(sample(&row, &narrow, &mut || 0.5), 3, "the 3 is outside the window");
        // Off by default and with the penalties off the row is not even copied.
        assert!(penalized(&logits, &Sampling::default(), &[3], &[]).is_none());
        // Frequency and presence are additive in logit space: three occurrences of 3 cost 3 * 0.5 and 0.2.
        let s = Sampling {
            frequency_penalty: 0.5,
            presence_penalty: 0.2,
            ..Default::default()
        };
        let row = penalized(&logits, &s, &[3, 3, 3], &[]).expect("penalty active");
        assert!((row[3] - (4.0 - 1.5 - 0.2)).abs() < 1e-6, "got {}", row[3]);
        // A draft accepted in a speculative round is part of the window like any other token.
        let s = Sampling {
            repeat_penalty: 1.5,
            ..Default::default()
        };
        assert!(penalized(&logits, &s, &[7], &[3]).is_some());
    }

    #[test]
    fn greedy_and_top_k() {
        let l = vec![0.0, 5.0, 1.0, 4.9];
        assert_eq!(sample(&l, &Sampling::default(), &mut || 0.5), 1);
        let s = Sampling {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        assert_eq!(sample(&l, &s, &mut || 0.99), 1);
        let s = Sampling {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        assert_eq!(sample(&l, &s, &mut || 0.99), 3);
    }

    #[test]
    fn token_probs_are_the_full_softmax_in_log_space() {
        // lse = ln(&Sigma; exp), and every reported probability is exp(logit - lse); the alternatives' probabilities
        // (and the token's own, which may be outside the top-N) are all proper probabilities.
        let l = vec![0.0, 1.0, 2.0, 3.0, -0.5];
        let p = token_probs(&l, 3, 3);
        assert_eq!(p.id, 3);
        let lse = (l.iter().map(|&v| (v as f64).exp()).sum::<f64>()).ln();
        let mut expect: Vec<(u32, f32)> = (0..l.len() as u32)
            .map(|i| (i, (l[i as usize] as f64 - lse) as f32))
            .collect();
        expect.sort_by(|&(ia, la), &(ib, lb)| (lb as f64 - la as f64).total_cmp(&0.0).then(ia.cmp(&ib)));
        assert_eq!(p.top, expect[0..3].to_vec(), "most likely first, top-N only");
        let own = expect.iter().find(|(i, _)| *i == p.id).unwrap();
        assert!((p.logprob - own.1).abs() < 1e-6, "own logprob follows the full softmax");
        let sum: f64 = p.top.iter().map(|&(_, lp)| lp.exp() as f64).sum();
        assert!(
            sum <= 1.0 + 1e-6,
            "the alternatives' probabilities sum, exp'd, stay below 1"
        );
        for (_, lp) in &p.top {
            assert!(*lp <= 1e-6, "no probability above 1");
        }
        // The sampled token's own probability is reported even when it is not in the top-N.
        let own4 = expect.iter().find(|(i, _)| *i == 4).unwrap();
        let p4 = token_probs(&l, 4, 1);
        assert!(
            (p4.logprob - own4.1).abs() < 1e-6,
            "id outside the top-N still gets its logprob"
        );
        assert_eq!(p4.top.len(), 1);
        // No alternatives asked: only the token's own log-probability.
        let p0 = token_probs(&l, 3, 0);
        assert!(p0.top.is_empty());
        assert!((p0.logprob - own.1).abs() < 1e-6);
    }

    #[test]
    fn speculative_sampling_keeps_the_distribution() {
        let l = vec![0.0, 1.0, 2.0, 0.5];
        let s = Sampling {
            temperature: 1.0,
            ..Default::default()
        };
        let mut rng = Rng(7);
        let n = 200_000;
        let mut counts = [0usize; 4];
        for i in 0..n {
            let (t, _) = sample_spec(&l, &s, (i % 4) as u32, &mut || rng.next_f32());
            counts[t as usize] += 1;
        }
        let z: f32 = l.iter().map(|v| v.exp()).sum();
        for (c, v) in counts.iter().zip(&l) {
            assert!((*c as f32 / n as f32 - v.exp() / z).abs() < 0.005, "{counts:?}");
        }
        assert_eq!(sample_spec(&l, &Sampling::default(), 2, &mut || 0.5), (2, true));
        assert_eq!(sample_spec(&l, &Sampling::default(), 1, &mut || 0.5), (2, false));
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::lookup_drafts;

    /// The lookup takes the longest continuation it can find, not the most recent: a tool result sits in the prompt
    /// with its line numbers, so the match there stops at the next line break while the original text keeps going.
    #[test]
    fn longest_continuation_wins() {
        let hist = [10u32, 11, 12, 20, 10, 11, 12]; // in-prompt match continues with one token
        let original = [10u32, 11, 12, 30, 31, 32, 33, 34];
        let got = lookup_drafts(&hist, &[original.to_vec()], 3, 5);
        assert_eq!(
            got,
            vec![30, 31, 32, 33, 34],
            "the short in-prompt match must not shadow the long one"
        );
    }

    #[test]
    fn falls_back_to_the_prompt() {
        let hist = [4u32, 5, 6, 100, 200, 300, 4, 5, 6];
        let got = lookup_drafts(&hist, &[vec![42, 43, 44]], 3, 3);
        assert_eq!(got, vec![100, 200, 300]);
    }

    #[test]
    fn no_match_and_short_history() {
        let hist = [1u32, 2, 3, 4];
        assert!(lookup_drafts(&hist, &[vec![9, 9, 9]], 3, 5).is_empty());
        assert!(lookup_drafts(&[1u32, 2], &[], 3, 5).is_empty());
        assert!(lookup_drafts(&[1u32, 2, 3, 1, 2, 3], &[], 3, 0).is_empty());
    }
}
