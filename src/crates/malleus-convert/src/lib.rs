#![forbid(unsafe_code)]

//! safetensors (bf16) → Malleus model file (`.malleus`: GGUF container with MLS4 / MLS8 / bf16 tensors).
//!
//! The converter first builds a plan of output tensors. Most are one source tensor, copied or quantized; some are
//! built from several sources or padded (rows and columns past the sources are zero):
//! - `<hc>.mix_down.weight` (qwen4_exp hyper-connections): `input_mix_weight_down` rows, then the
//!   `block_inject_weight` rows, padded to a multiple of 128 rows; `<hc>.mix_up.weight`: `input_mix_weight_up` with
//!   its K padded to the same width. Both MLS8.
//! - `<ple>.ngram_table` (qwen4_exp PLE): the shards of the hashed n-gram embedding stacked in order, one int8 row
//!   per embedding row followed by its f32 scale (little endian): [rows, dim + 4] I8. Read on the host, never uploaded.
//! - `model.layers.N.mlp.experts.{gate_up_proj,down_proj}` (Ling-3.0-flash): the per-expert tensors fused into one
//!   [E, rows, K] tensor per layer; in `gate_up` the gate rows of each expert come before its up rows (ADR 0002).

use anyhow::{Context, Result, bail, ensure};
use malleus_gguf::gguf::{ARCH_PREFIX, GgufWriter, Value};
use malleus_gguf::quant::{self, BLOCK, HADAMARD_SEED, LAYOUT_VERSION};
use malleus_gguf::safetensors::SafeTensors;
use malleus_gguf::{DType, TensorSource};
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Raw,
    Mls3,
    Mls4,
    Mls8,
    /// int8 rows with a trailing f32 scale each (n-gram lookup table).
    RowI8,
}

/// Which MoE layers store `gate`/`up` at 3 bit. MIGRATE_MODELS.md §2.4 keeps the sensitive last layers at 4 bit;
/// the exact range is a per-model decision, measured with `malleus eval` and written in its document.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mls3Layers {
    /// The range the family's model document chose, derived from `config.json`.
    #[default]
    Family,
    /// Every expert at 4 bit.
    Off,
    /// Inclusive layer range.
    Range(usize, usize),
}

/// Layers at the end of the stack that stay at 4 bit even with MLS3 on. Both model documents that use MLS3 promote
/// their last layers (Ling-3.0-flash: 35..41 plus the MTP; Qwen3.8-27B keeps its last 8 in MLS8 for prefill).
const MLS3_SKIP_LAST_LAYERS: usize = 8;

pub struct Options {
    /// Also store an MLS8 copy (`<name>.prefill`) of every MLS4 linear for prefill. **Off by default**: the copy
    /// doubles the file (~35 GB more for a 35B MoE) and buys prompt quality, not decode speed (measured: eval top-1
    /// 93.87 % without, 97.02 % with, prefill +11…29 % without). The name of the output should say which one it is:
    /// `-mls4` for a single copy, `-mls4-mls8` for the double one.
    pub prefill_int8: bool,
    /// Skip that copy for the routed experts only: prefill then reads their MLS4 nibbles with int8 activations
    /// (W4A8), halving the bytes of the widest block of the prefill and ~32 GB of device memory, while the dense
    /// linears keep their W8A8 path. Only worth it when those nibbles are GPTQ-calibrated: round-to-nearest loses
    /// ~3.9 points of top-1 on the eval set.
    pub no_prefill_int8_experts: bool,
    /// Layers whose FFN stays MLS8 in decode too.
    pub keep_int8_layers: Vec<usize>,
    /// Directory with calibrated MLS4 nibbles and scales (scripts/calibrate.py), used instead of round-to-nearest.
    pub mls4_from: Option<std::path::PathBuf>,
    /// Layers whose MoE `gate`/`up` go to MLS3; MLS3 tensors are always round-to-nearest, calibration is MLS4 only.
    pub mls3_layers: Mls3Layers,
    /// Also store the hyper-connection `mix_down`/`mix_up` as MLS4 for decode, keeping the MLS8 copy for prefill.
    /// Decode then reads 4 bit per weight instead of 8 on the 0.78 GB per token the mixing costs; `prefill` keeps
    /// working unchanged because `gemm_i8_hcmix` and `gemm_i8_sk` have no int4 form.
    pub hc_mls4: bool,
}

/// A model family `convert` can write: it fixes `general.architecture`, the name mapping and which tensor class
/// gets which format (MIGRATE_MODELS.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen3.5 and Qwen3.5-MoE (`qwen3_5`, `qwen3_5_moe`).
    Qwen35,
    /// Qwen3.8-Flash-Next (`qwen4_exp`): hyper-connections, PLE n-gram embedding, QSA indexer.
    Qwen4Exp,
    /// Ling-3.0-flash (`bailing_hybrid`, `BailingMoeV3`): KDA + MLA + 512 experts.
    Ling3,
    /// A DFlash speculative draft (`architectures: ["DFlashDraftModel"]`): 6 layers that propose a block of tokens
    /// for a target model to verify, fed by the target's hidden states and sharing its embedding and `lm_head`.
    /// Not a standalone model: `dflash_config.target_layer_ids` names the target layers it reads.
    Dflash,
}

impl Family {
    /// Picks the family from `model_type` in `config.json`.
    pub fn from_model_type(t: &str) -> Result<Self> {
        Ok(match t {
            "qwen3_5" | "qwen3_5_moe" => Family::Qwen35,
            "qwen4_exp" => Family::Qwen4Exp,
            "bailing_hybrid" => Family::Ling3,
            t => bail!("unsupported model_type `{t}`"),
        })
    }

    /// Family of a checkpoint, from `architectures` first: a DFlash draft declares `model_type: "qwen3"` (the model it
    /// drafts for), so the architecture name is the only thing that tells it apart.
    pub fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        let arch = |name: &str| {
            cfg["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(name)))
        };
        if arch("DFlashDraftModel") {
            return Ok(Family::Dflash);
        }
        Self::from_model_type(cfg["model_type"].as_str().context("config.json has no model_type")?)
    }

    /// Value of `general.architecture` without the `openmalleus.` prefix; `malleus-model` dispatches on it.
    pub fn arch(self) -> &'static str {
        match self {
            Family::Qwen35 => "qwen35",
            Family::Qwen4Exp => "qwen4exp",
            Family::Ling3 => "ling3",
            Family::Dflash => "dflash",
        }
    }

    /// Whether an MLS8 prefill copy of an MLS4 tensor fits the memory budget. qwen4_exp (123 B expert parameters)
    /// keeps it only outside the experts; Ling-3.0-flash (124 GB more) has none and prefills on W4A8.
    fn prefill_copy(self, expert: bool) -> bool {
        match self {
            Family::Qwen35 => true,
            Family::Qwen4Exp => !expert,
            Family::Ling3 => false,
            // 738 MB of draft: it never runs on more than a block of positions, so a prefill copy is dead weight.
            Family::Dflash => false,
        }
    }

    /// Format of one output tensor. `mls3` is the resolved inclusive layer range for the MoE `gate`/`up`.
    pub fn classify(self, name: &str, shape: &[usize], opts: &Options, mls3: Option<(usize, usize)>) -> Kind {
        if name == "lm_head.weight" {
            return Kind::Mls8;
        }
        if name.contains("visual.") {
            return Kind::Raw;
        }
        if self != Family::Ling3 && MLS8_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            return Kind::Mls8;
        }
        let mls4 = match self {
            Family::Qwen35 | Family::Qwen4Exp => QWEN_MLS4_SUFFIXES,
            Family::Ling3 => LING3_MLS4_SUFFIXES,
            Family::Dflash => DFLASH_MLS4_SUFFIXES,
        };
        if !mls4.iter().any(|s| name.ends_with(s)) {
            return Kind::Raw;
        }
        // The MLA output gate of Ling is [num_heads, hidden]: a gate, never quantized. The KDA one is [4096, 2560].
        if self == Family::Ling3 && name.ends_with("attention.g_proj.weight") && shape.first().is_some_and(|&n| n <= 64)
        {
            return Kind::Raw;
        }
        if name.contains(".mlp.")
            && !name.starts_with("mtp.")
            && layer_index(name).is_some_and(|i| opts.keep_int8_layers.contains(&i))
        {
            return Kind::Mls8;
        }
        // Only the fused gate/up of the routed experts goes to 3 bit: `down` and the shared expert stay at 4.
        if let Some((from, to)) = mls3
            && name.ends_with("mlp.experts.gate_up_proj")
            && layer_index(name).is_some_and(|i| (from..=to).contains(&i))
        {
            return Kind::Mls3;
        }
        Kind::Mls4
    }

    /// The MLS3 layer range the family's model document chose, or `None` if it keeps every expert at 4 bit.
    fn default_mls3(self, cfg: &serde_json::Value) -> Option<(usize, usize)> {
        match self {
            // docs/model-ling3-flash.md, configuration B: 3 bit for the routed gate/up except the last
            // MLS3_SKIP_LAST_LAYERS MoE layers, which dominate output quality and stay at 4.
            Family::Ling3 => {
                let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
                let from = u("first_k_dense_replace");
                let last = u("num_hidden_layers") + u("num_nextn_predict_layers");
                last.checked_sub(1 + MLS3_SKIP_LAST_LAYERS).map(|to| (from, to))
            }
            Family::Qwen35 | Family::Qwen4Exp | Family::Dflash => None,
        }
    }
}

/// Geometry of a MoE block, read from `config.json`; needed to fuse per-expert tensors.
#[derive(Clone, Copy, Debug)]
struct MoeGeom {
    experts: usize,
    inter: usize,
    hidden: usize,
}

/// Linears of a DFlash draft: the projection of the target's features plus the 6 layers. Norms and qk-norms stay bf16.
const DFLASH_MLS4_SUFFIXES: &[&str] = &[
    "fc.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

const QWEN_MLS4_SUFFIXES: &[&str] = &[
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.index_qk_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "mtp.fc.weight",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
];

/// Ling-3.0-flash linears that go to MLS4. `mlp.gate.weight` (the router) and `mlp.gate.expert_bias` do not end
/// with `mlp.gate_proj.weight` and stay raw, as MIGRATE_MODELS.md §2.4 requires; so do `attention.b_proj`, the three
/// `*_conv1d`, `A_log`, `dt_bias`, `attention.o_norm`, `kv_a_layernorm` and every other norm.
const LING3_MLS4_SUFFIXES: &[&str] = &[
    // KDA (35 layers)
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.f_proj.weight",
    "attention.g_proj.weight",
    "attention.o_proj.weight",
    // MLA (7 layers + MTP); `attention.q_proj.weight` is shared with KDA
    "attention.kv_a_proj_with_mqa.weight",
    "attention.kv_b_proj.weight",
    "attention.dense.weight",
    // Dense FFN (layers 0 and 1)
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    // MoE, fused by this converter into one 3-D tensor per layer
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    // MTP
    "eh_proj.weight",
];

/// qwen4_exp tensors that stay MLS8 in decode too.
const MLS8_SUFFIXES: &[&str] = &["ple.key_proj.weight", "ple.value_proj.weight"];

fn layer_index(name: &str) -> Option<usize> {
    let rest = name.split("layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
}

/// Rows `[row0, row0 + rows)` of an output tensor come from source tensor `src` (its first `rows` rows, `k` columns).
#[derive(Clone, Debug)]
struct Part {
    src: String,
    row0: usize,
    rows: usize,
    k: usize,
}

#[derive(Clone, Debug)]
struct Out {
    name: String,
    kind: Kind,
    /// Output shape; quantized tensors are [.., N, K] with K a multiple of 128.
    shape: Vec<usize>,
    parts: Vec<Part>,
    /// Store an MLS8 prefill copy next to an MLS4 tensor.
    prefill: bool,
}

/// Source rows converted per step (bf16 bytes): bounds the RAM used by large expert tensors.
const CHUNK_BYTES: usize = 256 << 20;

/// (rows, K) of a 2-D tensor or of a 3-D expert stack [E, N, K] seen as E * N rows.
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    let k = *shape.last().unwrap_or(&0);
    (shape.iter().rev().skip(1).product(), k)
}

/// Whether calibrated MLS4 tensors are available for `o`: under its own output name, or, for a composite the
/// converter assembles out of several sources, under the names of its parts.
fn is_calibrated(c: &SafeTensors, o: &Out) -> bool {
    c.info(&o.name).is_some() || o.parts.iter().any(|p| c.info(&p.src).is_some())
}

/// Shape and dtype of one calibrated tensor: `rows` x `qn` nibbles and `rows` x `sn` f16 scales.
fn check_calibrated(c: &SafeTensors, name: &str, rows: usize, qn: usize, sn: usize) -> Result<()> {
    let q = c.info(name).context("info")?;
    let s = c
        .info(&format!("{name}.scales"))
        .with_context(|| format!("{name}.scales missing"))?;
    ensure!(
        rows_cols(&q.shape) == (rows, qn) && rows_cols(&s.shape) == (rows, sn) && s.dtype == DType::F16,
        "{name}: calibrated tensors have the wrong shape"
    );
    Ok(())
}

/// Calibrated nibbles and scales of `o`, whose output is `rows` x `k`. A plain tensor is read by output name. A
/// composite is assembled from the calibrated parts at their row offsets, because `scripts/calibrate.py` only ever
/// sees the source tensors — `input_mix_weight_down` and `block_inject_weight` — while the padded `mix_down` row
/// layout is known here alone. Rows no part covers stay zero, which is what the writer pads them with anyway.
fn read_calibrated(c: &SafeTensors, o: &Out, rows: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let (qn, sn) = (k / 2, k / BLOCK);
    if c.info(&o.name).is_some() {
        check_calibrated(c, &o.name, rows, qn, sn)?;
        return Ok((c.read_all(&o.name)?, c.read_all(&format!("{}.scales", o.name))?));
    }
    let (mut q, mut s) = (vec![0u8; rows * qn], vec![0u8; rows * sn * 2]);
    for p in &o.parts {
        if c.info(&p.src).is_none() {
            continue;
        }
        check_calibrated(c, &p.src, p.rows, qn, sn)?;
        q[p.row0 * qn..(p.row0 + p.rows) * qn].copy_from_slice(&c.read_all(&p.src)?);
        let sc = c.read_all(&format!("{}.scales", p.src))?;
        s[p.row0 * sn * 2..(p.row0 + p.rows) * sn * 2].copy_from_slice(&sc);
    }
    Ok((q, s))
}

fn bf16_rows(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect()
}

fn quantize_mls4(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let g = k / BLOCK;
    let rows: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = Vec::with_capacity(k / 2);
            let mut s = Vec::with_capacity(g * 2);
            for grp in row.as_chunks::<BLOCK>().0 {
                let (nib, sc) = quant::quantize_group_int4(grp);
                q.extend_from_slice(&nib);
                s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
            }
            (q, s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k / 2);
    let mut s = Vec::with_capacity(n * g * 2);
    for (a, b) in rows {
        q.extend(a);
        s.extend(b);
    }
    (q, s)
}

fn quantize_mls3(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let g = k / BLOCK;
    let rows: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = Vec::with_capacity(k / 8 * 3);
            let mut s = Vec::with_capacity(g * 2);
            for grp in row.as_chunks::<BLOCK>().0 {
                let (bits, sc) = quant::quantize_group_int3(grp);
                q.extend_from_slice(&bits);
                s.extend_from_slice(&half::f16::from_f32(sc).to_le_bytes());
            }
            (q, s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k / 8 * 3);
    let mut s = Vec::with_capacity(n * g * 2);
    for (a, b) in rows {
        q.extend(a);
        s.extend(b);
    }
    (q, s)
}

fn quantize_mls8(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let signs = quant::hadamard_signs(HADAMARD_SEED);
    let rows: Vec<(Vec<u8>, f32)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut row = w[r * k..(r + 1) * k].to_vec();
            quant::rotate_row(&mut row, &signs);
            let mut q = vec![0i8; k];
            let s = quant::quantize_row_int8(&row, &mut q);
            (q.into_iter().map(|v| v as u8).collect(), s)
        })
        .collect();
    let mut q = Vec::with_capacity(n * k);
    let mut s = Vec::with_capacity(n * 4);
    for (a, b) in rows {
        q.extend(a);
        s.extend_from_slice(&b.to_le_bytes());
    }
    (q, s)
}

/// Unrotated int8 rows with a trailing f32 scale: [n, k + 4] bytes.
fn quantize_rows_i8(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * (k + 4)];
    out.par_chunks_mut(k + 4).enumerate().for_each(|(r, o)| {
        let mut q = vec![0i8; k];
        let s = quant::quantize_row_int8(&w[r * k..(r + 1) * k], &mut q);
        for (d, v) in o[..k].iter_mut().zip(q) {
            *d = v as u8;
        }
        o[k..].copy_from_slice(&s.to_le_bytes());
    });
    out
}

/// Hyper-connection low-rank width padded for the 128-wide rotation blocks of MLS8.
fn pad128(n: usize) -> usize {
    n.next_multiple_of(BLOCK)
}

fn shard_index(name: &str) -> Option<usize> {
    name.split("ngram_embedding.shard_")
        .nth(1)?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Output tensors for the sources, in source-name order (a composite tensor takes the place of its first source).
fn plan(
    st: &SafeTensors,
    names: &[String],
    opts: &Options,
    fam: Family,
    geom: Option<MoeGeom>,
    mls3: Option<(usize, usize)>,
) -> Result<Vec<Out>> {
    let mut outs = Vec::new();
    let mut seen_tables = std::collections::HashSet::new();
    let mut fused_layers = std::collections::HashSet::new();
    for name in names {
        let info = st.info(name).context("info")?;
        if let Some(g) = geom
            && let Some(prefix) = expert_layer(name)
        {
            // The first expert tensor of a layer emits both fused tensors; the others are skipped.
            if fused_layers.insert(prefix.clone()) {
                for gate_up in [true, false] {
                    let which = if gate_up { "gate_up_proj" } else { "down_proj" };
                    let name = format!("{prefix}.mlp.experts.{which}");
                    let shape = fused_shape(g, gate_up);
                    let kind = fam.classify(&name, &shape, opts, mls3);
                    ensure!(kind != Kind::Raw, "{name}: experts must be quantized");
                    let parts = expert_parts(g, &prefix, gate_up);
                    for p in &parts {
                        let pi = st
                            .info(&p.src)
                            .with_context(|| format!("{}: not in the checkpoint", p.src))?;
                        ensure!(
                            pi.dtype == DType::BF16 && rows_cols(&pi.shape) == (p.rows, p.k),
                            "{}: expected bf16 [{}, {}]",
                            p.src,
                            p.rows,
                            p.k
                        );
                    }
                    outs.push(Out {
                        name,
                        kind,
                        shape,
                        parts,
                        prefill: kind == Kind::Mls4 && opts.prefill_int8 && fam.prefill_copy(true),
                    });
                }
            }
            continue;
        }
        let one = |kind: Kind| -> Out {
            let (n, k) = rows_cols(&info.shape);
            Out {
                name: name.clone(),
                kind,
                shape: info.shape.clone(),
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows: n,
                    k,
                }],
                prefill: false,
            }
        };
        if let Some(hc) = name.strip_suffix("input_mix_weight_down.weight") {
            let (r, k) = rows_cols(&info.shape);
            let inject = format!("{hc}block_inject_weight.weight");
            let mut parts = vec![Part {
                src: name.clone(),
                row0: 0,
                rows: r,
                k,
            }];
            if let Some(ii) = st.info(&inject) {
                let (ri, ki) = rows_cols(&ii.shape);
                ensure!(ki == k, "{inject}: K {ki} != {k}");
                parts.push(Part {
                    src: inject,
                    row0: r,
                    rows: ri,
                    k,
                });
            }
            let rows = parts.iter().map(|p| p.row0 + p.rows).max().unwrap_or(0);
            outs.push(Out {
                name: format!("{hc}mix_down.weight"),
                kind: if opts.hc_mls4 { Kind::Mls4 } else { Kind::Mls8 },
                shape: vec![pad128(rows), k],
                parts,
                prefill: opts.hc_mls4 && opts.prefill_int8,
            });
            continue;
        }
        if name.ends_with("block_inject_weight.weight") {
            ensure!(
                st.info(&name.replace("block_inject_weight", "input_mix_weight_down"))
                    .is_some(),
                "{name} without its input_mix_weight_down"
            );
            continue;
        }
        if let Some(hc) = name.strip_suffix("input_mix_weight_up.weight") {
            let (n, k) = rows_cols(&info.shape);
            let down = st
                .info(&format!("{hc}input_mix_weight_down.weight"))
                .with_context(|| format!("{name} without its input_mix_weight_down"))?;
            ensure!(rows_cols(&down.shape).0 == k, "{name}: K {k} != down rows");
            outs.push(Out {
                name: format!("{hc}mix_up.weight"),
                kind: if opts.hc_mls4 { Kind::Mls4 } else { Kind::Mls8 },
                shape: vec![n, pad128(k)],
                parts: vec![Part {
                    src: name.clone(),
                    row0: 0,
                    rows: n,
                    k,
                }],
                prefill: opts.hc_mls4 && opts.prefill_int8,
            });
            continue;
        }
        if let Some(i) = name.find("ngram_embedding.shard_") {
            let base = &name[..i];
            if !seen_tables.insert(base.to_string()) {
                continue;
            }
            let mut shards: Vec<(usize, String)> = names
                .iter()
                .filter(|n| n.starts_with(base) && n.contains("ngram_embedding.shard_"))
                .map(|n| (shard_index(n).unwrap_or(usize::MAX), n.clone()))
                .collect();
            shards.sort();
            ensure!(
                shards.iter().enumerate().all(|(j, s)| s.0 == j),
                "{base}: n-gram shards are not numbered 0..{}",
                shards.len()
            );
            let mut parts = Vec::new();
            let mut row0 = 0;
            let mut dim = 0;
            for (_, s) in &shards {
                let si = st.info(s).context("info")?;
                let (r, k) = rows_cols(&si.shape);
                ensure!(dim == 0 || dim == k, "{s}: width {k} != {dim}");
                dim = k;
                parts.push(Part {
                    src: s.clone(),
                    row0,
                    rows: r,
                    k,
                });
                row0 += r;
            }
            outs.push(Out {
                name: format!("{base}ngram_table"),
                kind: Kind::RowI8,
                shape: vec![row0, dim + 4],
                parts,
                prefill: false,
            });
            continue;
        }
        let kind = fam.classify(name, &info.shape, opts, mls3);
        let mut o = one(kind);
        if kind != Kind::Raw {
            ensure!(
                info.dtype == DType::BF16,
                "{name}: only bf16 input is supported for now"
            );
            ensure!(
                matches!(info.shape.len(), 2 | 3),
                "{name}: quantized tensors must be 2-D or 3-D (experts)"
            );
            let k = *info.shape.last().unwrap_or(&0);
            if k % BLOCK != 0 {
                bail!("{name}: K={k} not a multiple of {BLOCK}");
            }
            let routed = name.contains(".mlp.experts.");
            let expert = routed || name.contains(".mlp.shared_expert");
            o.prefill = kind == Kind::Mls4
                && opts.prefill_int8
                && fam.prefill_copy(expert)
                && !(routed && opts.no_prefill_int8_experts);
        }
        outs.push(o);
    }
    Ok(outs)
}

pub fn convert(model_dir: &Path, out: &Path, opts: &Options, log: &dyn Fn(&str)) -> Result<()> {
    let st = SafeTensors::open_dir(model_dir)?;
    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    let read = |f: &str| std::fs::read_to_string(model_dir.join(f)).with_context(|| f.to_string());
    let config = read("config.json")?;
    let cfg: serde_json::Value = serde_json::from_str(&config)?;
    let fam = Family::from_config(&cfg)?;
    // Ling-3.0-flash stores the experts one tensor at a time: they are fused, one 3-D tensor per layer and projection.
    // Qwen stores them stacked, except the MTP layer of some checkpoints (Ornith): the geometry is only used for
    // names that match the per-expert pattern, and its keys live under `text_config` there.
    let geom = match fam {
        Family::Ling3 => {
            let u = |k: &str| -> Result<usize> {
                Ok(cfg[k].as_u64().with_context(|| format!("config.json has no {k}"))? as usize)
            };
            Some(MoeGeom {
                experts: u("num_experts")?,
                inter: u("moe_intermediate_size")?,
                hidden: u("hidden_size")?,
            })
        }
        // A DFlash draft has no MoE.
        Family::Dflash => None,
        Family::Qwen35 | Family::Qwen4Exp => {
            let t = if cfg["text_config"].is_object() {
                &cfg["text_config"]
            } else {
                &cfg
            };
            let u = |k: &str| t[k].as_u64().map(|v| v as usize);
            match (u("num_experts"), u("moe_intermediate_size"), u("hidden_size")) {
                (Some(experts), Some(inter), Some(hidden)) => Some(MoeGeom { experts, inter, hidden }),
                _ => None,
            }
        }
    };
    if opts.prefill_int8 && !fam.prefill_copy(true) {
        log("no MLS8 prefill copy of the experts: it does not fit the memory budget of this family");
    }
    let mut w = GgufWriter::new();
    w.meta(
        "general.architecture",
        Value::Str(format!("{ARCH_PREFIX}{}", fam.arch())),
    );
    w.meta(
        "general.name",
        Value::Str(model_dir.file_name().and_then(|s| s.to_str()).unwrap_or("model").into()),
    );
    w.meta("openmalleus.layout_version", Value::U32(LAYOUT_VERSION));
    w.meta("openmalleus.hadamard_seed", Value::U64(HADAMARD_SEED));
    w.meta("openmalleus.hf_config", Value::Str(config));
    if let Ok(g) = read("generation_config.json") {
        w.meta("openmalleus.generation_config", Value::Str(g));
    }
    if let Ok(p) = read("preprocessor_config.json") {
        w.meta("openmalleus.preprocessor_config", Value::Str(p));
    }
    // A speculative draft has no tokenizer of its own (it borrows the target's), so its absence is expected there
    // and fatal everywhere else: a model file that cannot tokenize is not a model.
    match (read("tokenizer.json"), read("chat_template.jinja")) {
        (Ok(tok), Ok(tpl)) => {
            w.meta("tokenizer.huggingface.json", Value::Str(tok));
            w.meta("tokenizer.chat_template", Value::Str(tpl));
        }
        (Err(e), _) if fam == Family::Dflash => {
            log(&format!("no tokenizer ({e}): a draft uses the target's, as expected"));
        }
        (Err(e), _) => return Err(e),
        (Ok(_), Err(e)) => return Err(e),
    }

    let mls3 = match opts.mls3_layers {
        Mls3Layers::Off => None,
        Mls3Layers::Range(from, to) => Some((from, to)),
        Mls3Layers::Family => fam.default_mls3(&cfg),
    };
    match mls3 {
        Some((from, to)) => log(&format!("MoE gate/up at MLS3 for layers {from}..={to}")),
        None => log("every expert at MLS4"),
    }

    let plan = plan(&st, &names, opts, fam, geom, mls3)?;
    let calib = opts.mls4_from.as_deref().map(SafeTensors::open_dir).transpose()?;
    let mut calibrated = 0;
    let mut bytes = 0usize;
    let mut tensors = 0usize;
    let mut add = |name: &str, dtype: DType, shape: &[usize]| {
        bytes += dtype.size_of_elems(shape.iter().product());
        tensors += 1;
        w.add_tensor(name, dtype, shape);
    };
    for o in &plan {
        let (lead, k) = o.shape.split_at(o.shape.len() - 1);
        let k = k[0];
        let dims = |last: &[usize]| [lead, last].concat();
        match o.kind {
            Kind::Raw => {
                let info = st.info(&o.name).context("info")?;
                add(&o.name, info.dtype, &info.shape);
            }
            Kind::Mls3 => {
                add(&o.name, DType::Mls3, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, &dims(&[k / BLOCK]));
            }
            Kind::Mls4 => {
                add(&o.name, DType::Mls4, &o.shape);
                add(&format!("{}.scales", o.name), DType::F16, &dims(&[k / BLOCK]));
                if o.prefill {
                    add(&format!("{}.prefill", o.name), DType::Mls8, &o.shape);
                    add(&format!("{}.prefill.scales", o.name), DType::F32, lead);
                }
            }
            Kind::Mls8 => {
                add(&o.name, DType::Mls8, &o.shape);
                add(&format!("{}.scales", o.name), DType::F32, lead);
            }
            Kind::RowI8 => add(&o.name, DType::I8, &o.shape),
        }
    }
    log(&format!(
        "{} plan entries -> {tensors} output tensors, {:.2} GB ({})",
        plan.len(),
        bytes as f64 / 1e9,
        if opts.prefill_int8 {
            "MLS4 for decode + MLS8 copies for prefill"
        } else {
            "MLS4 only: the prefill reads the same 4-bit weights"
        }
    ));
    // The name is the only thing a directory listing shows, so it has to say whether the file carries the second
    // copy. A warning and not an error: the file is correct either way, and renaming is the caller's business.
    let name = out.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if opts.prefill_int8 && !name.contains("mls8") {
        log(&format!(
            "warning: {name} carries the MLS8 prefill copies: name it `<model>-mls4-mls8.malleus` so a listing \
             does not hide that it is twice the size"
        ));
    } else if !opts.prefill_int8 && name.contains("mls8") {
        log(&format!(
            "warning: {name} says mls8 but carries a single MLS4 copy: name it `<model>-mls4.malleus`"
        ));
    }
    if let Some(c) = &calib {
        // Validate every calibrated tensor up front: a shape mismatch found after 100 GB of writing is useless.
        for o in &plan {
            if o.kind != Kind::Mls4 || !is_calibrated(c, o) {
                continue;
            }
            let (n, k) = rows_cols(&o.shape);
            let (qn, sn) = (k / 2, k / BLOCK);
            if c.info(&o.name).is_some() {
                check_calibrated(c, &o.name, n, qn, sn)?;
            } else {
                for p in &o.parts {
                    if c.info(&p.src).is_some() {
                        check_calibrated(c, &p.src, p.rows, qn, sn)?;
                    }
                }
            }
            calibrated += 1;
        }
        let total = plan.iter().filter(|p| p.kind == Kind::Mls4).count();
        log(&format!(
            "calibrated MLS4: {calibrated}/{total} tensors, the rest round-to-nearest"
        ));
        w.meta("openmalleus.mls4_calibrated", Value::U32(calibrated));
    }
    let mut sink = w.begin(out)?;
    let total = plan.len();
    let t0 = std::time::Instant::now();
    for (i, o) in plan.iter().enumerate() {
        if o.kind == Kind::Raw {
            let info = st.info(&o.name).context("info")?;
            let mut buf = vec![0u8; 64 << 20];
            let mut off = 0;
            while off < info.size {
                let n = buf.len().min(info.size - off);
                st.read(&o.name, off, &mut buf[..n])?;
                sink.write(&buf[..n])?;
                off += n;
            }
            continue;
        }
        let (n, k) = rows_cols(&o.shape);
        if o.kind == Kind::RowI8 {
            // Written part by part: the table is ~50 GB.
            let kk = k - 4;
            for p in &o.parts {
                let step = (CHUNK_BYTES / (2 * kk)).max(1);
                for r0 in (0..p.rows).step_by(step) {
                    let rows = step.min(p.rows - r0);
                    let mut raw = vec![0u8; rows * kk * 2];
                    st.read(&p.src, r0 * kk * 2, &mut raw)?;
                    sink.write(&quantize_rows_i8(&bf16_rows(&raw), rows, kk))?;
                }
            }
            log(&format!(
                "[{i}/{total}] {} ({n} rows, {:.0}s)",
                o.name,
                t0.elapsed().as_secs_f64()
            ));
            continue;
        }
        // Calibration files hold MLS4 tensors only: an MLS3 tensor is always round-to-nearest.
        let cal = calib.as_ref().filter(|c| o.kind == Kind::Mls4 && is_calibrated(c, o));
        let (mut qd, mut sd, mut q8, mut s8) = (vec![], vec![], vec![], vec![]);
        let step = (CHUNK_BYTES / (2 * k)).max(1);
        for r0 in (0..n).step_by(step) {
            let rows = step.min(n - r0);
            let mut wf = vec![0f32; rows * k];
            for p in &o.parts {
                let (a, b) = (p.row0.max(r0), (p.row0 + p.rows).min(r0 + rows));
                if a >= b {
                    continue;
                }
                let mut raw = vec![0u8; (b - a) * p.k * 2];
                st.read(&p.src, (a - p.row0) * p.k * 2, &mut raw)?;
                let src = bf16_rows(&raw);
                for r in 0..b - a {
                    let d = (a - r0 + r) * k;
                    wf[d..d + p.k].copy_from_slice(&src[r * p.k..(r + 1) * p.k]);
                }
            }
            if matches!(o.kind, Kind::Mls3 | Kind::Mls4) && cal.is_none() {
                let (q, s) = match o.kind {
                    Kind::Mls3 => quantize_mls3(&wf, rows, k),
                    _ => quantize_mls4(&wf, rows, k),
                };
                qd.extend(q);
                sd.extend(s);
            }
            if o.kind == Kind::Mls8 || o.prefill {
                let (q, s) = quantize_mls8(&wf, rows, k);
                q8.extend(q);
                s8.extend(s);
            }
        }
        if matches!(o.kind, Kind::Mls3 | Kind::Mls4) {
            if let Some(c) = cal {
                (qd, sd) = read_calibrated(c, o, n, k)?;
            }
            sink.write(&qd)?;
            sink.write(&sd)?;
        }
        if !q8.is_empty() {
            sink.write(&q8)?;
            sink.write(&s8)?;
        }
        if i % 25 == 0 {
            log(&format!(
                "[{i}/{total}] {} ({:.0}s)",
                o.name,
                t0.elapsed().as_secs_f64()
            ));
        }
    }
    sink.finish()?;
    Ok(())
}

/// Layer index of a per-expert tensor (`model.layers.5.mlp.experts.17.gate_proj.weight` → 5), or `None` for the
/// already-fused 3-D tensors of the other families (`model.layers.5.mlp.experts.gate_up_proj`).
/// Prefix of a layer whose routed experts come one tensor per expert: `model.layers.N` (Ling-3.0-flash) or
/// `mtp.layers.N` (the MTP layer of some Qwen checkpoints, e.g. Ornith, where the main stack is fused but the MTP is
/// not). The prefix names the fused output too, which is what the loader reads.
fn expert_layer(name: &str) -> Option<String> {
    let parse = |prefix: &str| -> Option<String> {
        let rest = name.strip_prefix(prefix)?;
        let (layer, tail) = rest.split_once('.')?;
        tail.strip_prefix("mlp.experts.")?
            .split('.')
            .next()?
            .parse::<usize>()
            .ok()?;
        layer.parse::<usize>().ok()?;
        Some(format!("{prefix}{layer}"))
    };
    parse("model.layers.").or_else(|| parse("mtp.layers."))
}

/// Shape of the tensor that fuses the experts of one layer: `[E, rows_per_expert, K]`.
fn fused_shape(g: MoeGeom, gate_up: bool) -> Vec<usize> {
    let (rows, k) = if gate_up {
        (2 * g.inter, g.hidden)
    } else {
        (g.hidden, g.inter)
    };
    vec![g.experts, rows, k]
}

/// Sources of a fused expert tensor in row order: per expert, gate then up (`gate_up`) or down.
fn expert_parts(g: MoeGeom, prefix: &str, gate_up: bool) -> Vec<Part> {
    let mut parts = Vec::new();
    for e in 0..g.experts {
        let p = format!("{prefix}.mlp.experts.{e}");
        if gate_up {
            for (j, w) in ["gate_proj", "up_proj"].iter().enumerate() {
                parts.push(Part {
                    src: format!("{p}.{w}.weight"),
                    row0: (2 * e + j) * g.inter,
                    rows: g.inter,
                    k: g.hidden,
                });
            }
        } else {
            parts.push(Part {
                src: format!("{p}.down_proj.weight"),
                row0: e * g.hidden,
                rows: g.hidden,
                k: g.inter,
            });
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Options {
        Options {
            prefill_int8: false,
            no_prefill_int8_experts: false,
            keep_int8_layers: vec![],
            mls4_from: None,
            mls3_layers: Mls3Layers::Off,
            hc_mls4: false,
        }
    }

    #[test]
    fn family_from_model_type() {
        let f = |t: &str| Family::from_model_type(t).unwrap();
        assert_eq!(f("qwen3_5"), Family::Qwen35);
        assert_eq!(f("qwen3_5_moe"), Family::Qwen35);
        assert_eq!(f("qwen4_exp"), Family::Qwen4Exp);
        assert_eq!(f("bailing_hybrid"), Family::Ling3);
        assert_eq!(Family::Ling3.arch(), "ling3");
        assert_eq!(Family::Qwen4Exp.arch(), "qwen4exp");
        assert!(Family::Qwen35.prefill_copy(true) && !Family::Ling3.prefill_copy(true));
        assert!(Family::Qwen4Exp.prefill_copy(false) && !Family::Qwen4Exp.prefill_copy(true));
        assert!(Family::from_model_type("llama").is_err());
    }

    /// Ling-3.0-flash: what stays raw is exactly what MIGRATE_MODELS.md §2.4 forbids to quantize, plus the MLA gate.
    #[test]
    fn ling3_classification() {
        let o = opts();
        let c = |name: &str, shape: &[usize]| Family::Ling3.classify(name, shape, &o, None);
        let l2 = "model.layers.2.";
        let l5 = "model.layers.5.";
        // KDA (layer 2)
        for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj"] {
            assert_eq!(
                c(&format!("{l2}attention.{p}.weight"), &[4096, 2560]),
                Kind::Mls4,
                "{p}"
            );
        }
        assert_eq!(c(&format!("{l2}attention.b_proj.weight"), &[32, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.q_conv1d.weight"), &[4096, 1, 4]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.A_log"), &[32]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.dt_bias"), &[4096]), Kind::Raw);
        assert_eq!(c(&format!("{l2}attention.o_norm.weight"), &[128]), Kind::Raw);
        // MLA (layer 5): the head gate is [32, 2560] and stays raw, the KDA one is [4096, 2560] and does not
        assert_eq!(c(&format!("{l5}attention.q_proj.weight"), &[6144, 2560]), Kind::Mls4);
        assert_eq!(
            c(&format!("{l5}attention.kv_a_proj_with_mqa.weight"), &[576, 2560]),
            Kind::Mls4
        );
        assert_eq!(c(&format!("{l5}attention.kv_b_proj.weight"), &[8192, 512]), Kind::Mls4);
        assert_eq!(c(&format!("{l5}attention.dense.weight"), &[2560, 4096]), Kind::Mls4);
        assert_eq!(c(&format!("{l5}attention.g_proj.weight"), &[32, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l5}attention.kv_a_layernorm.weight"), &[512]), Kind::Raw);
        // MoE
        assert_eq!(c(&format!("{l2}mlp.gate.weight"), &[512, 2560]), Kind::Raw);
        assert_eq!(c(&format!("{l2}mlp.gate.expert_bias"), &[512]), Kind::Raw);
        assert_eq!(
            c(&format!("{l2}mlp.experts.gate_up_proj"), &[512, 1536, 2560]),
            Kind::Mls4
        );
        assert_eq!(c(&format!("{l2}mlp.experts.down_proj"), &[512, 2560, 768]), Kind::Mls4);
        assert_eq!(
            c(&format!("{l2}mlp.shared_experts.gate_proj.weight"), &[768, 2560]),
            Kind::Mls4
        );
        assert_eq!(
            c(&format!("{l2}mlp.shared_experts.down_proj.weight"), &[2560, 768]),
            Kind::Mls4
        );
        // Dense FFN (layers 0 and 1)
        assert_eq!(c("model.layers.0.mlp.gate_proj.weight", &[6144, 2560]), Kind::Mls4);
        assert_eq!(c("model.layers.0.mlp.down_proj.weight", &[2560, 6144]), Kind::Mls4);
        // MTP and globals
        assert_eq!(c("model.layers.42.eh_proj.weight", &[2560, 5120]), Kind::Mls4);
        assert_eq!(c("model.layers.42.enorm.weight", &[2560]), Kind::Raw);
        assert_eq!(c("model.norm.weight", &[2560]), Kind::Raw);
        assert_eq!(c("model.word_embeddings.weight", &[157184, 2560]), Kind::Raw);
        assert_eq!(c("lm_head.weight", &[157184, 2560]), Kind::Mls8);
    }

    #[test]
    fn qwen35_classification_is_unchanged() {
        let o = opts();
        let c = |name: &str| Family::Qwen35.classify(name, &[4096, 2048], &o, None);
        assert_eq!(c("model.language_model.layers.0.self_attn.q_proj.weight"), Kind::Mls4);
        assert_eq!(c("model.language_model.layers.3.mlp.experts.gate_up_proj"), Kind::Mls4);
        assert_eq!(
            c("model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"),
            Kind::Mls4
        );
        assert_eq!(c("model.language_model.layers.0.linear_attn.A_log"), Kind::Raw);
        assert_eq!(c("model.language_model.layers.0.mlp.gate.weight"), Kind::Raw);
        assert_eq!(c("lm_head.weight"), Kind::Mls8);
        assert_eq!(c("model.visual.patch_embed.proj.weight"), Kind::Raw);
        let keep = Options {
            keep_int8_layers: vec![3],
            ..opts()
        };
        assert_eq!(
            Family::Qwen35.classify(
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                &[1, 1, 128],
                &keep,
                None
            ),
            Kind::Mls8
        );
    }

    /// Configuration B of docs/model-ling3-flash.md: 3 bit for the routed gate/up except the last 8 MoE layers.
    #[test]
    fn mls3_layer_policy() {
        let cfg = serde_json::json!({
            "first_k_dense_replace": 2,
            "num_hidden_layers": 42,
            "num_nextn_predict_layers": 1,
        });
        assert_eq!(Family::Ling3.default_mls3(&cfg), Some((2, 34)));
        assert_eq!(Family::Qwen35.default_mls3(&cfg), None);
        assert_eq!(Family::Qwen4Exp.default_mls3(&cfg), None);
        // A model with fewer layers than the promoted tail keeps everything at 4 bit instead of underflowing.
        let small = serde_json::json!({"first_k_dense_replace": 2, "num_hidden_layers": 6});
        assert_eq!(Family::Ling3.default_mls3(&small), None);

        let o = opts();
        let m3 = Some((2, 34));
        let c = |name: &str, shape: &[usize]| Family::Ling3.classify(name, shape, &o, m3);
        let gu = &[512, 1536, 2560][..];
        let dn = &[512, 2560, 768][..];
        assert_eq!(c("model.layers.2.mlp.experts.gate_up_proj", gu), Kind::Mls3);
        assert_eq!(c("model.layers.34.mlp.experts.gate_up_proj", gu), Kind::Mls3);
        // The promoted tail, the MTP layer, `down` and the shared expert stay at 4 bit.
        assert_eq!(c("model.layers.35.mlp.experts.gate_up_proj", gu), Kind::Mls4);
        assert_eq!(c("model.layers.42.mlp.experts.gate_up_proj", gu), Kind::Mls4);
        assert_eq!(c("model.layers.2.mlp.experts.down_proj", dn), Kind::Mls4);
        assert_eq!(
            c("model.layers.2.mlp.shared_experts.gate_proj.weight", &[768, 2560]),
            Kind::Mls4
        );
        assert_eq!(c("model.layers.2.attention.q_proj.weight", &[4096, 2560]), Kind::Mls4);
        assert_eq!(c("model.layers.0.mlp.gate_proj.weight", &[6144, 2560]), Kind::Mls4);
        // An explicit empty range turns MLS3 off everywhere.
        let none = |name: &str| Family::Ling3.classify(name, gu, &o, Some((1, 0)));
        assert_eq!(none("model.layers.2.mlp.experts.gate_up_proj"), Kind::Mls4);

        // 48 bytes of codes + 2 of scale per group of 128 = 3.125 bpw.
        let n = 512 * 1536 * 2560;
        assert_eq!(DType::Mls3.size_of_elems(n), n / 8 * 3);
        assert_eq!(
            DType::Mls3.size_of_elems(n) + DType::F16.size_of_elems(n / BLOCK),
            n * 25 / 64
        );
    }

    #[test]
    fn expert_tensor_detection() {
        let prefix = |s: &str| Some(s.to_string());
        assert_eq!(
            expert_layer("model.layers.41.mlp.experts.511.down_proj.weight"),
            prefix("model.layers.41")
        );
        assert_eq!(
            expert_layer("model.layers.0.mlp.experts.0.gate_proj.weight"),
            prefix("model.layers.0")
        );
        // Ornith stores the MTP experts one tensor at a time: same fusion, different prefix.
        assert_eq!(
            expert_layer("mtp.layers.0.mlp.experts.255.up_proj.weight"),
            prefix("mtp.layers.0")
        );
        assert_eq!(expert_layer("model.layers.5.mlp.experts.gate_up_proj"), None);
        assert_eq!(expert_layer("model.layers.5.mlp.shared_experts.gate_proj.weight"), None);
        assert_eq!(expert_layer("model.layers.5.mlp.gate.weight"), None);
        assert_eq!(
            expert_layer("model.language_model.layers.5.mlp.experts.3.gate_proj.weight"),
            None
        );
        let g = MoeGeom {
            experts: 512,
            inter: 768,
            hidden: 2560,
        };
        assert_eq!(fused_shape(g, true), vec![512, 1536, 2560]);
        assert_eq!(fused_shape(g, false), vec![512, 2560, 768]);
        let gu = expert_parts(g, "model.layers.2", true);
        assert_eq!(gu.len(), 1024);
        assert_eq!(gu.iter().map(|p| p.rows).sum::<usize>(), 512 * 1536);
        assert_eq!(gu[0].src, "model.layers.2.mlp.experts.0.gate_proj.weight");
        assert_eq!(
            (gu[1].src.as_str(), gu[1].row0),
            ("model.layers.2.mlp.experts.0.up_proj.weight", 768)
        );
        assert_eq!(
            (gu[2].src.as_str(), gu[2].row0),
            ("model.layers.2.mlp.experts.1.gate_proj.weight", 1536)
        );
        let d = expert_parts(g, "model.layers.2", false);
        assert_eq!(d.len(), 512);
        assert_eq!((d[1].row0, d[1].rows, d[1].k), (2560, 2560, 768));
    }
}
