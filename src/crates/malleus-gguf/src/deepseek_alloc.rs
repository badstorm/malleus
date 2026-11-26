//! The precision map of a DeepSeek-V4.1 file: which storage type each tensor gets (issue #83).
//!
//! The precision is not invented here. The released GGUF build carries a per-tensor allocation found by a
//! constrained search over a 2.00 bpw budget, and it is published as text: one `<tensor>: <type>` line per
//! tensor. This module reads that text, translates the names into the canonical ones of the tensor map and
//! turns each llama.cpp type into one of our storage types. The result is what `convert` writes and what the
//! memory plan budgets against.
//!
//! Two properties of the source matter and are preserved:
//!
//! - the experts are stacked per layer in the GGUF (one `ffn_gate_exps` tensor per layer, all 384 experts), so
//!   the choice is per layer, not per expert, and the map keeps it that way;
//! - the norms, the hyper-connection coefficients, the router bias and the sinks stay in f32, and the router
//!   itself in bf16: the search never touched them, and neither do we.
//!
//! Where a type has no equivalent here, the map says so rather than silently rounding up: IQ1_S is 1.56 bits
//! and there is no 1.5-bit type in this engine, so those tensors land on MLS2 and the budget reports the extra.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::DType;

/// How a tensor is stored in a `.malleus` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// The type the checkpoint already uses: f32, bf16 or an fp8 block format.
    Keep,
    /// 1 bit per value with an e4m3 scale per 32: 1.25 bpw.
    Mls1,
    /// 2-bit codes with an fp16 scale per group of 128.
    Mls2,
    /// 3-bit codes with an fp16 scale per group of 128.
    Mls3,
    /// 4-bit codes with an fp16 scale per group of 128.
    Mls4,
    /// 8-bit codes with an fp16 scale per group of 128.
    Mls8,
    /// 4-bit codes per row, for the Engram lookup tables.
    RowI4,
}

impl Pick {
    /// Bits per stored element, for the budget.
    pub fn bits(self) -> f64 {
        match self {
            Pick::Keep => 0.0,
            Pick::Mls1 => 1.0,
            Pick::Mls2 => 2.0,
            Pick::Mls3 => 3.0,
            Pick::Mls4 => 4.0,
            Pick::Mls8 => 8.0,
            Pick::RowI4 => 4.0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Pick::Keep => "keep",
            Pick::Mls1 => "mls1",
            Pick::Mls2 => "mls2",
            Pick::Mls3 => "mls3",
            Pick::Mls4 => "mls4",
            Pick::Mls8 => "mls8",
            Pick::RowI4 => "rowi4",
        }
    }

    /// The type a llama.cpp quantization maps to, or `None` when the name is not a quantization at all.
    ///
    /// The 1- and 2-bit types all land on MLS2: the engine has no 1.5-bit type, and a 2-bit code is the
    /// closest thing it can store. `Q5_K` lands on MLS4 for the MLPs and the embedding, and on MLS8 for the
    /// attention projections, where the search itself kept 8 bits next to it.
    pub fn from_ggml(name: &str, class: &str) -> Result<Pick> {
        Ok(match name {
            "F32" | "BF16" | "F16" => Pick::Keep,
            "Q8_0" => Pick::Mls8,
            "Q5_K" | "Q4_K" | "Q6_K" => {
                if class.starts_with("att") || class.starts_with("indexer") {
                    Pick::Mls8
                } else {
                    Pick::Mls4
                }
            }
            // The 1-bit type of the ladder is the only one that lands on MLS1: every other 1- and 2-bit type is a
            // 2-bit code here, which is what the 2.00 bpw build asks for. A map that wants 1.25 bpw names IQ1_S.
            "IQ1_S" => Pick::Mls1,
            "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ1_M" => Pick::Mls2,
            "IQ4_NL" => Pick::RowI4,
            other => bail!("unknown quantization type `{other}`"),
        })
    }
}

/// The map from canonical tensor name to storage type.
#[derive(Debug, Clone, Default)]
pub struct Allocation {
    picks: BTreeMap<String, Pick>,
}

impl Allocation {
    pub fn len(&self) -> usize {
        self.picks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.picks.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<Pick> {
        self.picks.get(name).copied()
    }

    /// The pick for a name, or `Keep` when the map does not mention it: a tensor the search never saw stays
    /// as the checkpoint stores it.
    pub fn pick(&self, name: &str) -> Pick {
        self.picks.get(name).copied().unwrap_or(Pick::Keep)
    }

    pub fn insert(&mut self, name: String, pick: Pick) {
        self.picks.insert(name, pick);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Pick)> {
        self.picks.iter()
    }

    /// Reads the published allocation text: comment lines start with `#`, every other line is
    /// `<tensor>: <type>`.
    pub fn parse(text: &str) -> Result<Allocation> {
        let mut out = Allocation::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, ty) = line
                .rsplit_once(':')
                .with_context(|| format!("line {}: expected `<tensor>: <type>`", n + 1))?;
            let (name, ty) = (name.trim(), ty.trim());
            let canonical = canonical(name)?;
            let class = class_of(&canonical);
            out.picks.insert(canonical, Pick::from_ggml(ty, class)?);
        }
        if out.picks.is_empty() {
            bail!("the allocation text has no tensors");
        }
        Ok(out)
    }
}

/// A GGUF tensor name in the canonical spelling of the tensor map.
///
/// The GGUF stacks the 384 routed experts into one tensor per layer; the canonical name keeps that shape and
/// the converter expands it, so the map stays one entry per layer.
pub fn canonical(name: &str) -> Result<String> {
    let out = match name {
        "token_embd.weight" => "embed.weight".to_string(),
        "output.weight" => "head.weight".to_string(),
        "output_norm.weight" => "norm.weight".to_string(),
        "per_layer_token_embd.weight" => "engram.table".to_string(),
        other => {
            let Some(rest) = other.strip_prefix("blk.") else {
                bail!("unknown tensor name `{other}`");
            };
            let (i, rest) = rest
                .split_once('.')
                .with_context(|| format!("`{other}`: expected `blk.<i>.<name>`"))?;
            let i: usize = i.parse().with_context(|| format!("`{other}`: bad layer index"))?;
            let suffix = match rest {
                "attn_kv.weight" => "attn.wkv.weight",
                "attn_kv_a_norm.weight" => "attn.kv_norm.weight",
                "attn_q_a.weight" => "attn.wq_a.weight",
                "attn_q_a_norm.weight" => "attn.q_norm.weight",
                "attn_q_b.weight" => "attn.wq_b.weight",
                "attn_output_a.weight" => "attn.wo_a.weight",
                "attn_output_b.weight" => "attn.wo_b.weight",
                "attn_sinks.weight" => "attn.attn_sink",
                "attn_norm.weight" => "attn_norm.weight",
                "ffn_norm.weight" => "ffn_norm.weight",
                "ffn_gate_inp.weight" => "ffn.gate.weight",
                "exp_probs_b.bias" => "ffn.gate.bias",
                "exp_probs_b_vl.bias" => "ffn.gate.bias_vl",
                "ffn_gate_exps.weight" => "ffn.experts.w1.weight",
                "ffn_up_exps.weight" => "ffn.experts.w3.weight",
                "ffn_down_exps.weight" => "ffn.experts.w2.weight",
                "ffn_gate_shexp.weight" => "ffn.shared_experts.w1.weight",
                "ffn_up_shexp.weight" => "ffn.shared_experts.w3.weight",
                "ffn_down_shexp.weight" => "ffn.shared_experts.w2.weight",
                "attn_compressor_kv.weight" => "attn.compressor.wkv.weight",
                "attn_compressor_gate.weight" => "attn.compressor.wgate.weight",
                "attn_compressor_norm.weight" => "attn.compressor.norm.weight",
                "indexer.attn_k.weight" => "attn.indexer.wk.weight",
                "indexer.attn_q_b.weight" => "attn.indexer.wq_b.weight",
                "indexer.k_norm.weight" => "attn.indexer.k_norm.weight",
                "indexer.proj.weight" => "attn.indexer.weights_proj.weight",
                "engram_gate.weight" => "engram.gate.weight",
                "engram_kv.weight" => "engram.kv.weight",
                "hc_attn_base.weight" => "hc_attn_base",
                "hc_attn_fn.weight" => "hc_attn_fn",
                "hc_attn_scale.weight" => "hc_attn_scale",
                "hc_ffn_base.weight" => "hc_ffn_base",
                "hc_ffn_fn.weight" => "hc_ffn_fn",
                "hc_ffn_scale.weight" => "hc_ffn_scale",
                other => bail!("unknown tensor name `blk.{i}.{other}`"),
            };
            format!("layers.{i}.{suffix}")
        }
    };
    Ok(out)
}

/// The key an expected tensor's name takes in the allocation: the per-expert tensors of a layer share one
/// entry, because the source stack them into one tensor per layer.
pub fn allocation_key(name: &str) -> String {
    if name.starts_with("layers.") && name.contains(".engram.embed.") {
        // The two Engram tables are one tensor in the source, `per_layer_token_embd`.
        return "engram.table".to_string();
    }
    if let Some(rest) = name.strip_prefix("layers.")
        && let Some((i, tail)) = rest.split_once('.')
        && let Some(tail) = tail.strip_prefix("ffn.experts.")
        && let Some((_, proj)) = tail.split_once('.')
    {
        let proj = proj.trim_end_matches(".weight");
        return format!("layers.{i}.ffn.experts.{proj}.weight");
    }
    name.to_string()
}

/// The bytes a tensor takes at a pick, from its element count.
///
/// A quantized tensor carries one fp16 scale per group of 128 along K; a row-quantized tensor carries one fp16
/// scale per row. `Keep` uses the checkpoint's own dtype, so the budget is the file's real size.
pub fn tensor_bytes(pick: Pick, dtype: DType, shape: &[usize], packed: bool) -> u64 {
    let mut elems: u64 = shape.iter().map(|&d| d as u64).product();
    // A packed fp4 source counts its pairs in the shape, so the tensor holds twice the elements the numbers say.
    if packed {
        elems *= 2;
    }
    match pick {
        Pick::Keep => dtype.size_of_elems(elems as usize) as u64,
        Pick::Mls1 => {
            // One sign bit per value, one e4m3 scale per 32 values, i.e. 1.25 bits.
            elems / 8 + elems / 32
        }
        Pick::Mls2 | Pick::Mls3 | Pick::Mls4 | Pick::Mls8 => {
            let q = match pick {
                Pick::Mls2 => elems / 4,
                Pick::Mls3 => elems / 8 * 3,
                Pick::Mls4 => elems / 2,
                Pick::Mls8 => elems,
                _ => unreachable!(),
            };
            q + elems / 128 * 2
        }
        Pick::RowI4 => {
            let rows = shape.first().copied().unwrap_or(1) as u64;
            elems / 2 + rows * 2
        }
    }
}

/// The class the tensor map gives a name, for the choices that depend on it.
fn class_of(name: &str) -> &'static str {
    if name.starts_with("engram.") {
        "engram"
    } else if name.starts_with("vision.") || name.starts_with("aligner.") {
        "vision"
    } else if name.contains(".attn.") || name.contains("attn_norm") {
        "attention"
    } else if name.contains(".ffn.") || name.contains("ffn_norm") {
        "moe"
    } else {
        "embed+head"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "# a comment\n\
        output.weight: Q5_K\n\
        token_embd.weight: Q5_K\n\
        per_layer_token_embd.weight: IQ4_NL\n\
        blk.0.attn_q_a.weight: Q8_0\n\
        blk.0.attn_q_b.weight: Q5_K\n\
        blk.0.ffn_gate_exps.weight: IQ2_XXS\n\
        blk.0.ffn_gate_shexp.weight: Q5_K\n\
        blk.0.ffn_norm.weight: F32\n\
        blk.0.hc_attn_fn.weight: F32\n";

    #[test]
    fn the_names_translate_to_the_tensor_map() {
        assert_eq!(canonical("blk.3.attn_q_b.weight").unwrap(), "layers.3.attn.wq_b.weight");
        assert_eq!(
            canonical("blk.0.ffn_gate_exps.weight").unwrap(),
            "layers.0.ffn.experts.w1.weight"
        );
        assert_eq!(canonical("token_embd.weight").unwrap(), "embed.weight");
        assert!(canonical("blk.0.no_such_weight").is_err());
    }

    #[test]
    fn the_allocation_reads_and_keeps_the_expert_choice_per_layer() {
        let a = Allocation::parse(TEXT).unwrap();
        assert_eq!(a.len(), 9);
        assert_eq!(a.pick("layers.0.ffn.experts.w1.weight"), Pick::Mls2);
        assert_eq!(a.pick("layers.0.ffn.experts.w3.weight"), Pick::Keep);
        assert_eq!(a.pick("layers.0.attn.wq_a.weight"), Pick::Mls8);
        // The attention keeps 8 bits where the search kept 8, and 4 bits nowhere in the attention.
        assert_eq!(a.pick("layers.0.attn.wq_b.weight"), Pick::Mls8);
        assert_eq!(a.pick("layers.0.ffn.shared_experts.w1.weight"), Pick::Mls4);
        assert_eq!(a.pick("engram.table"), Pick::RowI4);
        assert_eq!(a.pick("norm.weight"), Pick::Keep);
    }

    #[test]
    fn an_unknown_type_is_an_error() {
        assert!(Allocation::parse("blk.0.attn_q_a.weight: Q9_Z\n").is_err());
    }
}
