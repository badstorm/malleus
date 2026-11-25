//! The DSpark draft of DeepSeek-V4.1 (issues #83, #121): the stages the file keeps under `mtp.*`, on the host.
//!
//! A draft proposes `block` tokens after the last committed one in a single pass, and does not read the prompt's
//! KV: every stage keeps a sliding window of KV built from the backbone's attention input at the target layers,
//! seeded by the prefill and advanced one position per committed token. The `block` query rows all attend over that
//! window and over each other — the causal chain comes from the Markov bias, not from a mask — and the last stage
//! scores every row's confidence.
//!
//! The routed experts are read from the file when a row selects them: 7.93 GB there, against 163 GB if the three
//! stages held their 128 experts each in f32.

use anyhow::{Context, Result, ensure};
use malleus_gguf::TensorSource;

use crate::config::Config;
use crate::deepseek_text::{
    Hc, Lin, StreamedExperts, Weights, hc_post, hc_pre, identity_mix, load_lin, load_mat, load_matrix, load_vector,
    moe, rope_heads_at, rope_inv, rope_rows, sparse_attn,
};
use crate::hostmath::{dot, rmsnorm};

/// What one draft pass proposes.
#[derive(Clone, Debug)]
pub struct Proposal {
    /// The `block` tokens, in order.
    pub ids: Vec<u32>,
    /// The logits every proposal was taken from, `[block, vocab]`.
    pub logits: Vec<f32>,
    /// The last stage's confidence per proposal, `[block]`.
    pub confidence: Vec<f32>,
}

/// The three stages, the block they propose and the window they keep, in the order the file lists them.
pub struct Dspark {
    /// The model's config: the stages' experts read it for their geometry.
    cfg: Config,
    stages: Vec<Stage>,
    block: usize,
    /// Token every draft position but the first is embedded with.
    noise: u32,
    experts: usize,
    top_k: usize,
    rank: usize,
    hidden: usize,
    vocab: usize,
}

/// The Markov bias of the last stage: the previous token's row of `embed`, through its own `rank`, added to the
/// logits of the position that follows it.
struct Markov {
    embed: Vec<f32>,
    head: Lin,
}

struct Stage {
    hc_attn: Hc,
    hc_ffn: Hc,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn: DAttn,
    moe: crate::deepseek_text::Moe,
    /// First stage only: the projection and the norm that turn the backbone's concatenated hidden into the
    /// sub-layer input every stage reads.
    main: Option<(Lin, Vec<f32>)>,
    /// Last stage only: the final norm, the Markov bias and the confidence projection `[dim + rank]`.
    norm: Option<Vec<f32>>,
    markov: Option<Markov>,
    confidence: Option<Vec<f32>>,
    /// The window's KV, ring-indexed by position: `[window, head_dim]`, written from the backbone's hidden.
    kv: Vec<f32>,
    window: usize,
}

/// The draft's attention: a sliding window over the backbone's hidden states, the draft's own `block` rows, and
/// the grouped output projection. No compressor and no indexer (every DSpark layer is a window layer).
struct DAttn {
    wq_a: Lin,
    q_norm: Vec<f32>,
    wq_b: Lin,
    wkv: Lin,
    kv_norm: Vec<f32>,
    wo_a: Lin,
    wo_b: Lin,
    sink: Vec<f32>,
    heads: usize,
    head_dim: usize,
    rope: usize,
    q_lora: usize,
    groups: usize,
    o_lora: usize,
    /// Inverse RoPE frequencies of the window layers (the draft never extrapolates).
    inv: Vec<f32>,
}

impl Dspark {
    /// Reads the stages. The backbone's `embed` and `head` are not here: the draft shares them, so they come from
    /// the graph that owns them.
    pub fn load(src: &dyn TensorSource, c: &Config) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let s = d.dspark.as_ref().context("the config carries no DSpark draft")?;
        ensure!(s.stages > 0 && s.block > 0, "a draft needs stages and a block");
        let mut stages = Vec::with_capacity(s.stages);
        for i in 0..s.stages {
            stages.push(Stage::load(src, c, i)?);
        }
        Ok(Dspark {
            cfg: c.clone(),
            stages,
            block: s.block,
            noise: s.noise_token,
            experts: s.experts,
            top_k: s.top_k,
            rank: s.markov_rank,
            hidden: c.hidden,
            vocab: c.vocab,
        })
    }

    /// Tokens one pass proposes.
    pub fn block(&self) -> usize {
        self.block
    }

    /// The projection every stage reads, from the target layers' concatenated hidden `[t, layers * hidden]`.
    fn projected(&self, g: &Weights, main: &[f32], t: usize) -> Result<Vec<f32>> {
        let layers = g.target.len().max(1);
        ensure!(
            main.len() == t * layers * self.hidden,
            "the draft reads {} values as {t} positions of {layers} layers",
            main.len()
        );
        let (proj, norm) = self.stages[0]
            .main
            .as_ref()
            .context("the first stage has no main projection")?;
        let mut main_x = proj.mat(main, t);
        rmsnorm(&mut main_x, self.hidden, norm, g.eps);
        Ok(main_x)
    }

    /// Seeds the window with a prompt's positions: the KV of the backbone's hidden, rotated at 0..t.
    pub fn seed(&mut self, g: &Weights, main: &[f32], t: usize) -> Result<()> {
        let main_x = self.projected(g, main, t)?;
        for stage in &mut self.stages {
            stage.seed(&main_x, t, g.eps);
        }
        Ok(())
    }

    /// One committed token at `pos`: the stage windows take its row from `main` (one position), and the draft
    /// proposes the `block` tokens after it.
    pub fn propose(
        &mut self,
        src: &dyn TensorSource,
        g: &Weights,
        main: &[f32],
        last: u32,
        pos: usize,
    ) -> Result<Proposal> {
        self.propose_with(src, g, main, last, pos, |_, _| {})
    }

    /// The same, calling `on_stage(name, values)` after each stage's attention and after each stage's block: the
    /// stages the reference dumps, so a divergence names where it starts.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_with(
        &mut self,
        src: &dyn TensorSource,
        g: &Weights,
        main: &[f32],
        last: u32,
        pos: usize,
        mut on_stage: impl FnMut(&str, &[f32]),
    ) -> Result<Proposal> {
        let (dim, hc, block) = (self.hidden, g.c.hc_mult, self.block);
        ensure!((last as usize) < self.vocab, "token {last} outside the vocabulary");
        let main_x = self.projected(g, main, 1)?;
        // The draft's own positions: the token just committed, then the noise token.
        let mut draft = Vec::with_capacity(block);
        draft.push(last);
        draft.resize(block, self.noise);
        let mut h = vec![0f32; block * hc * dim];
        for (i, &id) in draft.iter().enumerate() {
            let row = g.embed_row(id as usize);
            for c in 0..hc {
                h[(i * hc + c) * dim..(i * hc + c + 1) * dim].copy_from_slice(row);
            }
        }
        let mut pre_mix = identity_mix(block, hc);
        for (i, stage) in self.stages.iter_mut().enumerate() {
            let am = stage.hc_attn.mixes(&h, block, dim, hc, g.c.hc_iters, g.c.hc_eps);
            let x = hc_pre(&h, &pre_mix, block, hc, dim);
            let mut xn = x.clone();
            rmsnorm(&mut xn, dim, &stage.attn_norm, g.eps);
            let attn_out = stage.attention(&xn, &main_x, pos, block, g.eps);
            on_stage(&format!("attn{i}"), &attn_out);
            h = hc_post(&attn_out, &h, &am.post, &am.comb, block, hc, dim);

            let fm = stage.hc_ffn.mixes(&h, block, dim, hc, g.c.hc_iters, g.c.hc_eps);
            let x = hc_pre(&h, &am.pre, block, hc, dim);
            let mut xn = x.clone();
            rmsnorm(&mut xn, dim, &stage.ffn_norm, g.eps);
            let mut bank =
                StreamedExperts::new(src, &self.cfg, i, self.top_k).with_base(format!("mtp.{i}.ffn.experts."));
            let ffn_out = moe(
                &stage.moe,
                &xn,
                block,
                dim,
                &g.c,
                self.experts,
                self.top_k,
                None,
                &mut bank,
                &mut |_, _| Ok(()),
                0,
            )?;
            h = hc_post(&ffn_out, &h, &fm.post, &fm.comb, block, hc, dim);
            on_stage(&format!("block{i}"), &h);
            pre_mix = fm.pre;
        }

        // The head: the streams collapsed, the final norm, the vocabulary, then the Markov chain over the block.
        let last_stage = self.stages.last().context("no stages")?;
        let x = hc_pre(&h, &pre_mix, block, hc, dim);
        let mut xh = x.clone();
        rmsnorm(
            &mut xh,
            dim,
            last_stage.norm.as_ref().context("the last stage has no norm")?,
            g.eps,
        );
        let mut logits = g.head_rows(&xh, block);
        let markov = last_stage
            .markov
            .as_ref()
            .context("the last stage has no Markov head")?;
        let rank = self.rank;
        let mut ids = Vec::with_capacity(block);
        let mut embeds = Vec::with_capacity(block * rank);
        let mut previous = last;
        for i in 0..block {
            let e = &markov.embed[previous as usize * rank..(previous as usize + 1) * rank];
            let bias = markov.head.mat(e, 1);
            for (l, b) in logits[i * self.vocab..(i + 1) * self.vocab].iter_mut().zip(&bias) {
                *l += b;
            }
            embeds.extend_from_slice(e);
            let row = &logits[i * self.vocab..(i + 1) * self.vocab];
            let mut best = 0;
            for (j, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = j;
                }
            }
            previous = best as u32;
            ids.push(previous);
        }
        let conf_w = last_stage
            .confidence
            .as_ref()
            .context("the last stage has no confidence head")?;
        ensure!(
            conf_w.len() == dim + rank,
            "the confidence head reads {} values, the rest gives {}",
            conf_w.len(),
            dim + rank
        );
        let mut confidence = Vec::with_capacity(block);
        for i in 0..block {
            let mut out = 0f32;
            for j in 0..dim {
                out += x[i * dim + j] * conf_w[j];
            }
            for j in 0..rank {
                out += embeds[i * rank + j] * conf_w[dim + j];
            }
            confidence.push(out);
        }
        Ok(Proposal {
            ids,
            logits,
            confidence,
        })
    }
}

impl Stage {
    fn load(src: &dyn TensorSource, c: &Config, i: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let s = d.dspark.as_ref().context("the config carries no DSpark draft")?;
        let (h, hc) = (c.hidden, d.hc_mult);
        let last = i + 1 == s.stages;
        let p = |x: &str| format!("mtp.{i}.{x}");
        let hc_load = |which: &str| -> Result<Hc> {
            Ok(Hc {
                f: load_mat(src, &p(&format!("{which}_fn")), (2 + hc) * hc, hc * h)?,
                base: load_vector(src, &p(&format!("{which}_base")), (2 + hc) * hc)?,
                scale: load_vector(src, &p(&format!("{which}_scale")), 3)?,
            })
        };
        let window = d.window.max(1);
        Ok(Stage {
            hc_attn: hc_load("hc_attn")?,
            hc_ffn: hc_load("hc_ffn")?,
            attn_norm: load_vector(src, &p("attn_norm.weight"), h)?,
            ffn_norm: load_vector(src, &p("ffn_norm.weight"), h)?,
            attn: DAttn::load(src, c, i)?,
            moe: crate::deepseek_text::Moe {
                gate: load_lin(src, &p("ffn.gate.weight"), s.experts, h)?,
                bias: load_vector(src, &p("ffn.gate.bias"), s.experts)?,
                bias_vl: None,
                shared: (
                    load_lin(src, &p("ffn.shared_experts.w1.weight"), d.moe_inter, h)?,
                    load_lin(src, &p("ffn.shared_experts.w2.weight"), h, d.moe_inter)?,
                    load_lin(src, &p("ffn.shared_experts.w3.weight"), d.moe_inter, h)?,
                ),
                experts: Vec::new(),
            },
            main: if i == 0 {
                Some((
                    load_lin(src, &p("main_proj.weight"), h, h * s.target_layers.len())?,
                    load_vector(src, &p("main_norm.weight"), h)?,
                ))
            } else {
                None
            },
            norm: if last {
                Some(load_vector(src, &p("norm.weight"), h)?)
            } else {
                None
            },
            markov: if last {
                Some(Markov {
                    embed: load_matrix(src, &p("markov_head.embed.weight"), c.vocab, s.markov_rank)?,
                    head: load_lin(src, &p("markov_head.head.weight"), c.vocab, s.markov_rank)?,
                })
            } else {
                None
            },
            confidence: if last {
                Some(load_matrix(
                    src,
                    &p("confidence_head.proj.weight"),
                    1,
                    h + s.markov_rank,
                )?)
            } else {
                None
            },
            kv: vec![0f32; window * d.head_dim],
            window,
        })
    }

    /// Writes the prompt's KV rows into the window, rotated at their own positions.
    fn seed(&mut self, main_x: &[f32], t: usize, eps: f32) {
        let a = &self.attn;
        let (hd, rd, win) = (a.head_dim, a.rope, self.window);
        let mut kv = a.wkv.mat(main_x, t);
        rmsnorm(&mut kv, hd, &a.kv_norm, eps);
        rope_rows(&mut kv, t, hd, rd, &a.inv, |i| i as f32, false);
        for i in 0..t {
            let at = (i % win) * hd;
            self.kv[at..at + hd].copy_from_slice(&kv[i * hd..(i + 1) * hd]);
        }
    }

    /// The attention of the `block` draft rows at positions `pos + 1 ..= pos + block`, reading the committed
    /// token's row (from the backbone's hidden) plus the window.
    fn attention(&mut self, xn: &[f32], main_x: &[f32], pos: usize, block: usize, eps: f32) -> Vec<f32> {
        let a = &self.attn;
        let (hd, rd, heads) = (a.head_dim, a.rope, a.heads);
        let win = self.window;
        // The committed token's own row, from the backbone's attention input.
        let mut cur = a.wkv.mat(main_x, 1);
        rmsnorm(&mut cur, hd, &a.kv_norm, eps);
        rope_rows(&mut cur, 1, hd, rd, &a.inv, |_| pos as f32, false);
        let at = (pos % win) * hd;
        self.kv[at..at + hd].copy_from_slice(&cur);

        let mut qr = a.wq_a.mat(xn, block);
        rmsnorm(&mut qr, a.q_lora, &a.q_norm, eps);
        let mut q = a.wq_b.mat(&qr, block);
        rope_rows(
            &mut q,
            block * heads,
            hd,
            rd,
            &a.inv,
            |i| (pos + 1 + i / heads) as f32,
            false,
        );
        let mut kv = a.wkv.mat(xn, block);
        rmsnorm(&mut kv, hd, &a.kv_norm, eps);
        rope_rows(&mut kv, block, hd, rd, &a.inv, |i| (pos + 1 + i) as f32, false);

        // Every draft row reads the whole reachable window and all the block's own rows.
        let reach = (pos + 1).min(win);
        let m = reach + block;
        let mut kv_all = Vec::with_capacity(m * hd);
        kv_all.extend_from_slice(&self.kv[..reach * hd]);
        kv_all.extend_from_slice(&kv);
        let mut idx = Vec::with_capacity(block * m);
        for _ in 0..block {
            idx.extend((0..reach as i32).chain(reach as i32..m as i32));
        }
        let mut o = sparse_attn(&q, &kv_all, m, &a.sink, &idx, block, m, heads, hd);
        for i in 0..block {
            rope_heads_at(
                &mut o[i * heads * hd..],
                heads,
                hd,
                rd,
                &a.inv,
                (pos + 1 + i) as f32,
                true,
            );
        }
        // The grouped output projection, one row at a time.
        let per_group = heads * hd / a.groups;
        let mut g = vec![0f32; block * a.groups * a.o_lora];
        for i in 0..block {
            for grp in 0..a.groups {
                let src = &o[i * heads * hd + grp * per_group..][..per_group];
                let w = &a.wo_a.w[grp * a.o_lora * per_group..][..a.o_lora * per_group];
                let out = &mut g[(i * a.groups + grp) * a.o_lora..][..a.o_lora];
                for r in 0..a.o_lora {
                    out[r] = dot(src, &w[r * per_group..][..per_group], per_group);
                }
            }
        }
        a.wo_b.mat(&g, block)
    }
}

impl DAttn {
    fn load(src: &dyn TensorSource, c: &Config, i: usize) -> Result<Self> {
        let d = c.deepseek.as_ref().context("no deepseek config")?;
        let (h, hd, rd) = (c.hidden, d.head_dim, d.rope);
        let heads = d.heads;
        let p = |x: &str| format!("mtp.{i}.attn.{x}");
        Ok(DAttn {
            wq_a: load_lin(src, &p("wq_a.weight"), d.q_lora, h)?,
            q_norm: load_vector(src, &p("q_norm.weight"), d.q_lora)?,
            wq_b: load_lin(src, &p("wq_b.weight"), heads * hd, d.q_lora)?,
            wkv: load_lin(src, &p("wkv.weight"), hd, h)?,
            kv_norm: load_vector(src, &p("kv_norm.weight"), hd)?,
            wo_a: load_lin(src, &p("wo_a.weight"), d.o_groups * d.o_lora, heads * hd / d.o_groups)?,
            wo_b: load_lin(src, &p("wo_b.weight"), h, d.o_groups * d.o_lora)?,
            sink: load_vector(src, &p("attn_sink"), heads)?,
            heads,
            head_dim: hd,
            rope: rd,
            q_lora: d.q_lora,
            groups: d.o_groups,
            o_lora: d.o_lora,
            inv: rope_inv(rd, d.rope_params.theta, 0, 1.0, 0.0, 0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_text::Streamer;
    use malleus_gguf::safetensors::SafeTensors;
    use std::path::Path;

    fn f32_file(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Cosine, the worst and the mean absolute difference — the same shape the other gates print.
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
        let cos = dot / (ng.sqrt() * nw.sqrt());
        eprintln!(
            "{what}: cosine {cos:.7}, max |diff| {max:.6}, mean |diff| {:.6}",
            sum / got.len() as f64
        );
        cos
    }

    /// The draft against the reference: the tiny checkpoint carries the three stages, and
    /// `scripts/deepseek_ref.py` dumps what the release's own `forward_spec` proposes one position after the
    /// prompt — the ids, the logits each was taken from and the confidence of each row.
    ///
    /// The reference prefills the prompt, seeds the draft with the whole prompt's target hidden, then re-feeds the
    /// last token as a decode step and drafts from it. The graph here does the same by prefilling one position less
    /// and stepping the last one, which leaves position `t-1` computed exactly once.
    ///
    ///   DEEPSEEK_REF=/golden/deepseek-ref/tiny
    #[test]
    #[ignore = "reads the reference checkpoint"]
    fn the_draft_matches_the_reference() {
        let dir = std::env::var("DEEPSEEK_REF").expect("DEEPSEEK_REF");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap()).unwrap();
        let ids: Vec<u32> = manifest["tokens"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let c = Config::from_hf(Path::new(&dir)).unwrap();
        let src = SafeTensors::open_dir(Path::new(&dir)).unwrap();
        let mut s = Streamer::open(Box::new(src), &c).unwrap();
        let t = ids.len();
        let mut cache = s.cache(8);
        s.prefill_cached(&mut cache, &ids[..t - 1]).unwrap();
        let main_prefill = s.main_hidden.clone();
        assert_eq!(main_prefill.len(), (t - 1) * s.graph().target.len() * c.hidden);
        s.step(&mut cache, &ids).unwrap();
        let main_step = s.main_hidden.clone();
        assert_eq!(main_step.len(), s.graph().target.len() * c.hidden);

        let mut draft = Dspark::load(s.source(), &c).unwrap();
        draft.seed(s.graph(), &main_prefill, t - 1).unwrap();
        let mut stages: Vec<(String, Vec<f32>)> = Vec::new();
        let p = draft
            .propose_with(s.source(), s.graph(), &main_step, ids[t - 1], t - 1, |name, v| {
                stages.push((name.to_string(), v.to_vec()));
            })
            .unwrap();
        // The inputs of stage 0. The reference feeds its own decode step's hidden, and that step does not agree
        // with its own prefill on two of the three target layers — 0.9974 on the hidden, `spec_prefill_hidden`
        // against `spec_main_hidden` — while this engine's step is the prefill's row by construction (gated).
        // That measured difference is the floor of everything below; the graph itself is exact on the tiny model.
        let want_hidden = f32_file(&format!("{dir}/spec_main_hidden.f32"));
        for (i, layer) in s.graph().target.iter().enumerate() {
            let h = c.hidden;
            let got: Vec<f32> = main_step[i * h..(i + 1) * h].to_vec();
            let cos = compare(
                &format!("main_hidden layer {layer}"),
                &got,
                &want_hidden[i * h..(i + 1) * h],
            );
            assert!(cos > 0.997, "the target's hidden diverges: {cos}");
        }
        let main_x = draft.projected(s.graph(), &main_step, 1).unwrap();
        let cos = compare("spec_main_x", &main_x, &f32_file(&format!("{dir}/spec_main_x.f32")));
        assert!(cos > 0.998, "the draft's projection diverges: {cos}");
        for (name, got) in &stages {
            let path = format!("{dir}/spec_{name}.f32");
            if std::path::Path::new(&path).exists() {
                let cos = compare(&format!("spec_{name}"), got, &f32_file(&path));
                assert!(cos > 0.999, "{name} diverges: {cos}");
            }
        }
        assert_eq!(p.ids.len(), draft.block());
        let cos = compare("spec_logits", &p.logits, &f32_file(&format!("{dir}/spec_logits.f32")));
        assert!(cos > 0.999, "the draft's logits diverge: {cos}");
        let cos = compare("spec_conf", &p.confidence, &f32_file(&format!("{dir}/spec_conf.f32")));
        assert!(cos > 0.999, "the confidence diverges: {cos}");
        // And the tokens themselves: the draft proposes exactly what the release's does.
        let want_ids = f32_file(&format!("{dir}/spec_ids.f32"));
        // The reference's first entry is the token it drafted from, not a proposal.
        let want: Vec<u32> = want_ids[1..].iter().map(|v| *v as u32).collect();
        assert_eq!(p.ids, want, "the draft proposes different tokens");
    }
}
