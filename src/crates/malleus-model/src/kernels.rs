use anyhow::{Context, Result, bail};
use malleus_hip::{Kernel, Module};
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));

pub const SUPPORTED_ARCHS: &[&str] = &["gfx1150", "gfx1151"];

const KERNEL_NAMES: &[(&str, &[&str])] = &[
    ("bench", &["bw_read", "bw_copy"]),
    (
        "attn",
        &[
            "attention_wmma",
            "attention_split",
            "attention_combine",
            "attention_gather",
            "attention_gather_q1",
            "attention_gather_u",
            "kv_pattern",
            "kv_grouped",
            "kv_k_only",
            "kv_v_only",
            "kv_v_local",
        ],
    ),
    ("deltanet", &["deltanet_chunk_prep", "deltanet_chunk_scan"]),
    (
        "wmma",
        &[
            "hadamard128_h",
            "gemm_wmma",
            "gemm_wmma_m64",
            "gemm_wmma_m32",
            "quant_rows_i8",
            "rot_quant_i8",
            "rot_quant_i8_w",
            "rot_quant_i4",
            "rot_quant_i4_w",
            "gemm_w4a4",
            "gemm_w4a4_m64",
            "gemm_w4a4_m32",
            "gemm_w4a8",
            "gemm_w4a8_m64",
            "gemm_w4a8_m32",
            "gemm_w8a8",
            "gemm_w8a8_m64",
            "gemm_w8a8_m32",
            "gemm_i8",
            "gemm_i8_m128",
            "gemm_i8_glu",
            "gemm_i8_glu_m128",
            "gemm_i8_sk",
            "gemm_i8_hcmix",
            "rot_quant_row",
            "rot_quant_i4_row",
            "rot_quant_i8_row",
            "moe_i8_glu",
            "moe_i8",
            "moe_w4a8_glu",
            "moe_w4a8_glu_m32",
            "moe_w4a8_m32",
            "moe_w4a8",
            "moe_w4a8_glu_m128",
            "moe_w4a8_m128",
            "moe_w4a4_glu",
            "moe_w4a4_glu_m32",
            "moe_w4a4_m32",
            "moe_w4a4",
            "moe_w4a4_glu_m128",
            "moe_w4a4_m128",
            "moe_gather_g",
            "moe_gather_g4",
            // Same down GEMMs with `y` in fp16 (MALLEUS_MOE_Y_HALF).
            "moe_w4a8_yh",
            "moe_w4a8_m32_yh",
            "moe_w4a8_m128_yh",
            "moe_w4a4_yh",
            "moe_w4a4_m32_yh",
            "moe_w4a4_m128_yh",
        ],
    ),
    ("vision", &["layernorm", "bias_act", "vision_rope", "vision_attention"]),
    (
        "qwen4",
        &[
            "hc_norm",
            "hc_inject_norm",
            "hc_gate",
            "hc_mix",
            "hc_inject",
            "hc_expand",
            "ple_gate",
            "ple_gate_norm",
            "ple_conv",
            "ple_state_save",
            "idx_prep",
            "idx_pool",
            "idx_tail",
            "idx_score",
            "idx_score_tile",
            "idx_select",
            "idx_select12",
            "idx_select12_1024",
            "logits_argmax",
            "idx_union",
            "hc_bcast_add",
            "copy_f32",
            "hc_pre_dec",
            "hc_down_dec",
            "hc_gate_dec",
            "hc_up_mix_dec",
            "hc_down_fused",
            "hc_up_fused",
            "hc_quant_row",
            "hc_mix_inv",
            "gemm_bf16_wmma",
            "idx_compact",
            "attention_sel",
        ],
    ),
    (
        "quant",
        &[
            "hadamard128",
            "gemv_mls4",
            "gemm_mls4",
            "gemv_mls4_l1",
            "gemv_mls4_l2",
            "gemv_mls4_l3",
            "gemv_mls4_l4",
            "gemv_mls4_l5",
            "gemv_mls4_l6",
            "gemv_mls4_r1",
            "gemv_mls4_m1",
            "gemv_mls4_m2",
            "gemv_mls4_m3",
            "gemv_mls4_m4",
            "gemv_mls4_m5",
            "gemv_mls4_m6",
            "gemv_mls8",
            "gemv_mls8_t2",
            "gemv_mls8_t3",
            "gemv_mls8_t4",
            "gemv_mls8_t5",
            "gemv_mls8_t6",
            "gemm_mls8",
            "moe_route",
            "moe_route_w",
            "moe_route_g",
            "moe_gate_up",
            "moe_act",
            "moe_down",
            "moe_sort",
            "moe_gather",
            "moe_combine",
            "moe_combine_yh",
        ],
    ),
    (
        "ref",
        &[
            "embed_bf16",
            "rmsnorm",
            "mtp_cat",
            "logits_top1",
            "gemm_bf16",
            "add_inplace",
            "silu_mul",
            "attn_prep",
            "attn_gate",
            "conv_qk_norm",
            "conv_state_save",
            "deltanet_gates",
            "deltanet_scan",
            "kda_gates",
            "rmsnorm_gated",
            "swiglu_clamp",
        ],
    ),
    ("mls3", &["gemv_mls3", "gemm_mls3", "moe_gate_up3", "moe_down3"]),
    ("kda", &["kda_scan", "kda_conv_norm"]),
    (
        "mla",
        &[
            "mla_cache_write",
            "mla_decode",
            "mla_absorb",
            "mla_split",
            "mla_combine",
            "mla_absorb_t",
            "mla_prefill",
            "mla_project",
            "mla_gate",
            "rope_ilv",
        ],
    ),
];

pub struct Kernels {
    pub arch: String,
    map: HashMap<&'static str, Kernel>,
}

impl Kernels {
    pub fn load(arch: &str) -> Result<Self> {
        let mut map = HashMap::new();
        for (module, names) in KERNEL_NAMES {
            let obj = KERNEL_OBJECTS
                .iter()
                .find(|(a, m, _)| *a == arch && m == module)
                .with_context(|| format!("no code object '{module}' built for {arch}"))?;
            let m = Module::load(obj.2)?;
            for n in *names {
                map.insert(*n, m.function(n)?);
            }
        }
        Ok(Kernels {
            arch: arch.to_string(),
            map,
        })
    }

    pub fn get(&self, name: &str) -> &Kernel {
        self.map.get(name).unwrap_or_else(|| panic!("kernel {name} not loaded"))
    }
}

/// Picks the single supported GPU visible to the process and makes it current.
pub fn select_device() -> Result<String> {
    let devs = malleus_hip::devices().context("HIP init failed: check /dev/kfd and the render node")?;
    let ok: Vec<_> = devs
        .iter()
        .filter(|d| SUPPORTED_ARCHS.contains(&d.arch.as_str()))
        .collect();
    match ok.as_slice() {
        [d] => {
            malleus_hip::set_device(d.index)?;
            Ok(d.arch.clone())
        }
        [] => bail!(
            "no supported GPU (gfx1150/gfx1151) found among {:?}; check /dev/kfd and the render node passed to the container",
            devs.iter().map(|d| d.arch.as_str()).collect::<Vec<_>>()
        ),
        _ => bail!("more than one supported GPU visible; restrict with HIP_VISIBLE_DEVICES"),
    }
}
