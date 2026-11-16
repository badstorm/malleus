use std::path::PathBuf;
use std::process::Command;

fn main() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let targets = std::env::var("MALLEUS_GPU_TARGETS").unwrap_or_else(|_| "gfx1150;gfx1151".into());
    let kdir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=MALLEUS_GPU_TARGETS");
    println!("cargo:rerun-if-changed={}", kdir.display());
    let mut srcs: Vec<_> = std::fs::read_dir(&kdir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "hip"))
        .collect();
    srcs.sort();
    let mut table = String::from("pub static KERNEL_OBJECTS: &[(&str, &str, &[u8])] = &[\n");
    for arch in targets.split(';').filter(|s| !s.is_empty()) {
        for src in &srcs {
            println!("cargo:rerun-if-changed={}", src.display());
            let stem = src.file_stem().unwrap().to_str().unwrap();
            let obj = out.join(format!("{stem}-{arch}.hsaco"));
            let st = Command::new(format!("{rocm}/bin/hipcc"))
                .args(["--genco", "-O3"])
                .arg(format!("--offload-arch={arch}"))
                .arg("-o")
                .arg(&obj)
                .arg(src)
                .status()
                .expect("failed to run hipcc");
            assert!(st.success(), "hipcc failed for {} ({arch})", src.display());
            table += &format!(
                "    ({arch:?}, {stem:?}, include_bytes!({:?})),\n",
                obj.display().to_string()
            );
        }
    }
    table += "];\n";
    std::fs::write(out.join("kernels.rs"), table).unwrap();
}
