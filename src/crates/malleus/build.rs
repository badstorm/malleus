//! Build-time flags for the binary.
//!
//! Two things come from here: the `rpath` the container image needs (`MALLEUS_RPATH=1`, so the binary finds the ROCm
//! libraries copied next to it) and the version it reports (`MALLEUS_BUILD_VERSION`, passed by the image build from
//! the git tag; a plain `cargo build` falls back to the crate version in `Cargo.toml`). The version goes out as a
//! `rustc-env` so `concat!(env!(...))` still works in the `LONG_VERSION` constant.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=MALLEUS_RPATH");
    if env::var("MALLEUS_RPATH").is_ok() {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../lib");
    }

    let version = env::var("MALLEUS_BUILD_VERSION")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION"));

    println!("cargo:rustc-env=MALLEUS_BUILD_VERSION={version}");
    println!("cargo:rerun-if-env-changed=MALLEUS_BUILD_VERSION");
}
