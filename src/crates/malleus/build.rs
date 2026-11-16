fn main() {
    println!("cargo:rerun-if-env-changed=MALLEUS_RPATH");
    if std::env::var("MALLEUS_RPATH").is_ok() {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../lib");
    }
}
