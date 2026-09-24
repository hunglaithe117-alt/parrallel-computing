//! Captures the build context (`rustc` version and `RUSTFLAGS`) into the binary
//! so the harness can print it to stderr per BENCH_CONTRACT.md rule 5.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Cargo exposes the effective flags to build scripts via the
    // `CARGO_ENCODED_RUSTFLAGS` variable (`\x1f`-separated); `RUSTFLAGS` is the
    // raw user value and is used as a fallback.
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .filter(|value| !value.is_empty())
        .map(|value| value.replace('\u{1f}', " "))
        .or_else(|| std::env::var("RUSTFLAGS").ok())
        .unwrap_or_default();

    println!("cargo:rustc-env=RUSTC_VERSION={version}");
    println!("cargo:rustc-env=BUILD_RUSTFLAGS={flags}");
}
