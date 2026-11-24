//! Build-time toolchain guard for this crate's nightly-gated feature.
//!
//! The `generic_const` feature turns on `#![feature(generic_const_exprs)]`
//! at the crate root, which the stable and beta compilers reject with the
//! bare
//! `error[E0554]: `#![feature]` may not be used on the stable release channel`.
//! That error says nothing about *which* feature of *which* crate needs a
//! nightly compiler. A build script runs before rustc ever sees the crate
//! attributes, so it can probe the compiler's release channel first and
//! refuse with targeted instructions. Nightly builds are unaffected apart
//! from one extra `rustc --version` invocation.

use std::process::Command;

fn main() {
    // Cargo mirrors the enabled features into `CARGO_FEATURE_<NAME>`
    // environment variables (upper-cased, `-` becomes `_`), so the active
    // feature set is read without parsing the manifest.
    let generic_const = std::env::var_os("CARGO_FEATURE_GENERIC_CONST").is_some();
    if !generic_const {
        return;
    }

    // `RUSTC_BOOTSTRAP` opts a non-nightly compiler into `#![feature]` on
    // purpose; whoever sets it owns the consequences, so treat it as nightly.
    let bootstrap = std::env::var_os("RUSTC_BOOTSTRAP").is_some();

    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let release = Command::new(&compiler)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned());

    // If the compiler cannot even be probed, fall back to rustc's own (ugly
    // but truthful) behaviour: refusing here risks rejecting a build that
    // would in fact have succeeded.
    let Some(release) = release else {
        return;
    };

    let nightly = release.contains("nightly") || release.contains("-dev");
    if nightly || bootstrap {
        return;
    }

    eprintln!(
        "symrust cannot be built on this toolchain:\n\n\
  - the enabled feature `generic_const` turns on rust-lang's nightly \
`generic_const_exprs`\n\n\
found:   {release}\n\
\n\
Re-run cargo through the nightly toolchain, for example:\n\
\n\
    RUSTUP_TOOLCHAIN=nightly cargo <the same arguments as before>\n\
\n\
or drop the nightly-only features to stay on the stable-supported surface."
    );
    std::process::exit(101);
}
