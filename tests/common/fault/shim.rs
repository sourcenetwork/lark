//! Builds the `LD_PRELOAD` interposer and reports whether it can run here.
//!
//! The shim source lives at `preload_shim.rs` next to this file and is
//! deliberately *not* a module of the test crate: it is compiled standalone
//! with a direct `rustc` call into a `cdylib`. Keeping it out of the
//! workspace keeps `cargo check --workspace`, clippy and the MSRV job on
//! pure library code, and means no new Cargo member and no C toolchain.
//!
//! The build is content-addressed and atomic: the object is named after a
//! hash of the source and installed with a rename, so several test
//! binaries running in parallel cannot race each other.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = include_str!("preload_shim.rs");

/// Interposition needs glibc's dynamic linker and glibc's `write`/`fsync`
/// symbols. Anywhere else the harness must say so rather than quietly
/// running an un-instrumented child.
pub const fn supported() -> bool {
    cfg!(all(target_os = "linux", target_env = "gnu"))
}

#[derive(Clone, Debug)]
pub enum ShimError {
    Unsupported(&'static str),
    Build { status: String, stderr: String },
    Io(String),
}

impl std::fmt::Display for ShimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShimError::Unsupported(why) => write!(f, "fault shim unsupported here: {why}"),
            ShimError::Build { status, stderr } => {
                write!(f, "fault shim failed to compile ({status}):\n{stderr}")
            }
            ShimError::Io(e) => write!(f, "fault shim io error: {e}"),
        }
    }
}

impl std::error::Error for ShimError {}

fn out_dir() -> PathBuf {
    // Cargo sets CARGO_TARGET_TMPDIR while *compiling* an integration
    // test, not while running one, so it has to be read with `option_env!`
    // rather than at runtime. It points inside `target/`, which keeps the
    // cached object out of a shared `/tmp` and lets `cargo clean` remove
    // it.
    match option_env!("CARGO_TARGET_TMPDIR") {
        Some(d) => PathBuf::from(d).join("lark-fault"),
        None => std::env::temp_dir().join("lark-fault"),
    }
}

fn source_hash() -> u64 {
    let mut h = DefaultHasher::new();
    SOURCE.hash(&mut h);
    // Rebuild when the toolchain changes: a cdylib built by a different
    // rustc may embed a different std.
    std::env::var("RUSTC").unwrap_or_default().hash(&mut h);
    h.finish()
}

/// Compile the shim if it is not already cached, and return its path.
pub fn build() -> Result<PathBuf, ShimError> {
    if !supported() {
        return Err(ShimError::Unsupported(
            "LD_PRELOAD interposition needs a linux-gnu target",
        ));
    }
    let dir = out_dir();
    std::fs::create_dir_all(&dir).map_err(|e| ShimError::Io(e.to_string()))?;
    let hash = source_hash();
    let lib = dir.join(format!("liblark_fault_shim_{hash:016x}.so"));
    if lib.is_file() {
        return Ok(lib);
    }

    let src = dir.join(format!("preload_shim_{hash:016x}.rs"));
    std::fs::write(&src, SOURCE).map_err(|e| ShimError::Io(e.to_string()))?;
    let staged = dir.join(format!(
        "staged_{hash:016x}_{}_{}.so",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc)
        .args(["--edition", "2021"])
        .args(["--crate-type", "cdylib"])
        .args(["--crate-name", "lark_fault_shim"])
        .args(["-C", "opt-level=1"])
        .args(["-C", "panic=abort"])
        .arg("-o")
        .arg(&staged)
        .arg(&src)
        .output()
        .map_err(|e| ShimError::Io(format!("spawning rustc: {e}")))?;
    if !out.status.success() {
        return Err(ShimError::Build {
            status: out.status.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    // Rename is atomic within a directory, so a parallel builder either
    // sees no file or sees a complete one.
    std::fs::rename(&staged, &lib).map_err(|e| ShimError::Io(e.to_string()))?;
    Ok(lib)
}

/// True when the shim compiles and can be preloaded on this machine.
pub fn available() -> bool {
    build().is_ok()
}

/// The shim path, or a panic that says exactly why power-loss testing
/// cannot run here. Tests call this rather than degrading silently to a
/// weaker model.
pub fn require() -> PathBuf {
    match build() {
        Ok(p) => p,
        Err(e) => panic!(
            "{e}\nPower-loss simulation needs the LD_PRELOAD shim. \
             Without it a crash test only models a process kill, which leaves unsynced \
             bytes in the page cache and proves far less."
        ),
    }
}

/// Prepend the shim to any inherited `LD_PRELOAD`.
pub fn preload_value(lib: &Path) -> String {
    match std::env::var("LD_PRELOAD") {
        Ok(existing) if !existing.is_empty() => format!("{}:{}", lib.display(), existing),
        _ => lib.display().to_string(),
    }
}
