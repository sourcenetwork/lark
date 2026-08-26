//! `lark-wasm-probe` - proves a lark database actually works on a
//! single-threaded host, not merely that it avoids a trap.
//!
//! ```sh
//! # native
//! cargo run -p lark-wasm-probe -- --report-memory
//!
//! # wasm32-wasip1 under wasmtime, with a host directory preopened
//! cargo build -p lark-wasm-probe --target wasm32-wasip1 --release
//! wasmtime run --dir="$(mktemp -d)::/data" \
//!     target/wasm32-wasip1/release/lark-wasm-probe.wasm -- --report-memory
//! ```
//!
//! The probe exits `0` only when every phase completed *and* every
//! byte it read back matched what it wrote, including after a close
//! and reopen from disk. Any mismatch prints `FAIL` with the offending
//! key and exits `1`.
//!
//! It has no dependencies beyond `lark-kv`: it is built for
//! `wasm32-wasip1` and checked on the 1.82 MSRV toolchain, so argument
//! parsing is hand-rolled rather than pulled from a crate that would
//! have to satisfy both.

mod capabilities;
mod check;
mod dataset;
mod host;
mod lifecycle;
mod mem;
mod queries;
mod report;
mod sustained;

use std::path::PathBuf;

use lark_kv::Options;

use report::Reporter;

/// Bulk records the lifecycle writes when `--records` is not given.
const DEFAULT_RECORDS: u64 = 5_000;

/// Memtable size for `--sustained`, small enough that a few thousand
/// writes cycle it many times.
const DEFAULT_SUSTAINED_BUFFER: usize = 32 * 1024;

const USAGE: &str = "\
lark-wasm-probe - full-lifecycle probe for single-threaded hosts

USAGE:
    lark-wasm-probe [OPTIONS]

OPTIONS:
    --dir <PATH>              database directory
                              (default: /data/lark-probe under wasi,
                               a temp directory otherwise)
    --profile <NAME>          embedded | default   (default: embedded)
    --background-compactions <N>
                              override max_background_compactions;
                              0 runs compaction on the calling thread
    --records <N>             bulk records to write (default: 5000)
    --sustained <N>           after the lifecycle, write N records with
                              a small memtable and no explicit
                              compaction, then verify a sample
    --write-buffer-bytes <N>  memtable size for --sustained
                              (default: 32768)
    --report-memory           print memory per phase and the high-water
                              mark
    --probe-host              print what each host primitive actually
                              does before running the lifecycle
    --keep                    leave the database directory in place
    -h, --help                print this help
";

struct Args {
    dir: PathBuf,
    profile: Profile,
    background_compactions: Option<usize>,
    records: u64,
    sustained: Option<u64>,
    write_buffer_bytes: usize,
    report_memory: bool,
    probe_host: bool,
    keep: bool,
}

#[derive(Clone, Copy)]
enum Profile {
    Embedded,
    Default,
}

impl Profile {
    fn options(self) -> Options {
        match self {
            Profile::Embedded => Options::embedded(),
            Profile::Default => Options::default(),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Profile::Embedded => "embedded",
            Profile::Default => "default",
        }
    }
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(message) => {
            eprintln!("FAIL  {message}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let args = match parse_args()? {
        Some(args) => args,
        None => {
            print!("{USAGE}");
            return Ok(());
        }
    };

    let mut reporter = Reporter::new(args.report_memory);
    reporter.pass("process baseline");

    // A stale database from an earlier run would not match the
    // expectations below, so the probe always starts from nothing.
    if args.dir.exists() {
        std::fs::remove_dir_all(&args.dir)
            .map_err(|e| format!("could not clear {}: {e}", args.dir.display()))?;
    }
    std::fs::create_dir_all(&args.dir)
        .map_err(|e| format!("could not create {}: {e}", args.dir.display()))?;

    // Measured unconditionally: the lifecycle compares these against
    // what `Db::capabilities` claims, and an honesty check that only
    // runs when asked for is not much of a check. `--probe-host` only
    // decides whether the table is printed.
    let host_root = args.dir.join("host-probe");
    let findings = host::probe(&host_root)
        .map_err(|e| format!("host probe could not use {}: {e}", host_root.display()))?;
    if args.probe_host {
        reporter.note("host primitives:");
        for finding in &findings {
            println!("      {:<40} {}", finding.name, finding.outcome.verdict());
        }
    }
    std::fs::remove_dir_all(&host_root)
        .map_err(|e| format!("could not clear {}: {e}", host_root.display()))?;

    let background = args.background_compactions;
    let profile = args.profile;
    let options = move || {
        let mut opts = profile.options();
        if let Some(n) = background {
            opts.max_background_compactions = n;
        }
        opts
    };

    reporter.note(&format!(
        "profile={} records={} background_compactions={}",
        profile.label(),
        args.records,
        options().max_background_compactions
    ));

    let db_dir = args.dir.join("db");
    lifecycle::run(&db_dir, &options, args.records, &findings, &mut reporter)?;

    if let Some(count) = args.sustained {
        let sustained_dir = args.dir.join("sustained");
        sustained::run(
            &sustained_dir,
            options(),
            args.write_buffer_bytes,
            count,
            &mut reporter,
        )?;
    }

    if !args.keep {
        std::fs::remove_dir_all(&args.dir)
            .map_err(|e| format!("could not remove {}: {e}", args.dir.display()))?;
        reporter.pass("cleanup");
    }

    reporter.summary();
    println!("OK    every phase passed and every byte matched");
    Ok(())
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = Args {
        dir: default_dir(),
        profile: Profile::Embedded,
        background_compactions: None,
        records: DEFAULT_RECORDS,
        sustained: None,
        write_buffer_bytes: DEFAULT_SUSTAINED_BUFFER,
        report_memory: false,
        probe_host: false,
        keep: false,
    };

    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        match flag.as_str() {
            // wasmtime forwards everything after the module path,
            // including the separator a caller writes to keep its own
            // flags apart from the runtime's.
            "--" => {}
            "-h" | "--help" => return Ok(None),
            "--dir" => args.dir = PathBuf::from(next(&mut raw, "--dir")?),
            "--profile" => {
                args.profile = match next(&mut raw, "--profile")?.as_str() {
                    "embedded" => Profile::Embedded,
                    "default" => Profile::Default,
                    other => return Err(format!("unknown --profile {other}")),
                }
            }
            "--background-compactions" => {
                args.background_compactions =
                    Some(parse_usize(&next(&mut raw, "--background-compactions")?)?)
            }
            "--records" => args.records = parse_u64(&next(&mut raw, "--records")?)?,
            "--sustained" => args.sustained = Some(parse_u64(&next(&mut raw, "--sustained")?)?),
            "--write-buffer-bytes" => {
                args.write_buffer_bytes = parse_usize(&next(&mut raw, "--write-buffer-bytes")?)?
            }
            "--report-memory" => args.report_memory = true,
            "--probe-host" => args.probe_host = true,
            "--keep" => args.keep = true,
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }
    Ok(Some(args))
}

fn next(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_u64(text: &str) -> Result<u64, String> {
    text.parse()
        .map_err(|_| format!("{text} is not a non-negative integer"))
}

fn parse_usize(text: &str) -> Result<usize, String> {
    text.parse()
        .map_err(|_| format!("{text} is not a non-negative integer"))
}

/// Under wasi the only writable location is whatever the host
/// preopened, and `/data` is the mapping the justfile and the CI job
/// both use. Elsewhere a temp directory is right.
#[cfg(target_os = "wasi")]
fn default_dir() -> PathBuf {
    PathBuf::from("/data/lark-probe")
}

#[cfg(not(target_os = "wasi"))]
fn default_dir() -> PathBuf {
    std::env::temp_dir().join("lark-wasm-probe")
}
