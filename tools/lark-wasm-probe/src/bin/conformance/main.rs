//! `conformance` - proves a lark database stores and returns the right
//! bytes on a target, rather than merely that it compiles and does not
//! trap.
//!
//! Each phase is a separate process so that "reopen" and "recover"
//! genuinely start from nothing but the directory on disk. The
//! transcript is designed to be diffed: every observable result is a
//! `CHECK` line, and running the same phases natively and under
//! wasmtime must produce byte-identical `CHECK` output.
//!
//! ```sh
//! # native
//! cargo run -p lark-wasm-probe --bin conformance -- --dir /tmp/d --phase write
//!
//! # wasm32-wasip1
//! wasmtime run --dir=/tmp/d::/data \
//!     target/wasm32-wasip1/release/conformance.wasm -- --dir /data --phase write
//! ```

#[path = "../../mem.rs"]
mod mem;

mod adversarial;
mod data;
mod lifecycle;
mod report;
mod restart;
mod verify;

use std::path::PathBuf;

use lark_kv::Options;

use report::Report;

/// Entries `crash-compact` lets through before taking the process
/// out, when `--budget` is not given. Large enough that the compaction
/// is well into writing an output file.
const DEFAULT_COMPACTION_KILL_BUDGET: u64 = 1_500;

const USAGE: &str = "\
conformance - native/wasm parity harness for lark

USAGE:
    conformance --dir <PATH> --phase <PHASE> [--profile <NAME>]

PHASES:
    write     open a fresh database, load, delete, walk, snapshot, close
    reopen    open what write left behind and verify every byte
    crash     write and exit without closing (always exits 9)
    recover   open after crash and account for every acknowledged write
    compact   force a full compaction and re-verify
    final     open after compaction and verify every byte again

OPTIONS:
    --dir <PATH>       parent directory; the database lives in <PATH>/db
    --phase <PHASE>    which phase to run (required)
    --profile <NAME>   embedded | default   (default: embedded)
    --budget <N>       entries crash-compact filters before exiting
    -h, --help         print this help
";

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(message) => {
            println!("FAIL  {message}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<bool, String> {
    let (dir, phase, profile, budget) = match parse_args()? {
        Some(parsed) => parsed,
        None => {
            print!("{USAGE}");
            return Ok(true);
        }
    };

    let db_dir = dir.join("db");
    let options = || match profile {
        Profile::Embedded => Options::embedded(),
        Profile::Default => Options::default(),
    };

    let mut report = Report::new();
    report.note(&format!(
        "phase={} profile={} dir={} background_compactions={}",
        phase.name(),
        profile.name(),
        db_dir.display(),
        options().max_background_compactions
    ));

    match phase {
        Phase::Write => {
            if db_dir.exists() {
                std::fs::remove_dir_all(&db_dir)
                    .map_err(|e| format!("could not clear {}: {e}", db_dir.display()))?;
            }
            std::fs::create_dir_all(&db_dir)
                .map_err(|e| format!("could not create {}: {e}", db_dir.display()))?;
            lifecycle::run(&db_dir, options(), &mut report)
        }
        Phase::Reopen => restart::reopen(&db_dir, options(), &mut report),
        Phase::Crash => restart::crash(&db_dir, options(), &mut report),
        Phase::Recover => restart::recover(&db_dir, options(), &mut report),
        Phase::Compact => restart::compact(&db_dir, options(), &mut report),
        Phase::Final => restart::final_check(&db_dir, options(), &mut report),
        Phase::CrashCompact => adversarial::crash_compact(&db_dir, options(), budget, &mut report),
        Phase::Survey => adversarial::survey(&db_dir, options(), &mut report),
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Write,
    Reopen,
    Crash,
    Recover,
    Compact,
    Final,
    CrashCompact,
    Survey,
}

impl Phase {
    fn parse(text: &str) -> Result<Self, String> {
        match text {
            "write" => Ok(Phase::Write),
            "reopen" => Ok(Phase::Reopen),
            "crash" => Ok(Phase::Crash),
            "recover" => Ok(Phase::Recover),
            "compact" => Ok(Phase::Compact),
            "final" => Ok(Phase::Final),
            "crash-compact" => Ok(Phase::CrashCompact),
            "survey" => Ok(Phase::Survey),
            other => Err(format!("unknown --phase {other}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Phase::Write => "write",
            Phase::Reopen => "reopen",
            Phase::Crash => "crash",
            Phase::Recover => "recover",
            Phase::Compact => "compact",
            Phase::Final => "final",
            Phase::CrashCompact => "crash-compact",
            Phase::Survey => "survey",
        }
    }
}

#[derive(Clone, Copy)]
enum Profile {
    Embedded,
    Default,
}

impl Profile {
    fn name(self) -> &'static str {
        match self {
            Profile::Embedded => "embedded",
            Profile::Default => "default",
        }
    }
}

type Parsed = (PathBuf, Phase, Profile, u64);

fn parse_args() -> Result<Option<Parsed>, String> {
    let mut dir: Option<PathBuf> = None;
    let mut phase: Option<Phase> = None;
    let mut profile = Profile::Embedded;
    let mut budget = DEFAULT_COMPACTION_KILL_BUDGET;

    let mut raw = std::env::args().skip(1);
    while let Some(flag) = raw.next() {
        match flag.as_str() {
            // wasmtime forwards the separator a caller writes to keep
            // its own flags apart from the runtime's.
            "--" => {}
            "-h" | "--help" => return Ok(None),
            "--dir" => dir = Some(PathBuf::from(next(&mut raw, "--dir")?)),
            "--phase" => phase = Some(Phase::parse(&next(&mut raw, "--phase")?)?),
            "--budget" => {
                let text = next(&mut raw, "--budget")?;
                budget = text
                    .parse()
                    .map_err(|_| format!("{text} is not a non-negative integer"))?;
            }
            "--profile" => {
                profile = match next(&mut raw, "--profile")?.as_str() {
                    "embedded" => Profile::Embedded,
                    "default" => Profile::Default,
                    other => return Err(format!("unknown --profile {other}")),
                }
            }
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }

    let dir = dir.ok_or("--dir is required")?;
    let phase = phase.ok_or("--phase is required")?;
    Ok(Some((dir, phase, profile, budget)))
}

fn next(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next().ok_or_else(|| format!("{flag} needs a value"))
}
