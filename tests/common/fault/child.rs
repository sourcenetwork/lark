//! Subprocess crash harness.
//!
//! A crash test needs a real process that really dies, so the workload runs
//! in a child. The child is the test binary re-executing itself: a separate
//! `bin` target would not see the test crate's code, so instead the parent
//! spawns `current_exe()` with `--exact --ignored <entry test>` and an
//! environment describing the workload. The entry test returns immediately
//! when that environment is absent, so a normal `cargo test` run never
//! notices it.
//!
//! Every test file that uses this harness declares the entry point once:
//!
//! ```ignore
//! mod common;
//!
//! #[test]
//! #[ignore = "child process entry point, re-executed by the crash harness"]
//! fn crash_child() {
//!     common::fault::child_entrypoint(common::fault::builtin_workload);
//! }
//! ```
//!
//! # Kill points
//!
//! Wall-clock kills are not reproducible and land nowhere in particular.
//! These land on a semantic boundary instead. [`Phase::AfterNPuts`] is the
//! workload killing itself once a chosen write has returned `Ok`. Every
//! other phase is byte-exact: the `LD_PRELOAD` shim counts the matching
//! `write`/`fsync` calls and raises `SIGKILL` on the nth one, so the crash
//! lands at the same byte on a fast machine and a slow one.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use lark_kv::{Db, DurabilityMode, Options, WriteBatch};

use super::journal::{Journal, journal_path_for, root_filter_for};
use super::prefix::{History, OpValue};
use super::shim;

/// Default workload seed. Fixed, so every run of every downstream test
/// replays byte for byte; override with `ChildSpec::seed`.
pub const DEFAULT_SEED: u64 = 0x1A12_5EED_C0FF_EE01;

pub const CHILD_ENV: &str = "LARK_CRASH_CHILD";
/// Name of the `#[test] #[ignore]` function the parent re-executes.
pub const CHILD_TEST: &str = "crash_child";

#[cfg(unix)]
unsafe extern "C" {
    fn getpid() -> i32;
    fn kill(pid: i32, sig: i32) -> i32;
}

/// Kill this process the way a power supply would kill it: no unwinding,
/// no destructors, no buffered output flushed.
pub fn kill_self() -> ! {
    #[cfg(unix)]
    unsafe {
        kill(getpid(), 9);
    }
    // SIGKILL cannot be caught, so this is only reached on a non-unix
    // platform where the harness is unsupported anyway.
    std::process::abort()
}

/// Where in the write path the child is meant to die.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Run the workload and shut down cleanly. The baseline every crash
    /// result is compared against.
    CleanExit,
    /// Die once `ChildSpec::ops` writes have returned `Ok`.
    AfterNPuts,
    /// Die part way through a single `WriteBatch`, while its WAL record is
    /// physically half written.
    MidWriteBatch,
    /// Die while a memtable flush is writing an SSTable.
    DuringFlush,
    /// Die while compaction is writing its output SSTable.
    DuringCompaction,
    /// Die while a `VersionEdit` is being appended to the MANIFEST.
    DuringManifestWrite,
    /// Die after the WAL append syscall has returned but before the write
    /// is applied to the memtable and `visible_seq` is published.
    BetweenWalAndApply,
    /// A workload supplied by the calling test crate.
    Custom(String),
}

impl Phase {
    pub fn as_str(&self) -> &str {
        match self {
            Phase::CleanExit => "clean_exit",
            Phase::AfterNPuts => "after_n_puts",
            Phase::MidWriteBatch => "mid_write_batch",
            Phase::DuringFlush => "during_flush",
            Phase::DuringCompaction => "during_compaction",
            Phase::DuringManifestWrite => "during_manifest_write",
            Phase::BetweenWalAndApply => "between_wal_and_apply",
            Phase::Custom(s) => s,
        }
    }

    pub fn parse(s: &str) -> Phase {
        match s {
            "clean_exit" => Phase::CleanExit,
            "after_n_puts" => Phase::AfterNPuts,
            "mid_write_batch" => Phase::MidWriteBatch,
            "during_flush" => Phase::DuringFlush,
            "during_compaction" => Phase::DuringCompaction,
            "during_manifest_write" => Phase::DuringManifestWrite,
            "between_wal_and_apply" => Phase::BetweenWalAndApply,
            other => Phase::Custom(other.to_string()),
        }
    }
}

/// Which syscall the shim counts, and whether it kills before or after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DieKind {
    Write,
    Fsync,
    Open,
    Truncate,
}

impl DieKind {
    fn as_str(&self) -> &'static str {
        match self {
            DieKind::Write => "write",
            DieKind::Fsync => "fsync",
            DieKind::Open => "open",
            DieKind::Truncate => "truncate",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// Never kill. The workload runs to completion.
    None,
    /// The workload kills itself at its own semantic point.
    Workload,
    /// The shim kills on the `nth` matching syscall.
    Syscall {
        kind: DieKind,
        path_contains: String,
        nth: u64,
        /// Kill before the real call instead of after it.
        before: bool,
    },
}

impl Trigger {
    pub fn wal_write(nth: u64) -> Trigger {
        Trigger::Syscall {
            kind: DieKind::Write,
            path_contains: "/wal/".to_string(),
            nth,
            before: false,
        }
    }
    pub fn sst_write(nth: u64) -> Trigger {
        Trigger::Syscall {
            kind: DieKind::Write,
            path_contains: ".sst".to_string(),
            nth,
            before: false,
        }
    }
    pub fn manifest_write(nth: u64) -> Trigger {
        Trigger::Syscall {
            kind: DieKind::Write,
            path_contains: "MANIFEST".to_string(),
            nth,
            before: false,
        }
    }
    pub fn wal_fsync(nth: u64) -> Trigger {
        Trigger::Syscall {
            kind: DieKind::Fsync,
            path_contains: "/wal/".to_string(),
            nth,
            before: false,
        }
    }
}

/// Everything the child needs to reproduce the workload, and everything
/// the parent needs to predict it. Crossing the process boundary as plain
/// environment variables keeps it debuggable and dependency free.
#[derive(Clone, Debug)]
pub struct ChildSpec {
    pub phase: Phase,
    pub db_path: PathBuf,
    pub seed: u64,
    /// Number of write operations in the workload.
    pub ops: usize,
    /// Writes per `WriteBatch`. 1 means plain `put`/`delete` calls.
    pub batch_size: usize,
    pub value_len: usize,
    /// Every nth operation deletes the previously written key, so tombstone
    /// recovery is exercised. 0 disables deletes.
    pub delete_every: usize,
    pub durability: DurabilityMode,
    pub write_buffer_size: usize,
    pub ack_path: PathBuf,
}

impl ChildSpec {
    pub fn new(phase: Phase, db_path: impl Into<PathBuf>) -> ChildSpec {
        let db_path = db_path.into();
        let ack_path = ack_path_for(&db_path);
        let (ops, batch_size, value_len, write_buffer_size) = match phase {
            // A batch has to exceed the WAL's 8 KiB BufWriter for a crash
            // to land physically inside one record rather than between two.
            Phase::MidWriteBatch => (256, 64, 1024, 1 << 20),
            Phase::DuringFlush => (2000, 1, 128, 8 * 1024),
            Phase::DuringCompaction => (20000, 1, 128, 8 * 1024),
            Phase::DuringManifestWrite => (2000, 1, 128, 8 * 1024),
            _ => (400, 1, 96, 1 << 20),
        };
        // MidWriteBatch defaults to Immediate so the batches before the
        // torn one are genuinely durable. Under Eventual the whole WAL is
        // unsynced, a power cut discards all of it, and "was the batch
        // atomic" becomes a question about an empty database.
        let durability = match phase {
            Phase::MidWriteBatch => DurabilityMode::Immediate,
            _ => DurabilityMode::Eventual,
        };
        ChildSpec {
            phase,
            db_path,
            seed: DEFAULT_SEED,
            ops,
            batch_size,
            value_len,
            delete_every: 7,
            durability,
            write_buffer_size,
            ack_path,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn ops(mut self, ops: usize) -> Self {
        self.ops = ops;
        self
    }
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }
    pub fn value_len(mut self, n: usize) -> Self {
        self.value_len = n;
        self
    }
    pub fn delete_every(mut self, n: usize) -> Self {
        self.delete_every = n;
        self
    }
    pub fn durability(mut self, d: DurabilityMode) -> Self {
        self.durability = d;
        self
    }
    pub fn write_buffer_size(mut self, n: usize) -> Self {
        self.write_buffer_size = n;
        self
    }

    pub fn options(&self) -> Options {
        Options {
            write_buffer_size: self.write_buffer_size,
            durability: self.durability,
            ..Options::default()
        }
    }

    /// The exact ordered history this spec produces, computed identically
    /// in the parent and the child so the parent never has to trust a
    /// process it just killed.
    pub fn history(&self) -> History {
        plan(self)
    }

    fn to_env(&self) -> Vec<(String, String)> {
        vec![
            (CHILD_ENV.into(), self.phase.as_str().into()),
            ("LARK_CRASH_DB".into(), self.db_path.display().to_string()),
            ("LARK_CRASH_SEED".into(), self.seed.to_string()),
            ("LARK_CRASH_OPS".into(), self.ops.to_string()),
            ("LARK_CRASH_BATCH".into(), self.batch_size.to_string()),
            ("LARK_CRASH_VLEN".into(), self.value_len.to_string()),
            ("LARK_CRASH_DEL".into(), self.delete_every.to_string()),
            (
                "LARK_CRASH_DUR".into(),
                match self.durability {
                    DurabilityMode::Immediate => "immediate".into(),
                    DurabilityMode::Eventual => "eventual".into(),
                },
            ),
            ("LARK_CRASH_WBS".into(), self.write_buffer_size.to_string()),
            ("LARK_CRASH_ACK".into(), self.ack_path.display().to_string()),
        ]
    }

    fn from_env() -> Option<ChildSpec> {
        let phase = Phase::parse(&std::env::var(CHILD_ENV).ok()?);
        let num = |k: &str, d: usize| -> usize {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Some(ChildSpec {
            phase,
            db_path: PathBuf::from(std::env::var("LARK_CRASH_DB").ok()?),
            seed: std::env::var("LARK_CRASH_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            ops: num("LARK_CRASH_OPS", 100),
            batch_size: num("LARK_CRASH_BATCH", 1),
            value_len: num("LARK_CRASH_VLEN", 64),
            delete_every: num("LARK_CRASH_DEL", 0),
            durability: match std::env::var("LARK_CRASH_DUR").as_deref() {
                Ok("immediate") => DurabilityMode::Immediate,
                _ => DurabilityMode::Eventual,
            },
            write_buffer_size: num("LARK_CRASH_WBS", 1 << 20),
            ack_path: PathBuf::from(std::env::var("LARK_CRASH_ACK").unwrap_or_default()),
        })
    }
}

/// Sidecar paths live beside the database directory, never inside it, so
/// the shim's root filter excludes them and a power-loss reconstruction
/// never rewrites the record of what the child was told.
pub fn ack_path_for(db_dir: &Path) -> PathBuf {
    sidecar(db_dir, "acks")
}

/// Marker the child drops the instant it recognises itself as a child.
/// Its absence after a run means the workload never started, which is the
/// one failure that would otherwise look like a clean pass.
pub fn started_path_for(db_dir: &Path) -> PathBuf {
    sidecar(db_dir, "started")
}

fn sidecar(db_dir: &Path, ext: &str) -> PathBuf {
    let name = db_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "db".to_string());
    db_dir.with_file_name(format!("{name}.{ext}"))
}

/// How the child process ended, plus everything observed about the run.
#[derive(Debug)]
pub struct ChildOutcome {
    pub spec: ChildSpec,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// The recorded syscall stream, empty when the shim did not run.
    pub journal: Journal,
    /// Indices of write operations that returned `Ok` before the crash.
    /// Durable by construction: written outside the database directory, so
    /// a power-loss reconstruction leaves them alone.
    pub acked: Vec<usize>,
    /// The history the workload intended to apply.
    pub history: History,
    pub trigger: Trigger,
}

impl ChildOutcome {
    pub fn was_killed(&self) -> bool {
        self.signal.is_some()
    }

    pub fn acked_count(&self) -> usize {
        self.acked.len()
    }

    /// Fail unless the child really died by signal. A crash test whose
    /// crash never fired is a false green, so this is not optional.
    pub fn assert_killed(&self) {
        assert!(
            self.was_killed(),
            "child for phase {:?} was expected to be killed but exited with {:?}.\n\
             The fault never fired, so this run proves nothing.\ntrigger: {:?}\n{}\nstderr:\n{}",
            self.spec.phase,
            self.exit_code,
            self.trigger,
            self.journal,
            self.stderr,
        );
    }

    /// Fail unless the child completed and shut down cleanly.
    pub fn assert_clean(&self) {
        assert_eq!(
            self.exit_code,
            Some(0),
            "child for phase {:?} did not exit cleanly (signal {:?})\nstderr:\n{}",
            self.spec.phase,
            self.signal,
            self.stderr,
        );
    }
}

/// Run `phase` against `db_path` in a child process and kill it at that
/// phase's kill point.
///
/// The database directory must already exist or be creatable by the child.
/// The parent removes any previous journal, ack file and captured output
/// so a stale run can never be mistaken for a fresh one.
pub fn run_child(phase: Phase, db_path: &Path) -> ChildOutcome {
    CrashRun::new(ChildSpec::new(phase, db_path)).run()
}

/// Builder for a crash run when the defaults are not what a test needs.
pub struct CrashRun {
    spec: ChildSpec,
    trigger: Trigger,
    entry_test: String,
    record_io: bool,
    timeout: Duration,
}

impl CrashRun {
    pub fn new(spec: ChildSpec) -> CrashRun {
        let trigger = default_trigger(&spec);
        CrashRun {
            spec,
            trigger,
            entry_test: CHILD_TEST.to_string(),
            record_io: true,
            timeout: Duration::from_secs(120),
        }
    }

    pub fn trigger(mut self, trigger: Trigger) -> Self {
        self.trigger = trigger;
        self
    }
    /// Name of the `#[test] #[ignore]` entry point in the calling crate.
    pub fn entry_test(mut self, name: impl Into<String>) -> Self {
        self.entry_test = name.into();
        self
    }
    /// Turn off syscall recording. Power-loss simulation needs it, so this
    /// is only for a run that is purely about the process dying.
    pub fn record_io(mut self, on: bool) -> Self {
        self.record_io = on;
        self
    }
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    pub fn run(self) -> ChildOutcome {
        let spec = self.spec;
        let db = spec.db_path.clone();
        std::fs::create_dir_all(&db).expect("crash run: create db dir");

        let journal = journal_path_for(&db);
        let stdout_path = sidecar(&db, "stdout");
        let stderr_path = sidecar(&db, "stderr");
        let started_path = started_path_for(&db);
        for p in [
            &journal,
            &spec.ack_path,
            &stdout_path,
            &stderr_path,
            &started_path,
        ] {
            let _ = std::fs::remove_file(p);
        }

        let exe = std::env::current_exe().expect("crash run: current_exe");
        let mut cmd = Command::new(exe);
        cmd.args(["--exact", "--nocapture", "--ignored", &self.entry_test])
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                File::create(&stdout_path).expect("crash run: stdout file"),
            ))
            .stderr(Stdio::from(
                File::create(&stderr_path).expect("crash run: stderr file"),
            ));
        for (k, v) in spec.to_env() {
            cmd.env(k, v);
        }
        cmd.env("LARK_CRASH_STARTED", &started_path);
        // A stale trigger inherited from the parent environment would kill
        // a child that was meant to run to completion.
        for k in [
            "LARK_FAULT_DIE_KIND",
            "LARK_FAULT_DIE_PATH",
            "LARK_FAULT_DIE_NTH",
            "LARK_FAULT_DIE_WHEN",
            "LARK_FAULT_JOURNAL",
        ] {
            cmd.env_remove(k);
        }

        let needs_shim = self.record_io || matches!(self.trigger, Trigger::Syscall { .. });
        if needs_shim {
            let lib = shim::require();
            cmd.env("LD_PRELOAD", shim::preload_value(&lib));
            cmd.env("LARK_FAULT_ROOT", root_filter_for(&db));
            if self.record_io {
                cmd.env("LARK_FAULT_JOURNAL", &journal);
            }
        }
        if let Trigger::Syscall {
            kind,
            path_contains,
            nth,
            before,
        } = &self.trigger
        {
            cmd.env("LARK_FAULT_DIE_KIND", kind.as_str())
                .env("LARK_FAULT_DIE_PATH", path_contains)
                .env("LARK_FAULT_DIE_NTH", nth.to_string())
                .env(
                    "LARK_FAULT_DIE_WHEN",
                    if *before { "before" } else { "after" },
                );
        }

        let mut child = cmd.spawn().expect("crash run: spawn child");
        let status = wait_with_deadline(&mut child, self.timeout);

        assert!(
            started_path.is_file(),
            "the child never entered the workload. The test crate needs the entry point:\n\n    \
             #[test]\n    #[ignore = \"child process entry point\"]\n    fn {}() {{\n        \
             common::fault::child_entrypoint(common::fault::builtin_workload);\n    }}\n\n\
             child stdout:\n{}\nchild stderr:\n{}",
            self.entry_test,
            std::fs::read_to_string(&stdout_path).unwrap_or_default(),
            std::fs::read_to_string(&stderr_path).unwrap_or_default(),
        );

        #[cfg(unix)]
        let signal = {
            use std::os::unix::process::ExitStatusExt;
            status.signal()
        };
        #[cfg(not(unix))]
        let signal = None;

        ChildOutcome {
            exit_code: status.code(),
            signal,
            stdout: std::fs::read_to_string(&stdout_path).unwrap_or_default(),
            stderr: std::fs::read_to_string(&stderr_path).unwrap_or_default(),
            journal: Journal::read(&journal).expect("crash run: read journal"),
            acked: read_acks(&spec.ack_path),
            history: spec.history(),
            trigger: self.trigger,
            spec,
        }
    }
}

/// Wait for the child with a deadline and bounded backoff, so a fast
/// machine returns immediately and a slow one is not cut off early. A
/// child that outlives the deadline is killed and the test fails loudly
/// rather than hanging the suite.
fn wait_with_deadline(
    child: &mut std::process::Child,
    timeout: Duration,
) -> std::process::ExitStatus {
    let deadline = Instant::now() + timeout;
    let mut backoff = Duration::from_micros(200);
    loop {
        match child.try_wait().expect("crash run: try_wait") {
            Some(status) => return status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("crash-harness child did not finish within {timeout:?}");
            }
            None => {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_millis(20));
            }
        }
    }
}

fn read_acks(path: &Path) -> Vec<usize> {
    match std::fs::read_to_string(path) {
        Ok(t) => t.lines().filter_map(|l| l.trim().parse().ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Where each phase crashes, counted in write syscalls to the file the
/// phase is about.
///
/// Two things make a syscall count different from an operation count,
/// and a trigger that ignores either one silently never fires, which
/// turns the probe into a false green rather than a failure:
///
/// - **The format stamp is a write.** A WAL and a MANIFEST each open
///   with a 12-byte stamp written by itself, so a record is one write
///   later than its ordinal.
/// - **Group commit coalesces records.** One vectored write carries a
///   whole group, so a batch phase produces a handful of writes rather
///   than one per operation.
///
/// The numbers below are therefore read off the journal for each phase,
/// not derived. `CrashOut::assert_killed` is what catches a stale one.
fn default_trigger(spec: &ChildSpec) -> Trigger {
    match spec.phase {
        Phase::CleanExit => Trigger::None,
        Phase::AfterNPuts => Trigger::Workload,
        // 64 writes of 1 KiB reach the log as a handful of grouped
        // writes, so this lands inside the record stream.
        Phase::MidWriteBatch => Trigger::wal_write(3),
        Phase::DuringFlush => Trigger::sst_write(3),
        Phase::DuringCompaction => Trigger::sst_write(40),
        // Write 1 is the REGOMAN stamp; write 3 is the second append.
        Phase::DuringManifestWrite => Trigger::manifest_write(3),
        // Write 1 is the WAL stamp, so writes 2..=37 are records 1..=36.
        Phase::BetweenWalAndApply => Trigger::wal_write(37),
        Phase::Custom(_) => Trigger::None,
    }
}

/// Deterministic workload plan. Called identically in the parent (to know
/// what should be there) and the child (to write it), so the two can never
/// drift.
pub fn plan(spec: &ChildSpec) -> History {
    let order = permutation(spec.ops, spec.seed);
    let mut h = History::new();
    let batch = spec.batch_size.max(1);
    let mut i = 0usize;
    while i < spec.ops {
        let n = batch.min(spec.ops - i);
        if n == 1 {
            let idx = i;
            if is_delete(spec, idx) {
                h.delete(key_for(order[idx.saturating_sub(1)]));
            } else {
                h.put(key_for(order[idx]), value_for(idx, spec));
            }
        } else {
            let ops: Vec<(Vec<u8>, OpValue)> = (i..i + n)
                .map(|idx| {
                    if is_delete(spec, idx) {
                        (key_for(order[idx.saturating_sub(1)]), OpValue::Delete)
                    } else {
                        (key_for(order[idx]), OpValue::Put(value_for(idx, spec)))
                    }
                })
                .collect();
            h.batch(ops);
        }
        i += n;
    }
    h
}

fn is_delete(spec: &ChildSpec, idx: usize) -> bool {
    spec.delete_every > 0 && idx > 0 && idx.is_multiple_of(spec.delete_every)
}

fn key_for(idx: usize) -> Vec<u8> {
    format!("key_{idx:08}").into_bytes()
}

/// The value carries the op index that wrote it, so a validator failure
/// names the exact write that went wrong instead of dumping bytes.
fn value_for(op: usize, spec: &ChildSpec) -> Vec<u8> {
    let head = format!("v{op:08}#");
    let mut v = head.into_bytes();
    let mut s = spec.seed ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    while v.len() < spec.value_len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(b'a' + (s % 26) as u8);
    }
    v
}

/// Seeded Fisher-Yates, so writes do not arrive in key order and the
/// memtable, the SSTable writer and recovery all see realistic churn.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.swap(i, (s % (i as u64 + 1)) as usize);
    }
    v
}

/// Child-side entry point. Returns immediately in the parent, and never
/// returns in the child.
pub fn child_entrypoint(dispatch: fn(&ChildSpec)) {
    let spec = match ChildSpec::from_env() {
        Some(s) => s,
        None => return,
    };
    if let Ok(marker) = std::env::var("LARK_CRASH_STARTED") {
        std::fs::write(&marker, spec.phase.as_str())
            .unwrap_or_else(|e| panic!("child: writing start marker {marker}: {e}"));
    }
    dispatch(&spec);
    std::process::exit(0);
}

struct Acker(Option<File>);

impl Acker {
    fn open(path: &Path) -> Acker {
        if path.as_os_str().is_empty() {
            return Acker(None);
        }
        Acker(OpenOptions::new().create(true).append(true).open(path).ok())
    }
    /// One unbuffered write per acknowledgement, so a `SIGKILL` a
    /// microsecond later cannot lose the record of what the caller was
    /// told.
    fn ack(&mut self, indices: impl IntoIterator<Item = usize>) {
        if let Some(f) = self.0.as_mut() {
            let mut line = String::new();
            for i in indices {
                line.push_str(&i.to_string());
                line.push('\n');
            }
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// The built-in workloads, one per [`Phase`]. Pass this to
/// [`child_entrypoint`] unless the test crate needs its own.
pub fn builtin_workload(spec: &ChildSpec) {
    let db = Db::open(&spec.db_path, spec.options()).expect("child: open db");
    let history = plan(spec);
    let mut acker = Acker::open(&spec.ack_path);

    let mut i = 0usize;
    let ops = history.ops();
    while i < ops.len() {
        let batch_id = ops[i].batch;
        let end = ops[i..]
            .iter()
            .position(|o| o.batch != batch_id)
            .map(|p| i + p)
            .unwrap_or(ops.len());
        let group = &ops[i..end];
        let result = if group.len() == 1 {
            match &group[0].value {
                OpValue::Put(v) => db.put(&group[0].key, v),
                OpValue::Delete => db.delete(&group[0].key),
            }
        } else {
            let mut wb = WriteBatch::new();
            for op in group {
                match &op.value {
                    OpValue::Put(v) => wb.put(&op.key, v),
                    OpValue::Delete => wb.delete(&op.key),
                }
            }
            db.write(wb)
        };
        result.expect("child: write failed");
        acker.ack(i..end);
        i = end;
    }

    match spec.phase {
        Phase::AfterNPuts => kill_self(),
        Phase::CleanExit => {
            db.close().expect("child: close");
        }
        // Every other phase is killed by the shim mid-flight. Reaching
        // here means the trigger never fired; shutting down cleanly lets
        // the parent's assert_killed report that honestly.
        _ => {
            db.close().expect("child: close");
        }
    }
}
