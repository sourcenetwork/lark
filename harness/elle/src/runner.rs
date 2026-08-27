//! Phase orchestration: steady-state concurrency, fault injection
//! through child processes, and post-recovery read-back.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use regolith::{DurabilityMode, Options};

use crate::cli::{Config, Fault, WorkerRole};
use crate::faults::{tear_wal_write, truncate_wal_tail, WalMark};
use crate::history::{
    close_dangling_invokes, read_worker_records, stream_record, write_history, Op, OpKind, Recorder,
};
use crate::model::{run_txn, Outcome, Rng, TxDb, TxnPlan, ValueSource};

/// Process id space reserved per child so a killed process is never
/// reused, which Jepsen requires after an indeterminate operation.
const CHILD_PROCESS_STRIDE: u64 = 1000;
/// Value space reserved per child so every append value stays globally
/// unique, which Elle requires to reconstruct version order.
const CHILD_VALUE_STRIDE: i64 = 1_000_000;

fn options(sync: bool) -> Options {
    Options {
        durability: if sync {
            DurabilityMode::Immediate
        } else {
            DurabilityMode::Eventual
        },
        ..Default::default()
    }
}

pub fn run(cfg: &Config) -> Result<usize, String> {
    prepare_dir(&cfg.dir)?;
    let recorder = Recorder::new(0);
    let values = ValueSource::new(cfg.value_base);

    {
        let db = TxDb::open(&cfg.dir, cfg.isolation, options(false)).map_err(|e| e.to_string())?;
        concurrent_phase(&db, cfg, &recorder, &values);
        read_back(&db, cfg, &recorder, cfg.threads);
        db.db().close().map_err(|e| e.to_string())?;
    }

    let mut recovery_failure = None;
    for (i, fault) in cfg.faults.iter().enumerate() {
        let child = i as u64;
        let note = match fault {
            Fault::Kill => kill_phase(cfg, &recorder, child)?,
            Fault::TornWrite => wal_phase(cfg, &recorder, child, Damage::Tear)?,
            Fault::TruncateWal => wal_phase(cfg, &recorder, child, Damage::Truncate)?,
        };
        eprintln!("fault: {}", note);

        // A database that will not reopen is itself the result. Stop
        // here rather than inventing read-back records for reads that
        // never ran, and report which fault it died on.
        match TxDb::open(&cfg.dir, cfg.isolation, options(false)) {
            Ok(db) => {
                read_back(&db, cfg, &recorder, cfg.threads + 1 + child);
                db.db().close().map_err(|e| e.to_string())?;
            }
            Err(err) => {
                recovery_failure = Some(format!("recovery failed after '{}': {}", note, err));
                break;
            }
        }
    }

    let mut ops = recorder.take();
    close_dangling_invokes(&mut ops, recorder.now());
    let written = write_history(&cfg.out, ops).map_err(|e| e.to_string())?;
    match recovery_failure {
        Some(failure) => Err(format!(
            "{} ({} operations written to {})",
            failure,
            written,
            cfg.out.display()
        )),
        None => Ok(written),
    }
}

fn concurrent_phase(db: &TxDb, cfg: &Config, recorder: &Recorder, values: &ValueSource) {
    std::thread::scope(|scope| {
        for process in 0..cfg.threads {
            scope.spawn(move || {
                let mut rng = Rng::new(cfg.seed ^ process.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                for _ in 0..cfg.txns {
                    let plan = TxnPlan::generate(cfg.model, cfg.keys, &mut rng, values);
                    let mut emit = |op: Op| recorder.push(op);
                    record_txn(db, cfg, &plan, recorder, process, false, &mut emit);
                }
            });
        }
    });
}

/// Read every key back on a single process. This is what lets a checker
/// see the final state of each list or register.
fn read_back(db: &TxDb, cfg: &Config, recorder: &Recorder, process: u64) {
    let plan = TxnPlan::read_all(cfg.keys);
    let mut emit = |op: Op| recorder.push(op);
    record_txn(db, cfg, &plan, recorder, process, false, &mut emit);
}

/// Run one transaction and emit its invoke and completion records.
///
/// `force_info` downgrades a commit to indeterminate. The doomed child
/// uses it because the log region holding its writes is about to be
/// damaged on purpose; claiming those writes committed would put a
/// fabricated fact into the history.
fn record_txn(
    db: &TxDb,
    cfg: &Config,
    plan: &TxnPlan,
    recorder: &Recorder,
    process: u64,
    force_info: bool,
    emit: &mut dyn FnMut(Op),
) {
    emit(Op::new(
        OpKind::Invoke,
        process,
        recorder.now(),
        plan.invoke_value(),
    ));
    let (kind, value) = match run_txn(db, cfg.model, plan) {
        Outcome::Committed(observed) => (OpKind::Ok, observed),
        Outcome::Aborted => (OpKind::Fail, plan.invoke_value()),
        Outcome::Unknown => (OpKind::Info, plan.invoke_value()),
    };
    let kind = if force_info && kind == OpKind::Ok {
        OpKind::Info
    } else {
        kind
    };
    emit(Op::new(kind, process, recorder.now(), value));
}

enum Damage {
    Tear,
    Truncate,
}

/// Spawn a worker, let it get transactions in flight, then SIGKILL it.
/// Its completed operations stay in the history; the one it had in
/// flight becomes indeterminate.
fn kill_phase(cfg: &Config, recorder: &Recorder, child: u64) -> Result<String, String> {
    let out = worker_path(cfg, child);
    let mut proc = spawn_worker(cfg, WorkerRole::Churn, recorder, child, &out)?;

    let observed = wait_for_records(&out, 8, Duration::from_secs(10));
    proc.kill().map_err(|e| format!("kill worker: {}", e))?;
    let status = proc.wait().map_err(|e| format!("reap worker: {}", e))?;

    merge_worker(recorder, &out)?;
    Ok(format!(
        "killed churn worker after {} records ({})",
        observed, status
    ))
}

/// Run a worker that crashes without flushing, then damage the log
/// region that only its writes occupy.
fn wal_phase(
    cfg: &Config,
    recorder: &Recorder,
    child: u64,
    damage: Damage,
) -> Result<String, String> {
    let out = worker_path(cfg, child);
    let proc = spawn_worker(cfg, WorkerRole::Doomed, recorder, child, &out)?;
    let finished = proc
        .wait_with_output()
        .map_err(|e| format!("reap worker: {}", e))?;

    merge_worker(recorder, &out)?;

    let stdout = String::from_utf8_lossy(&finished.stdout);
    let mark = stdout
        .lines()
        .find_map(WalMark::decode)
        .ok_or_else(|| "doomed worker did not report a write-ahead-log mark".to_string())?;

    let applied = match damage {
        Damage::Tear => tear_wal_write(&mark),
        Damage::Truncate => truncate_wal_tail(&mark),
    }
    .map_err(|e| format!("damage write-ahead log: {}", e))?;

    Ok(applied.unwrap_or_else(|| {
        format!(
            "skipped: {} did not grow past the {}-byte mark, so damaging it \
             would have destroyed committed data",
            mark.path.display(),
            mark.len
        )
    }))
}

fn worker_path(cfg: &Config, child: u64) -> PathBuf {
    let stem = format!("worker_{}.jsonl", child);
    match cfg.out.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(stem),
        _ => PathBuf::from(stem),
    }
}

fn spawn_worker(
    cfg: &Config,
    role: WorkerRole,
    recorder: &Recorder,
    child: u64,
    out: &Path,
) -> Result<std::process::Child, String> {
    let _ = std::fs::remove_file(out);
    let exe = std::env::current_exe().map_err(|e| format!("locate self: {}", e))?;
    let role_name = match role {
        WorkerRole::Churn => "churn",
        WorkerRole::Doomed => "doomed",
    };
    Command::new(exe)
        .arg("--worker")
        .arg(role_name)
        .arg("--dir")
        .arg(&cfg.dir)
        .arg("--model")
        .arg(cfg.model.as_str())
        .arg("--isolation")
        .arg(cfg.isolation.as_str())
        .arg("--keys")
        .arg(cfg.keys.to_string())
        .arg("--txns")
        .arg(cfg.txns.to_string())
        .arg("--seed")
        .arg((cfg.seed ^ (child + 1)).to_string())
        .arg("--worker-out")
        .arg(out)
        .arg("--process-base")
        .arg((CHILD_PROCESS_STRIDE * (child + 1)).to_string())
        .arg("--time-base")
        .arg(recorder.now().to_string())
        .arg("--value-base")
        .arg((CHILD_VALUE_STRIDE * (child as i64 + 1)).to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn worker: {}", e))
}

fn merge_worker(recorder: &Recorder, out: &Path) -> Result<(), String> {
    let records = read_worker_records(out).map_err(|e| format!("read worker history: {}", e))?;
    for op in records {
        recorder.push(op);
    }
    Ok(())
}

/// Poll until the worker has produced `want` records or the deadline
/// expires. A deadline plus bounded backoff keeps a fast machine quick
/// and a slow one correct.
fn wait_for_records(path: &Path, want: usize, deadline: Duration) -> usize {
    let start = Instant::now();
    let mut backoff = Duration::from_micros(200);
    loop {
        let seen = std::fs::read_to_string(path)
            .map(|raw| raw.lines().count())
            .unwrap_or(0);
        if seen >= want || start.elapsed() >= deadline {
            return seen;
        }
        std::thread::sleep(backoff);
        backoff = std::cmp::min(backoff * 2, Duration::from_millis(20));
    }
}

/// Child-process entry point.
pub fn run_worker(cfg: &Config) -> Result<(), String> {
    let role = cfg.worker.expect("worker role");
    let recorder = Recorder::new(cfg.time_base);
    let values = ValueSource::new(cfg.value_base);
    let mut sink = std::fs::File::create(&cfg.worker_out)
        .map_err(|e| format!("create worker history: {}", e))?;

    let db = TxDb::open(&cfg.dir, cfg.isolation, options(true)).map_err(|e| e.to_string())?;

    if role == WorkerRole::Doomed {
        let mark = WalMark::capture(&cfg.dir)
            .map_err(|e| format!("capture write-ahead-log mark: {}", e))?
            .ok_or_else(|| "no write-ahead log to mark".to_string())?;
        println!("{}", mark.encode());
    }

    let mut rng = Rng::new(cfg.seed);
    let budget = match role {
        WorkerRole::Churn => u64::MAX,
        WorkerRole::Doomed => cfg.txns,
    };
    for _ in 0..budget {
        let plan = TxnPlan::generate(cfg.model, cfg.keys, &mut rng, &values);
        let mut emit = |op: Op| {
            let _ = stream_record(&mut sink, &op);
        };
        record_txn(
            &db,
            cfg,
            &plan,
            &recorder,
            cfg.process_base,
            role == WorkerRole::Doomed,
            &mut emit,
        );
    }

    // Exit without closing: the database keeps its memtable unflushed
    // and its writes live only in the write-ahead log, which is the
    // state a crash leaves behind.
    std::process::exit(0);
}

fn prepare_dir(dir: &Path) -> Result<(), String> {
    match std::fs::read_dir(dir) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(format!("inspect {}: {}", dir.display(), err)),
        Ok(mut entries) => {
            let looks_like_regolith = std::fs::metadata(dir.join("MANIFEST")).is_ok();
            if entries.next().is_some() && !looks_like_regolith {
                return Err(format!(
                    "{} is not empty and does not look like a regolith database; \
                     refusing to delete it",
                    dir.display()
                ));
            }
            std::fs::remove_dir_all(dir).map_err(|e| format!("clear {}: {}", dir.display(), e))?;
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {}", dir.display(), e))
}
