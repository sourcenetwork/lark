//! Command-line surface for the generator.

use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Model {
    ListAppend,
    RwRegister,
}

impl Model {
    pub fn as_str(self) -> &'static str {
        match self {
            Model::ListAppend => "list-append",
            Model::RwRegister => "rw-register",
        }
    }
}

/// Requested isolation level.
///
/// lark exposes snapshot isolation only, through two transaction
/// flavors, so not every level is soundly reachable today. Snapshot
/// isolation is strictly stronger than read-committed, so checking a
/// snapshot-isolated history against read-committed is a sound
/// over-approximation: every anomaly reported is a genuine violation.
/// It is incomparable with repeatable-read (snapshot isolation permits
/// write skew, which repeatable-read forbids) and strictly weaker than
/// serializable, so for those two the harness runs the closest engine
/// mode and warns that the verdict cannot be read as a lark bug.
/// README.md carries the full reachability table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isolation {
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

impl Isolation {
    pub fn as_str(self) -> &'static str {
        match self {
            Isolation::ReadCommitted => "read-committed",
            Isolation::RepeatableRead => "repeatable-read",
            Isolation::Serializable => "serializable",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fault {
    Kill,
    TornWrite,
    TruncateWal,
}

/// Child-process role. The parent re-executes itself to get a process
/// it can kill or crash without taking its own history down with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkerRole {
    /// Churn until killed from the outside.
    Churn,
    /// Write a bounded batch, then exit without closing the database,
    /// leaving the batch in the write-ahead log only.
    Doomed,
}

pub struct Config {
    pub model: Model,
    pub isolation: Isolation,
    pub faults: Vec<Fault>,
    pub dir: PathBuf,
    pub out: PathBuf,
    pub threads: u64,
    pub txns: u64,
    pub keys: i64,
    pub seed: u64,
    pub worker: Option<WorkerRole>,
    pub worker_out: PathBuf,
    pub process_base: u64,
    pub time_base: u64,
    pub value_base: i64,
    pub verify_only: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: Model::ListAppend,
            isolation: Isolation::ReadCommitted,
            faults: Vec::new(),
            dir: PathBuf::from("db"),
            out: PathBuf::from("history.json"),
            threads: 8,
            txns: 50,
            keys: 4,
            seed: 0xA5A5_1234_DEAD_BEEF,
            worker: None,
            worker_out: PathBuf::from("worker.jsonl"),
            process_base: 0,
            time_base: 0,
            value_base: 1,
            verify_only: None,
        }
    }
}

pub const USAGE: &str = "\
elle-gen - generate a Jepsen history from concurrent lark transactions

Usage: elle-gen [options]

Options:
  --model <list-append|rw-register>   Workload model (default: list-append)
  --isolation <read-committed|repeatable-read|serializable>
                                      Requested isolation (default: read-committed)
  --faults <kill,torn-write,truncate-wal|all>
                                      Fault injection (default: none)
  --dir <path>                        Database directory (default: db)
  --out <path>                        History output (default: history.json)
  --threads <n>                       Concurrent processes (default: 8)
  --txns <n>                          Transactions per process (default: 50)
  --keys <n>                          Distinct keys (default: 4)
  --seed <n>                          Deterministic workload seed
  --verify <path>                     Analyze an existing history and exit
  --help                              Print this message
";

pub fn parse<I: Iterator<Item = String>>(mut args: I) -> Result<Config, String> {
    let mut cfg = Config::default();

    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("option {} needs a value", arg))
        };
        match arg.as_str() {
            "--model" => {
                cfg.model = match value()?.as_str() {
                    "list-append" => Model::ListAppend,
                    "rw-register" => Model::RwRegister,
                    other => return Err(format!("unknown model {}", other)),
                }
            }
            "--isolation" => {
                cfg.isolation = match value()?.as_str() {
                    "read-committed" => Isolation::ReadCommitted,
                    "repeatable-read" => Isolation::RepeatableRead,
                    "serializable" => Isolation::Serializable,
                    other => return Err(format!("unknown isolation level {}", other)),
                }
            }
            "--faults" => cfg.faults = parse_faults(&value()?)?,
            "--dir" => cfg.dir = PathBuf::from(value()?),
            "--out" => cfg.out = PathBuf::from(value()?),
            "--threads" => cfg.threads = parse_num(&value()?, "--threads")?,
            "--txns" => cfg.txns = parse_num(&value()?, "--txns")?,
            "--keys" => cfg.keys = parse_num(&value()?, "--keys")? as i64,
            "--seed" => cfg.seed = parse_num(&value()?, "--seed")?,
            "--verify" => cfg.verify_only = Some(PathBuf::from(value()?)),
            "--worker" => {
                cfg.worker = Some(match value()?.as_str() {
                    "churn" => WorkerRole::Churn,
                    "doomed" => WorkerRole::Doomed,
                    other => return Err(format!("unknown worker role {}", other)),
                })
            }
            "--worker-out" => cfg.worker_out = PathBuf::from(value()?),
            "--process-base" => cfg.process_base = parse_num(&value()?, "--process-base")?,
            "--time-base" => cfg.time_base = parse_num(&value()?, "--time-base")?,
            "--value-base" => cfg.value_base = parse_num(&value()?, "--value-base")? as i64,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option {}\n\n{}", other, USAGE)),
        }
    }

    if cfg.keys < 1 {
        return Err("--keys must be at least 1".to_string());
    }
    if cfg.threads < 1 {
        return Err("--threads must be at least 1".to_string());
    }
    Ok(cfg)
}

fn parse_faults(raw: &str) -> Result<Vec<Fault>, String> {
    if raw == "all" {
        return Ok(vec![Fault::Kill, Fault::TornWrite, Fault::TruncateWal]);
    }
    raw.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| match s {
            "kill" => Ok(Fault::Kill),
            "torn-write" => Ok(Fault::TornWrite),
            "truncate-wal" => Ok(Fault::TruncateWal),
            other => Err(format!("unknown fault {}", other)),
        })
        .collect()
}

fn parse_num(raw: &str, flag: &str) -> Result<u64, String> {
    raw.parse::<u64>()
        .map_err(|_| format!("{} expects a non-negative integer, got {}", flag, raw))
}
