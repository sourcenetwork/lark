//! Jepsen-format history records and the recorder that collects them.
//!
//! A history is a sequence of operation objects. Every logical
//! transaction contributes exactly two records: an `invoke` when the
//! process starts it and one of `ok` / `fail` / `info` when it
//! resolves. `ok` means committed, `fail` means definitely not
//! committed, `info` means indeterminate. Recording an indeterminate
//! outcome as `fail` would tell the checker a write definitely did not
//! happen when it may well have, so the fourth type is mandatory for a
//! sound history.

use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Outcome of a transaction as seen by the client.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OpKind {
    Invoke,
    Ok,
    Fail,
    Info,
}

/// The third element of a micro-operation: `null` on an invoke, the
/// observed list on a list-append read, a plain integer on a
/// rw-register read or write.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(untagged)]
pub enum MopVal {
    Null,
    Int(i64),
    List(Vec<i64>),
}

/// One micro-operation, serialized as `["append", key, value]`,
/// `["r", key, value]` or `["w", key, value]`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Mop(pub String, pub i64, pub MopVal);

/// One line of the history file.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Op {
    #[serde(rename = "type")]
    pub kind: OpKind,
    pub f: String,
    pub process: u64,
    pub time: u64,
    pub index: u64,
    pub value: Vec<Mop>,
}

impl Op {
    pub fn new(kind: OpKind, process: u64, time: u64, value: Vec<Mop>) -> Self {
        Self {
            kind,
            f: "txn".to_string(),
            process,
            time,
            index: 0,
            value,
        }
    }
}

/// Collects records from every worker thread of one process.
///
/// Records are pushed in completion order under a mutex, so a stable
/// sort by `time` later preserves the invoke-before-completion order
/// of any two records that share a timestamp.
pub struct Recorder {
    start: Instant,
    base_nanos: u64,
    ops: Mutex<Vec<Op>>,
}

impl Recorder {
    pub fn new(base_nanos: u64) -> Self {
        Self {
            start: Instant::now(),
            base_nanos,
            ops: Mutex::new(Vec::new()),
        }
    }

    /// Nanoseconds since this process started, shifted by the base the
    /// parent handed down so a child's records interleave correctly.
    pub fn now(&self) -> u64 {
        self.base_nanos + self.start.elapsed().as_nanos() as u64
    }

    pub fn push(&self, op: Op) {
        self.ops.lock().expect("recorder mutex").push(op);
    }

    pub fn take(&self) -> Vec<Op> {
        std::mem::take(&mut *self.ops.lock().expect("recorder mutex"))
    }
}

/// Append one record as a JSON line, flushing per line so a SIGKILL
/// leaves a readable prefix rather than an empty file.
pub fn stream_record(out: &mut impl Write, op: &Op) -> std::io::Result<()> {
    serde_json::to_writer(&mut *out, op)?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Sort by time, assign dense indices, and write both history files.
///
/// `path` gets a JSON array with one operation object per line, which
/// is what elle-cli 0.1.9 parses. The sibling `.jsonl` is the same
/// records as bare newline-delimited JSON for tooling that wants it.
pub fn write_history(path: &Path, mut ops: Vec<Op>) -> std::io::Result<usize> {
    ops.sort_by_key(|op| op.time);
    for (i, op) in ops.iter_mut().enumerate() {
        op.index = i as u64;
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(b"[")?;
    for (i, op) in ops.iter().enumerate() {
        if i > 0 {
            file.write_all(b",\n")?;
        }
        serde_json::to_writer(&mut file, op)?;
    }
    file.write_all(b"]\n")?;
    file.flush()?;

    let mut lines = std::io::BufWriter::new(std::fs::File::create(path.with_extension("jsonl"))?);
    for op in &ops {
        serde_json::to_writer(&mut lines, op)?;
        lines.write_all(b"\n")?;
    }
    lines.flush()?;

    Ok(ops.len())
}

/// Read a JSONL stream written by a killed child, dropping a trailing
/// partial line. A truncated final line means the child died mid-write;
/// that record is simply absent, and the parent synthesizes the
/// indeterminate completion.
pub fn read_worker_records(path: &Path) -> std::io::Result<Vec<Op>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    Ok(raw
        .lines()
        .filter_map(|line| serde_json::from_str::<Op>(line).ok())
        .collect())
}

/// Give every invoke that never resolved an `info` completion, so no
/// in-flight operation is silently dropped from the history.
pub fn close_dangling_invokes(ops: &mut Vec<Op>, at_time: u64) {
    let mut pending: Vec<Op> = Vec::new();
    for op in ops.iter() {
        match op.kind {
            OpKind::Invoke => pending.push(op.clone()),
            _ => pending.retain(|p| p.process != op.process),
        }
    }
    for (i, mut op) in pending.into_iter().enumerate() {
        op.kind = OpKind::Info;
        op.time = at_time + i as u64;
        ops.push(op);
    }
}
