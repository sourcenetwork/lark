//! A standalone witness detector for the anomaly this harness targets.
//!
//! elle-cli is the authority on whether a history is valid. This module
//! exists so the harness still produces evidence on a machine with no
//! JVM, and so a failure can be pointed at a concrete pair of
//! operations rather than a cycle diagram.
//!
//! Both checks are stated against real time and are therefore sound
//! under strict serializability, which is elle-cli's default
//! consistency model: once a transaction commits at time `t`, a
//! transaction that is *invoked* after `t` must not observe a state
//! older than that commit.

use std::collections::HashMap;
use std::path::Path;

use crate::cli::Model;
use crate::history::{MopVal, Op, OpKind};

/// One committed operation paired with the time its transaction was
/// invoked and the time it completed.
struct Completed {
    op: Op,
    invoked: u64,
}

pub struct Report {
    pub ops: usize,
    pub committed: usize,
    pub witnesses: Vec<String>,
}

impl Report {
    pub fn anomalous(&self) -> bool {
        !self.witnesses.is_empty()
    }
}

pub fn verify(path: &Path, model: Model) -> Result<Report, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let ops: Vec<Op> =
        serde_json::from_str(&raw).map_err(|e| format!("parse {}: {}", path.display(), e))?;

    let mut pending: HashMap<u64, u64> = HashMap::new();
    let mut committed: Vec<Completed> = Vec::new();
    for op in &ops {
        match op.kind {
            OpKind::Invoke => {
                pending.insert(op.process, op.time);
            }
            _ => {
                let invoked = pending.remove(&op.process).unwrap_or(op.time);
                if op.kind == OpKind::Ok {
                    committed.push(Completed {
                        op: op.clone(),
                        invoked,
                    });
                }
            }
        }
    }

    let witnesses = match model {
        Model::ListAppend => lost_appends(&committed),
        Model::RwRegister => stale_reads(&committed),
    };

    Ok(Report {
        ops: ops.len(),
        committed: committed.len(),
        witnesses,
    })
}

/// A committed append that a later transaction failed to observe, plus
/// the append that displaced it. The displacing append proves the read
/// side of a read-modify-write ran against a stale snapshot.
fn lost_appends(committed: &[Completed]) -> Vec<String> {
    let mut appended: Vec<(i64, i64, u64, u64)> = Vec::new();
    for entry in committed {
        for mop in &entry.op.value {
            if let ("append", MopVal::Int(v)) = (mop.0.as_str(), &mop.2) {
                appended.push((mop.1, *v, entry.op.time, entry.op.index));
            }
        }
    }

    let mut witnesses = Vec::new();
    for &(key, value, committed_at, index) in &appended {
        for entry in committed {
            if entry.invoked <= committed_at {
                continue;
            }
            for mop in &entry.op.value {
                let list = match (mop.0.as_str(), mop.1 == key, &mop.2) {
                    ("r", true, MopVal::List(list)) => list,
                    _ => continue,
                };
                if list.contains(&value) {
                    continue;
                }
                let displaced_by = appended
                    .iter()
                    .filter(|(k, v, at, _)| *k == key && *at > committed_at && list.contains(v))
                    .min_by_key(|(_, _, at, _)| *at);
                let blame = match displaced_by {
                    Some((_, v, at, idx)) => format!(
                        "; append {} on the same key committed later at t={} (index {}) \
                         and is present, so its read-modify-write started from a state \
                         that predates value {}",
                        v, at, idx, value
                    ),
                    None => String::new(),
                };
                witnesses.push(format!(
                    "lost update: append {} to key {} committed ok at t={} (index {}), \
                     but the read at index {} (invoked t={}) returned {:?} without it{}",
                    value, key, committed_at, index, entry.op.index, entry.invoked, list, blame
                ));
                break;
            }
            if witnesses.len() >= 32 {
                return witnesses;
            }
        }
    }
    witnesses
}

/// A read invoked after a write committed that returns a value older
/// than that write.
fn stale_reads(committed: &[Completed]) -> Vec<String> {
    let mut writes: HashMap<i64, Vec<(i64, u64, u64)>> = HashMap::new();
    for entry in committed {
        for mop in &entry.op.value {
            if let ("w", MopVal::Int(v)) = (mop.0.as_str(), &mop.2) {
                writes
                    .entry(mop.1)
                    .or_default()
                    .push((*v, entry.op.time, entry.op.index));
            }
        }
    }

    let mut witnesses = Vec::new();
    for entry in committed {
        for mop in &entry.op.value {
            if mop.0 != "r" {
                continue;
            }
            let key_writes = match writes.get(&mop.1) {
                Some(w) => w,
                None => continue,
            };
            let observed_at = match &mop.2 {
                MopVal::Null => Some(0),
                MopVal::Int(v) => key_writes
                    .iter()
                    .find(|(w, _, _)| w == v)
                    .map(|(_, at, _)| *at),
                MopVal::List(_) => None,
            };
            let observed_at = match observed_at {
                Some(at) => at,
                None => continue,
            };
            let newer = key_writes
                .iter()
                .filter(|(_, at, _)| *at > observed_at && *at < entry.invoked)
                .max_by_key(|(_, at, _)| *at);
            if let Some((w, at, idx)) = newer {
                witnesses.push(format!(
                    "stale read: index {} (invoked t={}) read key {} as {:?}, but write {} \
                     to that key committed ok at t={} (index {}) before the read was invoked",
                    entry.op.index, entry.invoked, mop.1, mop.2, w, at, idx
                ));
                if witnesses.len() >= 32 {
                    return witnesses;
                }
            }
        }
    }
    witnesses
}
