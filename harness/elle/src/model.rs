//! Workload generation and execution against lark transactions.
//!
//! A transaction is planned before it runs and the plan is reused
//! across retries, so the append values in the `invoke` record always
//! match the ones in the completion record even when an optimistic
//! transaction has to be replayed.

use lark_kv::{
    Db, OptimisticTransactionDb, Options, Transaction, TransactionDb, TransactionError, TxResult,
};
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};

use crate::cli::{Isolation, Model};
use crate::history::{Mop, MopVal};

/// Deterministic splitmix64. A dependency-free RNG is enough here: the
/// workload only needs reproducible key and shape choices.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

/// Hands out the globally unique values Elle needs to reconstruct
/// version order. Each process gets a disjoint base so a child's values
/// can never collide with the parent's.
pub struct ValueSource(AtomicI64);

impl ValueSource {
    pub fn new(base: i64) -> Self {
        Self(AtomicI64::new(base))
    }

    pub fn next(&self) -> i64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

#[derive(Clone, Debug)]
pub enum PlannedMop {
    Append { key: i64, val: i64 },
    Read { key: i64 },
    Write { key: i64, val: i64 },
}

/// A planned transaction: the micro-operations plus the values they
/// will write, fixed before the first attempt.
#[derive(Clone, Debug)]
pub struct TxnPlan {
    pub mops: Vec<PlannedMop>,
}

impl TxnPlan {
    pub fn generate(model: Model, keys: i64, rng: &mut Rng, values: &ValueSource) -> Self {
        let count = 1 + rng.below(3) as usize;
        let mut mops = Vec::with_capacity(count);
        for _ in 0..count {
            let key = rng.below(keys as u64) as i64;
            let write = rng.below(3) != 0;
            mops.push(match (model, write) {
                (Model::ListAppend, true) => PlannedMop::Append {
                    key,
                    val: values.next(),
                },
                (Model::RwRegister, true) => PlannedMop::Write {
                    key,
                    val: values.next(),
                },
                (_, false) => PlannedMop::Read { key },
            });
        }
        Self { mops }
    }

    /// Read-only transaction over every key, used for the post-recovery
    /// read-back phases.
    pub fn read_all(keys: i64) -> Self {
        Self {
            mops: (0..keys).map(|key| PlannedMop::Read { key }).collect(),
        }
    }

    /// The `invoke` value: writes carry their value, reads carry null.
    pub fn invoke_value(&self) -> Vec<Mop> {
        self.mops
            .iter()
            .map(|mop| match mop {
                PlannedMop::Append { key, val } => Mop("append".into(), *key, MopVal::Int(*val)),
                PlannedMop::Write { key, val } => Mop("w".into(), *key, MopVal::Int(*val)),
                PlannedMop::Read { key } => Mop("r".into(), *key, MopVal::Null),
            })
            .collect()
    }

    /// Run every micro-operation inside one lark transaction and return
    /// the completion value, with observed results filled into reads.
    ///
    /// Appends are read-modify-write against `get_for_update`, which is
    /// the API a caller reaches for precisely to make an increment safe.
    pub fn execute(&self, model: Model, tx: &mut Transaction<'_>) -> TxResult<Vec<Mop>> {
        let mut observed = Vec::with_capacity(self.mops.len());
        for mop in &self.mops {
            match mop {
                PlannedMop::Append { key, val } => {
                    let current = tx.get_for_update(&key_bytes(*key))?;
                    let mut list = decode_list(current.as_deref()).unwrap_or_default();
                    list.push(*val);
                    tx.put(&key_bytes(*key), &encode_list(&list))?;
                    observed.push(Mop("append".into(), *key, MopVal::Int(*val)));
                }
                PlannedMop::Write { key, val } => {
                    tx.put(&key_bytes(*key), val.to_string().as_bytes())?;
                    observed.push(Mop("w".into(), *key, MopVal::Int(*val)));
                }
                PlannedMop::Read { key } => {
                    let current = tx.get(&key_bytes(*key))?;
                    let value = match model {
                        Model::ListAppend => match decode_list(current.as_deref()) {
                            Some(list) => MopVal::List(list),
                            None => MopVal::Null,
                        },
                        Model::RwRegister => match decode_int(current.as_deref()) {
                            Some(v) => MopVal::Int(v),
                            None => MopVal::Null,
                        },
                    };
                    observed.push(Mop("r".into(), *key, value));
                }
            }
        }
        Ok(observed)
    }
}

/// The transaction flavor the requested isolation level maps onto.
pub enum TxDb {
    Pessimistic(TransactionDb),
    Optimistic(OptimisticTransactionDb),
}

impl TxDb {
    pub fn open(path: &Path, isolation: Isolation, opts: Options) -> lark_kv::Result<Self> {
        match isolation {
            Isolation::ReadCommitted => Ok(TxDb::Pessimistic(TransactionDb::open(path, opts)?)),
            Isolation::RepeatableRead | Isolation::Serializable => {
                Ok(TxDb::Optimistic(OptimisticTransactionDb::open(path, opts)?))
            }
        }
    }

    pub fn begin(&self) -> Transaction<'_> {
        match self {
            TxDb::Pessimistic(db) => db.begin_transaction(),
            TxDb::Optimistic(db) => db.begin_transaction(),
        }
    }

    pub fn db(&self) -> &Db {
        match self {
            TxDb::Pessimistic(db) => db.db(),
            TxDb::Optimistic(db) => db.db(),
        }
    }

    /// Optimistic transactions surface write-write conflicts to the
    /// caller, so the client retries them; pessimistic ones resolve
    /// contention by blocking and never report a conflict.
    pub fn retries(&self) -> u32 {
        match self {
            TxDb::Pessimistic(_) => 1,
            TxDb::Optimistic(_) => 16,
        }
    }
}

/// How a transaction attempt ended, mapped to a history record type.
pub enum Outcome {
    Committed(Vec<Mop>),
    /// Definitely not committed: rolled back after a conflict or a
    /// lock timeout.
    Aborted,
    /// Indeterminate: the commit itself failed with an I/O error, so
    /// the write may or may not be durable.
    Unknown,
}

pub fn run_txn(db: &TxDb, model: Model, plan: &TxnPlan) -> Outcome {
    let mut last_retryable = false;
    for _ in 0..db.retries() {
        let mut tx = db.begin();
        match plan.execute(model, &mut tx) {
            Ok(observed) => match tx.commit() {
                Ok(()) => return Outcome::Committed(observed),
                Err(TransactionError::Conflict { .. }) | Err(TransactionError::Busy(_)) => {
                    last_retryable = true;
                }
                Err(_) => return Outcome::Unknown,
            },
            Err(TransactionError::Conflict { .. }) | Err(TransactionError::Busy(_)) => {
                tx.rollback();
                last_retryable = true;
            }
            Err(_) => {
                tx.rollback();
                return Outcome::Unknown;
            }
        }
    }
    if last_retryable {
        Outcome::Aborted
    } else {
        Outcome::Unknown
    }
}

pub fn key_bytes(key: i64) -> Vec<u8> {
    format!("k{:08}", key).into_bytes()
}

fn encode_list(list: &[i64]) -> Vec<u8> {
    let mut out = String::new();
    for (i, v) in list.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.into_bytes()
}

fn decode_list(raw: Option<&[u8]>) -> Option<Vec<i64>> {
    let text = std::str::from_utf8(raw?).ok()?;
    if text.is_empty() {
        return Some(Vec::new());
    }
    text.split(',')
        .map(|part| part.parse::<i64>().ok())
        .collect()
}

fn decode_int(raw: Option<&[u8]>) -> Option<i64> {
    std::str::from_utf8(raw?).ok()?.parse::<i64>().ok()
}
