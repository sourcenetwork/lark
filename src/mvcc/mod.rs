//! Transactions on top of `kovan_mvcc`.
//!
//! kovan-mvcc owns the protocol: timestamps, prewrite, commit, conflict
//! detection and isolation. This module owns only the storage it drives,
//! so the transaction semantics come from one implementation rather than
//! from two that have to agree.
pub(crate) mod key;
pub(crate) mod layout;
pub(crate) mod storage;
pub(crate) mod txn;
