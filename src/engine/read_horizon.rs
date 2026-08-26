//! The published read horizon: the newest sequence whose data is both
//! durable and applied.
//!
//! # Why it is a type
//!
//! Two places have to agree about this number and they are on opposite
//! sides of a memory-ordering argument: the commit pipeline publishes it
//! after a group's WAL bytes are synced and every operation is in the
//! memtable, and every snapshot reads it before walking the memtable. If
//! either side used the wrong ordering, a snapshot could observe a
//! sequence whose memtable insert it cannot yet see, and report a
//! committed key as absent. Keeping both sides in one type means the
//! pair cannot drift, and it gives `tests/loom_memtable.rs` something
//! real to model-check rather than a transcription of the protocol.
//!
//! # Invariants
//!
//! - **H1 (release).** [`ReadHorizon::publish`] is an `AcqRel`
//!   read-modify-write, so every write the publishing thread performed
//!   first - the WAL append, the sync, the memtable inserts -
//!   happens-before any `Acquire` load that observes the new value.
//! - **H2 (acquire).** [`ReadHorizon::visible`] loads with `Acquire`, so
//!   a reader that observes sequence `s` also observes every memtable
//!   insert that the publisher of `s` had already made.
//! - **H3 (monotonic).** `publish` is a `fetch_max`, never a `store`, so
//!   two writers finishing out of order can never move the horizon
//!   backwards and expose a hole. [`ReadHorizon::reset`] is the one
//!   exception and it is only reachable from `drop_all`, which holds the
//!   pipeline mutex and has already discarded every memtable.

use super::sync::{AtomicU64, Ordering};

/// The published read horizon. See the module documentation for the
/// ordering invariants H1 to H3.
#[derive(Debug)]
pub(crate) struct ReadHorizon(AtomicU64);

impl ReadHorizon {
    /// A horizon that starts at `seq`, the last sequence recovery
    /// established as durable.
    pub(crate) fn new(seq: u64) -> Self {
        Self(AtomicU64::new(seq))
    }

    /// The newest sequence a snapshot may read (H2).
    pub(crate) fn visible(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    /// Publish `seq` as durable and applied (H1, H3).
    ///
    /// Never lowers the horizon, so a slow writer finishing after a
    /// faster one cannot re-hide the faster one's data.
    pub(crate) fn publish(&self, seq: u64) {
        self.0.fetch_max(seq, Ordering::AcqRel);
    }

    /// Drop the horizon back to `0` after every memtable and file has
    /// been discarded (H3).
    pub(crate) fn reset(&self) {
        self.0.store(0, Ordering::Release);
    }
}
