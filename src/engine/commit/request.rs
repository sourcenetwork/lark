//! What a writer hands to the commit leader, and how the leader turns it
//! into WAL bytes and memtable entries.

use super::super::wal::{encode_ops_record, encode_put_record, ops_record_len, put_record_len};
use super::super::{DurabilityMode, MemTable, apply_batch_op_to_memtable, batch_op_wal_bytes};
use crate::WriteBatchOp;

/// The unit of work one writer hands to the commit leader.
///
/// Every byte a request refers to is owned by the request, never borrowed
/// from the submitting frame. That is what makes G3 hold: the leader can
/// finish a ticket whose submitter has already unwound.
pub(crate) enum WriteRequest {
    /// No work. The resting state of a handoff slot, and what the leader
    /// leaves behind when it takes a request.
    Idle,
    /// A single put, the shape [`crate::Db::put`] produces.
    Put {
        /// Column-family-prefixed key.
        key: Vec<u8>,
        /// Value bytes.
        value: Vec<u8>,
        /// WAL fsync policy this writer asked for.
        durability: DurabilityMode,
        /// Skip the WAL entirely for this writer's operations.
        disable_wal: bool,
    },
    /// An ordered batch applied atomically.
    Batch {
        /// Operations in the order the caller recorded them.
        ops: Vec<WriteBatchOp>,
        /// WAL fsync policy this writer asked for.
        durability: DurabilityMode,
        /// Skip the WAL entirely for this writer's operations.
        disable_wal: bool,
    },
}

impl WriteRequest {
    /// How many sequence numbers this request consumes.
    pub(super) fn op_count(&self) -> u64 {
        match self {
            WriteRequest::Idle => 0,
            WriteRequest::Put { .. } => 1,
            WriteRequest::Batch { ops, .. } => ops.len() as u64,
        }
    }

    pub(super) fn durability(&self) -> DurabilityMode {
        match self {
            WriteRequest::Idle => DurabilityMode::Eventual,
            WriteRequest::Put { durability, .. } | WriteRequest::Batch { durability, .. } => {
                *durability
            }
        }
    }

    pub(super) fn skips_wal(&self) -> bool {
        match self {
            WriteRequest::Idle => true,
            WriteRequest::Put { disable_wal, .. } | WriteRequest::Batch { disable_wal, .. } => {
                *disable_wal
            }
        }
    }

    /// Framed WAL bytes this request will stage. Used only to cap a group.
    pub(super) fn staged_len(&self) -> usize {
        if self.skips_wal() {
            return 0;
        }
        match self {
            WriteRequest::Idle => 0,
            WriteRequest::Put { key, value, .. } => put_record_len(key, value),
            WriteRequest::Batch { ops, .. } => ops_record_len(ops),
        }
    }

    /// Most memtable bytes this request can add when it is applied.
    ///
    /// The leader admits tickets against the active memtable's remaining
    /// `write_buffer_size`, so this has to be an upper bound: over-stating
    /// a cost ends a group one ticket early, while under-stating one lets
    /// the group carry the memtable past its budget.
    ///
    /// A `DeleteRange` op costs range-tombstone heap rather than an arena
    /// node, and is charged the same way [`MemTable::delete_range`]
    /// charges it.
    pub(super) fn memtable_cost(&self) -> usize {
        match self {
            WriteRequest::Idle => 0,
            WriteRequest::Put { key, value, .. } => {
                MemTable::max_entry_size(key.len(), value.len())
            }
            WriteRequest::Batch { ops, .. } => ops.iter().map(batch_op_memtable_cost).sum(),
        }
    }

    /// Payload bytes reported to [`Ticker::WalBytesWritten`]. Counts the
    /// same key, value and sequence bytes the per-write path counted
    /// before group commit, so the ticker stays comparable across the
    /// change.
    pub(super) fn reported_bytes(&self) -> u64 {
        match self {
            WriteRequest::Idle => 0,
            WriteRequest::Put { key, value, .. } => (key.len() + value.len() + 8) as u64,
            WriteRequest::Batch { ops, .. } => ops.iter().map(batch_op_wal_bytes).sum(),
        }
    }

    pub(super) fn encode_wal(&self, out: &mut Vec<u8>, base_seq: u64) {
        match self {
            WriteRequest::Idle => {}
            WriteRequest::Put { key, value, .. } => encode_put_record(out, key, value, base_seq),
            WriteRequest::Batch { ops, .. } => encode_ops_record(out, ops, base_seq),
        }
    }

    pub(super) fn apply(&self, memtable: &MemTable, seq: &mut u64) {
        match self {
            WriteRequest::Idle => {}
            WriteRequest::Put { key, value, .. } => {
                memtable.put(key, value, *seq);
                *seq += 1;
            }
            WriteRequest::Batch { ops, .. } => {
                for op in ops {
                    apply_batch_op_to_memtable(memtable, op, *seq);
                    *seq += 1;
                }
            }
        }
    }
}

/// Upper bound on the memtable bytes one batch operation can add.
fn batch_op_memtable_cost(op: &WriteBatchOp) -> usize {
    match op {
        WriteBatchOp::Put { key, value } => MemTable::max_entry_size(key.len(), value.len()),
        WriteBatchOp::Delete { key } => MemTable::max_entry_size(key.len(), 0),
        WriteBatchOp::Merge { key, operand } => MemTable::max_entry_size(key.len(), operand.len()),
        WriteBatchOp::DeleteRange { start, end } => {
            start.len() + end.len() + size_of::<crate::engine::range_tombstone::RangeTombstone>()
        }
    }
}

/// Shape only, never payload bytes: a request's key and value are user
/// data and have no business in a panic message or a log line.
impl std::fmt::Debug for WriteRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteRequest::Idle => f.write_str("WriteRequest::Idle"),
            WriteRequest::Put {
                key,
                value,
                durability,
                disable_wal,
            } => f
                .debug_struct("WriteRequest::Put")
                .field("key_len", &key.len())
                .field("value_len", &value.len())
                .field("durability", durability)
                .field("disable_wal", disable_wal)
                .finish(),
            WriteRequest::Batch {
                ops,
                durability,
                disable_wal,
            } => f
                .debug_struct("WriteRequest::Batch")
                .field("ops", &ops.len())
                .field("durability", durability)
                .field("disable_wal", disable_wal)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(key: &[u8], value: &[u8]) -> WriteRequest {
        WriteRequest::Put {
            key: key.to_vec(),
            value: value.to_vec(),
            durability: DurabilityMode::Eventual,
            disable_wal: false,
        }
    }

    #[test]
    fn op_count_matches_the_sequence_numbers_a_request_consumes() {
        assert_eq!(WriteRequest::Idle.op_count(), 0);
        assert_eq!(put(b"k", b"v").op_count(), 1);
        assert_eq!(
            WriteRequest::Batch {
                ops: vec![
                    WriteBatchOp::Put {
                        key: b"a".to_vec(),
                        value: b"1".to_vec()
                    },
                    WriteBatchOp::Delete { key: b"b".to_vec() },
                ],
                durability: DurabilityMode::Eventual,
                disable_wal: false,
            }
            .op_count(),
            2
        );
    }

    #[test]
    fn staged_len_matches_the_bytes_actually_encoded() {
        let request = put(b"key", b"value");
        let mut out = Vec::new();
        request.encode_wal(&mut out, 7);
        assert_eq!(out.len(), request.staged_len());

        let batch = WriteRequest::Batch {
            ops: vec![
                WriteBatchOp::Put {
                    key: b"a".to_vec(),
                    value: b"1".to_vec(),
                },
                WriteBatchOp::Merge {
                    key: b"b".to_vec(),
                    operand: b"2".to_vec(),
                },
            ],
            durability: DurabilityMode::Immediate,
            disable_wal: false,
        };
        let mut out = Vec::new();
        batch.encode_wal(&mut out, 3);
        assert_eq!(out.len(), batch.staged_len());
    }

    #[test]
    fn a_wal_disabled_request_stages_nothing() {
        let request = WriteRequest::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            durability: DurabilityMode::Immediate,
            disable_wal: true,
        };
        assert_eq!(request.staged_len(), 0);
        assert!(request.skips_wal());
    }
}
