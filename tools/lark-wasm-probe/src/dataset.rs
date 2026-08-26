//! The deterministic dataset the lifecycle writes, and the single
//! predicate that says what the database must contain at the end.
//!
//! Every byte is a pure function of a record index, so the read-back
//! phase after a reopen recomputes what it expects instead of holding
//! a reference copy. That matters here: a reference `HashMap` of the
//! whole dataset would itself move the linear-memory high-water mark
//! the probe exists to report.
//!
//! Key namespaces are ordered on purpose - `b/` < `k/` < `s/` < `z/` -
//! so the iterator phase can assert an exact first and last key.

/// Bytes in every generated value.
pub const VALUE_LEN: usize = 64;

/// Bulk records written in the write phase must be at least this many.
/// Below it the deleted index, the range-deleted band, and the
/// overwritten index start to collide.
pub const MIN_RECORDS: u64 = 128;

/// Bulk index whose value is replaced after the initial write.
pub const OVERWRITE_INDEX: u64 = 7;

/// Added to the index when generating the replacement value, so an
/// engine that served the pre-overwrite version is caught.
pub const OVERWRITE_TAG: u64 = 1_000_000;

/// Number of consecutive bulk records removed by `delete_range`.
pub const RANGE_DELETE_LEN: u64 = 16;

/// Batch-written records, one of which the same batch deletes.
pub const BATCH_RECORDS: u64 = 8;

/// Index inside the batch that the batch itself deletes.
pub const BATCH_DELETED: u64 = 3;

/// Key for bulk record `i`. Fixed width so lexicographic and numeric
/// order agree, which the scan and iterator phases rely on.
pub fn bulk_key(i: u64) -> Vec<u8> {
    format!("k/{i:08}").into_bytes()
}

/// Key for batch record `i`.
pub fn batch_key(i: u64) -> Vec<u8> {
    format!("b/{i:04}").into_bytes()
}

/// Key for smoke record `i`. Every smoke key is deleted before the
/// database is closed.
pub fn smoke_key(i: u64) -> Vec<u8> {
    format!("s/smoke-{i}").into_bytes()
}

/// Key written while a snapshot is held, to prove the snapshot does
/// not see it and that it survives the reopen.
pub fn post_snapshot_key() -> Vec<u8> {
    b"z/after-snapshot".to_vec()
}

/// Value for record `i`: the index in the first eight bytes, then a
/// fill byte derived from it. A truncated or torn read fails the
/// length check; a wrong record fails the content check.
pub fn value(i: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(VALUE_LEN);
    v.extend_from_slice(&i.to_le_bytes());
    let fill = (i % 251) as u8;
    v.resize(VALUE_LEN, fill);
    v
}

/// First bulk index at or after `i` that is still live, or `None` when
/// none is.
pub fn first_live_at_or_after(i: u64, records: u64) -> Option<u64> {
    (i..records).find(|&candidate| bulk_expected(candidate, records).is_some())
}

/// What bulk record `i` must hold once every write phase has run, or
/// `None` when it must be absent.
///
/// This is the only place the expected end state is defined. The write
/// phases and the post-reopen read-back both consult it, so the two
/// cannot drift.
pub fn bulk_expected(i: u64, records: u64) -> Option<Vec<u8>> {
    let range_start = records / 4;
    if i == records / 2 {
        return None;
    }
    if i >= range_start && i < range_start + RANGE_DELETE_LEN {
        return None;
    }
    if i == OVERWRITE_INDEX {
        return Some(value(i + OVERWRITE_TAG));
    }
    Some(value(i))
}

/// What batch record `i` must hold, or `None` when the batch deleted
/// it.
pub fn batch_expected(i: u64) -> Option<Vec<u8>> {
    if i == BATCH_DELETED {
        None
    } else {
        Some(value(i))
    }
}

/// Verify `got` matches `want` exactly, naming the first discrepancy.
pub fn check(label: &str, want: &[u8], got: &[u8]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{label}: read back {} bytes, expected {}",
            got.len(),
            want.len()
        ));
    }
    if let Some(at) = got.iter().zip(want.iter()).position(|(a, b)| a != b) {
        return Err(format!(
            "{label}: byte {at} is {}, expected {}",
            got[at], want[at]
        ));
    }
    Ok(())
}
