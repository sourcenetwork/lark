//! Wall-clock TTL wrapper on top of [`crate::Db`].
//!
//! Every value written through [`DbWithTtl`] carries a 4-byte Unix
//! timestamp appended at the end:
//!
//! ```text
//! stored = [user_value][timestamp: u32_be]
//! ```
//!
//! Reads strip the trailing 4 bytes before returning and compare the
//! embedded timestamp against `now - ttl`; expired entries read back
//! as `None` even if a compaction has not yet physically removed them.
//!
//! Physical reclamation is driven by a [`TtlCompactionFilter`] installed
//! in [`crate::Options::compaction_filter`] by [`DbWithTtl::open`]: as
//! entries flow through compaction, any whose embedded timestamp is
//! older than the TTL is dropped.
//!
//! # Year 2106 problem
//!
//! The 4-byte timestamp overflows on 2106-02-07. A format bump to
//! 8-byte timestamps is a future change.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::options::{CompactionDecision, CompactionFilter};
use crate::{Db, Error, Options, Result, WriteBatch};

/// Byte width of the trailing timestamp suffix.
const TS_LEN: usize = 4;

/// A [`Db`] wrapper that attaches a wall-clock TTL to every written
/// value. Entries whose embedded timestamp is older than `ttl_seconds`
/// read as `None` and are physically reclaimed at the next compaction.
///
/// `ttl_seconds == 0` disables expiration entirely — the wrapper still
/// appends the timestamp suffix for format consistency, but the filter
/// keeps every entry and reads never treat any value as expired.
pub struct DbWithTtl {
    inner: Db,
    ttl_seconds: u64,
}

impl std::fmt::Debug for DbWithTtl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbWithTtl")
            .field("ttl_seconds", &self.ttl_seconds)
            .finish_non_exhaustive()
    }
}

impl DbWithTtl {
    /// Open or create a TTL-enabled database at `path`. The caller's
    /// `opts` are forwarded to [`Db::open`] with one override: the
    /// compaction filter slot is replaced with a [`TtlCompactionFilter`]
    /// bound to `ttl_seconds`. Any user-supplied filter is dropped and
    /// logged via `tracing::warn`.
    pub fn open<P: AsRef<Path>>(path: P, mut opts: Options, ttl_seconds: u64) -> Result<Self> {
        if opts.compaction_filter.is_some() {
            tracing::warn!(
                "DbWithTtl::open replaces the caller's compaction_filter with TtlCompactionFilter"
            );
        }
        opts.compaction_filter = Some(Arc::new(TtlCompactionFilter { ttl_seconds }));
        let inner = Db::open(path, opts)?;
        Ok(Self { inner, ttl_seconds })
    }

    /// Write `key → value` with the current wall-clock timestamp
    /// appended. The in-memory value is the stamped form until a
    /// matching [`DbWithTtl::get`] strips it off.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let stamped = stamp(value, now_seconds());
        self.inner.put(key, &stamped)
    }

    /// Look up `key`. Returns `Ok(None)` if the key is absent or if
    /// its embedded timestamp has aged beyond `ttl_seconds`. The
    /// trailing timestamp is always stripped from the returned value.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        match self.inner.get(key)? {
            Some(stamped) => Ok(self.filter_expired(stamped)),
            None => Ok(None),
        }
    }

    /// Delete `key`. Range deletes are delegated to the inner
    /// [`Db::delete`] — tombstones don't carry timestamps.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
    }

    /// Delete every key in `[start, end)`. Same semantics as
    /// [`Db::delete_range`]; TTL has no effect on range tombstones.
    pub fn delete_range(&self, start: &[u8], end: &[u8]) -> Result<()> {
        self.inner.delete_range(start, end)
    }

    /// Scan `[start, end)`, returning `(key, value)` pairs where
    /// every value has its trailing timestamp stripped and every
    /// expired entry is omitted. Ordering matches [`Db::scan`].
    pub fn scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let raw = self.inner.scan(start, end)?;
        let horizon = self.expiry_horizon();
        let mut out = Vec::with_capacity(raw.len());
        for (k, stamped) in raw {
            if let Some(v) = strip_if_live(stamped, horizon) {
                out.push((k, v));
            }
        }
        Ok(out)
    }

    /// Apply a batch of point writes atomically. Each `put` in the
    /// batch gets its value stamped with the current timestamp; each
    /// `delete` is passed through unchanged.
    ///
    /// The input batch's original contents are consumed and the
    /// stamped/delegated version is applied.
    pub fn write(&self, batch: WriteBatch) -> Result<()> {
        let ts = now_seconds();
        let mut stamped_batch = WriteBatch::new();
        // Source batch keys are already CF-prefixed by the public
        // `put`/`delete`/... methods. Pass them through via raw
        // inserts so we don't double-prefix.
        for (key, value) in batch.ops_iter() {
            match value {
                Some(v) => stamped_batch.insert_raw_put(key.to_vec(), stamp(v, ts)),
                None => stamped_batch.insert_raw_delete(key.to_vec()),
            }
        }
        for (start, end) in batch.range_deletes_iter() {
            stamped_batch.insert_raw_range_delete(start.to_vec(), end.to_vec());
        }
        for (key, operand) in batch.merges_iter() {
            stamped_batch.insert_raw_merge(key.to_vec(), operand.to_vec());
        }
        self.inner.write(stamped_batch)
    }

    /// Borrow the underlying [`Db`] for APIs that [`DbWithTtl`] does
    /// not wrap (streaming iterator, snapshots, compact_range).
    ///
    /// Values read through the inner `Db` still carry their trailing
    /// timestamp — callers are responsible for stripping it via
    /// [`strip_timestamp`] if they want the user payload.
    pub fn inner(&self) -> &Db {
        &self.inner
    }

    /// Synchronously compact every SSTable overlapping `[start, end)`
    /// down the tree. Expired entries are physically removed via the
    /// installed [`TtlCompactionFilter`].
    pub fn compact_range(&self, start: Option<&[u8]>, end: Option<&[u8]>) -> Result<()> {
        self.inner.compact_range(start, end)
    }

    /// Flush to disk and shut down background threads.
    pub fn close(&self) -> Result<()> {
        self.inner.close()
    }

    /// Compute the "entries written before this Unix second have
    /// expired" threshold. Returns `0` when `ttl_seconds == 0` so
    /// every timestamp compares as live.
    fn expiry_horizon(&self) -> u32 {
        if self.ttl_seconds == 0 {
            return 0;
        }
        let now = now_seconds();
        now.saturating_sub(self.ttl_seconds as u32)
    }

    fn filter_expired(&self, stamped: Vec<u8>) -> Option<Vec<u8>> {
        strip_if_live(stamped, self.expiry_horizon())
    }
}

/// Return the raw user value with the trailing TTL timestamp
/// removed. Returns `None` if `stamped` is shorter than the suffix.
/// Callers that hold a `stamped` buffer from a raw read through
/// [`DbWithTtl::inner`] can use this to recover the user payload.
pub fn strip_timestamp(stamped: &[u8]) -> Option<&[u8]> {
    if stamped.len() < TS_LEN {
        return None;
    }
    Some(&stamped[..stamped.len() - TS_LEN])
}

fn strip_if_live(stamped: Vec<u8>, horizon: u32) -> Option<Vec<u8>> {
    if stamped.len() < TS_LEN {
        return None;
    }
    let ts = timestamp_of(&stamped);
    // horizon == 0 means "no expiration" (ttl=0). Otherwise a value
    // whose timestamp is strictly below the horizon has aged out.
    if horizon > 0 && ts < horizon {
        return None;
    }
    let mut out = stamped;
    out.truncate(out.len() - TS_LEN);
    Some(out)
}

fn stamp(value: &[u8], ts: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + TS_LEN);
    out.extend_from_slice(value);
    out.extend_from_slice(&ts.to_be_bytes());
    out
}

fn timestamp_of(stamped: &[u8]) -> u32 {
    let start = stamped.len() - TS_LEN;
    u32::from_be_bytes(stamped[start..].try_into().unwrap())
}

fn now_seconds() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

// ─── TtlCompactionFilter ────────────────────────────────────────────────────

/// Compaction filter that drops every point entry whose embedded
/// Unix timestamp is older than `ttl_seconds` at compaction time.
///
/// Installed automatically by [`DbWithTtl::open`]. Can be used
/// standalone if a caller prefers to manage the wrapper themselves.
pub struct TtlCompactionFilter {
    ttl_seconds: u64,
}

impl TtlCompactionFilter {
    /// Construct a filter that expires values older than
    /// `ttl_seconds`. A TTL of `0` is a no-op: every entry is kept
    /// regardless of its timestamp.
    pub fn new(ttl_seconds: u64) -> Self {
        Self { ttl_seconds }
    }
}

impl CompactionFilter for TtlCompactionFilter {
    fn filter(&self, _level: usize, _key: &[u8], value: &[u8]) -> CompactionDecision {
        if self.ttl_seconds == 0 || value.len() < TS_LEN {
            return CompactionDecision::Keep;
        }
        let ts = timestamp_of(value);
        let now = now_seconds();
        // Saturating arithmetic so a clock skew into the past does
        // not accidentally expire everything.
        let horizon = now.saturating_sub(self.ttl_seconds as u32);
        if ts < horizon {
            CompactionDecision::Remove
        } else {
            CompactionDecision::Keep
        }
    }

    fn name(&self) -> &'static str {
        "lark_ttl_filter"
    }
}

// Internal accessors on `WriteBatch` used by `DbWithTtl::write`.
//
// `WriteBatch` stores point ops in a `BTreeMap` and range deletes in a
// `Vec`. `DbWithTtl::write` needs to rebuild a parallel batch with
// stamped values, so we expose read-only iterators instead of moving
// the fields to the public API.
impl WriteBatch {
    pub(crate) fn ops_iter(&self) -> impl Iterator<Item = (&[u8], Option<&[u8]>)> {
        self.ops.iter().map(|(k, v)| (k.as_slice(), v.as_deref()))
    }

    pub(crate) fn range_deletes_iter(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        self.range_deletes
            .iter()
            .map(|(s, e)| (s.as_slice(), e.as_slice()))
    }

    pub(crate) fn merges_iter(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        self.merges
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
    }
}

// `Error` is plumbed via `crate::Result`; silence unused-import lint
// in case Error becomes unreferenced after a future refactor.
#[allow(dead_code)]
type _ErrorAlias = Error;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tiny_opts() -> Options {
        Options {
            write_buffer_size: 4 * 1024,
            ..Options::default()
        }
    }

    fn force_flush(db: &DbWithTtl, tag: &str) {
        let payload = vec![0u8; 512];
        for i in 0..32 {
            let key = format!("__flush_{}_{:04}", tag, i);
            db.put(key.as_bytes(), &payload).unwrap();
        }
    }

    #[test]
    fn test_ttl_basic_put_get() {
        let dir = TempDir::new().unwrap();
        // Ttl is long enough that nothing expires mid-test.
        let db = DbWithTtl::open(dir.path(), Options::default(), 3600).unwrap();
        db.put(b"key1", b"value1").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(db.get(b"missing").unwrap(), None);
    }

    #[test]
    fn test_ttl_zero_disables_expiration() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), Options::default(), 0).unwrap();
        db.put(b"k", b"v").unwrap();
        // Even after a compaction nothing should be dropped.
        db.compact_range(None, None).unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn test_ttl_expired_read_returns_none() {
        // Simulate expiry by writing a past timestamp directly via
        // the raw inner Db, bypassing `put`'s "stamp with now" path.
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), Options::default(), 10).unwrap();
        // `now - 100s` is well past the 10s TTL horizon.
        let ancient = now_seconds().saturating_sub(100);
        let stamped = stamp(b"stale", ancient);
        db.inner().put(b"old", &stamped).unwrap();
        // Read goes through DbWithTtl and filters the expired entry.
        assert_eq!(db.get(b"old").unwrap(), None);

        // A fresh put with the current timestamp still reads back.
        db.put(b"fresh", b"new").unwrap();
        assert_eq!(db.get(b"fresh").unwrap(), Some(b"new".to_vec()));
    }

    #[test]
    fn test_ttl_compaction_removes_expired() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), tiny_opts(), 10).unwrap();
        // Prime with 20 "stale" entries backdated past the horizon.
        let ancient = now_seconds().saturating_sub(100);
        for i in 0..20 {
            let key = format!("old_{i:02}");
            let stamped = stamp(b"payload", ancient);
            db.inner().put(key.as_bytes(), &stamped).unwrap();
        }
        // And 20 fresh entries written normally.
        for i in 0..20 {
            db.put(format!("new_{i:02}").as_bytes(), b"payload")
                .unwrap();
        }

        force_flush(&db, "ttl");
        db.compact_range(None, None).unwrap();

        // Reads via DbWithTtl: stale is gone, fresh survives.
        for i in 0..20 {
            assert_eq!(db.get(format!("old_{i:02}").as_bytes()).unwrap(), None);
            assert_eq!(
                db.get(format!("new_{i:02}").as_bytes()).unwrap(),
                Some(b"payload".to_vec())
            );
        }

        // Persistence check on the raw inner Db: stale entries are
        // physically gone after compaction, not just filtered on read.
        for i in 0..20 {
            assert_eq!(
                db.inner().get(format!("old_{i:02}").as_bytes()).unwrap(),
                None,
                "stale old_{i:02} should be physically removed"
            );
        }
    }

    #[test]
    fn test_ttl_scan_strips_timestamp_and_hides_expired() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), Options::default(), 10).unwrap();
        let ancient = now_seconds().saturating_sub(100);
        db.inner()
            .put(b"a_old", &stamp(b"stale_a", ancient))
            .unwrap();
        db.put(b"b_new", b"fresh_b").unwrap();
        db.inner()
            .put(b"c_old", &stamp(b"stale_c", ancient))
            .unwrap();
        db.put(b"d_new", b"fresh_d").unwrap();

        let pairs = db.scan(None, None).unwrap();
        // Only the fresh entries appear; their values are the
        // stripped user payloads.
        assert_eq!(
            pairs,
            vec![
                (b"b_new".to_vec(), b"fresh_b".to_vec()),
                (b"d_new".to_vec(), b"fresh_d".to_vec()),
            ]
        );
    }

    #[test]
    fn test_ttl_write_batch_stamps_every_put() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), Options::default(), 3600).unwrap();
        let mut batch = WriteBatch::new();
        batch.put(b"a", b"1");
        batch.put(b"b", b"2");
        batch.delete(b"ghost");
        db.write(batch).unwrap();

        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
        // Raw inner value carries the suffix, proving the stamp was
        // applied.
        let raw = db.inner().get(b"a").unwrap().unwrap();
        assert_eq!(raw.len(), 1 + TS_LEN);
        assert_eq!(strip_timestamp(&raw), Some(b"1".as_slice()));
    }

    #[test]
    fn test_ttl_strip_timestamp_helper() {
        let v = stamp(b"hello", 42);
        assert_eq!(strip_timestamp(&v), Some(b"hello".as_slice()));
        assert_eq!(strip_timestamp(b""), None);
        assert_eq!(strip_timestamp(b"abc"), None); // shorter than suffix
    }

    #[test]
    fn test_ttl_filter_standalone_noop_when_zero() {
        let filter = TtlCompactionFilter::new(0);
        let ancient = stamp(b"v", 0);
        assert_eq!(filter.filter(0, b"k", &ancient), CompactionDecision::Keep);
    }

    #[test]
    fn test_ttl_filter_removes_stale() {
        let filter = TtlCompactionFilter::new(10);
        let ancient = stamp(b"v", now_seconds().saturating_sub(100));
        assert_eq!(filter.filter(0, b"k", &ancient), CompactionDecision::Remove);
        let fresh = stamp(b"v", now_seconds());
        assert_eq!(filter.filter(0, b"k", &fresh), CompactionDecision::Keep);
    }
}
