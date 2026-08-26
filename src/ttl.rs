//! Wall-clock TTL wrapper on top of [`crate::Db`].
//!
//! Every value written through [`DbWithTtl`] carries a versioned TTL
//! suffix appended at the end:
//!
//! ```text
//! stored = [user_value]["LTTL"][version: u8][timestamp: u64_be]
//! ```
//!
//! Reads strip the trailing suffix before returning and compare the
//! embedded 64-bit Unix timestamp against `now - ttl`; expired entries
//! read back as `None` even if a compaction has not yet physically
//! removed them.
//!
//! Physical reclamation is driven by a [`TtlCompactionFilter`] installed
//! in [`crate::Options::compaction_filter`] by [`DbWithTtl::open`]: as
//! entries flow through compaction, any whose embedded timestamp is
//! older than the TTL is dropped.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::options::{CompactionDecision, CompactionFilter};
use crate::{Db, DbSlice, Error, Options, Result, WriteBatch, WriteBatchOp};

const LEGACY_TS_LEN: usize = 4;
const TTL_MAGIC: [u8; 4] = *b"LTTL";
const TTL_MAGIC_LEN: usize = 4;
const TTL_FORMAT_VERSION: u8 = 1;
const TTL_TS_LEN: usize = 8;
const TTL_SUFFIX_LEN: usize = TTL_MAGIC_LEN + 1 + TTL_TS_LEN;

/// A [`Db`] wrapper that attaches a wall-clock TTL to every written
/// value. Entries whose embedded timestamp is older than `ttl_seconds`
/// read as `None` and are physically reclaimed at the next compaction.
///
/// `ttl_seconds == 0` disables expiration entirely - the wrapper still
/// appends the TTL suffix for format consistency, but the filter
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
    /// trailing TTL suffix is always stripped from the returned value.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        match self.inner.get(key)? {
            Some(stamped) => Ok(self.filter_expired(stamped)),
            None => Ok(None),
        }
    }

    /// [`DbWithTtl::get`] without copying the value.
    ///
    /// The TTL suffix is dropped by narrowing the view, so the result
    /// still borrows the bytes the database already owns. Same absence
    /// semantics as [`DbWithTtl::get`]: an expired entry, and a value
    /// with no decodable TTL suffix, both read as `Ok(None)`.
    pub fn get_slice(&self, key: &[u8]) -> Result<Option<DbSlice>> {
        let Some(stamped) = self.inner.get_slice(key)? else {
            return Ok(None);
        };
        let Some(decoded) = decode_ttl_suffix(stamped.as_slice()) else {
            return Ok(None);
        };
        let horizon = self.expiry_horizon();
        if horizon > 0 && decoded.timestamp < horizon {
            return Ok(None);
        }
        Ok(stamped.try_subslice(0..decoded.value_len))
    }

    /// Delete `key`. Range deletes are delegated to the inner
    /// [`Db::delete`] - tombstones don't carry timestamps.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
    }

    /// Delete every key in `[start, end)`. Same semantics as
    /// [`Db::delete_range`]; TTL has no effect on range tombstones.
    pub fn delete_range(&self, start: &[u8], end: &[u8]) -> Result<()> {
        self.inner.delete_range(start, end)
    }

    /// Scan `[start, end)`, returning `(key, value)` pairs where
    /// every value has its trailing TTL suffix stripped and every
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
        for op in batch.ops_iter() {
            match op {
                WriteBatchOp::Put { key, value } => {
                    stamped_batch.insert_raw_put(key.clone(), stamp(value, ts));
                }
                WriteBatchOp::Delete { key } => {
                    stamped_batch.insert_raw_delete(key.clone());
                }
                WriteBatchOp::DeleteRange { start, end } => {
                    stamped_batch.insert_raw_range_delete(start.clone(), end.clone());
                }
                WriteBatchOp::Merge { key, operand } => {
                    stamped_batch.insert_raw_merge(key.clone(), operand.clone());
                }
            }
        }
        self.inner.write(stamped_batch)
    }

    /// Borrow the underlying [`Db`] for APIs that [`DbWithTtl`] does
    /// not wrap (streaming iterator, snapshots, compact_range).
    ///
    /// Values read through the inner `Db` still carry their trailing
    /// TTL suffix - callers are responsible for stripping it via
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
    fn expiry_horizon(&self) -> u64 {
        if self.ttl_seconds == 0 {
            return 0;
        }
        let now = now_seconds();
        now.saturating_sub(self.ttl_seconds)
    }

    fn filter_expired(&self, stamped: Vec<u8>) -> Option<Vec<u8>> {
        strip_if_live(stamped, self.expiry_horizon())
    }
}

/// Return the raw user value with the trailing TTL suffix removed.
/// Returns `None` if `stamped` does not contain a supported suffix.
/// Callers that hold a `stamped` buffer from a raw read through
/// [`DbWithTtl::inner`] can use this to recover the user payload.
pub fn strip_timestamp(stamped: &[u8]) -> Option<&[u8]> {
    decode_ttl_suffix(stamped).map(|decoded| &stamped[..decoded.value_len])
}

fn strip_if_live(stamped: Vec<u8>, horizon: u64) -> Option<Vec<u8>> {
    let decoded = decode_ttl_suffix(&stamped)?;
    // horizon == 0 means "no expiration" (ttl=0). Otherwise a value
    // whose timestamp is strictly below the horizon has aged out.
    if horizon > 0 && decoded.timestamp < horizon {
        return None;
    }
    let mut out = stamped;
    out.truncate(decoded.value_len);
    Some(out)
}

fn stamp(value: &[u8], ts: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + TTL_SUFFIX_LEN);
    out.extend_from_slice(value);
    out.extend_from_slice(&TTL_MAGIC);
    out.push(TTL_FORMAT_VERSION);
    out.extend_from_slice(&ts.to_be_bytes());
    out
}

fn timestamp_of(stamped: &[u8]) -> Option<u64> {
    decode_ttl_suffix(stamped).map(|decoded| decoded.timestamp)
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct DecodedTtlSuffix {
    value_len: usize,
    timestamp: u64,
}

fn decode_ttl_suffix(stamped: &[u8]) -> Option<DecodedTtlSuffix> {
    if stamped.len() >= TTL_SUFFIX_LEN {
        let suffix_start = stamped.len() - TTL_SUFFIX_LEN;
        let suffix = &stamped[suffix_start..];
        if suffix[..TTL_MAGIC_LEN] == TTL_MAGIC {
            if suffix[TTL_MAGIC_LEN] != TTL_FORMAT_VERSION {
                return None;
            }
            let ts_start = TTL_MAGIC_LEN + 1;
            let timestamp =
                u64::from_be_bytes(suffix[ts_start..ts_start + TTL_TS_LEN].try_into().unwrap());
            return Some(DecodedTtlSuffix {
                value_len: suffix_start,
                timestamp,
            });
        }
    }

    if stamped.len() < LEGACY_TS_LEN {
        return None;
    }

    let start = stamped.len() - LEGACY_TS_LEN;
    let timestamp = u32::from_be_bytes(stamped[start..].try_into().unwrap()) as u64;
    Some(DecodedTtlSuffix {
        value_len: start,
        timestamp,
    })
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
        if self.ttl_seconds == 0 {
            return CompactionDecision::Keep;
        }
        let Some(ts) = timestamp_of(value) else {
            return CompactionDecision::Keep;
        };
        let now = now_seconds();
        // Saturating arithmetic so a clock skew into the past does
        // not accidentally expire everything.
        let horizon = now.saturating_sub(self.ttl_seconds);
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
// `DbWithTtl::write` needs to rebuild a parallel batch with stamped
// values, so expose a read-only ordered iterator instead of moving
// the storage field to the public API.
impl WriteBatch {
    pub(crate) fn ops_iter(&self) -> impl Iterator<Item = &WriteBatchOp> {
        self.ops.iter()
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

    fn legacy_stamp(value: &[u8], ts: u32) -> Vec<u8> {
        let mut out = Vec::from(value);
        out.extend_from_slice(&ts.to_be_bytes());
        out
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
        assert_eq!(raw.len(), 1 + TTL_SUFFIX_LEN);
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
    fn test_ttl_timestamp_uses_u64_encoding() {
        let far_future = u32::MAX as u64 + 100;
        let stamped = stamp(b"future", far_future);
        assert_eq!(timestamp_of(&stamped), Some(far_future));
        assert_eq!(strip_timestamp(&stamped), Some(b"future".as_slice()));
    }

    #[test]
    fn test_ttl_strip_timestamp_accepts_legacy_u32_suffix() {
        let legacy = legacy_stamp(b"hello", 42);
        assert_eq!(timestamp_of(&legacy), Some(42));
        assert_eq!(strip_timestamp(&legacy), Some(b"hello".as_slice()));
    }

    #[test]
    fn test_ttl_legacy_u32_suffix_still_expires() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), Options::default(), 10).unwrap();
        db.inner()
            .put(b"old_legacy", &legacy_stamp(b"stale", 1))
            .unwrap();
        assert_eq!(db.get(b"old_legacy").unwrap(), None);

        let filter = TtlCompactionFilter::new(10);
        assert_eq!(
            filter.filter(0, b"k", &legacy_stamp(b"stale", 1)),
            CompactionDecision::Remove
        );
    }

    #[test]
    fn get_slice_strips_the_suffix_without_copying() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), tiny_opts(), 0).unwrap();
        db.put(b"k", b"payload").unwrap();

        let slice = db.get_slice(b"k").unwrap().expect("present");
        assert_eq!(slice.as_slice(), b"payload");
        assert_eq!(Some(slice.to_vec()), db.get(b"k").unwrap());
        assert_eq!(db.get_slice(b"absent").unwrap(), None);
    }

    #[test]
    fn get_slice_hides_an_expired_entry() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), tiny_opts(), 1).unwrap();
        // Write straight through the inner `Db` with an ancient stamp so
        // the test does not have to wait a wall-clock second.
        db.inner()
            .put(
                b"stale",
                &stamp(b"payload", now_seconds().saturating_sub(1000)),
            )
            .unwrap();
        assert_eq!(db.get_slice(b"stale").unwrap(), None);
        assert_eq!(db.get(b"stale").unwrap(), None);
    }

    #[test]
    fn get_slice_hides_a_value_too_short_to_carry_a_suffix() {
        let dir = TempDir::new().unwrap();
        let db = DbWithTtl::open(dir.path(), tiny_opts(), 0).unwrap();
        db.inner().put(b"raw", b"ab").unwrap();
        assert_eq!(db.get_slice(b"raw").unwrap(), None);
        assert_eq!(db.get(b"raw").unwrap(), None);
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
