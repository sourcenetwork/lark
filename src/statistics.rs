//! Engine-wide counters and histograms for observability.
//!
//! A caller configures [`crate::Options::statistics`] with an
//! [`Arc<Statistics>`] and the engine increments tickers and
//! records histograms on every hot path it has instrumented. The
//! caller then polls the `Statistics` object (via
//! [`Statistics::get_ticker`], [`Statistics::get_histogram_snapshot`],
//! or [`Statistics::to_string`]) to export the values to their
//! monitoring stack of choice.
//!
//! # Cost when disabled
//!
//! `Options::statistics = None` short-circuits every instrumentation
//! site behind an `Option::is_some` check plus an `Arc` clone at
//! open time. The hot-path overhead is a single branch. Reaching
//! for a non-`None` statistics object adds one `fetch_add` per
//! ticker update and a short mutex for histogram updates.
//!
//! # Histograms
//!
//! The initial histogram implementation tracks `count`, `sum`,
//! `min`, and `max` only. `HistogramSnapshot::average` gives you
//! `sum / count`. Percentiles / buckets are intentionally out of
//! scope for v1 - adding an HDR-style bucket array is a follow-up
//! that drops in behind the existing API without breaking
//! callers.

use crate::portability::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Enumerated counters incremented by the engine. Every variant
/// is backed by one `AtomicU64` slot in [`Statistics`]; looking
/// up a ticker is `O(1)` and thread-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Ticker {
    /// Total key + value bytes written to the memtable. Counts
    /// raw user bytes; does not include internal encoding
    /// overhead, the CF prefix, or WAL framing.
    BytesWritten = 0,
    /// Total value bytes returned from point lookups that found
    /// a live value.
    BytesRead = 1,
    /// Number of successful `put` operations routed through
    /// `apply_batch`.
    KeysWritten = 2,
    /// Number of `get` calls (including lookups that returned
    /// `None`).
    KeysRead = 3,
    /// Number of `delete` operations.
    KeysDeleted = 4,
    /// Number of range-delete operations.
    RangeDeletesWritten = 5,
    /// Number of merge operands written.
    MergesWritten = 6,
    /// Block cache hits (the block was already resident).
    BlockCacheHit = 7,
    /// Block cache misses (the block had to be fetched from disk).
    BlockCacheMiss = 8,
    /// Block cache insertions (one per miss that actually
    /// populated the cache).
    BlockCacheAdd = 9,
    /// Bloom-filter "useful" hits - the filter correctly
    /// answered "not present" and spared a block read.
    BloomFilterUseful = 10,
    /// Bloom-filter "full positive" hits - the filter said
    /// "maybe", and the key was actually present in the block.
    BloomFilterFullPositive = 11,
    /// Bytes read by compaction (sum of input file sizes).
    CompactionBytesRead = 12,
    /// Bytes written by compaction (sum of output file sizes).
    CompactionBytesWritten = 13,
    /// Number of compaction jobs that have run.
    CompactionCount = 14,
    /// Bytes written by the flush path (output SSTable size).
    FlushBytesWritten = 15,
    /// Number of memtable → L0 flushes.
    FlushCount = 16,
    /// Bytes appended to the WAL.
    WalBytesWritten = 17,
    /// Number of `Wal::sync` calls.
    WalSyncCount = 18,
    /// Number of `Iter::seek` / `Iter::seek_to_first` / etc.
    IterSeekCount = 19,
    /// Number of `Iter::next` calls that produced a key.
    IterNextCount = 20,
    /// Microseconds the engine spent stalling a writer. Not
    /// populated yet - reserved for the write-stall plumbing.
    WriteStallMicros = 21,
    /// Number of snapshots registered via `Db::snapshot`.
    SnapshotsRegistered = 22,
    /// Number of snapshots released (dropped or explicitly
    /// released).
    SnapshotsReleased = 23,
    /// Number of incomplete trailing WAL records discarded during
    /// recovery. A non-zero value means a crash left a partly written
    /// record behind and the bytes from it to end-of-file were dropped;
    /// the discard is also logged with the file and the offset.
    WalTailDiscarded = 24,
}

const NUM_TICKERS: usize = 25;

/// Every defined ticker, in discriminant order. Used by
/// [`Statistics::dump`] to iterate all slots. Keep this in sync
/// with the [`Ticker`] enum - adding a variant without
/// appending here will silently drop it from the dump output.
const ALL_TICKERS: &[Ticker] = &[
    Ticker::BytesWritten,
    Ticker::BytesRead,
    Ticker::KeysWritten,
    Ticker::KeysRead,
    Ticker::KeysDeleted,
    Ticker::RangeDeletesWritten,
    Ticker::MergesWritten,
    Ticker::BlockCacheHit,
    Ticker::BlockCacheMiss,
    Ticker::BlockCacheAdd,
    Ticker::BloomFilterUseful,
    Ticker::BloomFilterFullPositive,
    Ticker::CompactionBytesRead,
    Ticker::CompactionBytesWritten,
    Ticker::CompactionCount,
    Ticker::FlushBytesWritten,
    Ticker::FlushCount,
    Ticker::WalBytesWritten,
    Ticker::WalSyncCount,
    Ticker::IterSeekCount,
    Ticker::IterNextCount,
    Ticker::WriteStallMicros,
    Ticker::SnapshotsRegistered,
    Ticker::SnapshotsReleased,
    Ticker::WalTailDiscarded,
];

impl Ticker {
    /// Stable string name for exporting to monitoring systems.
    pub fn name(&self) -> &'static str {
        match self {
            Ticker::BytesWritten => "lark.bytes_written",
            Ticker::BytesRead => "lark.bytes_read",
            Ticker::KeysWritten => "lark.keys_written",
            Ticker::KeysRead => "lark.keys_read",
            Ticker::KeysDeleted => "lark.keys_deleted",
            Ticker::RangeDeletesWritten => "lark.range_deletes_written",
            Ticker::MergesWritten => "lark.merges_written",
            Ticker::BlockCacheHit => "lark.block_cache_hit",
            Ticker::BlockCacheMiss => "lark.block_cache_miss",
            Ticker::BlockCacheAdd => "lark.block_cache_add",
            Ticker::BloomFilterUseful => "lark.bloom_filter_useful",
            Ticker::BloomFilterFullPositive => "lark.bloom_filter_full_positive",
            Ticker::CompactionBytesRead => "lark.compaction_bytes_read",
            Ticker::CompactionBytesWritten => "lark.compaction_bytes_written",
            Ticker::CompactionCount => "lark.compaction_count",
            Ticker::FlushBytesWritten => "lark.flush_bytes_written",
            Ticker::FlushCount => "lark.flush_count",
            Ticker::WalBytesWritten => "lark.wal_bytes_written",
            Ticker::WalSyncCount => "lark.wal_sync_count",
            Ticker::IterSeekCount => "lark.iter_seek_count",
            Ticker::IterNextCount => "lark.iter_next_count",
            Ticker::WriteStallMicros => "lark.write_stall_micros",
            Ticker::SnapshotsRegistered => "lark.snapshots_registered",
            Ticker::SnapshotsReleased => "lark.snapshots_released",
            Ticker::WalTailDiscarded => "lark.wal_tail_discarded",
        }
    }
}

/// Enumerated histograms recorded by the engine. Every variant
/// is backed by one histogram slot in [`Statistics`], guarded
/// by its own short mutex so recording is non-contending
/// across histograms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Histogram {
    /// Wall-clock microseconds per `Db::get` call.
    DbGet = 0,
    /// Wall-clock microseconds per `Db::write` / `apply_batch`.
    DbWrite = 1,
    /// Wall-clock microseconds per `Iter::seek*` call.
    DbIterSeek = 2,
    /// Wall-clock microseconds per `Iter::next` call that
    /// produced a key.
    DbIterNext = 3,
    /// Wall-clock microseconds per compaction job.
    CompactionTime = 4,
    /// Wall-clock microseconds per flush.
    FlushTime = 5,
    /// Wall-clock microseconds spent reading a single data
    /// block off disk (decompression included, cache lookup
    /// excluded).
    BlockReadTime = 6,
    /// Bytes returned per `Db::get` that found a live value.
    BytesPerRead = 7,
    /// Total bytes applied per `Db::write` (keys + values across
    /// every op in the batch).
    BytesPerWrite = 8,
    /// Wall-clock microseconds to append a batch to the WAL
    /// (including fsync when durability is Immediate).
    WalWriteTime = 9,
}

const NUM_HISTOGRAMS: usize = 10;

/// Every defined histogram, in discriminant order. Same pattern
/// as [`ALL_TICKERS`].
const ALL_HISTOGRAMS: &[Histogram] = &[
    Histogram::DbGet,
    Histogram::DbWrite,
    Histogram::DbIterSeek,
    Histogram::DbIterNext,
    Histogram::CompactionTime,
    Histogram::FlushTime,
    Histogram::BlockReadTime,
    Histogram::BytesPerRead,
    Histogram::BytesPerWrite,
    Histogram::WalWriteTime,
];

impl Histogram {
    /// Stable string name for exporting to monitoring systems.
    pub fn name(&self) -> &'static str {
        match self {
            Histogram::DbGet => "lark.db_get",
            Histogram::DbWrite => "lark.db_write",
            Histogram::DbIterSeek => "lark.db_iter_seek",
            Histogram::DbIterNext => "lark.db_iter_next",
            Histogram::CompactionTime => "lark.compaction_time",
            Histogram::FlushTime => "lark.flush_time",
            Histogram::BlockReadTime => "lark.block_read_time",
            Histogram::BytesPerRead => "lark.bytes_per_read",
            Histogram::BytesPerWrite => "lark.bytes_per_write",
            Histogram::WalWriteTime => "lark.wal_write_time",
        }
    }
}

/// Immutable snapshot of a single histogram's state. Callers
/// read this to export to their metrics pipeline or assert in
/// tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HistogramSnapshot {
    /// Number of samples recorded.
    pub count: u64,
    /// Sum of all recorded values.
    pub sum: u64,
    /// Minimum recorded value (0 when `count == 0`).
    pub min: u64,
    /// Maximum recorded value (0 when `count == 0`).
    pub max: u64,
}

impl HistogramSnapshot {
    /// Arithmetic mean. Returns 0 when `count == 0`.
    pub fn average(&self) -> u64 {
        self.sum.checked_div(self.count).unwrap_or(0)
    }
}

#[derive(Debug, Default)]
struct HistogramData {
    count: u64,
    sum: u64,
    min: u64,
    max: u64,
}

impl HistogramData {
    fn record(&mut self, value: u64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        self.count += 1;
        self.sum = self.sum.saturating_add(value);
    }

    fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            count: self.count,
            sum: self.sum,
            min: self.min,
            max: self.max,
        }
    }

    fn clear(&mut self) {
        *self = HistogramData::default();
    }
}

/// Engine-wide counters and histograms. Constructed by the
/// caller and passed to [`crate::Options::statistics`]. The
/// engine clones the `Arc` into the paths it wants to
/// instrument and updates it via lock-free atomic adds
/// (tickers) or short mutex sections (histograms).
pub struct Statistics {
    tickers: [AtomicU64; NUM_TICKERS],
    histograms: [Mutex<HistogramData>; NUM_HISTOGRAMS],
}

impl std::fmt::Debug for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Statistics").finish_non_exhaustive()
    }
}

impl Default for Statistics {
    fn default() -> Self {
        Self::new()
    }
}

impl Statistics {
    /// Construct a fresh `Statistics` with every ticker and
    /// histogram at zero.
    pub fn new() -> Self {
        // `AtomicU64` is not `Copy`, so we can't use the
        // `[x; N]` syntax. `std::array::from_fn` is the
        // cleanest alternative.
        let tickers = std::array::from_fn(|_| AtomicU64::new(0));
        let histograms = std::array::from_fn(|_| Mutex::new(HistogramData::default()));
        Self {
            tickers,
            histograms,
        }
    }

    /// Read the current value of a ticker. Guaranteed to be
    /// consistent with the corresponding `fetch_add` via
    /// `Ordering::Relaxed` - callers that need stronger
    /// ordering should wrap their own fences.
    pub fn get_ticker(&self, ticker: Ticker) -> u64 {
        self.tickers[ticker as usize].load(Ordering::Relaxed)
    }

    /// Return an immutable snapshot of a histogram's state.
    pub fn get_histogram_snapshot(&self, hist: Histogram) -> HistogramSnapshot {
        self.histograms[hist as usize].lock().snapshot()
    }

    /// Zero every ticker and clear every histogram.
    pub fn reset(&self) {
        for t in &self.tickers {
            t.store(0, Ordering::Relaxed);
        }
        for h in &self.histograms {
            h.lock().clear();
        }
    }

    /// Human-readable dump of every ticker and histogram. Meant
    /// for debug output, not machine parsing.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str("-- tickers --\n");
        for ticker in ALL_TICKERS {
            out.push_str(&format!(
                "{:40} {}\n",
                ticker.name(),
                self.tickers[*ticker as usize].load(Ordering::Relaxed)
            ));
        }
        out.push_str("-- histograms --\n");
        for hist in ALL_HISTOGRAMS {
            let snap = self.histograms[*hist as usize].lock().snapshot();
            out.push_str(&format!(
                "{:40} count={} sum={} min={} max={} avg={}\n",
                hist.name(),
                snap.count,
                snap.sum,
                snap.min,
                snap.max,
                snap.average(),
            ));
        }
        out
    }

    /// Add `amount` to `ticker`. `Ordering::Relaxed` - callers
    /// that need stronger ordering should provide their own.
    pub(crate) fn add(&self, ticker: Ticker, amount: u64) {
        self.tickers[ticker as usize].fetch_add(amount, Ordering::Relaxed);
    }

    /// Record a single sample into `hist`.
    pub(crate) fn record(&self, hist: Histogram, value: u64) {
        self.histograms[hist as usize].lock().record(value);
    }
}

/// Convenience RAII helper: creates a timer on construction and
/// records the elapsed wall-clock microseconds into `hist` on
/// `Drop`. If the statistics handle is `None` the helper is
/// optimized out - both construction and drop are no-ops.
pub(crate) struct TimeScope<'a> {
    /// Start reading in microseconds. `None` when no statistics
    /// handle is installed, and also `None` on a platform with no
    /// monotonic clock: the histogram then takes no sample at all
    /// rather than a zero that reads like a measurement.
    start: Option<u64>,
    stats: Option<&'a Statistics>,
    hist: Histogram,
}

impl<'a> TimeScope<'a> {
    pub(crate) fn new(stats: Option<&'a Statistics>, hist: Histogram) -> Self {
        Self {
            start: stats.and_then(|_| crate::env::platform_micros()),
            stats,
            hist,
        }
    }
}

impl Drop for TimeScope<'_> {
    fn drop(&mut self) {
        if let (Some(start), Some(stats), Some(now)) =
            (self.start, self.stats, crate::env::platform_micros())
        {
            stats.record(self.hist, now.saturating_sub(start));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticker_add_and_read() {
        let s = Statistics::new();
        assert_eq!(s.get_ticker(Ticker::BytesWritten), 0);
        s.add(Ticker::BytesWritten, 100);
        s.add(Ticker::BytesWritten, 50);
        assert_eq!(s.get_ticker(Ticker::BytesWritten), 150);
    }

    #[test]
    fn histogram_records_min_max_sum_count() {
        let s = Statistics::new();
        s.record(Histogram::DbGet, 10);
        s.record(Histogram::DbGet, 20);
        s.record(Histogram::DbGet, 5);
        let snap = s.get_histogram_snapshot(Histogram::DbGet);
        assert_eq!(snap.count, 3);
        assert_eq!(snap.sum, 35);
        assert_eq!(snap.min, 5);
        assert_eq!(snap.max, 20);
        assert_eq!(snap.average(), 11);
    }

    #[test]
    fn reset_zeroes_everything() {
        let s = Statistics::new();
        s.add(Ticker::BytesRead, 42);
        s.record(Histogram::FlushTime, 99);
        s.reset();
        assert_eq!(s.get_ticker(Ticker::BytesRead), 0);
        let snap = s.get_histogram_snapshot(Histogram::FlushTime);
        assert_eq!(snap, HistogramSnapshot::default());
    }

    #[test]
    fn dump_contains_every_ticker_and_histogram_name() {
        let s = Statistics::new();
        let out = s.dump();
        assert!(out.contains("lark.bytes_written"));
        assert!(out.contains("lark.block_cache_hit"));
        assert!(out.contains("lark.db_get"));
        assert!(out.contains("lark.flush_time"));
    }

    #[test]
    fn histogram_empty_snapshot_is_default() {
        let s = Statistics::new();
        assert_eq!(
            s.get_histogram_snapshot(Histogram::DbGet),
            HistogramSnapshot::default()
        );
    }

    #[test]
    fn time_scope_records_on_drop() {
        let s = Statistics::new();
        {
            let _t = TimeScope::new(Some(&s), Histogram::DbGet);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let snap = s.get_histogram_snapshot(Histogram::DbGet);
        assert_eq!(snap.count, 1);
        assert!(snap.sum > 0, "scope should have recorded non-zero micros");
    }

    #[test]
    fn time_scope_disabled_is_noop() {
        let _t = TimeScope::new(None, Histogram::DbGet);
        // Nothing to assert - the test is that Drop runs without
        // panicking when the stats handle is absent.
    }
}
