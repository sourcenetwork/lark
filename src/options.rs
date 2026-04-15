use std::sync::Arc;

/// Decision returned by a [`CompactionFilter`] for each entry it sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionDecision {
    /// Leave the entry untouched.
    Keep,
    /// Drop the entry. Compaction replaces it with a tombstone at the
    /// same sequence number so lower levels cannot resurrect the
    /// original value.
    Remove,
    /// Replace the entry's value with a new byte string. The key and
    /// sequence number are preserved.
    Change(Vec<u8>),
}

/// A user-supplied hook that runs during compaction and can drop or
/// rewrite entries in place. Typical uses: TTL expiration, application-
/// level GC, schema migrations.
///
/// # Determinism
///
/// Implementations must be deterministic functions of `(level, key,
/// value)` — the same input should always yield the same decision.
/// They must not read from the database (would deadlock) and should
/// avoid blocking.
///
/// # Snapshot isolation
///
/// Compaction filters currently run only when no live [`crate::Snapshot`]
/// is pinned. This guarantees that every `Snapshot` taken before
/// compaction still observes the pre-filter value until the snapshot
/// is dropped. When a snapshot is alive, compaction still runs but
/// skips the filter entirely. Finer-grained per-snapshot filtering is
/// a planned follow-up.
pub trait CompactionFilter: Send + Sync + 'static {
    /// Inspect a point entry. Called once per surviving
    /// `(user_key, value)` pair during compaction.
    fn filter(&self, level: usize, key: &[u8], value: &[u8]) -> CompactionDecision;

    /// Inspect a range tombstone. Default implementation keeps every
    /// range tombstone. `CompactionDecision::Change` is treated as
    /// `Keep` for range tombstones (there's no "value" to rewrite).
    fn filter_range_delete(&self, level: usize, start: &[u8], end: &[u8]) -> CompactionDecision {
        let _ = (level, start, end);
        CompactionDecision::Keep
    }

    /// A stable, human-readable identifier for this filter. Used by
    /// tracing and diagnostics.
    fn name(&self) -> &'static str;
}

/// A user-supplied associative merge operator.
///
/// A merge operator lets callers emit `Merge(key, operand)` records
/// cheaply instead of doing a read-modify-write. Readers and
/// compaction collapse a chain of merge operands by calling
/// [`MergeOperator::full_merge`] (combine a base value with an
/// ordered list of operands) or [`MergeOperator::partial_merge`]
/// (fold two adjacent operands, without a base).
///
/// # Determinism and safety
///
/// Implementations must be deterministic functions of their inputs.
/// They MUST NOT read from the database — doing so will deadlock —
/// and they should not block. Returning `None` from either method is
/// interpreted as a merge failure: readers surface it as
/// [`crate::Error::MergeFailed`], while compaction treats it
/// conservatively (keeps the raw chain intact rather than losing data).
///
/// # Operand order
///
/// `operands` in [`MergeOperator::full_merge`] is ordered oldest
/// first: `operands[0]` was written before `operands[1]`, which was
/// written before `operands[2]`, etc. The base value (if any) was
/// written before every operand.
pub trait MergeOperator: Send + Sync + 'static {
    /// Combine a `base` value (possibly `None` if the key didn't
    /// exist) with the ordered list of merge `operands`, returning
    /// the final value. Returning `None` signals merge failure.
    fn full_merge(&self, key: &[u8], base: Option<&[u8]>, operands: &[&[u8]]) -> Option<Vec<u8>>;

    /// Optionally fold two adjacent operands `left` (older) and
    /// `right` (newer) into a single equivalent operand without a
    /// base value. Compaction calls this to shrink merge chains in
    /// the middle of an LSM level. The default implementation
    /// returns `None`, which disables compaction-time collapse.
    fn partial_merge(&self, key: &[u8], left: &[u8], right: &[u8]) -> Option<Vec<u8>> {
        let _ = (key, left, right);
        None
    }

    /// A stable, human-readable identifier for this operator. Used
    /// by tracing and diagnostics.
    fn name(&self) -> &'static str;
}

/// Per-call knobs for point and batch writes. Overrides the database-
/// global [`Options::durability`] on a single operation so callers can
/// opt a critical write into synchronous fsync, or opt a bulk-load
/// phase out of the WAL, without flipping the whole database.
///
/// All fields default to `false`. The ergonomic builders below cover
/// the two knobs currently implemented.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    /// Fsync the WAL before returning. Equivalent to
    /// [`DurabilityMode::Immediate`] for this single call, regardless
    /// of the database-global default.
    pub sync: bool,
    /// Skip the WAL append entirely. The caller accepts that a crash
    /// before the next memtable flush loses the write. Used by
    /// bulk-load phases that will be ingested via SST file anyway.
    pub disable_wal: bool,
    /// Reserved for future use by a cooperative write-lock priority
    /// queue. Currently accepted but ignored.
    pub low_pri: bool,
    /// Reserved for future use by the write-stall / rate-limiter
    /// plumbing. Currently accepted but ignored — the engine never
    /// actually stalls today, so this knob is a no-op.
    pub no_slowdown: bool,
}

impl WriteOptions {
    /// Construct a `WriteOptions` with the default values (all
    /// fields `false`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Shortcut for a `WriteOptions` with `sync: true`.
    pub fn sync() -> Self {
        Self {
            sync: true,
            ..Self::default()
        }
    }

    /// Shortcut for a `WriteOptions` with `disable_wal: true`.
    pub fn disable_wal() -> Self {
        Self {
            disable_wal: true,
            ..Self::default()
        }
    }
}

/// Carves a prefix out of a user key so the SSTable bloom filter can
/// answer "does this file contain any key with prefix P?" queries.
///
/// A configured extractor is consulted when building an SSTable: in
/// addition to the user-key bloom filter (which powers point lookups),
/// the writer builds a second, prefix-keyed bloom filter. Prefix-bounded
/// range scans (`Iter::seek_prefix`) consult the prefix bloom and skip
/// SSTables that demonstrably cannot contain the requested prefix.
///
/// Point lookups are unaffected — they always consult the user-key
/// bloom filter and do not call the extractor.
pub trait PrefixExtractor: Send + Sync + 'static {
    /// Return the portion of `key` that the prefix bloom should index,
    /// or `None` if `key` cannot produce a prefix (e.g., it's shorter
    /// than a fixed-length extractor's width). Keys that return `None`
    /// are simply absent from the prefix bloom.
    fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]>;

    /// A stable, human-readable identifier for this extractor. Used
    /// by tracing and diagnostics.
    fn name(&self) -> &'static str;
}

/// A [`PrefixExtractor`] that takes the first `N` bytes of every key.
/// Keys shorter than `N` contribute no prefix.
#[derive(Debug, Clone, Copy)]
pub struct FixedLengthPrefix(pub usize);

impl PrefixExtractor for FixedLengthPrefix {
    fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]> {
        if key.len() >= self.0 {
            Some(&key[..self.0])
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "FixedLengthPrefix"
    }
}

/// Controls when data is flushed to disk after a write.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Flush to disk on every write. Safe against process and OS crashes.
    Immediate,
    /// Rely on the OS to flush eventually (default). Process crash is still
    /// safe due to WAL.
    #[default]
    Eventual,
}

/// Block compression codec applied to SSTable data blocks.
///
/// Each codec is identified by a 1-byte discriminator stored in the
/// block frame, so a single SSTable file can read blocks compressed
/// with different codecs (and a database can mix codecs across levels).
///
/// All variants are pure-Rust implementations — no C/C++ toolchain
/// or system libraries required.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression. Smallest CPU cost; largest on-disk footprint.
    None,
    /// Snappy. Fast, modest compression ratio. Pure-Rust via the
    /// `snap` crate.
    Snappy,
    /// LZ4. Slightly faster decompression than Snappy, comparable ratio.
    /// Pure-Rust via `lz4_flex`. This is the default.
    #[default]
    Lz4,
}

/// Configuration options for a lark database.
#[derive(Clone)]
pub struct Options {
    /// Write buffer (memtable) size before flush. Default: 64 MB.
    pub write_buffer_size: usize,
    /// Data block size in SSTables. Default: 16 KB.
    pub block_size: usize,
    /// Block cache size for decompressed blocks. Default: 512 MB.
    pub block_cache_size: usize,
    /// Bloom filter bits per key. Default: 10.
    pub bloom_bits_per_key: usize,
    /// Default block compression codec. Used at every level unless
    /// overridden by [`Options::compression_per_level`]. Default: LZ4.
    pub compression: CompressionType,
    /// Per-level compression override. When set, entry `i` selects the
    /// codec for level `i`. Levels beyond the vector's length fall
    /// back to [`Options::compression`]. `None` (default) means "use
    /// the default codec at every level".
    pub compression_per_level: Option<Vec<CompressionType>>,
    /// Number of L0 SSTables before triggering compaction. Default: 4.
    pub l0_compaction_trigger: usize,
    /// Target size for level 1. Default: 256 MB.
    pub level_base_bytes: u64,
    /// Size multiplier between levels. Default: 10.
    pub level_size_multiplier: u64,
    /// Target SSTable file size during compaction. Default: 64 MB.
    pub target_file_size: u64,
    /// Durability mode. Default: Eventual.
    pub durability: DurabilityMode,
    /// Optional user hook invoked during compaction for every point
    /// entry and range tombstone. See [`CompactionFilter`] for
    /// semantics and snapshot-isolation rules.
    pub compaction_filter: Option<Arc<dyn CompactionFilter>>,
    /// Optional prefix extractor. When set, SSTable writers build an
    /// additional prefix-keyed bloom filter that `Iter::seek_prefix`
    /// consults to skip files that cannot contain the scanned prefix.
    /// Point lookups are unaffected.
    pub prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
    /// Optional associative merge operator. When set, callers may
    /// emit merge operands via [`crate::Db::merge`] /
    /// [`crate::WriteBatch::merge`] instead of doing
    /// read-modify-write, and readers collapse the merge chain via
    /// [`MergeOperator::full_merge`] at visibility time.
    pub merge_operator: Option<Arc<dyn MergeOperator>>,
    /// Flag accepted for RocksDB API parity. Lark's column-family
    /// implementation is key-prefix based: every CF shares one
    /// memtable, one WAL, one manifest, and one flush path, so a
    /// multi-CF [`crate::WriteBatch`] is **always** atomic across
    /// CFs regardless of this flag's value. A flush either
    /// persists every participant's half of a batch or persists
    /// none of it — the flag exists so caller code ported from
    /// RocksDB compiles without modification.
    pub atomic_flush: bool,
    /// Event listeners subscribed to engine lifecycle events
    /// (flush, compaction, ingest, background errors). Dispatch
    /// is synchronous on the firing thread — listeners **must not
    /// block or re-enter the database**. See
    /// [`crate::EventListener`] for the full contract.
    pub listeners: Vec<Arc<dyn crate::EventListener>>,
    /// Optional statistics sink. When set, every hot path in
    /// the engine updates the provided [`crate::Statistics`]
    /// object with tickers and histograms. The caller polls the
    /// same object to export metrics to their monitoring stack.
    /// `None` (default) short-circuits every instrumentation site
    /// at a branch, so disabled stats cost almost nothing.
    pub statistics: Option<Arc<crate::Statistics>>,
    /// Optional rate limiter. When set, flush and compaction output
    /// writes are throttled via [`crate::RateLimiter::request`]
    /// before the engine moves on to the next job, capping the
    /// combined background-I/O rate at the limiter's configured
    /// bytes/second. Foreground (user) writes are not throttled.
    /// `None` (default) means background I/O is uncapped.
    pub rate_limiter: Option<Arc<dyn crate::RateLimiter>>,
    /// Start slowing foreground writes when the number of L0
    /// SSTables reaches this threshold. Each affected write
    /// incurs a small fixed delay, back-pressuring callers so
    /// background compaction can catch up. Default: 20.
    pub level0_slowdown_writes_trigger: usize,
    /// Stop foreground writes entirely when the number of L0
    /// SSTables reaches this threshold. Writers block on a
    /// condvar that compaction notifies once it reduces the
    /// count below the slowdown trigger. Default: 36.
    pub level0_stop_writes_trigger: usize,
    /// Start slowing writes when total bytes in L0 (lark's
    /// approximation of "pending compaction bytes") exceed this
    /// limit. Default: 64 GB.
    pub soft_pending_compaction_bytes_limit: u64,
    /// Stop writes when total bytes in L0 exceed this limit.
    /// Default: 256 GB.
    pub hard_pending_compaction_bytes_limit: u64,
    /// Soft cap on the number of in-memory memtables (active +
    /// frozen). Reaching this count slows writes; reaching
    /// `2 * max_write_buffer_number` stops them. Default: 2.
    pub max_write_buffer_number: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
            bloom_bits_per_key: 10,
            compression: CompressionType::Lz4,
            compression_per_level: None,
            l0_compaction_trigger: 4,
            level_base_bytes: 256 * 1024 * 1024,
            level_size_multiplier: 10,
            target_file_size: 64 * 1024 * 1024,
            durability: DurabilityMode::Eventual,
            compaction_filter: None,
            prefix_extractor: None,
            merge_operator: None,
            atomic_flush: false,
            listeners: Vec::new(),
            statistics: None,
            rate_limiter: None,
            level0_slowdown_writes_trigger: 20,
            level0_stop_writes_trigger: 36,
            soft_pending_compaction_bytes_limit: 64 * 1024 * 1024 * 1024,
            hard_pending_compaction_bytes_limit: 256 * 1024 * 1024 * 1024,
            max_write_buffer_number: 2,
        }
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("write_buffer_size", &self.write_buffer_size)
            .field("block_size", &self.block_size)
            .field("block_cache_size", &self.block_cache_size)
            .field("bloom_bits_per_key", &self.bloom_bits_per_key)
            .field("compression", &self.compression)
            .field("compression_per_level", &self.compression_per_level)
            .field("l0_compaction_trigger", &self.l0_compaction_trigger)
            .field("level_base_bytes", &self.level_base_bytes)
            .field("level_size_multiplier", &self.level_size_multiplier)
            .field("target_file_size", &self.target_file_size)
            .field("durability", &self.durability)
            .field(
                "compaction_filter",
                &self.compaction_filter.as_ref().map(|f| f.name()),
            )
            .field(
                "prefix_extractor",
                &self.prefix_extractor.as_ref().map(|p| p.name()),
            )
            .field(
                "merge_operator",
                &self.merge_operator.as_ref().map(|m| m.name()),
            )
            .field("atomic_flush", &self.atomic_flush)
            .field("listeners", &self.listeners.len())
            .field("statistics", &self.statistics.is_some())
            .field("rate_limiter", &self.rate_limiter.is_some())
            .field(
                "level0_slowdown_writes_trigger",
                &self.level0_slowdown_writes_trigger,
            )
            .field(
                "level0_stop_writes_trigger",
                &self.level0_stop_writes_trigger,
            )
            .field(
                "soft_pending_compaction_bytes_limit",
                &self.soft_pending_compaction_bytes_limit,
            )
            .field(
                "hard_pending_compaction_bytes_limit",
                &self.hard_pending_compaction_bytes_limit,
            )
            .field("max_write_buffer_number", &self.max_write_buffer_number)
            .finish()
    }
}

impl Options {
    pub(crate) fn to_engine_options(&self) -> crate::engine::EngineOptions {
        crate::engine::EngineOptions {
            write_buffer_size: self.write_buffer_size,
            block_size: self.block_size,
            block_cache_size: self.block_cache_size,
            bloom_bits_per_key: self.bloom_bits_per_key,
            compression: self.compression,
            compression_per_level: self.compression_per_level.clone(),
            l0_compaction_trigger: self.l0_compaction_trigger,
            level_base_bytes: self.level_base_bytes,
            level_size_multiplier: self.level_size_multiplier,
            target_file_size: self.target_file_size,
            compaction_filter: self.compaction_filter.clone(),
            prefix_extractor: self.prefix_extractor.clone(),
            merge_operator: self.merge_operator.clone(),
            listeners: self.listeners.clone(),
            statistics: self.statistics.clone(),
            rate_limiter: self.rate_limiter.clone(),
            level0_slowdown_writes_trigger: self.level0_slowdown_writes_trigger,
            level0_stop_writes_trigger: self.level0_stop_writes_trigger,
            soft_pending_compaction_bytes_limit: self.soft_pending_compaction_bytes_limit,
            hard_pending_compaction_bytes_limit: self.hard_pending_compaction_bytes_limit,
            max_write_buffer_number: self.max_write_buffer_number,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_length_prefix_extract() {
        let ex = FixedLengthPrefix(4);
        assert_eq!(ex.extract(b"tenant_001"), Some(&b"tena"[..]));
        assert_eq!(ex.extract(b"abcd"), Some(&b"abcd"[..]));
        assert_eq!(ex.extract(b"abc"), None);
        assert_eq!(ex.extract(b""), None);
        assert_eq!(ex.name(), "FixedLengthPrefix");
    }
}
