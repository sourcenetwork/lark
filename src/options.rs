use std::sync::Arc;

/// Default maximum user-key length accepted by write APIs: 8 MiB.
pub const DEFAULT_MAX_KEY_SIZE: usize = 8 * 1024 * 1024;

/// Default maximum value / merge-operand length accepted by write APIs: 64 MiB.
pub const DEFAULT_MAX_VALUE_SIZE: usize = 64 * 1024 * 1024;

/// Highest supported block-cache shard exponent.
pub const MAX_BLOCK_CACHE_SHARD_BITS: u32 = 8;

/// Highest supported Bloom-filter density. Larger values waste space
/// because the hash count is already capped internally.
pub const MAX_BLOOM_BITS_PER_KEY: usize = 64;

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

/// Compaction strategy used by the background compaction thread.
///
/// Lark currently ships two styles. More can be added in follow-up
/// work (universal / size-tiered is the obvious next one) without
/// breaking callers, since the field is consumed by name.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStyle {
    /// Standard leveled compaction: L0 collects flushes, then files
    /// are pushed down through L1..L6 with each level ~10× larger
    /// than the previous one. Best fit for mixed read/write
    /// workloads where space amplification matters more than
    /// write amplification. This is the default.
    #[default]
    Level,
    /// FIFO: lark never merges files. Once the total size of all
    /// SSTables exceeds [`FifoCompactionOptions::max_table_files_size`],
    /// the oldest SSTable is unlinked. Best fit for time-series and
    /// append-only log workloads where the oldest data is also the
    /// least valuable. Reads still consult every L0 file (there is
    /// no L1+) so read amplification grows with the number of
    /// retained files; this is the trade-off for ~zero write
    /// amplification.
    Fifo,
    /// Universal (size-tiered): all files live at L0 and are
    /// merged into progressively larger runs based on size ratios
    /// instead of fixed level targets. Each merge produces one
    /// larger L0 file; an over-sized database periodically
    /// triggers a full compaction that folds every existing file
    /// into a single run. Write amplification is roughly
    /// `log_ratio(total_size / memtable_size)`, much lower than
    /// leveled at the cost of higher space amplification during
    /// the merge and more read amplification than leveled. Best
    /// fit for write-heavy workloads where disk is cheap and read
    /// latency is tolerant.
    Universal,
}

/// Tunables for [`CompactionStyle::Fifo`]. Ignored when
/// [`Options::compaction_style`] is [`CompactionStyle::Level`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FifoCompactionOptions {
    /// Soft cap on the total bytes held by SSTables. After every
    /// flush, if the sum of file sizes exceeds this number, the
    /// background compaction thread unlinks the oldest SSTables
    /// (smallest `file_id` first) until the cap is satisfied or
    /// only one file remains. Must be greater than zero when FIFO
    /// compaction is selected. Default: 1 GiB.
    pub max_table_files_size: u64,
}

impl Default for FifoCompactionOptions {
    fn default() -> Self {
        Self {
            max_table_files_size: 1024 * 1024 * 1024,
        }
    }
}

/// Tunables for [`CompactionStyle::Universal`]. Ignored when the
/// style is not [`CompactionStyle::Universal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniversalCompactionOptions {
    /// Percent tolerance used by the size-ratio merge rule. Walk
    /// the L0 files newest-first; accumulate files while the
    /// running total is within `size_ratio` percent of the next
    /// candidate's size. When the accumulator hits
    /// [`UniversalCompactionOptions::min_merge_width`] files, the
    /// picker merges them into one new L0 file. A larger
    /// `size_ratio` groups more files per merge (lower write
    /// amplification, larger output); smaller is pickier. Must be
    /// greater than zero when universal compaction is selected.
    /// Default: 1 (percent).
    pub size_ratio: u32,
    /// Smallest number of files the size-ratio rule is willing to
    /// merge. Must be at least `2` when universal compaction is
    /// selected. Pass `2` for the most aggressive grouping.
    /// Default: 2.
    pub min_merge_width: u32,
    /// Cap on the number of files a single size-ratio merge can
    /// consume. Must be >= [`UniversalCompactionOptions::min_merge_width`]
    /// when universal compaction is selected. `u32::MAX` disables
    /// the cap. Default: `u32::MAX`.
    pub max_merge_width: u32,
    /// Size-amplification trigger, as a percent. When
    /// `total_size_of_all_older_files * 100 / size_of_oldest_file`
    /// exceeds this value, the picker forces a full merge of
    /// every L0 file into one run. Default: 200 (i.e., full merge
    /// fires once the accumulated non-oldest content is ~2× the
    /// oldest run). Must be greater than zero when universal
    /// compaction is selected.
    pub max_size_amplification_percent: u32,
}

impl Default for UniversalCompactionOptions {
    fn default() -> Self {
        Self {
            size_ratio: 1,
            min_merge_width: 2,
            max_merge_width: u32::MAX,
            max_size_amplification_percent: 200,
        }
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
    /// Write buffer (memtable) size before flush. Must be greater
    /// than zero. Default: 64 MB.
    pub write_buffer_size: usize,
    /// Data block size in SSTables. Must be greater than zero.
    /// Default: 16 KB.
    pub block_size: usize,
    /// Block cache size for decompressed blocks. Must be greater
    /// than zero. Default: 512 MB.
    pub block_cache_size: usize,
    /// Base-2 log of the block cache shard count. The block cache
    /// is split into `2^block_cache_num_shard_bits` shards keyed
    /// by `hash(file_id, offset)` so concurrent readers contend
    /// only with other readers that hash to the same shard.
    /// Must be <= [`MAX_BLOCK_CACHE_SHARD_BITS`]. Tiny cache budgets
    /// may use fewer effective shards so each shard has usable
    /// capacity. Default: 6 (64 shards).
    pub block_cache_num_shard_bits: u32,
    /// If `true`, the block cache refuses to admit a single entry
    /// that is larger than one shard's byte capacity; the caller
    /// uses the block directly without caching it. If `false`
    /// (default), an oversized entry evicts everything else in
    /// its shard and is admitted anyway.
    pub strict_capacity_limit: bool,
    /// Bloom filter bits per key. Must be in
    /// `1..=MAX_BLOOM_BITS_PER_KEY`. Default: 10.
    pub bloom_bits_per_key: usize,
    /// Default block compression codec. Used at every level unless
    /// overridden by [`Options::compression_per_level`]. Default: LZ4.
    pub compression: CompressionType,
    /// Per-level compression override. When set, entry `i` selects the
    /// codec for level `i`. Levels beyond the vector's length fall
    /// back to [`Options::compression`]. `None` (default) means "use
    /// the default codec at every level".
    pub compression_per_level: Option<Vec<CompressionType>>,
    /// Number of L0 SSTables before triggering compaction. Must be
    /// greater than zero. Default: 4.
    pub l0_compaction_trigger: usize,
    /// Target size for level 1. Must be greater than zero.
    /// Default: 256 MB.
    pub level_base_bytes: u64,
    /// Size multiplier between levels. Must be greater than zero.
    /// Default: 10.
    pub level_size_multiplier: u64,
    /// Target SSTable file size during compaction. Must be greater
    /// than zero. Default: 64 MB.
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
    /// Opt-in flag accepted for parity with storage engines that
    /// require an explicit switch to get atomic multi-CF flushes.
    /// Lark's column-family implementation is key-prefix based:
    /// every CF shares one memtable, one WAL, one manifest, and
    /// one flush path, so a multi-CF [`crate::WriteBatch`] is
    /// **always** atomic across CFs regardless of this flag's
    /// value. A flush either persists every participant's half of
    /// a batch or persists none of it.
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
    /// background compaction can catch up. `0` disables this
    /// trigger. If both L0 triggers are enabled, this must be <=
    /// [`Options::level0_stop_writes_trigger`]. Default: 20.
    pub level0_slowdown_writes_trigger: usize,
    /// Stop foreground writes entirely when the number of L0
    /// SSTables reaches this threshold. Writers block on a
    /// condvar that compaction notifies once it reduces the
    /// count below the slowdown trigger. `0` disables this
    /// trigger. Default: 36.
    pub level0_stop_writes_trigger: usize,
    /// Start slowing writes when total bytes in L0 (lark's
    /// approximation of "pending compaction bytes") reach this
    /// limit. `0` disables this trigger. If both pending-byte
    /// triggers are enabled, this must be <=
    /// [`Options::hard_pending_compaction_bytes_limit`].
    /// Default: 64 GB.
    pub soft_pending_compaction_bytes_limit: u64,
    /// Stop writes when total bytes in L0 reach this limit. `0`
    /// disables this trigger. Default: 256 GB.
    pub hard_pending_compaction_bytes_limit: u64,
    /// Soft cap on the number of in-memory memtables (active +
    /// frozen). Reaching this count slows writes; reaching
    /// `2 * max_write_buffer_number` stops them. `0` disables
    /// this trigger. Default: 2.
    pub max_write_buffer_number: usize,
    /// Compaction strategy. See [`CompactionStyle`] for the
    /// trade-offs. Default: [`CompactionStyle::Level`].
    pub compaction_style: CompactionStyle,
    /// Tunables for [`CompactionStyle::Fifo`]. Ignored when the
    /// style is [`CompactionStyle::Level`].
    pub fifo_compaction_options: FifoCompactionOptions,
    /// Tunables for [`CompactionStyle::Universal`]. Ignored when
    /// the style is not Universal.
    pub universal_compaction_options: UniversalCompactionOptions,
    /// Number of background threads available for compaction.
    ///
    /// When `> 1`, multiple non-overlapping compaction jobs can
    /// run concurrently (e.g. L1→L2 at key range `[a,m)` on one
    /// worker while L2→L3 at `[m,z)` runs on another). L0
    /// compactions are exclusive — only one L0 job runs at a
    /// time because L0 files can overlap arbitrarily.
    ///
    /// Must be greater than zero. `1` (default) keeps compaction
    /// single-threaded and matches pre-multi-worker behavior.
    pub max_background_compactions: usize,
    /// Accepted for compatibility with earlier releases.
    ///
    /// Compaction now streams a k-way merge with bounded memory
    /// and writes outputs on the compaction worker thread, so this
    /// knob does not change behavior.
    pub max_subcompactions: usize,
    /// Hint the OS page cache to drop pages backing SSTables
    /// that are read or written by background compaction.
    ///
    /// Without the hint, gigabytes of sequentially-consumed
    /// compaction data pollute the page cache and evict hot
    /// foreground reads. With the hint, the kernel is told to
    /// discard those pages immediately after the compaction
    /// finishes with them, billing the page cache cost strictly
    /// to foreground data.
    ///
    /// Currently implemented as `posix_fadvise(DONTNEED)` on
    /// Linux; on other targets the flag is accepted but the
    /// hint is a no-op. `false` by default — callers who care
    /// about foreground latency stability on Linux should turn
    /// it on.
    pub evict_compaction_data_from_page_cache: bool,
    /// Split the SSTable index into small leaf blocks on disk and keep
    /// only a compact top-level index in memory. Reduces resident
    /// memory when thousands of SSTables are open, at the cost of one
    /// extra disk read per point lookup (amortized by the OS page
    /// cache). Default: `false` (flat index loaded eagerly).
    pub partitioned_index: bool,
    /// Target size for each index leaf block when
    /// [`Options::partitioned_index`] is enabled. Must be greater
    /// than zero. Ignored when partitioned indexing is off.
    /// Default: 4096.
    pub metadata_block_size: usize,
    /// Open an existing database without creating files, rewriting
    /// recovered WALs, compacting, or allowing writes. Mutating APIs
    /// return [`crate::Error::ReadOnly`].
    ///
    /// Default: `false`.
    pub read_only: bool,
    /// Maximum user-key length accepted by write APIs. Default: 8 MiB.
    pub max_key_size: usize,
    /// Maximum value and merge-operand length accepted by write APIs.
    /// Default: 64 MiB.
    pub max_value_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
            block_cache_num_shard_bits: 6,
            strict_capacity_limit: false,
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
            compaction_style: CompactionStyle::Level,
            fifo_compaction_options: FifoCompactionOptions::default(),
            universal_compaction_options: UniversalCompactionOptions::default(),
            evict_compaction_data_from_page_cache: false,
            max_background_compactions: 1,
            max_subcompactions: 1,
            partitioned_index: false,
            metadata_block_size: 4096,
            read_only: false,
            max_key_size: DEFAULT_MAX_KEY_SIZE,
            max_value_size: DEFAULT_MAX_VALUE_SIZE,
        }
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("write_buffer_size", &self.write_buffer_size)
            .field("block_size", &self.block_size)
            .field("block_cache_size", &self.block_cache_size)
            .field(
                "block_cache_num_shard_bits",
                &self.block_cache_num_shard_bits,
            )
            .field("strict_capacity_limit", &self.strict_capacity_limit)
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
            .field("compaction_style", &self.compaction_style)
            .field("fifo_compaction_options", &self.fifo_compaction_options)
            .field(
                "universal_compaction_options",
                &self.universal_compaction_options,
            )
            .field(
                "evict_compaction_data_from_page_cache",
                &self.evict_compaction_data_from_page_cache,
            )
            .field(
                "max_background_compactions",
                &self.max_background_compactions,
            )
            .field("max_subcompactions", &self.max_subcompactions)
            .field("partitioned_index", &self.partitioned_index)
            .field("metadata_block_size", &self.metadata_block_size)
            .field("read_only", &self.read_only)
            .field("max_key_size", &self.max_key_size)
            .field("max_value_size", &self.max_value_size)
            .finish()
    }
}

impl Options {
    /// Validate option invariants before the database uses them.
    ///
    /// Public open paths call this automatically. Callers that build
    /// option sets dynamically can also call it directly to fail fast
    /// before touching the filesystem.
    pub fn validate(&self) -> crate::Result<()> {
        require_nonzero_usize("write_buffer_size", self.write_buffer_size)?;
        require_nonzero_usize("block_size", self.block_size)?;
        require_nonzero_usize("block_cache_size", self.block_cache_size)?;
        require_nonzero_usize("l0_compaction_trigger", self.l0_compaction_trigger)?;
        require_nonzero_u64("level_base_bytes", self.level_base_bytes)?;
        require_nonzero_u64("level_size_multiplier", self.level_size_multiplier)?;
        require_nonzero_u64("target_file_size", self.target_file_size)?;
        require_nonzero_usize("metadata_block_size", self.metadata_block_size)?;

        if self.block_cache_num_shard_bits > MAX_BLOCK_CACHE_SHARD_BITS {
            return invalid_option(
                "block_cache_num_shard_bits",
                format!("must be <= {MAX_BLOCK_CACHE_SHARD_BITS}"),
            );
        }

        if !(1..=MAX_BLOOM_BITS_PER_KEY).contains(&self.bloom_bits_per_key) {
            return invalid_option(
                "bloom_bits_per_key",
                format!("must be in 1..={MAX_BLOOM_BITS_PER_KEY}"),
            );
        }

        if self.level0_slowdown_writes_trigger > 0
            && self.level0_stop_writes_trigger > 0
            && self.level0_slowdown_writes_trigger > self.level0_stop_writes_trigger
        {
            return invalid_option(
                "level0_slowdown_writes_trigger",
                "must be <= level0_stop_writes_trigger when both triggers are nonzero",
            );
        }

        if self.soft_pending_compaction_bytes_limit > 0
            && self.hard_pending_compaction_bytes_limit > 0
            && self.soft_pending_compaction_bytes_limit > self.hard_pending_compaction_bytes_limit
        {
            return invalid_option(
                "soft_pending_compaction_bytes_limit",
                "must be <= hard_pending_compaction_bytes_limit when both limits are nonzero",
            );
        }

        if self.max_background_compactions == 0 {
            return invalid_option("max_background_compactions", "must be greater than 0");
        }

        match self.compaction_style {
            CompactionStyle::Level => {}
            CompactionStyle::Fifo => {
                require_nonzero_u64(
                    "fifo_compaction_options.max_table_files_size",
                    self.fifo_compaction_options.max_table_files_size,
                )?;
            }
            CompactionStyle::Universal => {
                let universal = self.universal_compaction_options;
                if universal.size_ratio == 0 {
                    return invalid_option(
                        "universal_compaction_options.size_ratio",
                        "must be greater than 0",
                    );
                }
                if universal.min_merge_width < 2 {
                    return invalid_option(
                        "universal_compaction_options.min_merge_width",
                        "must be at least 2",
                    );
                }
                if universal.max_merge_width < universal.min_merge_width {
                    return invalid_option(
                        "universal_compaction_options.max_merge_width",
                        "must be >= universal_compaction_options.min_merge_width",
                    );
                }
                if universal.max_size_amplification_percent == 0 {
                    return invalid_option(
                        "universal_compaction_options.max_size_amplification_percent",
                        "must be greater than 0",
                    );
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_engine_options(&self) -> crate::engine::EngineOptions {
        crate::engine::EngineOptions {
            write_buffer_size: self.write_buffer_size,
            block_size: self.block_size,
            block_cache_size: self.block_cache_size,
            block_cache_num_shard_bits: self.block_cache_num_shard_bits,
            strict_capacity_limit: self.strict_capacity_limit,
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
            compaction_style: self.compaction_style,
            fifo_compaction_options: self.fifo_compaction_options,
            universal_compaction_options: self.universal_compaction_options,
            evict_compaction_data_from_page_cache: self.evict_compaction_data_from_page_cache,
            max_background_compactions: self.max_background_compactions,
            partitioned_index: self.partitioned_index,
            metadata_block_size: self.metadata_block_size,
            read_only: self.read_only,
            max_key_size: self.max_key_size,
            max_value_size: self.max_value_size,
        }
    }
}

fn require_nonzero_usize(name: &'static str, value: usize) -> crate::Result<()> {
    if value == 0 {
        invalid_option(name, "must be greater than 0")
    } else {
        Ok(())
    }
}

fn require_nonzero_u64(name: &'static str, value: u64) -> crate::Result<()> {
    if value == 0 {
        invalid_option(name, "must be greater than 0")
    } else {
        Ok(())
    }
}

fn invalid_option(name: &'static str, requirement: impl Into<String>) -> crate::Result<()> {
    Err(crate::Error::invalid_argument(format!(
        "invalid option `{name}`: {}",
        requirement.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid_option(opts: Options, expected: &str) {
        match opts.validate().unwrap_err() {
            crate::Error::InvalidArgument(message) => {
                assert!(
                    message.contains(expected),
                    "expected invalid option message to contain {expected:?}, got {message:?}"
                );
            }
            other => panic!("expected invalid argument, got {other:?}"),
        }
    }

    #[test]
    fn fixed_length_prefix_extract() {
        let ex = FixedLengthPrefix(4);
        assert_eq!(ex.extract(b"tenant_001"), Some(&b"tena"[..]));
        assert_eq!(ex.extract(b"abcd"), Some(&b"abcd"[..]));
        assert_eq!(ex.extract(b"abc"), None);
        assert_eq!(ex.extract(b""), None);
        assert_eq!(ex.name(), "FixedLengthPrefix");
    }

    #[test]
    fn fixed_length_prefix_zero_accepts_any_key() {
        let ex = FixedLengthPrefix(0);
        assert_eq!(ex.extract(b"anything"), Some(&b""[..]));
        assert_eq!(ex.extract(b""), Some(&b""[..]));
    }

    #[test]
    fn compaction_decision_equality_and_clone() {
        assert_eq!(CompactionDecision::Keep, CompactionDecision::Keep);
        assert_ne!(CompactionDecision::Keep, CompactionDecision::Remove);
        let c = CompactionDecision::Change(b"new".to_vec());
        assert_eq!(c.clone(), c);
        assert_ne!(c, CompactionDecision::Change(b"other".to_vec()));
    }

    #[test]
    fn write_options_defaults_are_all_false() {
        let wo = WriteOptions::new();
        assert!(!wo.sync);
        assert!(!wo.disable_wal);
        assert!(!wo.low_pri);
        assert!(!wo.no_slowdown);
        assert_eq!(wo, WriteOptions::default());
    }

    #[test]
    fn write_options_sync_constructor_sets_only_sync() {
        let wo = WriteOptions::sync();
        assert!(wo.sync);
        assert!(!wo.disable_wal);
    }

    #[test]
    fn write_options_disable_wal_constructor_sets_only_disable_wal() {
        let wo = WriteOptions::disable_wal();
        assert!(wo.disable_wal);
        assert!(!wo.sync);
    }

    #[test]
    fn compaction_style_default_is_level() {
        assert_eq!(CompactionStyle::default(), CompactionStyle::Level);
    }

    #[test]
    fn fifo_compaction_options_default_is_one_gib() {
        let f = FifoCompactionOptions::default();
        assert_eq!(f.max_table_files_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn options_validate_accepts_defaults_and_disabled_stall_triggers() {
        Options::default().validate().unwrap();

        let opts = Options {
            level0_slowdown_writes_trigger: 0,
            level0_stop_writes_trigger: 0,
            soft_pending_compaction_bytes_limit: 0,
            hard_pending_compaction_bytes_limit: 0,
            max_write_buffer_number: 0,
            ..Options::default()
        };
        opts.validate().unwrap();
    }

    #[test]
    fn options_validate_rejects_zero_core_sizes() {
        assert_invalid_option(
            Options {
                write_buffer_size: 0,
                ..Options::default()
            },
            "write_buffer_size",
        );
        assert_invalid_option(
            Options {
                block_size: 0,
                ..Options::default()
            },
            "block_size",
        );
        assert_invalid_option(
            Options {
                block_cache_size: 0,
                ..Options::default()
            },
            "block_cache_size",
        );
        assert_invalid_option(
            Options {
                l0_compaction_trigger: 0,
                ..Options::default()
            },
            "l0_compaction_trigger",
        );
        assert_invalid_option(
            Options {
                level_base_bytes: 0,
                ..Options::default()
            },
            "level_base_bytes",
        );
        assert_invalid_option(
            Options {
                level_size_multiplier: 0,
                ..Options::default()
            },
            "level_size_multiplier",
        );
        assert_invalid_option(
            Options {
                target_file_size: 0,
                ..Options::default()
            },
            "target_file_size",
        );
        assert_invalid_option(
            Options {
                metadata_block_size: 0,
                ..Options::default()
            },
            "metadata_block_size",
        );
    }

    #[test]
    fn options_validate_rejects_unsupported_ranges() {
        assert_invalid_option(
            Options {
                block_cache_num_shard_bits: MAX_BLOCK_CACHE_SHARD_BITS + 1,
                ..Options::default()
            },
            "block_cache_num_shard_bits",
        );
        assert_invalid_option(
            Options {
                bloom_bits_per_key: 0,
                ..Options::default()
            },
            "bloom_bits_per_key",
        );
        assert_invalid_option(
            Options {
                bloom_bits_per_key: MAX_BLOOM_BITS_PER_KEY + 1,
                ..Options::default()
            },
            "bloom_bits_per_key",
        );
        assert_invalid_option(
            Options {
                max_background_compactions: 0,
                ..Options::default()
            },
            "max_background_compactions",
        );
    }

    #[test]
    fn options_validate_rejects_inconsistent_stall_thresholds() {
        assert_invalid_option(
            Options {
                level0_slowdown_writes_trigger: 10,
                level0_stop_writes_trigger: 5,
                ..Options::default()
            },
            "level0_slowdown_writes_trigger",
        );
        assert_invalid_option(
            Options {
                soft_pending_compaction_bytes_limit: 10,
                hard_pending_compaction_bytes_limit: 5,
                ..Options::default()
            },
            "soft_pending_compaction_bytes_limit",
        );
    }

    #[test]
    fn options_validate_checks_selected_compaction_style_options() {
        Options {
            fifo_compaction_options: FifoCompactionOptions {
                max_table_files_size: 0,
            },
            ..Options::default()
        }
        .validate()
        .unwrap();

        assert_invalid_option(
            Options {
                compaction_style: CompactionStyle::Fifo,
                fifo_compaction_options: FifoCompactionOptions {
                    max_table_files_size: 0,
                },
                ..Options::default()
            },
            "fifo_compaction_options.max_table_files_size",
        );
        assert_invalid_option(
            Options {
                compaction_style: CompactionStyle::Universal,
                universal_compaction_options: UniversalCompactionOptions {
                    size_ratio: 0,
                    ..UniversalCompactionOptions::default()
                },
                ..Options::default()
            },
            "universal_compaction_options.size_ratio",
        );
        assert_invalid_option(
            Options {
                compaction_style: CompactionStyle::Universal,
                universal_compaction_options: UniversalCompactionOptions {
                    min_merge_width: 1,
                    ..UniversalCompactionOptions::default()
                },
                ..Options::default()
            },
            "universal_compaction_options.min_merge_width",
        );
        assert_invalid_option(
            Options {
                compaction_style: CompactionStyle::Universal,
                universal_compaction_options: UniversalCompactionOptions {
                    min_merge_width: 4,
                    max_merge_width: 3,
                    ..UniversalCompactionOptions::default()
                },
                ..Options::default()
            },
            "universal_compaction_options.max_merge_width",
        );
        assert_invalid_option(
            Options {
                compaction_style: CompactionStyle::Universal,
                universal_compaction_options: UniversalCompactionOptions {
                    max_size_amplification_percent: 0,
                    ..UniversalCompactionOptions::default()
                },
                ..Options::default()
            },
            "universal_compaction_options.max_size_amplification_percent",
        );
    }

    #[test]
    fn options_to_engine_options_preserves_engine_fields() {
        // Defaults consumed by the engine must survive the translation
        // unchanged; compatibility-only public knobs are intentionally
        // not present in EngineOptions.
        let opts = Options::default();
        let eo = opts.to_engine_options();
        assert_eq!(eo.write_buffer_size, opts.write_buffer_size);
        assert_eq!(eo.block_size, opts.block_size);
        assert_eq!(eo.block_cache_size, opts.block_cache_size);
        assert_eq!(eo.bloom_bits_per_key, opts.bloom_bits_per_key);
        assert_eq!(eo.l0_compaction_trigger, opts.l0_compaction_trigger);
        assert_eq!(eo.level_base_bytes, opts.level_base_bytes);
        assert_eq!(eo.level_size_multiplier, opts.level_size_multiplier);
        assert_eq!(eo.target_file_size, opts.target_file_size);
        assert_eq!(eo.compaction_style, opts.compaction_style);
        assert_eq!(eo.read_only, opts.read_only);
        assert_eq!(eo.max_key_size, opts.max_key_size);
        assert_eq!(eo.max_value_size, opts.max_value_size);
        assert_eq!(
            eo.max_background_compactions,
            opts.max_background_compactions
        );
        assert_eq!(
            eo.evict_compaction_data_from_page_cache,
            opts.evict_compaction_data_from_page_cache
        );
        assert_eq!(eo.partitioned_index, opts.partitioned_index);
    }

    /// Custom `PrefixExtractor` that returns everything before the
    /// first `:` — proves the trait is user-implementable.
    #[test]
    fn custom_prefix_extractor_can_be_plugged_in() {
        struct UntilColon;
        impl PrefixExtractor for UntilColon {
            fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]> {
                key.iter().position(|&b| b == b':').map(|i| &key[..i])
            }
            fn name(&self) -> &'static str {
                "UntilColon"
            }
        }
        let ex: Arc<dyn PrefixExtractor> = Arc::new(UntilColon);
        assert_eq!(ex.extract(b"tenant:key"), Some(&b"tenant"[..]));
        assert_eq!(ex.extract(b"nocolon"), None);
        assert_eq!(ex.name(), "UntilColon");
    }
}
