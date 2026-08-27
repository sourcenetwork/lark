use std::sync::Arc;

use crate::engine::internal_key::INTERNAL_KEY_SUFFIX_LEN;

/// Default maximum user-key length accepted by write APIs: 8 MiB.
pub const DEFAULT_MAX_KEY_SIZE: usize = 8 * 1024 * 1024;

/// Default maximum value / merge-operand length accepted by write APIs: 64 MiB.
pub const DEFAULT_MAX_VALUE_SIZE: usize = 64 * 1024 * 1024;

/// Highest supported block-cache shard exponent.
pub const MAX_BLOCK_CACHE_SHARD_BITS: u32 = 8;

/// Highest supported Bloom-filter density. Larger values waste space
/// because the hash count is already capped internally.
pub const MAX_BLOOM_BITS_PER_KEY: usize = 64;

/// Default value of [`Options::max_background_compactions`]: one
/// background worker, or none on a target that has no threads.
///
/// No wasm build can carry a background compaction worker, so this is
/// `0` on every wasm target rather than only on the single-threaded
/// ones. Two independent reasons, either of which alone is decisive:
/// without the `atomics` target feature the module has exactly one
/// thread and `std::thread::spawn` reports
/// [`std::io::ErrorKind::Unsupported`]; and the worker's blocking wait
/// is built on thread-parking primitives that do not exist for
/// `wasm32` at all. `0` runs compaction on the calling thread instead.
///
/// This is a compile-time property of the target, not a runtime
/// fallback: the build cannot grow a second thread later.
#[cfg(target_family = "wasm")]
pub const DEFAULT_MAX_BACKGROUND_COMPACTIONS: usize = 0;

/// Default value of [`Options::max_background_compactions`]: one
/// background worker, or none on a target that has no threads.
#[cfg(not(target_family = "wasm"))]
pub const DEFAULT_MAX_BACKGROUND_COMPACTIONS: usize = 1;

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
/// value)` - the same input should always yield the same decision.
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
/// They MUST NOT read from the database - doing so will deadlock -
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
    /// Fail the write instead of waiting when the engine is stalling.
    ///
    /// A write that would otherwise block on L0 or memtable pressure
    /// returns [`crate::Error::Busy`] straight away, carrying the reason
    /// it would have waited for. When the engine is not stalling this has
    /// no effect.
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
/// Point lookups are unaffected - they always consult the user-key
/// bloom filter and do not call the extractor.
pub trait PrefixExtractor: Send + Sync + 'static {
    /// Return the portion of `key` that the prefix bloom should index,
    /// or `None` if `key` cannot produce a prefix (e.g., it's shorter
    /// than a fixed-length extractor's width). Keys that return `None`
    /// are simply absent from the prefix bloom.
    fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]>;

    /// Return the bloom key that is safe to probe for a caller's
    /// prefix-bounded scan. The default implementation only permits
    /// exact extracted-prefix queries, preserving correctness for
    /// custom extractors whose output may depend on more than a fixed
    /// byte width. Extractors with stronger prefix-stability guarantees
    /// can override this to enable broader SSTable skipping.
    fn extract_query<'a>(&self, prefix: &'a [u8]) -> Option<&'a [u8]> {
        let extracted = self.extract(prefix)?;
        (extracted == prefix).then_some(extracted)
    }

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

    fn extract_query<'a>(&self, prefix: &'a [u8]) -> Option<&'a [u8]> {
        self.extract(prefix)
    }

    fn name(&self) -> &'static str {
        "FixedLengthPrefix"
    }
}

/// Compaction strategy used by the background compaction thread.
///
/// Regolith currently ships two styles. More can be added in follow-up
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
    /// FIFO: regolith never merges files. Once the total size of all
    /// SSTables exceeds [`FifoCompactionOptions::max_table_files_size`],
    /// the oldest SSTable is unlinked. Best fit for time-series and
    /// append-only log workloads where the oldest data is also the
    /// least valuable. Reads still consult every L0 file (there is
    /// no L1+) so read amplification grows with the number of
    /// retained files; this is the trade-off for ~zero write
    /// amplification.
    ///
    /// Because nothing merges L0, the L0 *count* triggers
    /// ([`Options::level0_slowdown_writes_trigger`] and
    /// [`Options::level0_stop_writes_trigger`]) cannot be relieved by
    /// compaction under this style. Set them to `0` unless the byte
    /// cap is guaranteed to bite first.
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
    ///
    /// Merging is driven by [`UniversalCompactionOptions`], not by the
    /// L0 *count* triggers: a well-formed size tier legitimately holds
    /// many L0 files without merging any of them. Set
    /// [`Options::level0_stop_writes_trigger`] and
    /// [`Options::level0_slowdown_writes_trigger`] to `0` under this
    /// style, or raise them well past the tier depth the workload
    /// produces.
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
/// All variants are pure-Rust implementations - no C/C++ toolchain
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

pub use crate::engine::arena::ArenaProfile;

/// Configuration options for a regolith database.
#[derive(Clone)]
pub struct Options {
    /// Write buffer (memtable) size before flush. Must be greater
    /// than zero. Default: 64 MB.
    ///
    /// This bounds the memtable's arena bytes: the node header, tower,
    /// internal key and value of every entry, rounded to alignment. A
    /// memtable's arena reserves at most
    /// `write_buffer_size + max(arena_profile.max_chunk_size,
    /// largest single entry)` between writes, because the engine rotates
    /// as soon as the budget is reached. A single `WriteBatch` larger
    /// than the remaining budget overshoots by that batch's size, so
    /// batch size is the caller's to bound.
    pub write_buffer_size: usize,
    /// Chunk sizing policy for the memtable arena. Default:
    /// [`ArenaProfile::SERVER`]; [`Options::embedded`] selects the
    /// small-footprint preset.
    pub arena_profile: ArenaProfile,
    /// Data block size in SSTables. Must be greater than zero.
    /// Default: 16 KB.
    pub block_size: usize,
    /// Block cache size in bytes for decompressed data blocks.
    /// `0` disables the block cache entirely: nothing is allocated
    /// for it, no block is retained, every read goes to the file,
    /// and the block-cache tickers stay at zero. Default: 512 MB.
    ///
    /// What the cache allocates tracks this budget, not the shard
    /// count: shard maps start empty and each shard caps its entry
    /// count at its share of the budget.
    pub block_cache_size: usize,
    /// Base-2 log of the block cache shard count. The block cache
    /// is split into `2^block_cache_num_shard_bits` shards keyed
    /// by `hash(file_id, offset)` so concurrent readers contend
    /// only with other readers that hash to the same shard.
    /// Must be <= [`MAX_BLOCK_CACHE_SHARD_BITS`]. Tiny cache budgets
    /// may use fewer effective shards so each shard has usable
    /// capacity, and a [`Options::block_cache_size`] of 0 makes this
    /// setting irrelevant because no shard is allocated.
    /// Default: 6 (64 shards).
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
    /// Regolith's column-family implementation is key-prefix based:
    /// every CF shares one memtable, one WAL, one manifest, and
    /// one flush path, so a multi-CF [`crate::WriteBatch`] is
    /// **always** atomic across CFs regardless of this flag's
    /// value. A flush either persists every participant's half of
    /// a batch or persists none of it.
    pub atomic_flush: bool,
    /// Event listeners subscribed to engine lifecycle events
    /// (flush, compaction, ingest, background errors). Dispatch
    /// is synchronous on the firing thread - listeners **must not
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
    ///
    /// Level-style, with the same caveat as
    /// [`Options::level0_stop_writes_trigger`]: under FIFO and
    /// universal compaction the L0 file count is not what the picker
    /// reduces, so this delay can become permanent rather than
    /// transient.
    pub level0_slowdown_writes_trigger: usize,
    /// Stop foreground writes entirely when the number of L0
    /// SSTables reaches this threshold. Writers block on a
    /// condvar that compaction notifies once it reduces the
    /// count below the slowdown trigger. `0` disables this
    /// trigger. Default: 36.
    ///
    /// # This is a level-style trigger
    ///
    /// Only [`CompactionStyle::Level`] reduces the L0 *file count*
    /// in response to this threshold. Under [`CompactionStyle::Fifo`]
    /// nothing ever merges L0 files (only the byte cap unlinks them),
    /// and under [`CompactionStyle::Universal`] the picker merges on
    /// its own size-ratio and amplification rules, which a healthy
    /// size tier can satisfy while sitting above this count. With
    /// either of those styles the threshold can therefore be reached
    /// and never relieved, and writes then fail with
    /// [`crate::Error::Busy`] until the configuration changes. The
    /// error message names the style and this field. Set this to `0`
    /// with those styles and bound memory with
    /// [`Options::max_write_buffer_number`] and
    /// [`Options::hard_pending_compaction_bytes_limit`] instead,
    /// which both apply to every style.
    pub level0_stop_writes_trigger: usize,
    /// Start slowing writes when total bytes in L0 (regolith's
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
    /// `0` starts no background worker at all. Compaction then runs
    /// on whichever thread asks for it: a writer that reaches a
    /// write-stall threshold performs a compaction job itself before
    /// retrying instead of parking on a condvar nobody would ever
    /// signal. This is the only mode that works on a single-threaded
    /// host such as `wasm32-wasip1`, where `std::thread::spawn`
    /// reports [`std::io::ErrorKind::Unsupported`], and it is what
    /// [`Options::embedded`] selects.
    ///
    /// When `> 1`, multiple non-overlapping compaction jobs can
    /// run concurrently (e.g. L1→L2 at key range `[a,m)` on one
    /// worker while L2→L3 at `[m,z)` runs on another). L0
    /// compactions are exclusive - only one L0 job runs at a
    /// time because L0 files can overlap arbitrarily.
    ///
    /// `1` keeps compaction single-threaded and matches
    /// pre-multi-worker behavior. It is the default off wasm; see
    /// [`DEFAULT_MAX_BACKGROUND_COMPACTIONS`] for why every wasm
    /// target defaults to `0` instead, so that [`crate::Db::open`]
    /// with [`Options::default`] works there too. On wasm any other
    /// value is rejected by [`Options::validate`].
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
    /// hint is a no-op. `false` by default - callers who care
    /// about foreground latency stability on Linux should turn
    /// it on.
    pub evict_compaction_data_from_page_cache: bool,
    /// Split the SSTable index into small leaf blocks on disk and keep
    /// only a compact top-level index in memory. Reduces resident
    /// memory when thousands of SSTables are open, at the cost of one
    /// extra disk read per point lookup (amortized by the OS page
    /// cache). Default: `false` (flat index loaded eagerly).
    pub partitioned_index: bool,
    /// Charge SSTable index and filter blocks to the block cache
    /// instead of pinning them in each open reader.
    ///
    /// With this off (the default), every open SSTable holds its whole
    /// index and its bloom filters resident for the reader's lifetime,
    /// outside [`Options::block_cache_size`]. With it on they are read
    /// through the cache and are evictable, so `block_cache_size`
    /// bounds index and filter bytes as well as data bytes. The cost is
    /// one cache lookup on the point-read path and a re-read from disk
    /// whenever an evicted filter is next consulted, which is why the
    /// default is off.
    ///
    /// Independent of [`Options::partitioned_index`]: the *leaves* of a
    /// partitioned index always go through the cache, and a partitioned
    /// file's top-level index is always pinned. This option decides
    /// where a flat file's whole index and every file's filter region
    /// live.
    ///
    /// `Db::get_int_property("regolith.pinned-metadata-bytes")` reports
    /// what the open files are holding outside the cache budget either
    /// way.
    ///
    /// Default: `false`.
    pub cache_index_and_filter_blocks: bool,
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
    /// The host platform this database runs on: its filesystem, its
    /// clock, and its threads.
    ///
    /// Defaults to [`crate::env::StdEnv`], which is `std::fs` +
    /// `std::time` + `std::thread` and behaves exactly as regolith did
    /// before this field existed. Replace it to run on a filesystem
    /// regolith does not know about, or on [`crate::env::MemEnv`] to keep
    /// a database entirely in memory.
    ///
    /// What the environment can actually do is reported by
    /// [`crate::env::Env::capabilities`] and handed back to callers
    /// through [`crate::Db::capabilities`].
    pub env: Arc<dyn crate::env::Env>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            arena_profile: ArenaProfile::SERVER,
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
            max_background_compactions: DEFAULT_MAX_BACKGROUND_COMPACTIONS,
            max_subcompactions: 1,
            partitioned_index: false,
            cache_index_and_filter_blocks: false,
            metadata_block_size: 4096,
            read_only: false,
            max_key_size: DEFAULT_MAX_KEY_SIZE,
            max_value_size: DEFAULT_MAX_VALUE_SIZE,
            env: crate::env::std_env(),
        }
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("write_buffer_size", &self.write_buffer_size)
            .field("arena_profile", &self.arena_profile)
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
            .field(
                "cache_index_and_filter_blocks",
                &self.cache_index_and_filter_blocks,
            )
            .field("metadata_block_size", &self.metadata_block_size)
            .field("read_only", &self.read_only)
            .field("max_key_size", &self.max_key_size)
            .field("max_value_size", &self.max_value_size)
            .field("env", &self.env)
            .finish()
    }
}

impl Options {
    /// Tuning for a device or module whose whole working set has to
    /// fit in roughly 1-4 MiB: a Linux-class embedded board (Cortex-A,
    /// or an ESP32-S3 under esp-idf), or a wasm module, where every
    /// 64 KiB page linear memory ever touches stays committed for the
    /// life of the instance.
    ///
    /// [`Options::default`] is tuned for a server and reserves far
    /// more than such a host has: a 64 MiB write buffer, a 512 MiB
    /// block cache, and a 64 MiB target file size. This profile
    /// replaces every one of those with a value that is bounded by the
    /// budget rather than by throughput.
    ///
    /// # The budget, term by term
    ///
    /// Steady-state resident cost, which is what the 1-4 MiB figure
    /// has to cover:
    ///
    /// - **Memtables, ~0.5 MiB of entry bytes.**
    ///   [`Options::write_buffer_size`] is 256 KiB and
    ///   [`Options::max_write_buffer_number`] is 2, so at most one
    ///   active plus one frozen memtable is resident. Skip-list node
    ///   overhead sits on top of the entry bytes, so budget more than
    ///   512 KiB for this term, not exactly 512 KiB.
    /// - **Block cache, exactly 0 bytes.**
    ///   [`Options::block_cache_size`] of 0 allocates no shard at all,
    ///   retains no block, and leaves the cache tickers at zero. A
    ///   host with an OS page cache absorbs the resulting re-reads;
    ///   a wasm module serves them out of bytes it already holds.
    ///   Raise it to 256 KiB if the budget is nearer 4 MiB than 1 MiB.
    /// - **Per-SSTable metadata, linear in the number of open files.**
    ///   Each open SSTable holds its index block and Bloom filter
    ///   resident. With a 256 KiB [`Options::target_file_size`], a
    ///   4 KiB [`Options::block_size`], and 10
    ///   [`Options::bloom_bits_per_key`], that is roughly 4-6 KiB per
    ///   file, so about 0.25 MiB for a 16 MiB database. This term is
    ///   the one that grows without bound as the database grows; see
    ///   the note on [`Options::partitioned_index`] below.
    /// - **Transient flush and compaction buffers**, bounded by
    ///   [`Options::target_file_size`] and [`Options::block_size`]
    ///   rather than by the size of the level being merged.
    ///
    /// # Every value, and what it buys
    ///
    /// - [`Options::write_buffer_size`] `256 KiB`: the largest
    ///   steady-state allocation, and the bound on how much WAL
    ///   accumulates between rotations.
    /// - [`Options::max_write_buffer_number`] `2`: caps in-memory
    ///   entry bytes at two write buffers.
    /// - [`Options::block_size`] `4 KiB`: matches LittleFS and SPIFFS
    ///   page granularity, and caps the transient decompression buffer
    ///   for an uncached read at 4 KiB instead of the default 16 KiB.
    ///   That transient matters more here precisely because there is
    ///   no block cache to amortize it.
    /// - [`Options::block_cache_size`] `0` and
    ///   [`Options::block_cache_num_shard_bits`] `0`: no cache, and
    ///   the shard count pinned so that raising the budget later does
    ///   not silently allocate 64 shards.
    /// - [`Options::target_file_size`] `256 KiB`: bounds one
    ///   compaction output file, and keeps compaction holding a block
    ///   at a time rather than a file at a time.
    /// - [`Options::level_base_bytes`] `1 MiB` with
    ///   [`Options::level_size_multiplier`] `10`: L1 1 MiB, L2 10 MiB,
    ///   L3 100 MiB - a shape suited to flash measured in tens of MiB.
    /// - [`Options::l0_compaction_trigger`] `2`,
    ///   [`Options::level0_slowdown_writes_trigger`] `4`,
    ///   [`Options::level0_stop_writes_trigger`] `8`: compact early
    ///   and often. A writer that compacts on its own thread should
    ///   pay in small installments rather than take one long stall,
    ///   and a low L0 count also caps how many files a point lookup
    ///   has to consult.
    /// - [`Options::soft_pending_compaction_bytes_limit`] `4 MiB` and
    ///   [`Options::hard_pending_compaction_bytes_limit`] `16 MiB`:
    ///   the byte-denominated back-pressure, scaled to flash rather
    ///   than to the default's 64 GiB / 256 GiB.
    /// - [`Options::max_background_compactions`] `0`: compaction runs
    ///   on the calling thread. Required on a single-threaded host;
    ///   on a board with a spare core, set this to `1`.
    /// - [`Options::evict_compaction_data_from_page_cache`] `true`: on
    ///   a device whose page cache is a few MiB, compaction streaming
    ///   through it would evict every hot foreground page. The hint is
    ///   `posix_fadvise(DONTNEED)` on Linux and a no-op elsewhere.
    /// - [`Options::max_key_size`] `16 KiB` and
    ///   [`Options::max_value_size`] `256 KiB`: the defaults accept an
    ///   8 MiB key and a 64 MiB value, either of which exceeds the
    ///   whole budget on its own. Capping a value at one write buffer
    ///   keeps a single accepted write from being one the memtable
    ///   cannot hold.
    /// - [`Options::compression`] stays [`CompressionType::Lz4`],
    ///   inherited from [`Options::default`]. It is pure Rust, its
    ///   block format needs no allocator tricks, and it reduces flash
    ///   writes, which is the resource that wears out.
    /// - [`Options::bloom_bits_per_key`] stays `10`: about a 1% false
    ///   positive rate for 1.25 bytes per key. Cutting it saves little
    ///   memory and costs real reads.
    /// - [`Options::partitioned_index`] stays `false`, so each open
    ///   SSTable's index is resident. That is the cheaper choice while
    ///   the per-file metadata term above stays small. Past roughly
    ///   100 MiB of database - about 400 files at this file size, or
    ///   ~1 MiB of resident index - set it to `true` to keep only a
    ///   top-level index in memory, at the cost of one extra read per
    ///   point lookup. [`Options::metadata_block_size`] is already set
    ///   to 1 KiB here so that switch does not also need retuning.
    ///
    /// # Not a promise about your workload
    ///
    /// These values bound what regolith reserves. They do not bound what
    /// your keys and values cost, nor what the allocator does with
    /// fragmentation. Measure on the target.
    ///
    /// ```
    /// use regolith::{Db, Options};
    ///
    /// let dir = tempfile::TempDir::new()?;
    /// let db = Db::open(dir.path(), Options::embedded())?;
    /// db.put(b"k", b"v")?;
    /// assert_eq!(db.get(b"k")?.as_deref(), Some(&b"v"[..]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Why the memtable term stays flat
    ///
    /// Memtable-attributable memory is bounded by
    /// `2 * M * (W + c) + M * W` where `W = write_buffer_size`,
    /// `c = arena_profile.max_chunk_size` and
    /// `M = max_write_buffer_number`: at most `2 * M` memtables are in
    /// memory at once (the write stall stops writers there), each arena
    /// holds at most `W + c`, and the recycling pool parks at most
    /// `M * W`.
    ///
    /// That flatness is the point on wasm32, where linear memory grows
    /// in 64 KiB pages and never shrinks, so every peak is permanent.
    /// [`ArenaProfile::EMBEDDED`] caps a chunk at exactly one page and
    /// the pool hands the same pages back to the next memtable, so
    /// `memory.grow` runs once per size class rather than once per
    /// flush-and-refill cycle.
    pub fn embedded() -> Self {
        Self {
            write_buffer_size: 256 * 1024,
            arena_profile: ArenaProfile::EMBEDDED,
            block_size: 4 * 1024,
            block_cache_size: 0,
            block_cache_num_shard_bits: 0,
            l0_compaction_trigger: 2,
            level_base_bytes: 1024 * 1024,
            target_file_size: 256 * 1024,
            level0_slowdown_writes_trigger: 4,
            level0_stop_writes_trigger: 8,
            soft_pending_compaction_bytes_limit: 4 * 1024 * 1024,
            hard_pending_compaction_bytes_limit: 16 * 1024 * 1024,
            max_write_buffer_number: 2,
            max_background_compactions: 0,
            evict_compaction_data_from_page_cache: true,
            metadata_block_size: 1024,
            max_key_size: 16 * 1024,
            max_value_size: 256 * 1024,
            ..Self::default()
        }
    }

    /// Tuning for a wasm module: `wasm32-unknown-unknown` in a
    /// browser, or `wasm32-wasip1` under a wasi host.
    ///
    /// wasm is not just a small machine, and that is why this is a
    /// separate profile from [`Options::embedded`] rather than an
    /// alias for it. Three properties of the platform drive every
    /// value that differs.
    ///
    /// # What makes wasm different
    ///
    /// - **There are no threads, and there is no way to wait for
    ///   one.** [`Options::max_background_compactions`] is `0` and on
    ///   a wasm target [`Options::validate`] rejects anything else,
    ///   so the failure arrives at the option rather than out of the
    ///   middle of [`crate::Db::open`]. Compaction runs on whichever
    ///   thread asks for it. Every other value here is chosen so that
    ///   the writer paying that cost pays it in small installments.
    /// - **There is no OS page cache.** This is the sharpest split
    ///   from [`Options::embedded`], which sets
    ///   [`Options::block_cache_size`] to `0` precisely because a
    ///   Linux-class board has a page cache to absorb the re-reads. A
    ///   wasm module has none, so a cache miss costs a host call and,
    ///   because [`Options::compression`] defaults to
    ///   [`CompressionType::Lz4`], a fresh decompression every time.
    ///   The cache here holds decompressed blocks, so it buys back
    ///   both. It is 1 MiB, and one shard, because a single-threaded
    ///   host never contends on it.
    /// - **Linear memory grows in 64 KiB pages and never shrinks.**
    ///   Every peak is permanent for the life of the instance, so the
    ///   figure to keep down is the high-water mark, not the average.
    ///   [`ArenaProfile::EMBEDDED`] caps a chunk at exactly one page
    ///   and the memtable pool hands the same pages to the next
    ///   memtable, so `memory.grow` runs once per size class rather
    ///   than once per flush-and-refill cycle.
    ///
    /// # Where the budget goes
    ///
    /// Up to 2 MiB of memtable entry bytes (a 1 MiB write buffer, at
    /// most one active plus one frozen), 1 MiB of block cache, and
    /// per-SSTable index and Bloom residency that grows with the file
    /// count. Skip-list node overhead sits on top of the entry bytes.
    /// Past roughly 100 MiB of database, set
    /// [`Options::partitioned_index`] to `true` to stop the per-file
    /// index term from growing without bound;
    /// [`Options::metadata_block_size`] is already 1 KiB here so that
    /// switch needs no further retuning.
    ///
    /// Measured, not estimated. The `embedded_profile` example run on
    /// `wasm32-wasip1` under wasmtime (open, 20000 x 128 B puts,
    /// sampled reads, page scan, compact, close, reopen, read back)
    /// reaches **4480 KiB** of linear memory, of which 3392 KiB is
    /// over the pre-open baseline, against **1600 KiB** for
    /// [`Options::embedded`] and **11456 KiB** for
    /// [`Options::default`] on the same run. Most of the gap over
    /// `embedded` is the block cache this profile buys and `embedded`
    /// declines, plus the larger write buffer. Linear memory only
    /// grows, so for a wasm module the high-water mark *is* the
    /// footprint. Reproduce with `just wasm-budget`.
    ///
    /// [`Options::evict_compaction_data_from_page_cache`] is `false`,
    /// unlike [`Options::embedded`]: the hint is `posix_fadvise` on
    /// Linux and a no-op everywhere else, so asking for it on wasm
    /// would imply an effect the target cannot deliver.
    ///
    /// [`Options::max_value_size`] is 1 MiB, exactly one write
    /// buffer, which is the same rule [`Options::embedded`] follows:
    /// a write the API accepts must fit in a memtable that can hold
    /// it. Raising it above [`Options::write_buffer_size`] is
    /// supported and costs a flush per outsized value, which on wasm
    /// is also paid in pages that never come back.
    ///
    /// # Not a promise about your workload
    ///
    /// These values bound what regolith reserves. They do not bound what
    /// your keys and values cost, nor what the allocator does with
    /// fragmentation. Measure in the target runtime.
    ///
    /// # It does not choose a filesystem for you
    ///
    /// On `wasm32-wasip1` the default [`Options::env`] works: wasi has
    /// a real `std::fs` behind a preopened directory. On
    /// `wasm32-unknown-unknown` it does not, because that target has no
    /// filesystem at all and every `std::fs` call reports
    /// [`std::io::ErrorKind::Unsupported`]. In a browser, mount OPFS
    /// and set the env yourself:
    ///
    /// ```ignore
    /// let opfs = OpfsEnv::mount("my-db", OpfsOptions::default()).await?;
    /// let mut options = Options::wasm();
    /// options.env = opfs.as_env();
    /// let db = Db::open(opfs.db_path(), options)?;
    /// ```
    ///
    /// Mounting is async, which is why the profile cannot do it.
    ///
    /// The profile is a plain value with no wasm-only fields, so it
    /// builds and opens on the host too, which is what lets it be
    /// exercised by the normal test suite:
    ///
    /// ```
    /// use regolith::{Db, Options};
    ///
    /// let dir = tempfile::TempDir::new()?;
    /// let db = Db::open(dir.path(), Options::wasm())?;
    /// db.put(b"k", b"v")?;
    /// assert_eq!(db.get(b"k")?.as_deref(), Some(&b"v"[..]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn wasm() -> Self {
        Self {
            write_buffer_size: 1024 * 1024,
            arena_profile: ArenaProfile::EMBEDDED,
            block_size: 4 * 1024,
            block_cache_size: 1024 * 1024,
            block_cache_num_shard_bits: 0,
            l0_compaction_trigger: 4,
            level_base_bytes: 8 * 1024 * 1024,
            target_file_size: 1024 * 1024,
            level0_slowdown_writes_trigger: 8,
            level0_stop_writes_trigger: 12,
            soft_pending_compaction_bytes_limit: 32 * 1024 * 1024,
            hard_pending_compaction_bytes_limit: 128 * 1024 * 1024,
            max_write_buffer_number: 2,
            max_background_compactions: 0,
            evict_compaction_data_from_page_cache: false,
            metadata_block_size: 1024,
            max_key_size: 16 * 1024,
            max_value_size: 1024 * 1024,
            ..Self::default()
        }
    }

    /// Validate option invariants before the database uses them.
    ///
    /// Public open paths call this automatically. Callers that build
    /// option sets dynamically can also call it directly to fail fast
    /// before touching the filesystem.
    pub fn validate(&self) -> crate::Result<()> {
        require_nonzero_usize("write_buffer_size", self.write_buffer_size)?;
        require_nonzero_usize("block_size", self.block_size)?;
        require_nonzero_usize("l0_compaction_trigger", self.l0_compaction_trigger)?;
        require_nonzero_u64("level_base_bytes", self.level_base_bytes)?;
        require_nonzero_u64("level_size_multiplier", self.level_size_multiplier)?;
        require_nonzero_u64("target_file_size", self.target_file_size)?;
        require_nonzero_usize("metadata_block_size", self.metadata_block_size)?;

        // No wasm build can carry a compaction worker: `Env::spawn`
        // reports `Unsupported` there and the worker's blocking wait
        // has no wasm implementation to build against. Reject it here
        // so the reason arrives with the option that caused it rather
        // than as an io error from the middle of `open`.
        #[cfg(target_family = "wasm")]
        if self.max_background_compactions > 0 {
            return invalid_option(
                "max_background_compactions",
                "must be 0 on wasm, which has no threads; compaction then runs on the \
                 calling thread. Options::wasm() sets this and the rest of the profile",
            );
        }

        if !self.arena_profile.is_valid() {
            return invalid_option(
                "arena_profile",
                "initial_chunk_size and max_chunk_size must be powers of two with initial <= max",
            );
        }

        // The memtable stores each entry's key and value length in a
        // `u32` inside its arena node, so an entry that could not be
        // encoded must be rejected at the boundary rather than at the
        // write. What the node holds is the *internal* key, which is the
        // caller's key plus the sequence and value-type suffix, so the
        // key ceiling is that much lower: allowing a key of exactly
        // `u32::MAX` would let a write pass validation, reach the WAL and
        // take a sequence number, and then be dropped by the memtable.
        const MAX_ENCODABLE_KEY: usize = u32::MAX as usize - INTERNAL_KEY_SUFFIX_LEN;
        if self.max_key_size > MAX_ENCODABLE_KEY {
            return invalid_option(
                "max_key_size",
                "must be <= u32::MAX minus the 9-byte internal-key suffix",
            );
        }
        if self.max_value_size > u32::MAX as usize {
            return invalid_option("max_value_size", "must be <= u32::MAX");
        }

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

        // Fail loud rather than accepting a durability the platform
        // cannot keep: an env without `durable_sync` returns `Ok(())`
        // from `sync_all` without making anything durable, so honouring
        // `Immediate` there would report a guarantee that does not
        // exist. OPFS in mirror mode is the case this catches.
        if matches!(self.durability, DurabilityMode::Immediate)
            && !self.env.capabilities().durable_sync
        {
            return invalid_option(
                "durability",
                "DurabilityMode::Immediate needs an Env whose \
                 Capabilities::durable_sync is true; this one reports false, \
                 so a synced write would not be durable",
            );
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
            arena_profile: self.arena_profile,
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
            cache_index_and_filter_blocks: self.cache_index_and_filter_blocks,
            metadata_block_size: self.metadata_block_size,
            read_only: self.read_only,
            max_key_size: self.max_key_size,
            max_value_size: self.max_value_size,
            env: Arc::clone(&self.env),
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
    fn options_validate_accepts_zero_block_cache_size() {
        Options {
            block_cache_size: 0,
            ..Options::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn zero_block_cache_db_opens_and_reads_correctly() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = crate::Db::open(
            dir.path(),
            Options {
                block_cache_size: 0,
                write_buffer_size: 4 * 1024,
                ..Options::default()
            },
        )
        .unwrap();
        for i in 0..500u32 {
            db.put(
                format!("key{i:04}").as_bytes(),
                format!("value{i}").as_bytes(),
            )
            .unwrap();
        }
        // Small write buffer forces flushes, so most of these reads
        // come from SSTables and exercise the uncached block path.
        for i in 0..500u32 {
            assert_eq!(
                db.get(format!("key{i:04}").as_bytes()).unwrap(),
                Some(format!("value{i}").into_bytes())
            );
        }
        assert!(db.get(b"absent").unwrap().is_none());
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
    }

    #[test]
    fn zero_background_compactions_is_a_supported_configuration() {
        Options {
            max_background_compactions: 0,
            ..Options::default()
        }
        .validate()
        .expect("zero background workers selects foreground compaction, not an error");
    }

    #[test]
    fn embedded_profile_validates() {
        Options::embedded()
            .validate()
            .expect("the shipped embedded profile must satisfy its own invariants");
    }

    #[test]
    fn embedded_profile_bounds_every_resident_term() {
        let o = Options::embedded();

        // Entry bytes held in memory at once: two write buffers.
        assert_eq!(o.write_buffer_size, 256 * 1024);
        assert_eq!(o.max_write_buffer_number, 2);

        // The cache must allocate nothing at all, not merely a little.
        assert_eq!(o.block_cache_size, 0);
        assert_eq!(o.block_cache_num_shard_bits, 0);

        // Per-file metadata and transient buffers.
        assert_eq!(o.block_size, 4 * 1024);
        assert_eq!(o.target_file_size, 256 * 1024);
        assert_eq!(o.metadata_block_size, 1024);

        // A single accepted write must not exceed one write buffer,
        // or the memtable cannot hold what the API just admitted.
        assert!(o.max_value_size <= o.write_buffer_size);
        assert!(o.max_key_size < o.write_buffer_size);

        // Compaction runs on the calling thread.
        assert_eq!(o.max_background_compactions, 0);
    }

    #[test]
    fn wasm_profile_validates() {
        Options::wasm()
            .validate()
            .expect("the shipped wasm profile must satisfy its own invariants");
    }

    #[test]
    fn wasm_profile_bounds_every_resident_term() {
        let o = Options::wasm();

        // Entry bytes held in memory at once: two write buffers.
        assert_eq!(o.write_buffer_size, 1024 * 1024);
        assert_eq!(o.max_write_buffer_number, 2);

        // A chunk is exactly one wasm page, so a chunk never straddles
        // more pages than its contents need.
        assert_eq!(o.arena_profile.max_chunk_size, 64 * 1024);

        // Unlike `embedded`, the cache is real: wasm has no OS page
        // cache behind it. One shard, because nothing contends.
        assert_eq!(o.block_cache_size, 1024 * 1024);
        assert_eq!(o.block_cache_num_shard_bits, 0);

        // The same admission rule `embedded` holds: a write the API
        // accepts must fit in a memtable that can hold it.
        assert!(o.max_value_size <= o.write_buffer_size);
        assert!(o.max_key_size < o.write_buffer_size);

        // Compaction runs on the calling thread. This is not a tuning
        // choice on wasm; `validate` rejects anything else there.
        assert_eq!(o.max_background_compactions, 0);

        // The page-cache hint is a no-op on wasm, so the profile does
        // not ask for an effect the target cannot deliver.
        assert!(!o.evict_compaction_data_from_page_cache);

        // Already set for the `partitioned_index` switch the doc
        // points at, so flipping it later needs no further retuning.
        assert_eq!(o.metadata_block_size, 1024);
    }

    #[test]
    fn wasm_profile_is_smaller_than_the_default_but_roomier_than_embedded() {
        let d = Options::default();
        let w = Options::wasm();
        let e = Options::embedded();

        assert!(w.write_buffer_size < d.write_buffer_size);
        assert!(w.block_cache_size < d.block_cache_size);
        assert!(w.max_value_size < d.max_value_size);
        assert!(w.hard_pending_compaction_bytes_limit < d.hard_pending_compaction_bytes_limit);

        // Roomier than `embedded` everywhere it differs, because a
        // browser tab or a wasi host is not an ESP32.
        assert!(w.write_buffer_size > e.write_buffer_size);
        assert!(w.target_file_size > e.target_file_size);
        assert!(w.level_base_bytes > e.level_base_bytes);
        assert!(w.max_value_size > e.max_value_size);

        // The one term that goes the other way, and the reason this is
        // a separate profile rather than an alias for `embedded`.
        assert_eq!(e.block_cache_size, 0);
        assert!(w.block_cache_size > e.block_cache_size);
    }

    #[test]
    fn default_profile_opens_on_every_target_it_is_compiled_for() {
        // `Db::open` validates, so a default that a target rejects
        // would make the zero-configuration path unopenable there.
        Options::default()
            .validate()
            .expect("Options::default must be valid on the target it was built for");
        #[cfg(target_family = "wasm")]
        assert_eq!(Options::default().max_background_compactions, 0);
    }

    #[test]
    fn embedded_profile_is_strictly_smaller_than_the_default() {
        let d = Options::default();
        let e = Options::embedded();
        assert!(e.write_buffer_size < d.write_buffer_size);
        assert!(e.block_cache_size < d.block_cache_size);
        assert!(e.block_size < d.block_size);
        assert!(e.target_file_size < d.target_file_size);
        assert!(e.level_base_bytes < d.level_base_bytes);
        assert!(e.max_key_size < d.max_key_size);
        assert!(e.max_value_size < d.max_value_size);
        assert!(e.level0_stop_writes_trigger < d.level0_stop_writes_trigger);
        assert!(e.hard_pending_compaction_bytes_limit < d.hard_pending_compaction_bytes_limit);
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
    fn embedded_preset_is_valid_and_small() {
        let opts = Options::embedded();
        opts.validate()
            .expect("the preset must be a valid option set");
        assert_eq!(opts.arena_profile, ArenaProfile::EMBEDDED);
        assert_eq!(
            opts.to_engine_options().arena_profile,
            ArenaProfile::EMBEDDED
        );
        // The documented bound: 2 * M * (W + c) + M * W.
        let w = opts.write_buffer_size;
        let c = opts.arena_profile.max_chunk_size;
        let m = opts.max_write_buffer_number;
        assert_eq!(2 * m * (w + c) + m * w, 1792 * 1024);
    }

    #[test]
    fn arena_profile_validation_rejects_bad_shapes() {
        let bad = [
            ArenaProfile {
                initial_chunk_size: 3000,
                max_chunk_size: 64 * 1024,
            },
            ArenaProfile {
                initial_chunk_size: 4096,
                max_chunk_size: 3000,
            },
            ArenaProfile {
                initial_chunk_size: 128 * 1024,
                max_chunk_size: 64 * 1024,
            },
            ArenaProfile {
                initial_chunk_size: 0,
                max_chunk_size: 64 * 1024,
            },
        ];
        for profile in bad {
            assert!(!profile.is_valid(), "{profile:?} must be rejected");
            let opts = Options {
                arena_profile: profile,
                ..Options::default()
            };
            match opts.validate() {
                Err(crate::Error::InvalidArgument(message)) => {
                    assert!(message.contains("arena_profile"), "{message}")
                }
                other => panic!("expected an arena_profile error, got {other:?}"),
            }
        }
        assert!(ArenaProfile::SERVER.is_valid());
        assert!(ArenaProfile::EMBEDDED.is_valid());
        assert_eq!(ArenaProfile::default(), ArenaProfile::SERVER);
    }

    #[test]
    fn key_and_value_size_caps_are_encodable() {
        // The memtable stores both lengths in a `u32` inside its arena
        // node, so an unencodable limit has to fail at the boundary.
        for (name, opts) in [
            (
                "max_key_size",
                Options {
                    max_key_size: u32::MAX as usize + 1,
                    ..Options::default()
                },
            ),
            (
                "max_value_size",
                Options {
                    max_value_size: u32::MAX as usize + 1,
                    ..Options::default()
                },
            ),
        ] {
            match opts.validate() {
                Err(crate::Error::InvalidArgument(message)) => {
                    assert!(message.contains(name), "{message}")
                }
                other => panic!("expected a {name} error, got {other:?}"),
            }
        }
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
        assert_eq!(
            eo.cache_index_and_filter_blocks,
            opts.cache_index_and_filter_blocks
        );
    }

    /// Custom `PrefixExtractor` that returns everything before the
    /// first `:` - proves the trait is user-implementable.
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
