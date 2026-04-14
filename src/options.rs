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
        }
    }
}
