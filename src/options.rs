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

/// Configuration options for a lark database.
#[derive(Debug, Clone)]
pub struct Options {
    /// Write buffer (memtable) size before flush. Default: 64 MB.
    pub write_buffer_size: usize,
    /// Data block size in SSTables. Default: 16 KB.
    pub block_size: usize,
    /// Block cache size for decompressed blocks. Default: 512 MB.
    pub block_cache_size: usize,
    /// Bloom filter bits per key. Default: 10.
    pub bloom_bits_per_key: usize,
    /// Enable LZ4 block compression. Default: true.
    pub compression: bool,
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
}

impl Default for Options {
    fn default() -> Self {
        Self {
            write_buffer_size: 64 * 1024 * 1024,
            block_size: 16 * 1024,
            block_cache_size: 512 * 1024 * 1024,
            bloom_bits_per_key: 10,
            compression: true,
            l0_compaction_trigger: 4,
            level_base_bytes: 256 * 1024 * 1024,
            level_size_multiplier: 10,
            target_file_size: 64 * 1024 * 1024,
            durability: DurabilityMode::Eventual,
        }
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
            l0_compaction_trigger: self.l0_compaction_trigger,
            level_base_bytes: self.level_base_bytes,
            level_size_multiplier: self.level_size_multiplier,
            target_file_size: self.target_file_size,
        }
    }
}
