# Lark

A pure Rust, embedded key-value store built from scratch on an LSM-tree architecture inspired by LevelDB. The public API is shaped to slot into common embedded-KV abstraction layers, so callers that already hide a storage backend behind a trait can add lark as another implementation with minimal glue.

> **Status:** pre-1.0. The public surface (`Db`, `Snapshot`, `WriteBatch`, `Options`, `Iter`, `TailingIter`, transactions, column families) is small and stable-shaped, but breakage is allowed until 1.0.

## Features

- **Pure Rust** — no C/C++ dependencies, no FFI, no linker surprises
- **LSM-tree architecture** — write-optimized with background compaction
- **Three compaction styles** — leveled (default), FIFO, and universal / size-tiered
- **MVCC snapshots** — point-in-time consistent reads via sequence numbers
- **WAL crash recovery** — write-ahead log with xxhash-checksummed records
- **Column families** — isolated keyspaces within one `Db`, atomic across CFs
- **Transactions** — both optimistic (OCC) and pessimistic (lock-based) flavors
- **Merge operator** — user-defined associative merges, compaction-aware
- **Tailing iterators** — forward-only iterators that pick up new writes
- **Range deletes** — `delete_range` as a single range tombstone, not N point tombstones
- **Pluggable compression** — LZ4 (default) or Snappy, both pure-Rust
- **Bloom filters** — 10 bits/key for fast negative lookups, plus optional prefix bloom
- **Sharded block cache** — 64-shard LRU with byte-accurate capacity tracking
- **Back-pressure** — write stalls, token-bucket rate limiter, per-write `no_slowdown`
- **Observability** — tickers + histograms, properties API, `EventListener` callbacks
- **Checkpoint / BackupEngine** — hardlinked snapshots and content-addressed backups
- **Lock-free reads** — concurrent readers via a crossbeam skip list memtable
- **No async runtime required** — compaction runs on a dedicated OS thread

## Usage

```rust
use lark_kv::{Db, Options, WriteBatch};

let db = Db::open("/tmp/my_db", Options::default())?;

// Write
db.put(b"hello", b"world")?;

// Read
assert_eq!(db.get(b"hello")?, Some(b"world".to_vec()));

// Delete
db.delete(b"hello")?;

// Batch write (atomic)
let mut batch = WriteBatch::new();
batch.put(b"key1", b"val1");
batch.put(b"key2", b"val2");
batch.delete(b"key3");
db.write(batch)?;

// Snapshot isolation
let snap = db.snapshot();
db.put(b"key1", b"new_val")?;
assert_eq!(snap.get(b"key1")?, Some(b"val1".to_vec())); // still sees the old value

// Range scan
let results = db.scan(Some(b"a"), Some(b"z"))?;
```

## Architecture

```text
                   ┌──────────────┐
         writes ──>│  MemTable    │──> reads
                   │ (skip list)  │
                   └──────┬───────┘
                          │ flush
                   ┌──────▼───────┐
                   │   L0 SSTs    │  (may overlap)
                   └──────┬───────┘
                          │ compaction
                   ┌──────▼───────┐
                   │   L1 SSTs    │  (sorted, non-overlapping)
                   ├──────────────┤
                   │   L2 SSTs    │  (10× larger)
                   ├──────────────┤
                   │     ...      │
                   └──────────────┘
```

**Write path:** WAL append → MemTable insert → (when full) flush to an L0 SSTable → background compaction merges levels.

**Read path:** active MemTable → frozen MemTables → L0 SSTables (bloom-filter pre-check) → L1+ SSTables (binary search over non-overlapping files).

## Configuration

```rust
use lark_kv::{CompressionType, Options};

let opts = Options {
    write_buffer_size: 64 * 1024 * 1024,    // 64 MB memtable
    block_cache_size: 512 * 1024 * 1024,    // 512 MB block cache
    block_size: 16 * 1024,                  // 16 KB data blocks
    bloom_bits_per_key: 10,                 // ~1% false positive rate
    compression: CompressionType::Lz4,      // LZ4 block compression
    l0_compaction_trigger: 4,               // compact after 4 L0 files
    level_base_bytes: 256 * 1024 * 1024,    // 256 MB L1 target
    level_size_multiplier: 10,              // 10× between levels
    target_file_size: 64 * 1024 * 1024,     // 64 MB SSTable target
    ..Default::default()
};

let db = lark_kv::Db::open("/path/to/db", opts)?;
```

## On-disk format

| Component    | Format                                                                  |
| ------------ | ----------------------------------------------------------------------- |
| WAL          | `[len:u32][type:u8][data][crc:u32]` per record                          |
| SSTable      | Data blocks + bloom filter + index block + 48-byte footer               |
| Data block   | Prefix-compressed entries with restart points, block-level compression  |
| Bloom filter | Double-hashing with xxh3, 10 bits/key default                           |
| Manifest     | Append-only log of `VersionEdit` records with checksums                 |

## License

Apache-2.0 OR MIT
