# Lark

A pure Rust, embedded key-value store built from scratch on an LSM-tree architecture inspired by LevelDB. The public API is shaped to slot into common embedded-KV abstraction layers, so callers that already hide a storage backend behind a trait can add lark as another implementation with minimal glue.

> **Status:** pre-1.0. The public surface (`Db`, `Snapshot`, `WriteBatch`, `Options`, `Iter`, `TailingIter`, transactions, column families) is small and stable-shaped, but breakage is allowed until 1.0.

## Features

- **Pure Rust** - no C/C++ dependencies, no FFI, no linker surprises
- **LSM-tree architecture** - write-optimized with background compaction
- **Three compaction styles** - leveled (default), FIFO, and universal / size-tiered
- **MVCC snapshots** - point-in-time consistent reads via sequence numbers
- **WAL crash recovery** - write-ahead log with xxhash-checksummed records. The default
  `DurabilityMode::Eventual` does not fsync, so a committed write survives a process crash
  but not a power loss or OS crash; `DurabilityMode::Immediate` fsyncs every write
- **Column families** - isolated keyspaces within one `Db`, atomic across CFs
- **Transactions** - optimistic (OCC, write-write conflicts detected at commit) and
  pessimistic (exclusive key locks held until commit or rollback). Both provide snapshot
  isolation and both prevent lost updates: a key that a transaction read and then wrote is
  validated at commit against the sequence it was read at, so a concurrent write aborts the
  transaction instead of being overwritten. `get_for_update` additionally takes the key lock
  before it reads, which turns contention into waiting rather than retrying. Not
  serializable: a key that is read and never written is not validated
- **Merge operator** - user-defined associative merges, compaction-aware
- **Tailing iterators** - forward-only iterators that pick up new writes
- **Range deletes** - `delete_range` as a single range tombstone, not N point tombstones
- **Pluggable compression** - LZ4 (default) or Snappy, both pure-Rust
- **Bloom filters** - 10 bits/key for fast negative lookups, plus optional prefix bloom
- **Sharded block cache** - 64 LRU shards by default (fewer for a small cache).
  `block_cache_size` is a hard bound: every entry is charged its block bytes plus its
  bookkeeping, nothing is preallocated, and the total the cache holds does not move with the
  shard count, the block size, or the value size
- **Back-pressure** - write stalls, token-bucket rate limiter, per-write `no_slowdown`.
  These shape write admission; they do not bound process memory, which is dominated by the
  memtables, the block cache, and in-flight values
- **Observability** - tickers + histograms, properties API, `EventListener` callbacks
- **Checkpoint / BackupEngine** - hardlinked snapshots and content-addressed backups
- **Lock-free reads** - concurrent readers via a crossbeam skip list memtable. Measured:
  5.31x scaling from 1 to 8 threads, 3.6M reads/s at 8 threads, reproduced across 4 sessions
- **No async runtime required** - compaction runs on a dedicated OS thread. A platform that
  cannot spawn that thread (a single-threaded target such as wasm) makes `Db::open` return an
  error instead of aborting

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
    max_value_size: 64 * 1024 * 1024,       // 64 MiB cap on a single value
    ..Default::default()
};

let db = lark_kv::Db::open("/path/to/db", opts)?;
```

### Value size

`max_value_size` defaults to 64 MiB; a write above the limit is rejected with an error.
Raising it works, but lark stores values inline with keys, with no key-value separation: a
large value is rewritten in full by every compaction that touches its key, and writing a
1 GiB value peaked at 3719 MiB RSS.

## On-disk format

| Component    | Format                                                                  |
| ------------ | ----------------------------------------------------------------------- |
| WAL          | `[len:u32][type:u8][data][crc:u32]` per record                          |
| SSTable      | Data blocks + bloom filter + index block + 64-byte footer               |
| Data block   | Prefix-compressed entries with restart points, block-level compression  |
| Bloom filter | Double-hashing with xxh3, 10 bits/key default                           |
| Manifest     | Append-only log of `VersionEdit` records with checksums                 |

## License

Apache-2.0 OR MIT
