# Lark

A pure Rust, embedded key-value store built on an LSM-tree architecture. Designed as a lightweight, zero-dependency-on-C alternative to RocksDB.

## Features

- **Pure Rust** -- no C/C++ dependencies, no FFI
- **LSM-tree architecture** -- write-optimized with level-based compaction
- **MVCC snapshots** -- point-in-time consistent reads via sequence numbers
- **WAL crash recovery** -- write-ahead log with xxhash checksums
- **LZ4 compression** -- fast block compression via `lz4_flex`
- **Bloom filters** -- 10 bits/key for fast negative lookups
- **Lock-free reads** -- concurrent readers via crossbeam skip list memtable
- **Background compaction** -- dedicated OS thread, no async runtime required

## Usage

```rust
use lark_kv::{Db, Options};

let db = Db::open("/tmp/my_db", Options::default())?;

// Write
db.put(b"hello", b"world")?;

// Read
assert_eq!(db.get(b"hello")?, Some(b"world".to_vec()));

// Delete
db.delete(b"hello")?;

// Batch write (atomic)
let mut batch = lark_kv::WriteBatch::new();
batch.put(b"key1", b"val1");
batch.put(b"key2", b"val2");
batch.delete(b"key3");
db.write(batch)?;

// Snapshot isolation
let snap = db.snapshot();
db.put(b"key1", b"new_val")?;
assert_eq!(snap.get(b"key1")?, Some(b"val1".to_vec())); // old value

// Range scan
let results = db.scan(Some(b"a"), Some(b"z"))?;
```

## Architecture

```
                    ┌─────────────┐
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
                   │   L2 SSTs    │  (10x larger)
                   ├──────────────┤
                   │     ...      │
                   └──────────────┘
```

**Write path:** WAL append -> MemTable insert -> (when full) flush to L0 SSTable -> background compaction merges levels.

**Read path:** MemTable -> frozen MemTables -> L0 SSTables (bloom filter check) -> L1+ SSTables (binary search).

## Configuration

```rust
use lark_kv::Options;

let opts = Options {
    write_buffer_size: 64 * 1024 * 1024,   // 64 MB memtable
    block_cache_size: 512 * 1024 * 1024,    // 512 MB block cache
    block_size: 16 * 1024,                   // 16 KB data blocks
    bloom_bits_per_key: 10,                  // ~1% false positive rate
    compression: true,                        // LZ4 compression
    l0_compaction_trigger: 4,                // compact after 4 L0 files
    level_base_bytes: 256 * 1024 * 1024,     // 256 MB L1 target
    level_size_multiplier: 10,               // 10x between levels
    target_file_size: 64 * 1024 * 1024,      // 64 MB SSTable target
    ..Default::default()
};

let db = lark_kv::Db::open("/path/to/db", opts)?;
```

## On-disk format

| Component | Format |
|-----------|--------|
| WAL | `[len:u32][type:u8][data][crc:u32]` per record |
| SSTable | Data blocks + bloom filter + index block + 48-byte footer |
| Data block | Prefix-compressed entries with restart points, LZ4 compressed |
| Bloom filter | Double-hashing with xxhash, 10 bits/key |
| Manifest | Append-only log of `VersionEdit` records with checksums |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
