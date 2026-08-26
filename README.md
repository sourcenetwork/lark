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

### Write back-pressure and compaction style

`level0_slowdown_writes_trigger` and `level0_stop_writes_trigger` count L0 *files*, and
only `CompactionStyle::Level` reduces that count in response. FIFO never merges L0 files
at all, and universal merges on its own size-ratio and amplification rules, which a
healthy size tier satisfies while holding more L0 files than the trigger allows. Under
either of those styles the trigger can be reached and never relieved, and writes then fail
with `Error::Busy` whose message names the style and the knob. Set both triggers to `0`
with FIFO or universal, and bound memory with `max_write_buffer_number` and
`hard_pending_compaction_bytes_limit`, which apply to every style. This is long-standing
behaviour, not new; what changed is that the writer now returns rather than blocking
forever.

### Value size

`max_value_size` defaults to 64 MiB; a write above the limit is rejected with an error.
Raising it works, but lark stores values inline with keys, with no key-value separation: a
large value is rewritten in full by every compaction that touches its key, and writing a
1 GiB value peaked at 3719 MiB RSS.

### Embedded and small-memory hosts

Two profiles replace the server-shaped defaults with values bounded by a memory budget
rather than by throughput.

`Options::embedded()` targets a host whose whole working set has to fit in roughly
1-4 MiB: a Linux-class board (Cortex-A, ESP32-S3 under esp-idf). A 256 KiB write buffer,
no block cache at all, 4 KiB blocks, 256 KiB SSTables, an L0 stop trigger of 8, and
`max_background_compactions: 0`, which runs compaction on the calling thread instead of a
background worker.

`Options::wasm()` targets a wasm module, in a browser or under a wasi host. It is not an
alias for `embedded()`, because wasm is not merely a small machine:

- **No threads, and no way to wait for one.** `max_background_compactions` is `0`, and on
  a wasm target `Options::validate()` *rejects* anything else, so the failure arrives at
  the option rather than out of the middle of `Db::open`. This holds on every wasm target,
  with or without the `atomics` feature, so `Options::default()` also opens there.
- **No OS page cache.** This is the sharpest split from `embedded()`, which sets
  `block_cache_size` to `0` precisely *because* a Linux-class board has a page cache to
  absorb the re-reads. A wasm module has none, so `wasm()` keeps a 1 MiB single-shard
  cache of decompressed blocks, buying back both the host call and the LZ4 decompression.
- **Linear memory grows in 64 KiB pages and never shrinks**, so the high-water mark *is*
  the footprint. The arena caps a chunk at exactly one page and the memtable pool recycles
  pages, so `memory.grow` runs once per size class rather than once per flush.

```rust
let db = lark_kv::Db::open("/path/to/db", lark_kv::Options::embedded())?;
let db = lark_kv::Db::open("/path/to/db", lark_kv::Options::wasm())?;
```

Every value and the reasoning behind it is documented on `Options::embedded` and
`Options::wasm` themselves.

#### Measured budget

Every figure comes from a checked-in example, re-run by the `embedded-budget` CI job
rather than quoted from here. Reproduce with:

```sh
just wasm-budget    # both columns of the table below, all three profiles
cargo run --release --example stack_depth -- /tmp/lark-stack
cargo run --release --example read_scaling -- /tmp/lark-scale 500000 3
```

Full lifecycle (open, 20k x 128 B puts, sampled reads, page scan, compact, close, reopen,
read back), x86_64 Linux and wasm32-wasip1 under wasmtime, one run per cell:

| profile      | Linux RSS attributable to lark | wasm linear memory high-water            |
| ------------ | ------------------------------ | ---------------------------------------- |
| `embedded()` | 1.12 MiB                       | 1.56 MiB total, 0.50 MiB over baseline   |
| `wasm()`     | 4.00 MiB                       | 4.38 MiB total, 3.31 MiB over baseline   |
| `default()`  | 7.42 MiB                       | 11.19 MiB total, 10.13 MiB over baseline |

RSS is machine- and allocator-dependent and moves between runs; the wasm column does not,
because linear memory only ever grows. Treat the wasm figures as the reproducible ones.

#### Stack requirement

**A release build of lark needs at least 16 KiB of stack on any thread that calls it. A
debug build needs at least 64 KiB.** Worst-case measured depth is 11.6 KiB for `Db::open`,
on x86_64 with the release profile. Every measured path exceeds the 4 KiB that a small
Cortex-M stack provides:

| path                                 | bytes |
| ------------------------------------ | ----- |
| `Db::open` (multi-level, WAL replay) | 11840 |
| 8000 puts crossing a flush boundary  | 11424 |
| compaction merge (`compact_range`)   | 11400 |
| iterator seek + 50-entry walk        |  8328 |
| `scan_page` of 1000 rows             |  7695 |
| point read                           |  4552 |

The table is the release profile. An unoptimized build does not inline the same frames and
measures roughly 4x deeper: `examples/stack_probe.rs` runs each path on a thread of a
chosen size, and in debug every one of them overflows a 32 KiB stack and fits a 64 KiB one.
That probe cannot resolve below 16 KiB, because `std::thread::Builder` raises any smaller
request to the platform minimum; the table above comes from the painting harness, which
can.
Provision for the profile you are actually flashing, and remember that bring-up is
usually the debug one.

On Linux, where the default thread stack is 8 MiB, this costs nothing. On an esp-idf task
it is a `CONFIG_ESP_MAIN_TASK_STACK_SIZE` you must set. These are host numbers: frame
layout is target-specific, so re-run `stack_depth` on the target before treating any of
them as a budget for it. They have not been measured on ARM or RISC-V.

### Target tiers

| tier | targets | status |
| ---- | ------- | ------ |
| Tier A, Linux-class with `std` | `aarch64-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf` | supported, checked in CI |
| Tier A, same by construction | ESP32-S3 under esp-idf | unverified, no toolchain in CI |
| wasm | `wasm32-wasip1` | supported, full lifecycle run under wasmtime in CI |
| Tier B, bare metal | `thumbv7em-none-eabi`, `riscv32imac-unknown-none-elf` | not supported; lark is `std`-only |

Tier B needs `no_std` plus `alloc`. lark does not build there today, and the first failure
is a dependency (`crossbeam-utils`), not lark's own code.

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
