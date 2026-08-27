<h1 align="center">
    <img src="https://github.com/sourcenetwork/regolith/raw/master/art/regolith_banner_2048x512.png"/>
</h1>
<div align="center">
 <strong>
   ACID, performance oriented, embedded key-value database for edge systems
 </strong>
<hr>

[![Crates.io](https://img.shields.io/crates/v/regolith.svg)](https://crates.io/crates/regolith)
[![Documentation](https://docs.rs/regolith/badge.svg)](https://docs.rs/regolith)

[![CI](https://github.com/sourcenetwork/regolith/actions/workflows/ci.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/ci.yml)
[![Transactional Verification](https://github.com/sourcenetwork/regolith/actions/workflows/transactional-verification.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/transactional-verification.yml)
[![Durability](https://github.com/sourcenetwork/regolith/actions/workflows/durability.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/durability.yml)
[![Concurrency Model Checking](https://github.com/sourcenetwork/regolith/actions/workflows/model-checking.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/model-checking.yml)
[![Chaos](https://github.com/sourcenetwork/regolith/actions/workflows/chaos.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/chaos.yml)
[![WASM](https://github.com/sourcenetwork/regolith/actions/workflows/wasm.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/wasm.yml)
[![Embedded](https://github.com/sourcenetwork/regolith/actions/workflows/embedded.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/embedded.yml)

</div>

> **Pre-1.0.** The public surface is small and stable-shaped, but breakage is allowed until 1.0.

## Install

```toml
[dependencies]
regolith = "0.1"
```

## Quick start

```rust
use regolith::{Db, Options, WriteBatch};

let db = Db::open("/tmp/my_db", Options::default())?;

db.put(b"hello", b"world")?;
assert_eq!(db.get(b"hello")?.as_deref(), Some(&b"world"[..]));
db.delete(b"hello")?;

// A batch applies atomically: all of it, or none of it.
let mut batch = WriteBatch::new();
batch.put(b"a", b"1");
batch.put(b"b", b"2");
db.write(batch)?;

// A snapshot is a point in time. Later writes are invisible to it.
let snap = db.snapshot();
db.put(b"a", b"changed")?;
assert_eq!(snap.get(b"a")?.as_deref(), Some(&b"1"[..]));

for (key, value) in db.scan(Some(b"a"), Some(b"z"))? {
    println!("{key:?} = {value:?}");
}
```

Reads that should not copy use `get_slice`, which borrows the bytes the database already
holds:

```rust
if let Some(slice) = db.get_slice(b"a")? {
    assert_eq!(&*slice, b"1");
}
```

## Transactions

Pick the isolation a unit of work actually needs. The level decides how much of the
transaction's footprint is validated at commit, so a stricter level refuses more and
commits fewer.

```rust
use regolith::{IsolationLevel, OptimisticTransactionDb, Options};

let db = OptimisticTransactionDb::open("/tmp/txn_db", Options::default())?;

let mut txn = db.begin_transaction_with(IsolationLevel::Serializable);
let balance = txn.get(b"account")?.unwrap_or_default();
txn.put(b"account", b"debited")?;
txn.commit()?;
```

| Level | Lost update | Read skew | Write skew |
|---|---|---|---|
| `ReadCommitted` | prevented | possible | possible |
| `SnapshotIsolation` (default) | prevented | prevented | possible |
| `Serializable` | prevented | prevented | prevented |

`TransactionDb` is the pessimistic flavour: it takes key locks, so contention waits
instead of retrying. `OptimisticTransactionDb` validates at commit and retries.

Commits report the sequence they landed at, so an upper layer can order its own versions
against the store without a lock of its own:

```rust
let seq = db.write_sequenced(batch)?;
assert!(db.snapshot().sequence() >= seq);
```

## Durability

`DurabilityMode::Eventual` (default) hands every write to the kernel but does not fsync: a
committed write survives a process crash, not a power loss. `DurabilityMode::Immediate`
fsyncs, and every write that returned `Ok` survives a power cut.

Either way, recovery reaches a **valid prefix** of the write history: some number of
writes applied, in order, no gaps, no half-applied batch. A `WriteBatch` is atomic under
both modes.

## Platforms

| Target | Status |
|---|---|
| Linux, macOS (x86_64, aarch64) | full |
| `wasm32-wasip1` | full, via a preopened directory |
| `wasm32-unknown-unknown` | full, via OPFS (`Options::wasm()`) |
| Embedded Linux (Cortex-A, ESP32-S3) | `Options::embedded()`, ~1-4 MiB working set |

Pure Rust throughout: no C toolchain, no FFI, no linker surprises. Compaction runs on an
ordinary OS thread; no async runtime is required. On a target without threads, set
`max_background_compactions = 0` and compaction runs on the calling thread.

## Configuration

```rust
use regolith::{CompressionType, Options};

let opts = Options {
    write_buffer_size: 64 * 1024 * 1024,
    block_cache_size: 512 * 1024 * 1024,
    compression: CompressionType::Lz4,
    ..Default::default()
};
```

Three ready-made profiles: `Options::default()` for a server, `Options::embedded()` for a
1-4 MiB budget, and `Options::wasm()` for a browser or wasi module. Every value and the
reasoning behind it is documented on the profile itself.

Two knobs deserve a warning:

- **Value size.** `max_value_size` defaults to 64 MiB. Values are stored inline, with no
  key-value separation, so a large value is rewritten in full by every compaction that
  touches its key. A 1 GiB value peaked at 3719 MiB RSS.
- **Back-pressure under FIFO and universal compaction.** `level0_stop_writes_trigger`
  counts L0 files, and only `CompactionStyle::Level` reduces that count. Under the other
  styles the trigger can be reached and never relieved, and writes then fail with
  `Error::Busy` naming the knob. Set both L0 triggers to `0` there and bound memory with
  `max_write_buffer_number` and `hard_pending_compaction_bytes_limit`.

## How it works

```text
         writes ──> WAL ──> MemTable ──> reads
                             │ flush
                        ┌────▼─────┐
                        │  L0 SSTs │  may overlap
                        └────┬─────┘
                             │ compaction
                        ┌────▼─────┐
                        │ L1..L6   │  sorted, non-overlapping, 10x each
                        └──────────┘
```

**Write:** append to the WAL, insert into an arena-backed skip-list memtable, and when it
fills, flush to an L0 SSTable. Background compaction merges levels.

**Read:** active memtable, then frozen memtables, then L0 (bloom pre-check), then L1+ by
binary search. First hit wins.

**MVCC:** every write takes a sequence number. A snapshot captures the current one and
ignores anything newer, so reads need no locks.

## Testing

The suite is the argument for trusting any of the above.

```sh
just gate       # format, lint, docs, tests, dependency audit
just test       # the whole suite under cargo-nextest
just loom-all   # exhaustive model checking of the publication protocols
just elle       # Elle consistency checking of transaction histories
just chaos      # the full-size read-view chaos workload
just wasm       # the wasm32-wasip1 lifecycle under wasmtime
```

Each badge above is a workflow of its own, so a red one names exactly which property
regressed.

## License

Dual-licensed under Apache-2.0 or MIT, at your option.
