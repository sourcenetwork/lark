<h1 align="center">
    <img src="https://github.com/sourcenetwork/regolith/raw/main/art/regolith_banner_2048x512.png"/>
</h1>
<div align="center">
 <strong>
   ACID, performance oriented, embedded key-value database engine for edge systems
 </strong>
<hr>

[![Crates.io](https://img.shields.io/crates/v/regolith.svg)](https://crates.io/crates/regolith)
[![Documentation](https://docs.rs/regolith/badge.svg)](https://docs.rs/regolith)

[![CI](https://github.com/sourcenetwork/regolith/actions/workflows/ci.yml/badge.svg)](https://github.com/sourcenetwork/regolith/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sourcenetwork/regolith/branch/main/graph/badge.svg)](https://codecov.io/gh/sourcenetwork/regolith)
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

A `DbSlice` also adopts a buffer you already own, without copying it, so bytes the
database handed back and bytes you assembled yourself travel as one type:

```rust
let assembled: DbSlice = b"built elsewhere".to_vec().into();
```

## Streaming

A cursor is lazy and iterates as an ordinary Rust iterator, so a scan holds one
entry rather than the whole range. Values come back as `DbSlice`, so iterating
copies keys but never value bytes:

```rust
let snap = db.snapshot();
for (key, value) in snap.owned_iter() {
    // `value` borrows the bytes the database already holds.
    println!("{} = {} bytes", String::from_utf8_lossy(&key), value.len());
}

// Positioned scans, reverse scans, and early exit all work:
let mut cursor = snap.owned_iter();
cursor.seek(b"user:");
let first_ten: Vec<_> = cursor.entries().take(10).collect();
```

`Db::scan` returns a `Vec` and reads the whole range up front. Prefer a cursor,
or `scan_page` for explicit page-sized reads.

regolith stays synchronous and needs no async runtime, so it does not implement
`Stream`. Building one is a one-liner where the async context already is:

```rust
let stream = futures::stream::iter(db.snapshot().owned_iter());
```

Writes stream too. A `WriteBatch` costs memory proportional to its input, which
is wrong for a stream whose length the caller does not control. A
`StreamingWriter` costs a fixed budget instead:

```rust
use regolith::StreamOptions;

let mut writer = db.streaming_writer(StreamOptions {
    max_buffered_bytes: 1 << 20,
    ..Default::default()
});
for (key, value) in huge_source {
    writer.put_owned(&key, value)?;  // takes the buffer, does not copy it
}
let sequence = writer.finish()?;
```

Peak footprint is the budget plus one operation, whatever the stream's length.
The tradeoff is explicit: each flush is atomic, the stream as a whole is not.
A crash partway through leaves a valid prefix of the stream, never a
half-applied flush. Work that must land all-or-nothing wants a single
`WriteBatch` and has to pay the memory for it.

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

Buffering a read or a write takes `&self`, so one transaction can be shared across
threads without a lock around it. The write buffer and the read set are lock-free, and
every read is folded into the commit-time validation set no matter which thread recorded
it. Only `commit`, `rollback`, and the savepoint calls need exclusive access, because
those are the points where the buffer stops changing.

```rust
use std::sync::Arc;
use regolith::{IsolationLevel, OptimisticTransactionDb, Options};

let db = Arc::new(OptimisticTransactionDb::open("/tmp/shared_db", Options::default())?);

// `begin_transaction` borrows the database, so the transaction cannot outlive it or be
// stored in a `'static` container. `begin_transaction_owned` returns an
// `OwnedTransaction`, which carries an `Arc` on the database instead and can be boxed,
// shared, and moved freely.
let txn = Arc::new(db.begin_transaction_owned(IsolationLevel::Serializable));

std::thread::scope(|scope| {
    for thread in 0..4 {
        let txn = txn.clone();
        scope.spawn(move || txn.put(format!("k{thread}").as_bytes(), b"v"));
    }
});

txn.into_inner().expect("threads joined").commit()?;
```

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

<sub><sup>Here men from the [planet Earth](https://open.spotify.com/track/6qZthmNcaK0jlrkMZ3khmy) first set foot upon the Moon July 1969, A.D. We came in peace for all mankind.<sup><sub>
