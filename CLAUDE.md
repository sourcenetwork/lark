# Development Principles

## 0. What Lark Is

**lark** (crate: `lark-kv`, v0.1.0) is a pure Rust, embedded LSM-tree key-value store built from scratch. The architecture follows the LevelDB design (memtable → WAL → leveled SSTables → background compaction); the public API is shaped to slot into common embedded-KV abstraction layers so consuming applications can swap lark in alongside other backends through the same trait.

Early-stage. The public API (`Db`, `Snapshot`, `WriteBatch`, `Options`) is small and stable-shaped, but breakage is allowed pre-1.0.

### Design goals

- **Pure Rust, no FFI** — no C/C++ toolchain, no `bindgen`, no linker surprises
- **LSM-tree** — write-optimized with level-based background compaction
- **MVCC** — point-in-time consistent reads via global sequence numbers
- **Crash recovery** — WAL with xxhash-checksummed records
- **Concurrent reads** — lock-free memtable via `crossbeam-skiplist`
- **No async runtime required** — compaction runs on a dedicated OS thread

---

## 1. Information Hygiene

This codebase is designed for **AI-human pair programming**. Every structural choice optimizes for **rapid context acquisition**.

**Context clarity is oxygen for productive collaboration.**

## 2. Temporal Boundaries

| Zone        | Contains                | Lives in                        |
| ----------- | ----------------------- | ------------------------------- |
| **Past**    | How we got here         | Git history, closed issues/PRs  |
| **Present** | What the code does now  | Working tree                    |
| **Future**  | What we might do next   | GitHub issues                   |

**No commented-out code. No TODO comments (create issues instead). No speculative docs.**

## 3. No Documentation Files

Only allowed: `README.md`, `CLAUDE.md`, `Cargo.toml`, `LICENSE-*`.

No `ROADMAP.md`, `DEVELOPMENT.md`, `docs/` directories, or planning documents.

## 4. File Organization

**One concept per file. Small files over large files.**

### Module Map

Single crate (`lark-kv`), all code under `src/`.

```text
src/
├── lib.rs              # Public API: Db, Snapshot, WriteBatch, re-exports
├── error.rs            # Error enum, Result alias
├── options.rs          # Options, DurabilityMode
└── engine/
    ├── mod.rs          # LarkEngine — orchestration, read/write paths, recovery
    ├── memtable.rs     # Lock-free skip list memtable (MVCC-encoded keys)
    ├── wal.rs          # Write-ahead log: append, replay, CRC records
    ├── block.rs        # Data blocks: prefix compression, restart points, varint
    ├── bloom.rs        # Bloom filter (double-hashed xxh3)
    ├── sstable.rs      # SSTable reader/writer, footer, index block
    ├── block_cache.rs  # LRU cache for decompressed SSTable blocks
    ├── manifest.rs     # VersionSet, VersionEdit log, level tracking
    ├── compaction.rs   # Level-based compaction scheduler (background thread)
    └── snapshot.rs     # Snapshot token (captured sequence number)
```

### Public API surface

`Db`, `Snapshot`, `WriteBatch`, `Options`, `DurabilityMode`, `Error`, `Result` — all re-exported from `lib.rs`. Anything not re-exported is internal.

### File Size Guidelines

- Under 200 lines: Fine
- 200–400 lines: Check if doing one thing
- Over 400 lines: Consider splitting

## 5. Naming Conventions

| Thing         | Convention           | Example              |
| ------------- | -------------------- | -------------------- |
| Files/Modules | snake_case           | `block_cache.rs`     |
| Types         | PascalCase           | `WriteBatch`         |
| Functions     | snake_case           | `flush_memtable()`   |
| Constants     | SCREAMING_SNAKE_CASE | `BLOCK_FOOTER_SIZE`  |

## 6. Comments Policy

**Minimal comments. Code should be self-documenting.**

✅ Comment: Non-obvious WHY, safety invariants, on-disk format descriptions, public API docs (`///`)

❌ Don't: What the code does, TODO/FIXME, commented-out code, change history

## 7. Architecture at a Glance

**Write path:** `put`/`delete`/`write` → append to WAL → insert into active memtable → when memtable is full, swap to immutable and flush to an L0 SSTable → background compaction merges L0→L1 and promotes levels (each 10× larger than the last).

**Read path:** active memtable → frozen memtables → L0 SSTables (bloom filter pre-check, overlap possible) → L1..L6 (sorted, non-overlapping, binary search). The first hit wins.

**MVCC:** every write increments a global sequence number. `snapshot()` captures the current seq; reads through a `Snapshot` ignore any key whose seq is greater. This is how point-in-time isolation works without locks on the read path.

**Durability:** `DurabilityMode::Immediate` fsyncs the WAL per write; `Eventual` (default) leaves it to the OS.

## 8. Git Worktree Workflow

```bash
git worktree add ../lark-foo -b feat/foo    # Work on feature foo
git worktree add ../lark-bar -b feat/bar    # Work on feature bar
git worktree remove ../lark-foo             # Clean up
```

Each worktree is isolated, no branch-switching overhead.

## Build Dependencies

- **Rust** 1.82+ (see `rust-version` in `Cargo.toml`)

That's it. No `protoc`, no `cbindgen`, no system libraries.

## Common Commands

```bash
cargo test                         # Run all tests (inline in src/lib.rs)
cargo clippy -- -D warnings        # Lint (matches CI)
cargo fmt -- --check               # Format check (matches CI)
cargo fmt                          # Apply formatting
cargo build --release              # Optimized build (LTO, strip)
```

Tests live inline with `#[cfg(test)]` in the modules they cover — there is no separate `tests/` directory.

## Before Committing

1. `cargo test` passes
2. `cargo clippy -- -D warnings` clean
3. `cargo fmt -- --check` clean

These are exactly what CI runs (`.github/workflows/ci.yml`).

## Goal

**A new contributor should be able to read this file, skim `src/lib.rs`, and start making productive changes to the engine within an hour.**

Fast context acquisition → Confident changes → Productive iteration.
