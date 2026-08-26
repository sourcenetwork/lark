

default:
    @just --list

# Everything CI enforces on every push, in the order it fails fastest.
gate: fmt lint doc test deny

fmt:
    cargo fmt --all -- --check

fmt-fix:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets -- -D warnings

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

test:
    cargo test --workspace

deny:
    cargo deny check

msrv:
    cargo +1.82 check --workspace

cov:
    cargo llvm-cov --workspace --html
    @echo "report: target/llvm-cov/html/index.html"

# ── suites CI runs as their own jobs, split by mechanism ─────

test-fault:
    cargo test --test fault_smoke

# The `#[ignore]`d fault tests: each spawns child processes and simulates a
# power cut. Measured at well under a second each, but kept out of the
# default run so `cargo test` stays quick.

test-fault-slow:
    cargo test --test fault_smoke -- --ignored --skip crash_child

# The `#[ignore]`d resource-exhaustion and extremes tests: six-figure key
# counts, a 64 MiB value, megabyte keys, a six-level cascade, and a real
# ENOSPC on a tmpfs mounted in a private user namespace. About 21 s in
# total on a debug build. `--test-threads=1` is required, not cosmetic:
# each test resets the kernel's peak-RSS counter to report its own memory
# high-water mark, and a concurrent test would inflate that number.

test-crash:
    cargo test --test crash_recovery

# Handle lifecycle: open, close, reopen under changed Options, locking,
# and every misuse of a closed or read-only handle. Measured at 0.4s, so
# every test in the file also runs in the default `cargo test`; nothing
# here is `#[ignore]`d.

test-power:
    cargo test --test power_loss -- --skip crash_child --nocapture

# Process-crash recovery: kill -9 at every point in the write path. Every
# test in the file is in the default run already; this recipe just runs the
# file on its own. Measured at 0.6s, spawning 41 child processes.

test-corruption-slow:
    cargo test --test corruption_exhaustive -- --ignored --skip crash_child

# The power-loss durability tests, with output shown so the measured cost of
# the default DurabilityMode::Eventual is visible. Every test here spawns a
# child process, crashes it at a byte-exact point and simulates a power cut.
# Measured at 0.6s in total, so it also runs in the default `cargo test`.

test-durability-slow:
    cargo test --test proptest_durability -- --ignored --skip crash_child

# Every ignored test in the workspace, including the scheduled stress runs.

test-extremes:
    cargo test --test resource_limits -- --ignored --nocapture --test-threads 1 --skip crash_child

# The `#[ignore]`d durability property test: 128 randomized operation
# sequences, each run in a child process that is killed part way through
# and then power-cut by discarding every byte it never fsynced. Measured
# at 1.1 s. Needs the LD_PRELOAD fault shim, so it is Linux only.

test-lifecycle:
    cargo test --test lifecycle -- --skip crash_child

# MVCC and concurrency invariants: snapshot stability under concurrent
# writers and compaction, WriteBatch atomicity seen by concurrent readers,
# monotonic reads, version integrity across delete/compact/reopen, and
# iterators pinned across compactions that unlink their files. Measured at
# 0.5s, so every test in the fast set also runs in the default `cargo test`.

test-slow:
    cargo test --workspace -- --ignored --skip crash_child

# Rebuild the LD_PRELOAD fault shim from scratch by dropping its cache.

fault-shim-clean:
    rm -rf target/tmp/lark-fault

# The `#[ignore]`d exhaustive corruption sweeps: every byte offset and
# every single-bit flip of a WAL, an SSTable and a MANIFEST. 14,265
# trials, measured at 1.2s of wall time.

mvcc:
    cargo test --test mvcc_invariants -- --skip crash_child

# The `#[ignore]`d full-scale MVCC soaks: 120,000 writes racing a snapshot
# that pins every version of them, 30,000 WriteBatch generations checked by
# four readers, 1.9M monotonic point reads, and the focused gate for the
# user-thread `compact_range` read race. Measured at 13.8s + 9.3s + 4.5s +
# 26s on a debug build.
#
# This recipe is RED today, and that is the point: the focused gate finds a
# real read-path defect. See the doc comment on
# `a_user_thread_compact_range_never_makes_a_read_travel_backwards`.

mvcc-slow:
    cargo test --test mvcc_invariants -- --ignored --nocapture --skip crash_child

set shell := ["bash", "-uc"]

gains_dir := justfile_directory() / "../larkgains"

label     := env_var_or_default("LABEL", "wip")

py        := env_var_or_default("LARKGAINS_PY", "python3")

# NOTE: the commit sha is resolved INSIDE the recipe, never as a top-level
# `sha := `git ...`` assignment. just evaluates those eagerly at parse time, so a
# top-level backtick makes every `just --list` fail outside a git checkout.

# ---------- the gate ----------

fuzz target time="300":
    cargo +nightly fuzz run {{target}} -- -max_total_time={{time}}

# ---------- benchmark gating ----------

# Exit 1 when the host is too busy to trust a measurement. A dependency, not advice.

loadguard:
    {{py}} {{gains_dir}}/loadguard.py

# ---------- benchmarks ----------

# Capture the reference baseline. Run ONCE, before any stack code lands.

bench-baseline: loadguard
    cargo bench --bench point_read --bench write_durable --bench write_buffered \
                --bench scan --bench batch --bench transaction --bench isolation \
                --bench large_value -- --save-baseline pre

# Compare against the `pre` baseline. This is what a PR pastes into its body.

bench: loadguard
    cargo bench -- --baseline pre

bench-one name: loadguard
    cargo bench --bench {{name}} -- --baseline pre

# RSS soak. Default 360s; pass seconds and an Options variant tag.

soak secs="360" wb="64" cache="64" shard_bits="6" tag="default": loadguard
    cargo run --release --bench soak -- {{secs}} {{wb}} {{cache}} {{shard_bits}} {{tag}}

# The two soak variants section 1.2 compares. Deterministic: no loadguard needed.

soak-pair:
    just soak 360 64 64 6 default
    just soak 360 64 64 0 cache-budgeted

# Binary size, native and both wasm targets, against the section 6.5 budget.

size:
    cargo run --release --bench size

# Point memory probes from section 6.5.

mem:
    cargo run --release --bench memory

# The MVCC regression probes from section 6.1. Must stay at zero violations.

ycsb workload="a" records="1000000" ops="1000000": loadguard
    cargo run --release -p lark-ycsb -- \
        --workload {{workload}} --records {{records}} --operations {{ops}}

ycsb-all: loadguard
    for w in a b c d e f; do just ycsb $w; done

stress secs="600":
    cargo run --release -p lark-stress -- --duration {{secs}}

# ---------- consistency ----------

# Elle consistency checking. NOT WIRED UP YET: needs `harness/elle`, a
# history generator that drives lark and emits an Elle history, plus
# `elle-cli.jar` beside it. Neither is in the tree, so this recipe fails
# until they are added; it is here as the interface they must satisfy.
elle model="list-append":
    cargo run --release --manifest-path harness/elle/Cargo.toml -- \
        --model {{model}} --out /tmp/lark-history.json
    java -jar harness/elle/elle-cli.jar --model {{model}} /tmp/lark-history.json

elle-fault model="list-append":
    cargo run --release --manifest-path harness/elle/Cargo.toml -- \
        --model {{model}} --faults --out /tmp/lark-history-fault.json
    java -jar harness/elle/elle-cli.jar --model {{model}} /tmp/lark-history-fault.json

# ---------- portability ----------

# Full wasm32-wasip1 lifecycle under wasmtime. Non-zero on the first wrong byte.

wasm records="5000" sustained="20000":
    #!/usr/bin/env bash
    set -euo pipefail
    # open, put, get, delete, batch, scan, snapshot, iterate, compact,
    # close, REOPEN, read back - then `--sustained` writes past the L0
    # stop trigger with a 32 KiB memtable and no explicit compaction,
    # which is the case that wedges when nothing compacts on the
    # calling thread.
    cargo build --release -p lark-wasm-probe --target wasm32-wasip1
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    wasmtime run --dir="$d::/data" \
        target/wasm32-wasip1/release/lark-wasm-probe.wasm -- \
        --profile embedded --records {{records}} --sustained {{sustained}} \
        --probe-host --report-memory

# The same lifecycle natively, to tell a lark bug apart from a wasm one.

wasm-native records="5000" sustained="20000":
    cargo run --release -p lark-wasm-probe -- \
        --profile embedded --records {{records}} --sustained {{sustained}} \
        --probe-host --report-memory

wasm-browser:
    wasm-pack test --headless --chrome

embedded:
    cargo run --release --bench memory -- --profile embedded

# ---------- the gains figures ----------

# Collect every family into ONE run file and re-render. A PR runs this, then pastes.

gains: loadguard
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git rev-parse --short HEAD)
    cargo bench -- --baseline pre --save-baseline "$sha"
    cargo run --release --bench collect -- \
        --out {{gains_dir}}/runs/"$sha"-{{label}}.json \
        --commit "$sha" --label {{label}}
    just gains-render

gains-render:
    {{py}} {{gains_dir}}/render.py
    {{py}} {{gains_dir}}/render_rss.py

gains-diff base current:
    {{py}} {{gains_dir}}/render.py --baseline {{base}} --current {{current}}
    {{py}} {{gains_dir}}/render_rss.py --baseline {{base}} --current {{current}}

gains-list:
    {{py}} {{gains_dir}}/runs.py --list
