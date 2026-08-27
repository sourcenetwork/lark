

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

# nextest, not `cargo test`: each test gets its own process, so a
# wedged test fails on the profile's slow-timeout instead of holding the
# whole binary. See `.config/nextest.toml`.

test:
    cargo nextest run --workspace
    cargo test --workspace --doc

deny:
    cargo deny check

msrv:
    cargo "+$(grep -m1 '^rust-version' Cargo.toml | cut -d'"' -f2)" check --workspace

# The summary line CI annotates a build with.
# Through nextest, like the gate: one process per test, so a wedged
# test fails on the profile's slow-timeout instead of holding the whole
# instrumented run open until the job's ceiling.
cov-summary:
    cargo llvm-cov nextest --summary-only --workspace

# The browsable HTML report, for reading locally.
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
    cargo nextest run --test fault_smoke

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
    cargo nextest run --test corruption_exhaustive

# The power-loss durability tests, with output shown so the measured cost of
# the default DurabilityMode::Eventual is visible. Every test here spawns a
# child process, crashes it at a byte-exact point and simulates a power cut.
# Measured at 0.6s in total, so it also runs in the default `cargo test`.

test-durability-slow:
    cargo nextest run --test proptest_durability

# Every ignored test in the workspace, including the scheduled stress runs.

test-extremes:
    cargo nextest run --test resource_limits --no-capture --test-threads 1

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
    cargo nextest run --workspace --release --no-capture

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
    cargo nextest run --test mvcc_invariants --no-capture

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
                --bench scan --bench batch --bench transaction \
                --bench large_value -- --save-baseline pre

# Compare against the `pre` baseline. This is what a PR pastes into its body.

bench: loadguard
    cargo bench -- --baseline pre

# The sweep the CI perf artifact is built from: every family, into the
# JSON Lines file `collect` assembles a run file out of.
bench-collect: loadguard
    cargo bench --bench point_read --bench write_durable --bench write_buffered \
                --bench scan --bench batch --bench transaction \
                --bench large_value --bench memory --bench size

bench-one name: loadguard
    cargo bench --bench {{name}} -- --baseline pre

# RSS soak. Default 360s; pass seconds and an Options variant tag.

soak secs="360" wb="64" cache="64" shard_bits="6" tag="default": loadguard
    cargo run --release --bench soak -- {{secs}} {{wb}} {{cache}} {{shard_bits}} {{tag}}

# The two soak variants compared. Deterministic: no loadguard needed.

soak-pair:
    just soak 360 64 64 6 default
    just soak 360 64 64 0 cache-budgeted

# Binary size, native and both wasm targets, against the budget.

size:
    cargo run --release --bench size

# The memory table in the README: every profile, on both hosts, one
# workload. The wasm column is the reproducible one because linear
# memory only ever grows; RSS moves between runs.

wasm-budget puts="20000":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release --example embedded_profile --target wasm32-wasip1
    for profile in embedded wasm default; do
        echo "== $profile, x86_64 Linux =="
        d=$(mktemp -d)
        cargo run --release --quiet --example embedded_profile -- "$d" "$profile" {{puts}}
        rm -rf "$d"
        echo "== $profile, wasm32-wasip1 under wasmtime =="
        d=$(mktemp -d)
        wasmtime run --dir="$d::/data" \
            target/wasm32-wasip1/release/examples/embedded_profile.wasm \
            /data "$profile" {{puts}}
        rm -rf "$d"
    done

# Point memory probes.

mem:
    cargo run --release --bench memory

# The MVCC regression probes. Must stay at zero violations.

ycsb workload="a" records="1000000" ops="1000000": loadguard
    cargo run --release -p lark-ycsb -- \
        --workload {{workload}} --records {{records}} --operations {{ops}}

ycsb-all: loadguard
    for w in a b c d e f; do just ycsb $w; done

stress secs="600":
    cargo run --release -p lark-stress -- --duration {{secs}}

# ---------- model checking ----------

# Loom model checks for the arena memtable, read horizon and version
# handoffs. `--cfg loom` swaps the primitives in `src/engine/sync.rs`
# for loom's instrumented ones; without it the whole target compiles
# away, so an ordinary `cargo test` neither builds loom nor runs these.
#
# | recipe        | models | profile | measured |
# |---------------|--------|---------|----------|
# | `loom`        | 14     | release | 19.7s    |
# | `loom-debug`  | 15     | debug   | 133s     |
# | `loom-all`    | both   | both    | ~153s    |
#
# The debug run carries one extra calibration: the skip list's
# single-writer guard (S2) is a `debug_assert`, so the model proving it
# fires is compiled out of a release build. Six of the models are
# `should_panic` calibrations that deliberately get the ordering wrong;
# they are what make the passes mean anything.

loom:
    RUSTFLAGS="--cfg loom" cargo test --release --test loom_memtable

loom-debug:
    RUSTFLAGS="--cfg loom" cargo test --test loom_memtable

loom-all: loom loom-debug

# The read-view chaos workload at full size: 6 instances x 2 rounds x 400
# versions. Measured at over 20 minutes wall and 4h of CPU unoptimized,
# which is why `cargo test` runs a smaller default and this recipe
# carries the full one. Release, because debug is where the cost is.
#
# Sized to finish, not to be maximal. Cost is roughly
# instances x rounds x versions x compaction passes, and the compaction
# passes each rewrite a database that grows with the version count, so
# raising `versions` raises the run time faster than linearly: 400 does
# not complete inside seven minutes, 120 completes in seconds. What the
# workload is hunting is overlap between a compaction and a read, and
# the overlap count is already in the thousands per instance here.

chaos instances="4" rounds="2" versions="120" min_rounds="20":
    LARK_CHAOS_INSTANCES={{instances}} LARK_CHAOS_ROUNDS={{rounds}} \
    LARK_CHAOS_VERSIONS={{versions}} LARK_CHAOS_MIN_ROUNDS={{min_rounds}} \
        cargo test --release --test read_view_chaos_workload -- --nocapture

# ---------- consistency ----------

# Elle consistency checking. `model` is the workload (list-append or
# rw-register); `level` is the consistency model to check it against.
#
# The two are separate axes and elle-cli spells both `--model`-ish, which
# is easy to get wrong: passing an isolation level as --model throws
# "No matching clause". Hence the explicit --consistency-models here.
elle model="list-append" level="snapshot-isolation" isolation="repeatable-read":
    cargo run --release --manifest-path harness/elle/Cargo.toml --bin elle-gen -- \
        --model {{model}} --isolation {{isolation}} \
        --threads 8 --txns 50 --keys 4 \
        --out /tmp/lark-history.json --dir /tmp/lark-elle-db
    java -jar harness/elle/elle-cli.jar --model {{model}} \
        --consistency-models {{level}} /tmp/lark-history.json

# The same, with the fault injection the harness supports.
elle-fault model="list-append" level="snapshot-isolation":
    cargo run --release --manifest-path harness/elle/Cargo.toml --bin elle-gen -- \
        --model {{model}} --isolation repeatable-read --faults \
        --threads 8 --txns 50 --keys 4 \
        --out /tmp/lark-history-fault.json --dir /tmp/lark-elle-fault-db
    java -jar harness/elle/elle-cli.jar --model {{model}} \
        --consistency-models {{level}} /tmp/lark-history-fault.json

# Every level lark claims, checked in one go.
#
# Each line prints `true` or `false`; a `false` on a level lark claims is
# a real failure and the recipe exits non-zero.
elle-matrix:
    #!/usr/bin/env bash
    set -uo pipefail
    cd harness/elle
    cargo build --release --bin elle-gen
    fail=0
    check() {
        local name="$1" model="$2" level="$3"; shift 3
        ./target/release/elle-gen --model "$model" "$@" \
            --out "/tmp/elle-$name.json" --dir "/tmp/elle-db-$name" >/dev/null
        # elle-cli prints "<path>\t<true|false>"; take the last field and
        # strip surrounding whitespace, so a stray tab cannot read as a
        # failure on a history that actually passed.
        local v
        v=$(java -jar elle-cli.jar --model "$model" \
            --consistency-models "$level" "/tmp/elle-$name.json" \
            | tail -1 | awk '{print $NF}')
        printf '  %-42s %s\n' "$name [$level]" "$v"
        if [ "$v" != "true" ]; then fail=1; fi
    }
    # Optimistic transactions are snapshot isolation. That is the level
    # lark claims, so a false here is a defect.
    check optimistic-si       list-append snapshot-isolation --isolation repeatable-read --threads 8 --txns 50 --keys 4 --seed 4
    check optimistic-rw       rw-register snapshot-isolation --isolation repeatable-read --threads 8 --txns 50 --keys 4 --seed 5
    # Pessimistic transactions are checked at the level they request.
    check pessimistic-rc      list-append read-committed     --isolation read-committed --threads 8 --txns 50 --keys 4 --seed 2
    check pessimistic-hotkey  list-append read-committed     --isolation read-committed --threads 8 --txns 50 --keys 1 --seed 1
    # Serializable validates the whole read set, so the strongest model
    # Elle offers must hold.
    check serializable        list-append strict-serializable --isolation serializable --threads 8 --txns 60 --keys 4 --seed 11
    check serializable-rw     rw-register strict-serializable --isolation serializable --threads 8 --txns 60 --keys 4 --seed 12
    exit $fail

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
    # Both shipped profiles a wasm module can open with, on the real
    # target: `embedded` runs with no block cache, `wasm` with one.
    for profile in embedded wasm; do
        echo "== profile $profile =="
        d=$(mktemp -d)
        wasmtime run --dir="$d::/data" \
            target/wasm32-wasip1/release/lark-wasm-probe.wasm -- \
            --profile "$profile" --records {{records}} --sustained {{sustained}} \
            --probe-host --report-memory
        rm -rf "$d"
    done

# The same lifecycle natively, to tell a lark bug apart from a wasm one.

wasm-native records="5000" sustained="20000":
    #!/usr/bin/env bash
    set -euo pipefail
    for profile in embedded wasm; do
        echo "== profile $profile =="
        cargo run --release -p lark-wasm-probe -- \
            --profile "$profile" --records {{records}} --sustained {{sustained}} \
            --probe-host --report-memory
    done

# The OPFS contract against a real browser. `wasm-pack test` cannot
# drive these: it appends `--tests`, which builds every target in
# `tests/`, and all but the three `wasm_opfs*` files are native-only.
# The runner is named per target instead, so only the named test
# binaries are built.
wasm-browser:
    #!/usr/bin/env bash
    set -euo pipefail
    export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
    for suite in wasm_opfs wasm_opfs_main wasm_opfs_memory; do
        echo "== $suite =="
        cargo test --target wasm32-unknown-unknown --test "$suite"
    done

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
