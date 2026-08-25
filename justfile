set shell := ["bash", "-uc"]

gains_dir := justfile_directory() / "../larkgains"
label     := env_var_or_default("LABEL", "wip")
py        := env_var_or_default("LARKGAINS_PY", "python3")

# NOTE: the commit sha is resolved INSIDE the recipe, never as a top-level
# `sha := `git ...`` assignment. just evaluates those eagerly at parse time, so a
# top-level backtick makes every `just --list` fail outside a git checkout.

# ---------- the gate ----------

default: gate

gate: fmt lint doc test deny

fmt:
    cargo fmt --all -- --check

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
mvcc:
    cargo test --release --test mvcc_invariants -- --nocapture

# YCSB workload a-f. F (read-modify-write) is the one that exercises G1.
ycsb workload="a" records="1000000" ops="1000000": loadguard
    cargo run --release -p lark-ycsb -- \
        --workload {{workload}} --records {{records}} --operations {{ops}}

ycsb-all: loadguard
    for w in a b c d e f; do just ycsb $w; done

stress secs="600":
    cargo run --release -p lark-stress -- --duration {{secs}}

# ---------- consistency ----------

elle model="list-append":
    cargo run --release --manifest-path harness/elle/Cargo.toml -- \
        --model {{model}} --out /tmp/lark-history.json
    java -jar harness/elle/elle-cli.jar --model {{model}} /tmp/lark-history.json

elle-fault model="list-append":
    cargo run --release --manifest-path harness/elle/Cargo.toml -- \
        --model {{model}} --faults --out /tmp/lark-history-fault.json
    java -jar harness/elle/elle-cli.jar --model {{model}} /tmp/lark-history-fault.json

# ---------- portability ----------

wasm:
    cargo build --release --target wasm32-wasip1
    wasmtime --dir=.:/data target/wasm32-wasip1/release/examples/smoke.wasm

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
