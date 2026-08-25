# lark-kv task runner.
#
# `just` with no recipe lists what is available.

default:
    @just --list

# The gates CI runs. Make this green before asking to push.
gate: fmt-check lint test doc

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

doc:
    cargo doc --workspace --no-deps

# The default suite: library, integration and tool tests. Stays fast.
test:
    cargo test --workspace

# Fault-injection substrate only.
test-fault:
    cargo test --test fault_smoke

# The `#[ignore]`d fault tests: each spawns child processes and simulates a
# power cut. Measured at well under a second each, but kept out of the
# default run so `cargo test` stays quick.
test-fault-slow:
    cargo test --test fault_smoke -- --ignored --skip crash_child

# Every ignored test in the workspace, including the scheduled stress runs.
test-slow:
    cargo test --workspace -- --ignored --skip crash_child

# Rebuild the LD_PRELOAD fault shim from scratch by dropping its cache.
fault-shim-clean:
    rm -rf target/tmp/lark-fault
