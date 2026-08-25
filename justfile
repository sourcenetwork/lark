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

# The `#[ignore]`d resource-exhaustion and extremes tests: six-figure key
# counts, a 64 MiB value, megabyte keys, a six-level cascade, and a real
# ENOSPC on a tmpfs mounted in a private user namespace. About 21 s in
# total on a debug build. `--test-threads=1` is required, not cosmetic:
# each test resets the kernel's peak-RSS counter to report its own memory
# high-water mark, and a concurrent test would inflate that number.
test-extremes:
    cargo test --test resource_limits -- --ignored --nocapture --test-threads 1 --skip crash_child

# The `#[ignore]`d durability property test: 128 randomized operation
# sequences, each run in a child process that is killed part way through
# and then power-cut by discarding every byte it never fsynced. Measured
# at 1.1 s. Needs the LD_PRELOAD fault shim, so it is Linux only.
test-durability-slow:
    cargo test --test proptest_durability -- --ignored --skip crash_child

# Every ignored test in the workspace, including the scheduled stress runs.
test-slow:
    cargo test --workspace -- --ignored --skip crash_child

# Rebuild the LD_PRELOAD fault shim from scratch by dropping its cache.
fault-shim-clean:
    rm -rf target/tmp/lark-fault

# The `#[ignore]`d exhaustive corruption sweeps: every byte offset and
# every single-bit flip of a WAL, an SSTable and a MANIFEST. 14,265
# trials, measured at 1.2s of wall time.
test-corruption-slow:
    cargo test --test corruption_exhaustive -- --ignored --skip crash_child

# The power-loss durability tests, with output shown so the measured cost of
# the default DurabilityMode::Eventual is visible. Every test here spawns a
# child process, crashes it at a byte-exact point and simulates a power cut.
# Measured at 0.6s in total, so it also runs in the default `cargo test`.
test-power:
    cargo test --test power_loss -- --skip crash_child --nocapture

# Process-crash recovery: kill -9 at every point in the write path. Every
# test in the file is in the default run already; this recipe just runs the
# file on its own. Measured at 0.6s, spawning 41 child processes.
test-crash:
    cargo test --test crash_recovery

# Handle lifecycle: open, close, reopen under changed Options, locking,
# and every misuse of a closed or read-only handle. Measured at 0.4s, so
# every test in the file also runs in the default `cargo test`; nothing
# here is `#[ignore]`d.
test-lifecycle:
    cargo test --test lifecycle -- --skip crash_child
