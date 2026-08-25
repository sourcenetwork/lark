//! MVCC and concurrency invariants, promoted from one-off probes into
//! permanent regression tests.
//!
//! `tests/concurrency.rs` covers the scenario shapes ported from
//! `db_test.cc`. This file covers the *engine* invariants underneath
//! them, the ones a scenario test can pass without ever exercising:
//!
//! 1. A snapshot's view is byte-identical for its whole life, whatever
//!    concurrent writers, deleters and compactions do to the keys it
//!    covers.
//! 2. A `WriteBatch` that overwrites a key set is never observed
//!    half-applied, so a reader never sees a mix of generations.
//! 3. A repeated read of one key never travels backwards in version.
//! 4. Delete, compact and reopen preserve the exact surviving version
//!    of every key.
//! 5. An iterator keeps serving its view after compaction unlinks the
//!    SSTable files beneath it (the pinned-`Arc` contract).
//! 6. A snapshot pins every version its reads need, including when an
//!    older snapshot is the one that has to hold the GC horizon back.
//!
//! # Why none of these can flake
//!
//! Every concurrent test here has the same shape. The writers do a
//! **bounded** amount of work and then leave. The readers do a fixed
//! minimum number of checks and then keep going until every writer has
//! left, so a slower machine gets *more* overlap, never less, and the
//! run still terminates. No assertion is on a count of anything the
//! scheduler decides: the counts are returned and reported, and only
//! floors ("at least the minimum ran") are asserted. There is no
//! `sleep` anywhere in this file.
//!
//! Bounding the writers matters for more than runtime. A long-lived
//! snapshot pins every version written under it, so a reader-paced
//! writer loop feeds back on itself: more retained versions make each
//! snapshot scan slower, which lets the writers write more. Fixing the
//! writer op count breaks that loop and keeps the data volume flat.
//!
//! Every workload is generated from a fixed seed, so a failure
//! reproduces byte for byte.
//!
//! The `#[ignore]`d tests are the full-scale versions of the same
//! properties; see each one's doc comment for its measured runtime.
//! `just mvcc` runs the fast set, `just mvcc-slow` the full-scale set.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use lark_kv::{Db, Snapshot, WriteBatch};
use tempfile::TempDir;

mod common;

use common::fault;

/// The shared workload harness. It lives in its own file so neither
/// half grows past the size where it stops being readable in one
/// sitting; the `#[path]` attribute is what keeps
/// `tests/mvcc_invariants/` from being picked up as a second test
/// target.
#[path = "mvcc_invariants/harness.rs"]
mod harness;

use harness::{
    assert_background_work_happened, assert_same_view, drain_iter, instrumented, key_at,
    open_instrumented, run_batch_atomicity, run_monotonic_reads, run_monotonic_reads_in_parallel,
    run_snapshot_stability, stamp_of, stamped_value, AtomicityScale, Entries, Live, MonotonicScale,
    StabilityScale,
};

/// Child-process entry point required of every test crate that links
/// the fault-injection harness. This file injects no faults, so the
/// entry point is never re-executed; it exists so the crate satisfies
/// the harness contract uniformly with its siblings.
#[test]
#[ignore = "child process entry point, re-executed by the crash harness"]
fn crash_child() {
    fault::child_entrypoint(fault::builtin_workload);
}

/// A snapshot's view is byte-identical for its whole life.
///
/// Property: for a snapshot taken at seq `S`, every later read through
/// it (full range scan and point get) returns exactly the bytes the
/// first scan returned, whatever concurrent writers, deleters and full
/// compactions do to the same keys.
///
/// Catches: a compaction that drops a version a live snapshot still
/// needs, a GC horizon computed from the newest rather than the oldest
/// live snapshot, a flush that reorders visibility, and any read path
/// that consults a source without filtering it by the snapshot's seq.
#[test]
fn a_snapshots_view_is_byte_identical_for_its_whole_life() {
    let counts = run_snapshot_stability(&StabilityScale {
        keys: 400,
        writers: 6,
        ops_per_writer: 400,
        min_reads: 20,
        min_compactions: 3,
    });
    assert!(counts.snapshot_reads >= 20);
    assert!(counts.entries_compared >= 20 * 400);
    assert!(counts.compactions >= 3);
    assert_eq!(counts.writer_ops, 2_400);
}

/// Full-scale version of
/// [`a_snapshots_view_is_byte_identical_for_its_whole_life`], the
/// shape the original probe ran at: 2000 keys, 6 writer threads and
/// 120000 writes racing a snapshot that pins every version of them.
///
/// Measured runtime is in the `#[ignore]` reason. Kept out of the
/// default run so `cargo test` stays fast; `just mvcc-slow` runs it.
#[test]
#[ignore = "full-scale MVCC soak, measured at 13.8s; run with `just mvcc-slow`"]
fn snapshot_stability_at_full_scale() {
    let counts = run_snapshot_stability(&StabilityScale {
        keys: 2_000,
        writers: 6,
        ops_per_writer: 20_000,
        min_reads: 100,
        min_compactions: 10,
    });
    println!(
        "snapshot stability: {} scans, {} entries compared, {} writer ops, {} full compactions",
        counts.snapshot_reads, counts.entries_compared, counts.writer_ops, counts.compactions,
    );
}

/// A `WriteBatch` that rewrites a whole key set is never observed
/// half-applied.
///
/// Property: at any instant every key in the batch carries the *same*
/// generation stamp. `tests/concurrency.rs` already checks that a
/// batch of *fresh* keys is all-present or all-absent; this checks the
/// harder overwrite case, where a torn batch leaves the key set
/// present but at mixed generations, over three independent read
/// surfaces (`multi_get`, snapshot point reads, snapshot range scan).
///
/// Catches: publishing the read horizon before the last key of a batch
/// is applied, a per-key sequence number where a per-batch one is
/// required, a `multi_get` that captures a fresh seq per key instead
/// of one for the call, and a scan that disagrees with the point reads
/// taken from the same snapshot.
#[test]
fn a_reader_never_observes_a_write_batch_half_applied() {
    let (checks, generations) = run_batch_atomicity(&AtomicityScale {
        width: 12,
        readers: 4,
        min_checks_per_reader: 150,
        generations: 1_500,
    });
    assert!(checks >= 4 * 150 * 3);
    assert_eq!(generations, 1_500);
}

/// Full-scale version of
/// [`a_reader_never_observes_a_write_batch_half_applied`].
///
/// Measured runtime is in the `#[ignore]` reason. Run with
/// `just mvcc-slow`.
#[test]
#[ignore = "full-scale batch-atomicity soak, measured at 9.3s; run with `just mvcc-slow`"]
fn batch_atomicity_at_full_scale() {
    let (checks, generations) = run_batch_atomicity(&AtomicityScale {
        width: 24,
        readers: 4,
        min_checks_per_reader: 2_000,
        generations: 30_000,
    });
    println!("batch atomicity: {checks} checks over {generations} generations, 0 torn");
}

/// A repeated read of one key never travels backwards in version.
///
/// Property: each key has exactly one writer, which stamps strictly
/// increasing version numbers into the value. A reader that observes
/// version `n` must never subsequently observe a version below `n`,
/// and must never observe the key as absent, across memtable rotation,
/// flush to L0 and full compaction.
///
/// Catches: a read that consults an SSTable before the memtable and
/// returns the older version, a flush that publishes an L0 file before
/// the memtable it replaces is retired, a compaction that installs a
/// version whose newest entry lost to an older one in the merge, and a
/// block-cache entry served after its file was rewritten.
#[test]
fn a_repeated_read_of_one_key_never_travels_backwards() {
    let outcome = run_monotonic_reads(&MonotonicScale {
        writers: 4,
        keys_per_writer: 8,
        readers: 3,
        min_rounds_per_reader: 150,
        versions: 250,
    });
    outcome.assert_clean("monotonic reads");
    assert!(outcome.reads >= 3 * 150 * 4 * 8);
    assert_eq!(outcome.writes, 250 * 32);
}

/// Full-scale version of
/// [`a_repeated_read_of_one_key_never_travels_backwards`].
///
/// This test currently **fails intermittently** against the engine, and
/// the failure is real, not a test defect: see
/// [`a_user_thread_compact_range_never_makes_a_read_travel_backwards`]
/// for the measured reproduction and the mechanism. Measured red in 2
/// of 16 runs at this scale, which is why the focused gate exists.
///
/// Measured runtime is in the `#[ignore]` reason. Run with
/// `just mvcc-slow`.
#[test]
#[ignore = "full-scale monotonic-read soak, measured at 4.5s; run with `just mvcc-slow`"]
fn monotonic_reads_at_full_scale() {
    let outcome = run_monotonic_reads(&MonotonicScale {
        writers: 4,
        keys_per_writer: 12,
        readers: 3,
        min_rounds_per_reader: 2_000,
        versions: 4_000,
    });
    println!(
        "monotonic reads: {} reads across {} writes, {} regressions",
        outcome.reads,
        outcome.writes,
        outcome.violations.len(),
    );
    outcome.assert_clean("monotonic reads at full scale");
}

/// A `compact_range` driven from a user thread never makes a
/// concurrent read travel backwards.
///
/// This test currently **fails** against the engine. The failure is a
/// real defect, and this test is the focused gate for it. It runs the
/// minimal reproducing shape on 60 databases, 4 at a time, because the
/// window is short and only opens under real contention.
///
/// Property: with writers overwriting their own keys, readers polling
/// them, and one thread calling `Db::compact_range`, no reader may see
/// a key move backwards in version or read back as absent.
///
/// The defect: `LarkEngine::get` (`src/engine/mod.rs`) takes the active
/// memtable, the frozen memtable list and the version under three
/// separate lock acquisitions, releasing each before taking the next,
/// so no reader ever observes a consistent set of sources. A reader can
/// miss a key's newest version in every source it looks at and fall
/// through to an older one, or briefly find it in none of them.
///
/// Measured: 11 violating instances out of 300, over five independent
/// 60-instance batches (4, 3, 3, 1, 0), so about 4 runs in 5 are red.
/// The same workload with the user-thread `compact_range` removed gave
/// 0/60, and the same workload with a live `Snapshot` pinning the GC
/// horizon also gave 0/60. That pair is what narrows the cause to the
/// read path racing a compaction that is free to drop versions, rather
/// than to the flush path, which installs the new L0 file *before* it
/// removes the memtable and so leaves no gap.
#[test]
#[ignore = "focused regression gate for the user-thread compact_range read race, measured at 26s; currently red, see the doc comment; run with `just mvcc-slow`"]
fn a_user_thread_compact_range_never_makes_a_read_travel_backwards() {
    // 15 rounds of 4 concurrent databases: the exact shape the defect
    // was measured on, so the rate this prints is comparable with the
    // 4/60 and 3/60 recorded in the doc comment.
    let scale = MonotonicScale {
        writers: 4,
        keys_per_writer: 12,
        readers: 3,
        min_rounds_per_reader: 50,
        versions: 1_500,
    };
    let mut violations = Vec::new();
    let mut reads = 0u64;
    let mut instances = 0usize;
    let mut dirty = 0usize;
    for _ in 0..15 {
        for o in run_monotonic_reads_in_parallel(&scale, 4) {
            instances += 1;
            reads += o.reads;
            if !o.violations.is_empty() {
                dirty += 1;
                violations.extend(o.violations);
            }
        }
    }
    println!(
        "compact_range read race: {dirty} of {instances} instances violated, over {reads} reads",
    );
    assert!(
        violations.is_empty(),
        "{dirty} of {instances} instances saw a read travel backwards under a user-thread \
         compact_range:\n  {}",
        violations.join("\n  "),
    );
}

// ── 4. version integrity across delete, compact, reopen ────────────

/// Delete, compact and reopen preserve the exact surviving version of
/// every key.
///
/// Property: over 5000 keys written at version 1, a third overwritten
/// to version 2 and a third deleted, the surviving state after a full
/// compaction is exactly `{i%3==0 -> v1, i%3==1 -> v2}`, the range
/// scan and the iterator agree on it, and it is byte-identical after
/// closing and reopening the database and compacting again.
///
/// Catches: a compaction that keeps a shadowed older version, a
/// tombstone dropped above a live entry it must still shadow, a
/// resurrection across WAL replay, and a scan whose order or content
/// diverges from point reads after recovery.
#[test]
fn delete_then_compact_then_reopen_keeps_every_surviving_version() {
    let dir = TempDir::new().unwrap();
    let total = 5_000usize;

    let expected: Vec<Option<Vec<u8>>> = (0..total)
        .map(|i| match i % 3 {
            0 => Some(stamped_value(1)),
            1 => Some(stamped_value(2)),
            _ => None,
        })
        .collect();
    let live = expected.iter().filter(|e| e.is_some()).count();

    let (before_scan, before_iter) = {
        let (db, stats) = open_instrumented(dir.path(), 16 * 1024);
        for i in 0..total {
            db.put(&key_at(i), &stamped_value(1)).unwrap();
        }
        for i in (1..total).step_by(3) {
            db.put(&key_at(i), &stamped_value(2)).unwrap();
        }
        for i in (2..total).step_by(3) {
            db.delete(&key_at(i)).unwrap();
        }
        db.compact_range(None, None).unwrap();

        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                db.get(&key_at(i)).unwrap().as_deref(),
                want.as_deref(),
                "pre-reopen: key {i} has the wrong version",
            );
        }
        assert_background_work_happened(&stats, "delete-compact-reopen");
        (db.scan(None, None).unwrap(), drain_iter(&db))
    };

    assert_eq!(
        before_scan.len(),
        live,
        "pre-reopen scan returned {} entries, expected {live}",
        before_scan.len(),
    );
    assert_eq!(
        before_iter, before_scan,
        "pre-reopen: the iterator and the range scan disagree",
    );

    let (db, _stats) = open_instrumented(dir.path(), 16 * 1024);
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(
            db.get(&key_at(i)).unwrap().as_deref(),
            want.as_deref(),
            "post-reopen: key {i} has the wrong version",
        );
    }
    assert_same_view(
        &before_scan,
        &db.scan(None, None).unwrap(),
        "post-reopen scan",
    );
    assert_same_view(&before_scan, &drain_iter(&db), "post-reopen iterator");

    // Compacting the recovered database must not change its content.
    db.compact_range(None, None).unwrap();
    assert_same_view(
        &before_scan,
        &db.scan(None, None).unwrap(),
        "post-reopen recompaction scan",
    );
}

// ── 5. iterators pinned across compactions that unlink their files ──

/// An iterator keeps serving its view after compaction unlinks the
/// SSTable files it is reading from.
///
/// Property: an iterator captures an `Arc<Version>` whose `LiveSst`
/// readers hold open file descriptors. A later compaction rewrites
/// every input file, evicts their blocks from the block cache and
/// unlinks them; the half-consumed iterator must still produce the
/// exact remaining tail of the view it started with, and must not
/// observe the newer values written after it was created.
///
/// Catches: an iterator that reopens an SSTable by path instead of
/// holding its reader, a version released while an iterator still
/// references it, a block-cache eviction that leaves the iterator
/// unable to refill from the unlinked inode, and an iterator that
/// leaks later writes from the shared active memtable instead of
/// filtering them by its own sequence number. The test fails loudly if
/// no file was actually unlinked, so it can never pass vacuously.
#[test]
fn an_iterator_survives_the_compaction_that_unlinks_its_files() {
    let dir = TempDir::new().unwrap();
    let (mut opts, stats) = instrumented(4 * 1024);
    // Small output files so the iterator's view spans several of them.
    opts.target_file_size = 16 * 1024;
    let db = Db::open(dir.path(), opts).unwrap();

    let total = 1_200usize;
    for i in 0..total {
        db.put(&key_at(i), &stamped_value(1)).unwrap();
    }
    // Land the whole view in files, so there is something to unlink.
    db.compact_range(None, None).unwrap();

    let baseline = drain_iter(&db);
    assert_eq!(baseline.len(), total);

    let files_before = fault::find_ssts(dir.path());
    assert!(
        !files_before.is_empty(),
        "the seed data never reached an SSTable",
    );

    let mut it = db.iter();
    it.seek_to_first();
    let mut seen = Entries::new();
    for _ in 0..total / 2 {
        assert!(it.valid(), "iterator ended early at entry {}", seen.len());
        seen.push((it.key().unwrap().to_vec(), it.value().unwrap().to_vec()));
        it.next();
    }

    // Rewrite everything underneath the half-consumed iterator.
    for i in 0..total {
        db.put(&key_at(i), &stamped_value(2)).unwrap();
    }
    db.compact_range(None, None).unwrap();

    let unlinked = files_before.iter().filter(|p| !p.exists()).count();
    assert!(
        unlinked > 0,
        "none of the {} SSTables the iterator started over was unlinked, so the pinned-Arc \
         contract was never exercised",
        files_before.len(),
    );

    while it.valid() {
        seen.push((it.key().unwrap().to_vec(), it.value().unwrap().to_vec()));
        it.next();
    }
    it.status()
        .expect("iterator errored after its files were unlinked");

    assert_same_view(
        &baseline,
        &seen,
        "iterator held across an unlinking compaction",
    );
    assert_background_work_happened(&stats, "iterator pinning");

    // The live database moved on; only the iterator stayed behind.
    assert_eq!(
        db.get(&key_at(0)).unwrap().as_deref(),
        Some(stamped_value(2).as_slice()),
    );
}

/// Concurrent iterators are unaffected by compactions running
/// underneath them.
///
/// Property: three threads repeatedly walk the whole keyspace while a
/// fourth rewrites every key as one atomic batch and compacts. Every
/// walk must return the full key set in order, and every value in one
/// walk must carry the same generation, because each walk reads at a
/// single sequence number and each rewrite is one batch.
///
/// Catches: a version freed while another thread iterates it, a shared
/// block-cache entry invalidated under a concurrent reader, and an
/// iterator whose view drifts forward mid-walk as files are replaced.
#[test]
fn concurrent_iterators_are_unaffected_by_compactions_beneath_them() {
    let dir = TempDir::new().unwrap();
    let (db, stats) = open_instrumented(dir.path(), 16 * 1024);
    let db = Arc::new(db);

    let total = 800usize;
    let mut seed_batch = WriteBatch::new();
    for i in 0..total {
        seed_batch.put(&key_at(i), &stamped_value(1));
    }
    db.write(seed_batch).unwrap();
    db.compact_range(None, None).unwrap();

    let baseline = Arc::new(db.scan(None, None).unwrap());
    assert_eq!(baseline.len(), total);

    let readers_count = 3usize;
    let min_walks = 10usize;
    let generations = 8u64;
    let live = Live::new(1);
    let walks = Arc::new(AtomicU64::new(0));
    let gate = Arc::new(Barrier::new(readers_count + 1));

    let compactor = {
        let db = Arc::clone(&db);
        let live = live.clone();
        let gate = Arc::clone(&gate);
        thread::spawn(move || {
            gate.wait();
            for generation in 2..=generations + 1 {
                let mut batch = WriteBatch::new();
                for i in 0..total {
                    batch.put(&key_at(i), &stamped_value(generation));
                }
                db.write(batch).unwrap();
                db.compact_range(None, None).unwrap();
            }
            live.done_one();
        })
    };

    let mut readers = Vec::new();
    for r in 0..readers_count {
        let db = Arc::clone(&db);
        let live = live.clone();
        let gate = Arc::clone(&gate);
        let baseline = Arc::clone(&baseline);
        let walks = Arc::clone(&walks);
        readers.push(thread::spawn(move || {
            gate.wait();
            let mut round = 0usize;
            loop {
                let view = drain_iter(&db);
                assert_eq!(
                    view.len(),
                    baseline.len(),
                    "iterator r{r} round {round} saw {} keys, expected {}",
                    view.len(),
                    baseline.len(),
                );
                for (i, (want, got)) in baseline.iter().zip(view.iter()).enumerate() {
                    assert_eq!(
                        want.0, got.0,
                        "iterator r{r} round {round}: entry {i} has the wrong key",
                    );
                }
                let first = stamp_of(&view[0].1);
                for (k, v) in &view {
                    assert_eq!(
                        stamp_of(v),
                        first,
                        "iterator r{r} round {round}: key {:?} is at version {} while the first \
                         key is at version {first}; the walk crossed a write boundary",
                        String::from_utf8_lossy(k),
                        stamp_of(v),
                    );
                }
                walks.fetch_add(1, Ordering::Relaxed);
                round += 1;
                if round >= min_walks && !live.any() {
                    break;
                }
            }
        }));
    }

    let outcomes: Vec<_> = readers
        .into_iter()
        .chain(std::iter::once(compactor))
        .map(|h| h.join())
        .collect();
    for o in outcomes {
        o.unwrap();
    }

    assert!(walks.load(Ordering::Relaxed) >= (readers_count * min_walks) as u64);
    assert_background_work_happened(&stats, "concurrent iterators");
}

// ── 6. snapshots pin the versions their reads need ─────────────────

/// A snapshot pins every version its reads need, including when an
/// older snapshot is the one holding the GC horizon back.
///
/// Property: three snapshots taken at three different generations each
/// keep returning their own generation after every intervening version
/// has been overwritten, deleted, rewritten and compacted away nine
/// times over. Dropping the newest two must not disturb the oldest,
/// and dropping the oldest must not disturb the live database.
///
/// Catches: a GC horizon computed from the newest live snapshot rather
/// than the oldest, a snapshot registry that releases a pin on the
/// wrong sequence, and a compaction that drops a version below the
/// pinned sequence.
#[test]
fn a_snapshot_pins_every_version_its_reads_need() {
    let dir = TempDir::new().unwrap();
    let (db, stats) = open_instrumented(dir.path(), 8 * 1024);
    let keys = 300usize;
    let last_generation = 12u64;

    let mut snapshots: Vec<(u64, Snapshot, Entries)> = Vec::new();
    for generation in 1..=3u64 {
        for i in 0..keys {
            db.put(&key_at(i), &stamped_value(generation)).unwrap();
        }
        let snap = db.snapshot();
        let view = snap.scan(None, None).unwrap();
        assert_eq!(view.len(), keys);
        snapshots.push((generation, snap, view));
    }
    assert_eq!(
        db.get_int_property("lark.num-snapshots"),
        Some(3),
        "the engine did not register all three snapshot pins",
    );

    // Bury every pinned version: overwrite, delete half, rewrite, and
    // compact after each round so the compactor gets every chance to
    // drop what the snapshots still need.
    for generation in 4..=last_generation {
        for i in 0..keys {
            db.put(&key_at(i), &stamped_value(generation)).unwrap();
        }
        for i in (0..keys).step_by(2) {
            db.delete(&key_at(i)).unwrap();
        }
        for i in (0..keys).step_by(2) {
            db.put(&key_at(i), &stamped_value(generation)).unwrap();
        }
        db.compact_range(None, None).unwrap();
    }

    for (generation, snap, view) in &snapshots {
        assert_same_view(
            view,
            &snap.scan(None, None).unwrap(),
            &format!("snapshot at generation {generation}"),
        );
        for i in [0usize, keys / 2, keys - 1] {
            assert_eq!(
                snap.get(&key_at(i)).unwrap().as_deref(),
                Some(stamped_value(*generation).as_slice()),
                "snapshot at generation {generation} lost key {i}",
            );
        }
    }

    // Dropping the two newer pins must leave the oldest intact. If the
    // horizon were taken from the newest live snapshot, what the
    // oldest still needs is exactly what the next compaction drops.
    let newest = snapshots.pop().unwrap();
    let middle = snapshots.pop().unwrap();
    drop(newest);
    drop(middle);
    assert_eq!(db.get_int_property("lark.num-snapshots"), Some(1));
    db.compact_range(None, None).unwrap();

    let (generation, snap, view) = snapshots.pop().unwrap();
    assert_eq!(generation, 1);
    assert_same_view(
        &view,
        &snap.scan(None, None).unwrap(),
        "oldest snapshot after the newer ones were dropped",
    );

    drop(snap);
    assert_eq!(db.get_int_property("lark.num-snapshots"), Some(0));
    db.compact_range(None, None).unwrap();
    for i in 0..keys {
        assert_eq!(
            db.get(&key_at(i)).unwrap().as_deref(),
            Some(stamped_value(last_generation).as_slice()),
            "live database lost key {i} once every snapshot was released",
        );
    }
    assert_background_work_happened(&stats, "snapshot pinning");
}
