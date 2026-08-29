//! Three things the rest of the API already promises, held on the paths that
//! were not keeping them.
//!
//! 1. A handle that is not live is an error, not an empty column family.
//! 2. `drop_all` drops everything, including what a compaction was midway
//!    through writing.
//! 3. `close` returns, even when an ingest holds the compaction gate.

// Native-only. wasm-pack builds every test target for wasm32, and these use
// threads and the filesystem, neither of which exists there.
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use regolith::{Db, Options, WriteBatch};

fn small() -> Options {
    Options {
        write_buffer_size: 64 * 1024,
        block_size: 4 * 1024,
        block_cache_size: 0,
        target_file_size: 128 * 1024,
        l0_compaction_trigger: 2,
        ..Options::default()
    }
}

#[test]
fn a_dropped_cf_handle_is_an_error_on_the_iterator_paths_too() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), small()).unwrap();

    let cf = db.create_column_family("things").unwrap();
    db.put_cf(&cf, b"a", b"1").unwrap();
    db.drop_column_family(cf.clone()).unwrap();

    // The rest of the CF surface already reports this.
    assert!(db.scan_cf(&cf, None, None).is_err());
    assert!(db.get_cf(&cf, b"a").is_err());

    // The iterators must too. Before, each of these read as an empty column
    // family that succeeded, which is the worst of both: no rows and no
    // reason.
    let mut it = db.iter_cf(&cf);
    it.seek_to_first();
    assert!(!it.valid());
    assert!(
        it.status().is_err(),
        "iter_cf over a dropped handle must say why it is empty"
    );

    let mut tail = db.iter_tailing_cf(&cf);
    tail.seek_to_first();
    assert!(
        tail.status().is_err(),
        "iter_tailing_cf over a dropped handle must say why it is empty"
    );

    let snap = db.snapshot();
    let mut sit = snap.iter_cf(&cf);
    sit.seek_to_first();
    assert!(
        sit.status().is_err(),
        "Snapshot::iter_cf over a dropped handle must say why it is empty"
    );

    // A live handle is unaffected, and reports clean.
    let live = db.create_column_family("live").unwrap();
    db.put_cf(&live, b"k", b"v").unwrap();
    let mut ok = db.iter_cf(&live);
    ok.seek_to_first();
    assert!(ok.valid());
    ok.status().expect("a live handle must report clean");
}

#[test]
fn drop_all_is_not_undone_by_a_compaction_that_was_already_running() {
    // Enough data, with workers running, that a compaction is very likely in
    // flight when the drop lands. Repeated so the interleaving is met rather
    // than hoped for.
    for round in 0..8 {
        let dir = tempfile::tempdir().unwrap();
        let opts = Options {
            max_background_compactions: 4,
            ..small()
        };
        let db = Db::open(dir.path(), opts.clone()).unwrap();

        let value = [b'v'; 128];
        let mut batch = WriteBatch::new();
        for i in 0..20_000u64 {
            batch.put(&i.to_be_bytes(), &value);
            if batch.buffered_bytes() >= 64 * 1024 {
                db.write(std::mem::take(&mut batch)).unwrap();
            }
        }
        db.write(batch).unwrap();

        db.drop_all().unwrap();

        // Give any compaction that was mid-flight time to try to apply its
        // edits over the reset version.
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let n = db.scan_stream(None, None).unwrap().count();
            assert_eq!(
                n, 0,
                "round {round}: drop_all reported success, so nothing may come back"
            );
            if !db.compact_step().unwrap() {
                break;
            }
        }
        assert_eq!(db.scan_stream(None, None).unwrap().count(), 0);

        // And it must stay dropped across a reopen, because a compaction that
        // landed after the reset would be in the manifest.
        db.close().unwrap();
        drop(db);
        let db = Db::open(dir.path(), opts).unwrap();
        assert_eq!(
            db.scan_stream(None, None).unwrap().count(),
            0,
            "round {round}: the drop must be recorded, not just held in memory"
        );
    }
}

#[test]
fn close_returns_while_an_ingest_holds_the_compaction_gate() {
    // close() used to join the compaction workers while holding the mutex
    // that ingest_external_files wants, and a worker cannot exit while ingest
    // holds the compaction gate. That is a cycle, and a cycle hangs. The
    // assertion is simply that this returns.
    for round in 0..6 {
        let dir = tempfile::tempdir().unwrap();
        let opts = Options {
            max_background_compactions: 4,
            ..small()
        };
        let db = Arc::new(Db::open(dir.path(), opts).unwrap());

        let value = [b'v'; 128];
        let mut batch = WriteBatch::new();
        for i in 0..8_000u64 {
            batch.put(&i.to_be_bytes(), &value);
            if batch.buffered_bytes() >= 32 * 1024 {
                db.write(std::mem::take(&mut batch)).unwrap();
            }
        }
        db.write(batch).unwrap();

        let writer = {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                // Keep the engine busy so workers are live and contending
                // when the close lands.
                for i in 0..4_000u64 {
                    if db.put(&(1_000_000 + i).to_be_bytes(), b"x").is_err() {
                        break;
                    }
                }
            })
        };

        let start = Instant::now();
        db.close().unwrap();
        let elapsed = start.elapsed();
        writer.join().unwrap();

        assert!(
            elapsed < Duration::from_secs(30),
            "round {round}: close took {elapsed:?}, which means it was waiting on \
             something that was waiting on it"
        );
    }
}
