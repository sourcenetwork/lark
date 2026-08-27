//! Reproduction probe: a checkpoint taken while a writer is running can
//! miss data that was acknowledged before the writer ever started.
//!
//! `LarkEngine::checkpoint_capture` flushes the *active* memtable when
//! it is non-empty, but never looks at the frozen list. A concurrent
//! `rotate_memtable` leaves a window where the active memtable is fresh
//! and empty while the sealed one is still being flushed; a checkpoint
//! sampled there skips the flush and then records the manifest length
//! before the flush's `AddFile` lands.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use lark_kv::{Db, Options, WriteBatch};
use tempfile::TempDir;

fn tiny() -> Options {
    Options {
        write_buffer_size: 4 * 1024,
        ..Options::default()
    }
}

fn one_instance(seed_keys: usize, checkpoints: usize) -> Option<String> {
    let src = TempDir::new().expect("tempdir");
    let tgt = TempDir::new().expect("tempdir");
    let db = Arc::new(Db::open(src.path(), tiny()).expect("open"));

    for i in 0..seed_keys {
        let k = format!("seed_{i:03}");
        db.put(k.as_bytes(), k.as_bytes()).expect("seed put");
    }

    let stop = Arc::new(AtomicBool::new(false));
    let wdb = Arc::clone(&db);
    let wstop = Arc::clone(&stop);
    let writer = thread::spawn(move || {
        let mut i = 0u64;
        while !wstop.load(Ordering::Relaxed) {
            let mut b = WriteBatch::new();
            let k = format!("live_{i:06}");
            b.put(k.as_bytes(), k.as_bytes());
            wdb.write(b).expect("live write");
            i += 1;
        }
    });

    let mut missing = None;
    for round in 0..checkpoints {
        let _ = std::fs::remove_dir_all(tgt.path());
        std::fs::create_dir_all(tgt.path()).expect("mkdir");
        db.checkpoint(tgt.path()).expect("checkpoint");
        let reopened = Db::open(tgt.path(), Options::default()).expect("reopen checkpoint");
        for i in 0..seed_keys {
            let k = format!("seed_{i:03}");
            if reopened.get(k.as_bytes()).expect("get") != Some(k.clone().into_bytes()) {
                missing = Some(format!(
                    "checkpoint round {round}: {k} was acknowledged before the writer thread \
                     started and is missing from the checkpoint"
                ));
                break;
            }
        }
        drop(reopened);
        if missing.is_some() {
            break;
        }
    }

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("writer");
    missing
}

#[test]
fn a_checkpoint_taken_under_a_concurrent_writer_keeps_every_earlier_write() {
    let instances: usize = std::env::var("CP_INSTANCES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let rounds: usize = std::env::var("CP_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut bad = Vec::new();
    let outs: Vec<Option<String>> = thread::scope(|s| {
        let hs: Vec<_> = (0..instances)
            .map(|_| s.spawn(move || one_instance(50, rounds)))
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("instance"))
            .collect()
    });
    for o in outs.into_iter().flatten() {
        bad.push(o);
    }
    println!(
        "checkpoint probe: {instances} instances x {rounds} checkpoints, {} violation(s)",
        bad.len()
    );
    assert!(bad.is_empty(), "{}", bad.join("\n  "));
}
