//! A memtable must never be retired before the SSTable holding its
//! contents is in the published version.
//!
//! Two paths reach a flush without excluding each other: a writer's
//! rotation, which runs under the commit pipeline's mutex, and
//! `drain_memtables`, which a checkpoint drives and which releases that
//! mutex before it flushes so a whole SSTable write does not block
//! writers. If both are inside the flush at once they take the same
//! victim, `frozen[0]`, and both retire "index 0" on the way out. The
//! second retirement drops a memtable nothing has flushed, so every
//! write it held disappears: a key that was only ever overwritten reads
//! as an older version, or as absent.
//!
//! This is the end-to-end half. It runs the two paths against each
//! other for real: many rounds of checkpoint-driven drain against
//! writers rotating a 16 KiB memtable constantly. Every key is written
//! once with a value naming its own index and never deleted, so any
//! disagreement at the end is a lost write and nothing else, both
//! before and after a reopen.
//!
//! It is a soak, not the regression gate. The interleaving is narrow
//! enough that this workload does not reliably reproduce it, which is
//! why the gate is the deterministic pair in
//! `src/engine/read_view.rs`: they construct the frozen list a racing
//! retirement leaves behind and check the retirement takes the right
//! memtable out of it.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use regolith::{Db, DurabilityMode, Options};
use tempfile::TempDir;

const WRITERS: usize = 4;
const KEYS_PER_WRITER: u64 = 900;

fn key_of(writer: usize, i: u64) -> Vec<u8> {
    format!("w{writer}k{i:06}").into_bytes()
}

fn value_of(writer: usize, i: u64) -> Vec<u8> {
    format!("v{writer}-{i}").into_bytes()
}

fn opts() -> Options {
    Options {
        // Small enough that the writers rotate constantly, which is what
        // puts a flush in flight for the drain to collide with.
        write_buffer_size: 16 * 1024,
        max_write_buffer_number: 4,
        durability: DurabilityMode::Eventual,
        ..Options::default()
    }
}

#[test]
fn a_checkpoint_drain_racing_a_rotation_never_loses_an_acknowledged_write() {
    let dir = TempDir::new().expect("tempdir");
    // Checkpoints land outside the database directory: a target inside
    // it would be walked by the engine's own file scans.
    let cp_root = TempDir::new().expect("tempdir");
    let db = Arc::new(Db::open(dir.path(), opts()).expect("open"));

    let stop = Arc::new(AtomicBool::new(false));
    let drains = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();

    for w in 0..WRITERS {
        let db = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..KEYS_PER_WRITER {
                db.put(&key_of(w, i), &value_of(w, i)).expect("put");
            }
        }));
    }

    // The drain side. `checkpoint` calls `drain_memtables(Always)`,
    // which is the path that flushes without the pipeline mutex.
    {
        let (db, stop, drains) = (Arc::clone(&db), Arc::clone(&stop), Arc::clone(&drains));
        let target = cp_root.path().to_path_buf();
        handles.push(thread::spawn(move || {
            let mut round = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let into = target.join(format!("cp{round}"));
                // A checkpoint can legitimately refuse while the
                // engine is busy; the race is in the drain it ran
                // before refusing, so a refusal is not a failure here.
                if db.checkpoint(&into).is_ok() {
                    drains.fetch_add(1, Ordering::Relaxed);
                    let _ = std::fs::remove_dir_all(&into);
                }
                round += 1;
            }
        }));
    }

    for h in handles.drain(..WRITERS) {
        h.join().expect("writer");
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("drain");
    }

    println!("drains completed: {}", drains.load(Ordering::Relaxed));
    assert!(
        drains.load(Ordering::Relaxed) > 0,
        "no checkpoint completed, so the drain path never ran and this proves nothing",
    );

    let mut lost = Vec::new();
    for w in 0..WRITERS {
        for i in 0..KEYS_PER_WRITER {
            let k = key_of(w, i);
            match db.get(&k).expect("get") {
                Some(v) if v == value_of(w, i) => {}
                other => lost.push(format!(
                    "{} -> {:?}, expected {:?}",
                    String::from_utf8_lossy(&k),
                    other.as_deref().map(String::from_utf8_lossy),
                    String::from_utf8_lossy(&value_of(w, i)),
                )),
            }
        }
    }
    assert!(
        lost.is_empty(),
        "{} acknowledged writes did not survive a concurrent flush and drain; first few:\n  {}",
        lost.len(),
        lost.iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );

    // The same must hold across a reopen: a memtable retired without its
    // SSTable also loses the WAL that backed it.
    db.close().expect("close");
    drop(db);
    let reopened = Db::open(dir.path(), opts()).expect("reopen");
    for w in 0..WRITERS {
        for i in 0..KEYS_PER_WRITER {
            assert_eq!(
                reopened.get(&key_of(w, i)).expect("get").as_deref(),
                Some(&value_of(w, i)[..]),
                "w{w}k{i:06} did not survive the reopen",
            );
        }
    }
}
