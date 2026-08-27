//! Measures what the checksummed footer costs at open: the footer probe is
//! now two reads instead of one, and every metadata region is hashed
//! once. Run in both trees and compare.

use std::time::Instant;

use lark_kv::{Db, Options};
use tempfile::TempDir;

fn bench_opts() -> Options {
    Options {
        write_buffer_size: 16 * 1024,
        l0_compaction_trigger: 1_000_000,
        level0_slowdown_writes_trigger: 1_000_000,
        level0_stop_writes_trigger: 1_000_000,
        ..Options::default()
    }
}

#[test]
#[ignore = "measurement, not a gate"]
fn open_cost_with_many_sstables() {
    let keys: usize = std::env::var("BENCH_KEYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000);
    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);

    let dir = TempDir::new().unwrap();
    {
        let db = Db::open(dir.path(), bench_opts()).unwrap();
        for i in 0..keys {
            db.put(format!("k_{i:08}").as_bytes(), &[b'v'; 64]).unwrap();
        }
        db.close().unwrap();
        drop(db);
    }
    let ssts = std::fs::read_dir(dir.path().join("sst")).unwrap().count();
    let bytes: u64 = std::fs::read_dir(dir.path().join("sst"))
        .unwrap()
        .flatten()
        .map(|e| e.metadata().unwrap().len())
        .sum();

    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let db = Db::open(dir.path(), bench_opts()).unwrap();
        times.push(t.elapsed());
        drop(db);
    }
    times.sort();
    println!(
        "open with {ssts} sstables ({bytes} bytes on disk): median {:?}, min {:?}, max {:?}",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1],
    );
}
