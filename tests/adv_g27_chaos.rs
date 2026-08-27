//! Chaos probe for the published read view (G27), against the
//! background operations the shipped adversarial suite does not run
//! alongside its readers: column-family creation and drop, external SST
//! ingestion, checkpoint capture, and a block cache small enough that
//! every scan has to go back to the file descriptors a compaction has
//! already unlinked.
//!
//! Invariant: every key has exactly one writer and is only ever
//! overwritten, in a column family nothing else touches. A read that
//! answers "absent", or answers with a stamp below one the same reader
//! already saw, is a read-path violation.
//!
//! A watchdog thread fails the test rather than hanging if the workload
//! stops making progress, so a lock-order inversion between the version
//! store and the read view's publish mutex surfaces as a failure
//! instead of a timeout.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use lark_kv::{Db, IngestOptions, Options, SstFileWriter};
use tempfile::TempDir;

const WRITERS: usize = 4;
const KEYS_PER_WRITER: usize = 12;
/// Reader threads per instance. Overridable so a wedge can be attributed
/// to the read path or to the write path.
fn readers() -> usize {
    env("LARK_CHAOS_READERS", 4) as usize
}
/// External tables ingested per instance. Bounded so the key space the
/// readers walk stays flat while the writers keep overwriting.
const INGESTS_PER_INSTANCE: u64 = 32;
/// Checkpoints per instance. Each one rotates the memtable and holds the
/// compaction lock while it copies the whole table set, so an unbounded
/// loop starves the writers instead of racing them.
const CHECKPOINTS_PER_INSTANCE: u64 = 32;
/// Foreground `compact_range` passes per instance, bounded for the same
/// reason.
const COMPACT_PASSES_PER_INSTANCE: u64 = 256;

fn key_of(w: usize, i: usize) -> Vec<u8> {
    format!("w{w:03}k{i:04}").into_bytes()
}

fn value_of(stamp: u64) -> Vec<u8> {
    format!("v{stamp:016}").into_bytes()
}

fn stamp_of(v: &[u8]) -> Option<u64> {
    std::str::from_utf8(v.get(1..)?).ok()?.parse().ok()
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Which read surface a reader thread hammers.
#[derive(Clone, Copy)]
enum Surface {
    Get,
    MultiGet,
    Iter,
}

fn read(db: &Db, surface: Surface, keys: &[Vec<u8>]) -> Vec<Option<u64>> {
    match surface {
        Surface::Get => keys
            .iter()
            .map(|k| db.get(k).expect("get").as_deref().and_then(stamp_of))
            .collect(),
        Surface::MultiGet => {
            let refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
            db.multi_get(&refs)
                .expect("multi_get")
                .iter()
                .map(|v| v.as_deref().and_then(stamp_of))
                .collect()
        }
        Surface::Iter => {
            let mut it = db.iter();
            it.seek_to_first();
            let mut pairs = Vec::new();
            while it.valid() {
                pairs.push((
                    it.key().expect("key").to_vec(),
                    it.value().expect("value").to_vec(),
                ));
                it.next();
            }
            it.status().expect("iter status");
            keys.iter()
                .map(|k| {
                    pairs
                        .iter()
                        .find(|(pk, _)| pk == k)
                        .and_then(|(_, v)| stamp_of(v))
                })
                .collect()
        }
    }
}

fn external_table(dir: &std::path::Path, n: u64) -> std::path::PathBuf {
    let path = dir.join(format!("ext_{n}.sst"));
    let mut w = SstFileWriter::create(&path, &Options::default()).expect("create sst");
    for i in 0..64u64 {
        w.put(
            format!("zzz_{n:04}_{i:04}").as_bytes(),
            format!("ext_{n}").as_bytes(),
        )
        .expect("sst put");
    }
    w.finish().expect("finish");
    path
}

fn run_instance(versions: u64, min_rounds: u64) -> Vec<String> {
    let dir = TempDir::new().expect("tempdir");
    let db = Arc::new(
        Db::open(
            dir.path(),
            Options {
                write_buffer_size: 8 * 1024,
                block_cache_size: 4 * 1024,
                max_write_buffer_number: env("LARK_CHAOS_MAX_MEMTABLES", 2) as usize,
                level0_stop_writes_trigger: env("LARK_CHAOS_L0_STOP", 36) as usize,
                ..Options::default()
            },
        )
        .expect("open"),
    );
    let ext_dir = TempDir::new().expect("tempdir");
    let cp_dir = TempDir::new().expect("tempdir");

    let keys: Vec<Vec<u8>> = (0..WRITERS)
        .flat_map(|w| (0..KEYS_PER_WRITER).map(move |i| key_of(w, i)))
        .collect();
    for k in &keys {
        db.put(k, &value_of(0)).expect("seed");
    }

    let live = Arc::new(AtomicU64::new(WRITERS as u64));
    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));
    let bad: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mask = env("LARK_CHAOS_MASK", 0b1111);
    let enabled = |bit: u64| mask & bit != 0;
    // `1 +` is the stall probe below, which is spawned unconditionally.
    let chaos_threads = 1 + [1u64, 2, 4, 8].iter().filter(|b| enabled(**b)).count();
    let gate_participants = WRITERS + readers() + chaos_threads + 1;
    let gate = Arc::new(Barrier::new(gate_participants));
    let mut handles = Vec::new();

    for w in 0..WRITERS {
        let (db, live, gate) = (Arc::clone(&db), Arc::clone(&live), Arc::clone(&gate));
        handles.push(thread::spawn(move || {
            gate.wait();
            for v in 1..=versions {
                for i in 0..KEYS_PER_WRITER {
                    db.put(&key_of(w, i), &value_of(v)).expect("put");
                }
            }
            live.fetch_sub(1, Ordering::AcqRel);
        }));
    }

    // Chaos 1: user-thread compact_range.
    if enabled(1) {
        let (db, live, gate, stop) = (
            Arc::clone(&db),
            Arc::clone(&live),
            Arc::clone(&gate),
            Arc::clone(&stop),
        );
        handles.push(thread::spawn(move || {
            gate.wait();
            let mut n = 0u64;
            loop {
                db.compact_range(None, None).expect("compact_range");
                n += 1;
                if n >= COMPACT_PASSES_PER_INSTANCE
                    || (n >= 2 && live.load(Ordering::Acquire) == 0 && stop.load(Ordering::Acquire))
                {
                    break;
                }
            }
        }));
    }

    // Chaos 2: column families created and dropped underneath the readers.
    if enabled(2) {
        let (db, live, gate, stop) = (
            Arc::clone(&db),
            Arc::clone(&live),
            Arc::clone(&gate),
            Arc::clone(&stop),
        );
        handles.push(thread::spawn(move || {
            gate.wait();
            let mut n = 0u64;
            loop {
                let name = format!("cf_{n}");
                if let Ok(h) = db.create_column_family(&name) {
                    db.put_cf(&h, b"x", b"y").expect("put_cf");
                    let _ = db.list_column_families();
                    db.drop_column_family(h).expect("drop_cf");
                }
                n += 1;
                if n >= 2 && live.load(Ordering::Acquire) == 0 && stop.load(Ordering::Acquire) {
                    break;
                }
            }
        }));
    }

    // Chaos 3: external SST ingestion. Bounded, because each ingest adds
    // keys the readers then have to walk, and an unbounded ingest loop
    // turns the reader cost quadratic in the run length.
    if enabled(4) {
        let (db, live, gate, stop) = (
            Arc::clone(&db),
            Arc::clone(&live),
            Arc::clone(&gate),
            Arc::clone(&stop),
        );
        let ext = ext_dir.path().to_path_buf();
        handles.push(thread::spawn(move || {
            gate.wait();
            let mut n = 0u64;
            loop {
                let p = external_table(&ext, n);
                db.ingest_external_files(&[p], IngestOptions::default())
                    .expect("ingest");
                n += 1;
                if n >= INGESTS_PER_INSTANCE
                    || (n >= 2 && live.load(Ordering::Acquire) == 0 && stop.load(Ordering::Acquire))
                {
                    break;
                }
            }
        }));
    }

    // Chaos 4: checkpoint capture, which rotates the memtable and holds
    // the compaction lock.
    if enabled(8) {
        let (db, live, gate, stop) = (
            Arc::clone(&db),
            Arc::clone(&live),
            Arc::clone(&gate),
            Arc::clone(&stop),
        );
        let cp = cp_dir.path().to_path_buf();
        handles.push(thread::spawn(move || {
            gate.wait();
            let mut n = 0u64;
            loop {
                let target = cp.join(format!("cp_{n}"));
                db.checkpoint(&target).expect("checkpoint");
                std::fs::remove_dir_all(&target).expect("rm checkpoint");
                n += 1;
                if n >= CHECKPOINTS_PER_INSTANCE
                    || (n >= 2 && live.load(Ordering::Acquire) == 0 && stop.load(Ordering::Acquire))
                {
                    break;
                }
            }
        }));
    }

    for r in 0..readers() {
        let surface = match r % 3 {
            0 => Surface::Get,
            1 => Surface::MultiGet,
            _ => Surface::Iter,
        };
        let (db, live, gate, bad, progress, keys) = (
            Arc::clone(&db),
            Arc::clone(&live),
            Arc::clone(&gate),
            Arc::clone(&bad),
            Arc::clone(&progress),
            keys.clone(),
        );
        handles.push(thread::spawn(move || {
            let mut seen = vec![0u64; keys.len()];
            gate.wait();
            let mut round = 0u64;
            loop {
                for (idx, obs) in read(&db, surface, &keys).iter().enumerate() {
                    match obs {
                        None => bad.lock().expect("lock").push(format!(
                            "reader {r} round {round}: {} read back ABSENT (last seen {})",
                            String::from_utf8_lossy(&keys[idx]),
                            seen[idx],
                        )),
                        Some(s) => {
                            if *s < seen[idx] {
                                bad.lock().expect("lock").push(format!(
                                    "reader {r} round {round}: {} went BACKWARDS {} -> {s}",
                                    String::from_utf8_lossy(&keys[idx]),
                                    seen[idx],
                                ));
                            }
                            seen[idx] = seen[idx].max(*s);
                        }
                    }
                }
                round += 1;
                progress.fetch_add(1, Ordering::Relaxed);
                if round >= min_rounds && live.load(Ordering::Acquire) == 0 {
                    break;
                }
            }
        }));
    }

    // Diagnostic: when the writers stop advancing, report what the
    // engine's stall inputs look like, so a stalled run says which
    // threshold is holding it rather than only that it is slow.
    {
        let (db, stop) = (Arc::clone(&db), Arc::clone(&stop));
        let live = Arc::clone(&live);
        let gate = Arc::clone(&gate);
        handles.push(thread::spawn(move || {
            // Counted by the `1 +` in `chaos_threads`, so it must reach
            // the gate: a participant the barrier is sized for but that
            // never arrives wedges every other thread forever.
            gate.wait();
            let mut ticks = 0u32;
            while !stop.load(Ordering::Relaxed) && live.load(Ordering::Acquire) > 0 {
                thread::sleep(Duration::from_secs(5));
                ticks += 1;
                if ticks.is_multiple_of(2) {
                    // `no_slowdown` turns any active stall condition into
                    // `Error::Busy(reason)`, which names the threshold.
                    let mut probe_opts = lark_kv::WriteOptions::new();
                    probe_opts.no_slowdown = true;
                    let busy = db.put_opt(&probe_opts, b"__stall_probe", b"1");
                    eprintln!("stall reason: {busy:?}");
                    eprintln!(
                        "stall probe: L0={:?} imm_memtables={:?} all_memtable_bytes={:?} live_writers={}",
                        db.get_property("lark.num-files-at-level0"),
                        db.get_property("lark.num-entries-imm-mem-tables"),
                        db.get_property("lark.cur-size-all-mem-tables"),
                        live.load(Ordering::Acquire),
                    );
                }
            }
        }));
    }

    // Watchdog: a lock-order inversion shows up as no progress at all.
    let watchdog_stop = Arc::new(AtomicBool::new(false));
    let wd = {
        let (progress, watchdog_stop) = (Arc::clone(&progress), Arc::clone(&watchdog_stop));
        thread::spawn(move || {
            let mut last = 0u64;
            let mut stalled_since = Instant::now();
            while !watchdog_stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(200));
                let now = progress.load(Ordering::Relaxed);
                if now != last {
                    last = now;
                    stalled_since = Instant::now();
                } else if stalled_since.elapsed() > Duration::from_secs(120) {
                    return Some(format!(
                        "no reader made progress for 120s at {now} rounds; the workload is wedged"
                    ));
                }
            }
            None
        })
    };

    // Every gate participant is also pushed into `handles`, and the
    // watchdog is not, so this is the whole invariant. A barrier sized
    // for a thread that never arrives is a permanent, silent wedge -
    // exactly what happened when the stall probe was added and counted
    // here without being given a `gate.wait()`. Fail loudly here
    // instead, before anything blocks.
    assert_eq!(
        handles.len() + 1,
        gate_participants,
        "barrier is sized for {gate_participants} threads but {} will reach it \
         (+1 coordinator); every thread counted by the gate must call gate.wait()",
        handles.len(),
    );

    gate.wait();
    for h in handles.drain(..) {
        if !h.is_finished() {
            stop.store(true, Ordering::Release);
        }
        h.join().expect("worker panicked");
    }
    stop.store(true, Ordering::Release);
    watchdog_stop.store(true, Ordering::Relaxed);

    let mut out = std::mem::take(&mut *bad.lock().expect("lock"));
    if let Some(w) = wd.join().expect("watchdog") {
        out.push(w);
    }
    out
}

#[test]
fn the_read_view_survives_compaction_cf_churn_ingest_and_checkpoint() {
    // Defaults sized for the ordinary gate, measured at 5s. The full
    // workload (6 / 2 / 400 / 40) is `just chaos`: over 20 minutes wall
    // and 4h of CPU unoptimized, which is not what `cargo test` should
    // run. Every value is overridable, so a wedge can be reproduced at
    // whatever size exposed it.
    let instances = env("LARK_CHAOS_INSTANCES", 2) as usize;
    let rounds = env("LARK_CHAOS_ROUNDS", 1);
    let versions = env("LARK_CHAOS_VERSIONS", 50);
    let min_rounds = env("LARK_CHAOS_MIN_ROUNDS", 10);

    let mut bad = Vec::new();
    for _ in 0..rounds {
        let outs: Vec<Vec<String>> = thread::scope(|s| {
            let hs: Vec<_> = (0..instances)
                .map(|_| s.spawn(move || run_instance(versions, min_rounds)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("instance"))
                .collect()
        });
        for o in outs {
            bad.extend(o);
        }
    }
    println!(
        "chaos: {} instances x {rounds} rounds, {} violation(s)",
        instances,
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{}",
        bad.iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
