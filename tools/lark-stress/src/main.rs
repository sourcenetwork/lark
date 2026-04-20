//! `lark-stress` — random-op correctness harness that cross-checks
//! lark against an in-memory `BTreeMap` reference.
//!
//! ```sh
//! cargo run --release -p lark-stress -- --num-ops=100000
//! ```
//!
//! Every read (get, scan) is verified against the reference. Any
//! mismatch prints the seed and the operation index so the
//! failure is reproducible with `--seed=<N>`.
//!
//! # Shadow log (db_stress-style crash verification)
//!
//! When `--shadow-log PATH` is supplied every committed write is
//! also appended to a line-delimited log file in the form
//! `PUT key_hex value_hex\n` or `DEL key_hex\n`. A sibling
//! `--verify-only PATH` mode replays the log against the DB and
//! asserts that every committed write is visible with the expected
//! value. The two combine to emulate RocksDB's `db_stress` pattern:
//! run the stressor under a SIGKILL harness, then run this tool
//! with `--verify-only` against the same DB + shadow log to prove
//! that no committed write was lost during the crash.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use lark_kv::{Db, Options};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Random-op correctness harness for lark-kv.
#[derive(Parser)]
#[command(name = "lark-stress")]
struct Args {
    /// Total number of operations to execute.
    #[arg(long, default_value_t = 100_000)]
    num_ops: u64,

    /// Number of distinct keys in the key space. Smaller values
    /// produce more overwrites and exercises dedup/compaction.
    #[arg(long, default_value_t = 1000)]
    key_range: u64,

    /// Size of each value in bytes.
    #[arg(long, default_value_t = 100)]
    value_size: usize,

    /// RNG seed. 0 means random.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Database path. Temp dir if not specified.
    #[arg(long)]
    db: Option<PathBuf>,

    /// Trigger a compact_range every N write operations.
    #[arg(long, default_value_t = 5000)]
    compact_every: u64,

    /// Reopen the database every N operations to exercise
    /// WAL replay / recovery.
    #[arg(long, default_value_t = 0)]
    reopen_every: u64,

    /// Append every committed write to the given line-delimited
    /// file. Pair with `--verify-only` to check DB consistency
    /// after a crash.
    #[arg(long)]
    shadow_log: Option<PathBuf>,

    /// Don't generate random ops. Instead, open the DB at `--db`,
    /// replay the shadow log at the given path, and assert that
    /// every PUT is visible with the logged value and every DELETE
    /// left the key absent.
    #[arg(long)]
    verify_only: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    // verify_only short-circuits the random-op loop entirely.
    if let Some(ref log_path) = args.verify_only {
        let db_path = args.db.clone().expect("--verify-only requires --db");
        std::process::exit(verify_from_shadow_log(&db_path, log_path));
    }

    let seed = if args.seed == 0 {
        rand::random()
    } else {
        args.seed
    };

    println!(
        "lark-stress  ops={}  key_range={}  value_size={}  seed={}",
        args.num_ops, args.key_range, args.value_size, seed
    );

    let _tmpdir;
    let db_path = match &args.db {
        Some(p) => p.clone(),
        None => {
            _tmpdir = tempfile::TempDir::new().expect("create temp dir");
            _tmpdir.path().to_path_buf()
        }
    };

    let opts = Options {
        write_buffer_size: 64 * 1024,
        ..Options::default()
    };

    let mut db = Db::open(&db_path, opts.clone()).expect("open database");
    let mut reference: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut writes_since_compact: u64 = 0;
    let mut mismatches: u64 = 0;

    // Shadow-log writer: buffered append-mode file opened eagerly
    // so every committed op reaches disk even if we SIGKILL.
    let mut shadow: Option<std::io::BufWriter<std::fs::File>> = args.shadow_log.as_ref().map(|p| {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .expect("open shadow log");
        std::io::BufWriter::new(f)
    });

    for op_idx in 0..args.num_ops {
        // Reopen path: close and reopen the db to exercise
        // recovery. The reference stays in memory — every
        // committed write should survive the reopen.
        if args.reopen_every > 0 && op_idx > 0 && op_idx % args.reopen_every == 0 {
            drop(db);
            db = Db::open(&db_path, opts.clone()).expect("reopen database");
            // After reopen, verify the entire reference is still
            // readable.
            for (k, v) in &reference {
                match db.get(k) {
                    Ok(Some(got)) => {
                        if got != *v {
                            eprintln!(
                                "MISMATCH after reopen at op {op_idx}: key {:?} expected {:?}, got {:?}  [seed={seed}]",
                                k, v, got
                            );
                            mismatches += 1;
                        }
                    }
                    Ok(None) => {
                        eprintln!(
                            "MISSING after reopen at op {op_idx}: key {:?} expected {:?}  [seed={seed}]",
                            k, v
                        );
                        mismatches += 1;
                    }
                    Err(e) => {
                        eprintln!(
                            "ERROR after reopen at op {op_idx}: key {:?}: {e}  [seed={seed}]",
                            k
                        );
                        mismatches += 1;
                    }
                }
            }
        }

        let op: u8 = rng.random_range(0..10);
        match op {
            // put (50% of ops)
            0..=4 => {
                let key = random_key(&mut rng, args.key_range);
                let val = random_value(&mut rng, args.value_size);
                db.put(&key, &val).unwrap();
                if let Some(w) = shadow.as_mut() {
                    writeln!(w, "PUT {} {}", hex(&key), hex(&val)).unwrap();
                    // Flush every write so a SIGKILL can't lose the
                    // tail — the shadow log must be a strict
                    // subset of what's durable in lark.
                    w.flush().unwrap();
                }
                reference.insert(key, val);
                writes_since_compact += 1;
            }
            // delete (15%)
            5 => {
                let key = random_key(&mut rng, args.key_range);
                let _ = db.delete(&key);
                if let Some(w) = shadow.as_mut() {
                    writeln!(w, "DEL {}", hex(&key)).unwrap();
                    w.flush().unwrap();
                }
                reference.remove(&key);
                writes_since_compact += 1;
            }
            // get + cross-check (20%)
            6 | 7 => {
                let key = random_key(&mut rng, args.key_range);
                let got = db.get(&key).unwrap();
                let expected = reference.get(&key).cloned();
                if got != expected {
                    eprintln!(
                        "MISMATCH at op {op_idx}: get({:?}) expected {:?}, got {:?}  [seed={seed}]",
                        key,
                        expected.as_ref().map(|v| v.len()),
                        got.as_ref().map(|v| v.len()),
                    );
                    mismatches += 1;
                }
            }
            // scan + cross-check (10%)
            8 => {
                let a = random_key(&mut rng, args.key_range);
                let b = random_key(&mut rng, args.key_range);
                let (start, end) = if a <= b { (a, b) } else { (b, a) };
                let got = db.scan(Some(&start), Some(&end)).unwrap();
                let expected: Vec<(Vec<u8>, Vec<u8>)> = reference
                    .range(start.clone()..end.clone())
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if got != expected {
                    eprintln!(
                        "MISMATCH at op {op_idx}: scan({:?}..{:?}) expected {} entries, got {}  [seed={seed}]",
                        start, end, expected.len(), got.len(),
                    );
                    mismatches += 1;
                }
            }
            // snapshot get (5%)
            _ => {
                let snap = db.snapshot();
                let key = random_key(&mut rng, args.key_range);
                let got = snap.get(&key).unwrap();
                let expected = reference.get(&key).cloned();
                if got != expected {
                    eprintln!(
                        "MISMATCH at op {op_idx}: snap.get({:?}) expected {:?}, got {:?}  [seed={seed}]",
                        key,
                        expected.as_ref().map(|v| v.len()),
                        got.as_ref().map(|v| v.len()),
                    );
                    mismatches += 1;
                }
            }
        }

        // Periodic compaction to exercise the background merge
        // path against the reference.
        if args.compact_every > 0 && writes_since_compact >= args.compact_every {
            db.compact_range(None, None).unwrap();
            writes_since_compact = 0;
        }
    }

    if mismatches > 0 {
        eprintln!("\nFAILED: {mismatches} mismatch(es)  [seed={seed}]");
        std::process::exit(1);
    }

    println!("OK  {} ops, 0 mismatches", args.num_ops);
}

fn random_key(rng: &mut SmallRng, key_range: u64) -> Vec<u8> {
    let i: u64 = rng.random_range(0..key_range);
    format!("key{i:012}").into_bytes()
}

fn random_value(rng: &mut SmallRng, size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    rng.fill(&mut buf[..]);
    buf
}

// ── shadow-log helpers ─────────────────────────────────────────

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn unhex(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16).unwrap_or(0);
        let lo = (bytes[i + 1] as char).to_digit(16).unwrap_or(0);
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    out
}

/// Replay the shadow log at `log_path` against the DB at
/// `db_path`. Returns an exit code: 0 on success, 1 on mismatch.
fn verify_from_shadow_log(db_path: &std::path::Path, log_path: &std::path::Path) -> i32 {
    let db = Db::open(db_path, Options::default()).expect("open database");
    // Replay the log into an authoritative view: last write wins
    // per key. The DB must match this view when we're done.
    let mut expected: BTreeMap<Vec<u8>, Option<Vec<u8>>> = BTreeMap::new();

    let f = OpenOptions::new()
        .read(true)
        .open(log_path)
        .expect("open shadow log");
    let mut lines = 0u64;
    for line in BufReader::new(f).lines() {
        let line = line.expect("read log line");
        lines += 1;
        let mut it = line.split_whitespace();
        match it.next() {
            Some("PUT") => {
                let k = unhex(it.next().unwrap_or(""));
                let v = unhex(it.next().unwrap_or(""));
                expected.insert(k, Some(v));
            }
            Some("DEL") => {
                let k = unhex(it.next().unwrap_or(""));
                expected.insert(k, None);
            }
            Some(tag) => eprintln!("unknown log tag {tag} at line {lines}"),
            None => {}
        }
    }

    let mut mismatches = 0u64;
    for (k, expected_v) in &expected {
        let got = db.get(k).unwrap();
        match (got.as_ref(), expected_v) {
            (Some(a), Some(b)) if a == b => {}
            (None, None) => {}
            (a, b) => {
                eprintln!(
                    "MISMATCH for key {:?}: expected {:?}, got {:?}",
                    k,
                    b.as_ref().map(|v| v.len()),
                    a.map(|v| v.len()),
                );
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        eprintln!("\nFAILED: {mismatches} mismatch(es) over {lines} log line(s)");
        1
    } else {
        println!("OK  {lines} log line(s), 0 mismatches");
        0
    }
}
