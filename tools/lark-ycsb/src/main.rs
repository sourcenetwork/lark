//! `lark-ycsb` — Rust-native YCSB driver for lark.
//!
//! Implements the Yahoo Cloud Serving Benchmark workloads A–F
//! with a Zipfian key distribution, so lark can be benchmarked
//! against the industry-standard methodology.
//!
//! ```sh
//! # Load 100K records, then run workload A (50/50 read/update)
//! cargo run --release -p lark-ycsb -- \
//!     --workload=a --num=100000 --ops=100000
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use lark_kv::{Db, Options};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// YCSB benchmark driver for lark-kv.
#[derive(Parser)]
#[command(name = "lark-ycsb")]
struct Args {
    /// Workload: a, b, c, d, e, f.
    #[arg(long, default_value = "a")]
    workload: String,

    /// Number of records to load in the load phase.
    #[arg(long, default_value_t = 100_000)]
    num: u64,

    /// Number of operations in the run phase.
    #[arg(long, default_value_t = 100_000)]
    ops: u64,

    /// Size of each field/value in bytes (YCSB default: 1000).
    #[arg(long, default_value_t = 1000)]
    field_size: usize,

    /// Zipfian constant (default 0.99 per YCSB spec).
    #[arg(long, default_value_t = 0.99)]
    zipfian_constant: f64,

    /// Database path. Temp dir if not specified.
    #[arg(long)]
    db: Option<PathBuf>,

    /// RNG seed. 0 = random.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let seed = if args.seed == 0 {
        rand::random()
    } else {
        args.seed
    };

    let _tmpdir;
    let db_path = match &args.db {
        Some(p) => p.clone(),
        None => {
            _tmpdir = tempfile::TempDir::new().expect("create temp dir");
            _tmpdir.path().to_path_buf()
        }
    };

    let opts = Options::default();
    let db = Db::open(&db_path, opts).expect("open database");

    let workload = parse_workload(&args.workload);
    println!(
        "YCSB workload {} : read={:.0}% update={:.0}% insert={:.0}% scan={:.0}% rmw={:.0}%",
        args.workload.to_uppercase(),
        workload.read_proportion * 100.0,
        workload.update_proportion * 100.0,
        workload.insert_proportion * 100.0,
        workload.scan_proportion * 100.0,
        workload.rmw_proportion * 100.0,
    );
    println!(
        "Records: {}  Ops: {}  FieldSize: {}  Seed: {}",
        args.num, args.ops, args.field_size, seed
    );
    println!("{:-<72}", "");

    // Load phase.
    let mut rng = SmallRng::seed_from_u64(seed);
    load_phase(&db, args.num, args.field_size, &mut rng);

    // Run phase.
    let mut zipf = ZipfianGenerator::new(args.num, args.zipfian_constant);
    run_phase(&db, &workload, &args, &mut rng, &mut zipf);
}

// ── workload definitions ──────────────────────────────────────

struct Workload {
    read_proportion: f64,
    update_proportion: f64,
    insert_proportion: f64,
    scan_proportion: f64,
    rmw_proportion: f64,
}

fn parse_workload(name: &str) -> Workload {
    match name.to_lowercase().as_str() {
        "a" => Workload {
            read_proportion: 0.5,
            update_proportion: 0.5,
            insert_proportion: 0.0,
            scan_proportion: 0.0,
            rmw_proportion: 0.0,
        },
        "b" => Workload {
            read_proportion: 0.95,
            update_proportion: 0.05,
            insert_proportion: 0.0,
            scan_proportion: 0.0,
            rmw_proportion: 0.0,
        },
        "c" => Workload {
            read_proportion: 1.0,
            update_proportion: 0.0,
            insert_proportion: 0.0,
            scan_proportion: 0.0,
            rmw_proportion: 0.0,
        },
        "d" => Workload {
            read_proportion: 0.95,
            update_proportion: 0.0,
            insert_proportion: 0.05,
            scan_proportion: 0.0,
            rmw_proportion: 0.0,
        },
        "e" => Workload {
            read_proportion: 0.0,
            update_proportion: 0.0,
            insert_proportion: 0.05,
            scan_proportion: 0.95,
            rmw_proportion: 0.0,
        },
        "f" => Workload {
            read_proportion: 0.5,
            update_proportion: 0.0,
            insert_proportion: 0.0,
            scan_proportion: 0.0,
            rmw_proportion: 0.5,
        },
        other => {
            eprintln!("unknown workload: {other}, defaulting to A");
            parse_workload("a")
        }
    }
}

// ── phases ────────────────────────────────────────────────────

fn load_phase(db: &Db, num: u64, field_size: usize, rng: &mut SmallRng) {
    let start = Instant::now();
    for i in 0..num {
        let key = ycsb_key(i);
        let val = random_value(rng, field_size);
        db.put(&key, &val).unwrap();
    }
    let elapsed = start.elapsed();
    report("load", elapsed, num, num * field_size as u64);
}

fn run_phase(
    db: &Db,
    workload: &Workload,
    args: &Args,
    rng: &mut SmallRng,
    zipf: &mut ZipfianGenerator,
) {
    let mut next_insert_key = args.num;
    let mut read_latencies: Vec<Duration> = Vec::new();
    let mut write_latencies: Vec<Duration> = Vec::new();

    let start = Instant::now();
    for _ in 0..args.ops {
        let r: f64 = rng.random();
        if r < workload.read_proportion {
            // Read
            let idx = zipf.next(rng);
            let key = ycsb_key(idx);
            let t = Instant::now();
            let _ = db.get(&key);
            read_latencies.push(t.elapsed());
        } else if r < workload.read_proportion + workload.update_proportion {
            // Update
            let idx = zipf.next(rng);
            let key = ycsb_key(idx);
            let val = random_value(rng, args.field_size);
            let t = Instant::now();
            let _ = db.put(&key, &val);
            write_latencies.push(t.elapsed());
        } else if r < workload.read_proportion
            + workload.update_proportion
            + workload.insert_proportion
        {
            // Insert
            let key = ycsb_key(next_insert_key);
            next_insert_key += 1;
            let val = random_value(rng, args.field_size);
            let t = Instant::now();
            let _ = db.put(&key, &val);
            write_latencies.push(t.elapsed());
        } else if r < workload.read_proportion
            + workload.update_proportion
            + workload.insert_proportion
            + workload.scan_proportion
        {
            // Scan (short range, 1-100 records per YCSB spec)
            let idx = zipf.next(rng);
            let start_key = ycsb_key(idx);
            let scan_len: u64 = rng.random_range(1..=100);
            let end_key = ycsb_key(idx + scan_len);
            let t = Instant::now();
            let _ = db.scan(Some(&start_key), Some(&end_key));
            read_latencies.push(t.elapsed());
        } else {
            // Read-modify-write
            let idx = zipf.next(rng);
            let key = ycsb_key(idx);
            let t = Instant::now();
            let _ = db.get(&key);
            let val = random_value(rng, args.field_size);
            let _ = db.put(&key, &val);
            write_latencies.push(t.elapsed());
        }
    }
    let elapsed = start.elapsed();

    report("run", elapsed, args.ops, args.ops * args.field_size as u64);

    // Latency summary.
    if !read_latencies.is_empty() {
        read_latencies.sort();
        let avg = avg_duration(&read_latencies);
        let p99 = read_latencies[read_latencies.len() * 99 / 100];
        println!(
            "  read  latency: avg={:.1} us  p99={:.1} us  ({} ops)",
            avg.as_secs_f64() * 1e6,
            p99.as_secs_f64() * 1e6,
            read_latencies.len(),
        );
    }
    if !write_latencies.is_empty() {
        write_latencies.sort();
        let avg = avg_duration(&write_latencies);
        let p99 = write_latencies[write_latencies.len() * 99 / 100];
        println!(
            "  write latency: avg={:.1} us  p99={:.1} us  ({} ops)",
            avg.as_secs_f64() * 1e6,
            p99.as_secs_f64() * 1e6,
            write_latencies.len(),
        );
    }
}

// ── helpers ───────────────────────────────────────────────────

fn ycsb_key(i: u64) -> Vec<u8> {
    format!("user{i:012}").into_bytes()
}

fn random_value(rng: &mut SmallRng, size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    rng.fill(&mut buf[..]);
    buf
}

fn report(phase: &str, elapsed: Duration, ops: u64, bytes: u64) {
    let secs = elapsed.as_secs_f64();
    let ops_sec = if secs > 0.0 { ops as f64 / secs } else { 0.0 };
    let mb_sec = if secs > 0.0 {
        bytes as f64 / secs / (1024.0 * 1024.0)
    } else {
        0.0
    };
    println!(
        "{:<20} : {:>10.0} ops/sec; {:>8.1} MB/s  ({:.2}s)",
        phase, ops_sec, mb_sec, secs,
    );
}

fn avg_duration(sorted: &[Duration]) -> Duration {
    let total: Duration = sorted.iter().sum();
    total / sorted.len() as u32
}

// ── Zipfian generator ─────────────────────────────────────────

/// Simple Zipfian random number generator following the YCSB
/// spec. Generates integers in `[0, item_count)` with a
/// power-law distribution controlled by `theta` (typically 0.99).
struct ZipfianGenerator {
    item_count: u64,
    theta: f64,
    zeta_n: f64,
    alpha: f64,
    eta: f64,
}

impl ZipfianGenerator {
    fn new(item_count: u64, theta: f64) -> Self {
        let zeta_2 = zeta(2, theta);
        let zeta_n = zeta(item_count, theta);
        let alpha = 1.0 / (1.0 - theta);
        let eta = (1.0 - (2.0 / item_count as f64).powf(1.0 - theta)) / (1.0 - zeta_2 / zeta_n);
        Self {
            item_count,
            theta,
            zeta_n,
            alpha,
            eta,
        }
    }

    fn next(&mut self, rng: &mut SmallRng) -> u64 {
        let u: f64 = rng.random();
        let uz = u * self.zeta_n;
        if uz < 1.0 {
            return 0;
        }
        if uz < 1.0 + 0.5_f64.powf(self.theta) {
            return 1;
        }
        let val =
            (self.item_count as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as u64;
        val.min(self.item_count - 1)
    }
}

fn zeta(n: u64, theta: f64) -> f64 {
    let mut sum = 0.0;
    for i in 1..=n.min(10_000) {
        sum += 1.0 / (i as f64).powf(theta);
    }
    // For large n, approximate the tail.
    if n > 10_000 {
        sum += ((n as f64) / 10_000.0).ln() / (1.0 - theta);
    }
    sum
}
