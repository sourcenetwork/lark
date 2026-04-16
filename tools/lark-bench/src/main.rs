//! `lark-bench` — a simple benchmark driver for lark, modeled
//! after the workloads in db_bench.
//!
//! ```sh
//! cargo run --release -p lark-bench -- \
//!     --benchmarks=fillseq,readrandom \
//!     --num=1_000_000 \
//!     --value-size=100
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use lark_kv::{CompressionType, Db, Options};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Simple benchmark driver for lark-kv.
#[derive(Parser)]
#[command(name = "lark-bench")]
struct Args {
    /// Comma-separated list of benchmarks to run.
    #[arg(
        long,
        default_value = "fillseq,fillrandom,readseq,readrandom,readmissing"
    )]
    benchmarks: String,

    /// Number of key-value pairs per benchmark.
    #[arg(long, default_value_t = 1_000_000)]
    num: u64,

    /// Size of each value in bytes.
    #[arg(long, default_value_t = 100)]
    value_size: usize,

    /// Size of each key in bytes.
    #[arg(long, default_value_t = 16)]
    key_size: usize,

    /// Path to the benchmark database directory. A temp directory
    /// is used if not specified.
    #[arg(long)]
    db: Option<PathBuf>,

    /// Compression codec.
    #[arg(long, default_value = "lz4")]
    compression: String,

    /// Write buffer (memtable) size in bytes.
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    write_buffer_size: usize,

    /// Block size in bytes.
    #[arg(long, default_value_t = 4096)]
    block_size: usize,

    /// Bloom filter bits per key.
    #[arg(long, default_value_t = 10)]
    bloom_bits: usize,

    /// RNG seed for reproducibility. 0 means random.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let compression = match args.compression.as_str() {
        "none" => CompressionType::None,
        "snappy" => CompressionType::Snappy,
        _ => CompressionType::Lz4,
    };

    let opts = Options {
        write_buffer_size: args.write_buffer_size,
        block_size: args.block_size,
        bloom_bits_per_key: args.bloom_bits,
        compression,
        ..Options::default()
    };

    let seed = if args.seed == 0 {
        rand::random()
    } else {
        args.seed
    };

    let benchmarks: Vec<&str> = args.benchmarks.split(',').map(str::trim).collect();

    println!(
        "Keys:   {} bytes    Values: {} bytes    Entries: {}",
        args.key_size, args.value_size, args.num
    );
    println!(
        "Compression: {:?}    WriteBuffer: {} MB    BlockSize: {} B    Seed: {}",
        compression,
        args.write_buffer_size / (1024 * 1024),
        args.block_size,
        seed,
    );
    println!("{:-<72}", "");

    for bench_name in &benchmarks {
        // Each benchmark gets a fresh database.
        let _tmpdir;
        let db_path = match &args.db {
            Some(p) => p.clone(),
            None => {
                _tmpdir = tempfile::TempDir::new().expect("create temp dir");
                _tmpdir.path().to_path_buf()
            }
        };

        let db = Db::open(&db_path, opts.clone()).expect("open database");

        match *bench_name {
            "fillseq" => run_fillseq(&db, &args, seed),
            "fillrandom" => run_fillrandom(&db, &args, seed),
            "readseq" => run_readseq(&db, &args, seed),
            "readrandom" => run_readrandom(&db, &args, seed),
            "readmissing" => run_readmissing(&db, &args, seed),
            "overwrite" => run_overwrite(&db, &args, seed),
            "deleterandom" => run_deleterandom(&db, &args, seed),
            "seekrandom" => run_seekrandom(&db, &args, seed),
            "compact" => run_compact(&db, &args, seed),
            other => {
                eprintln!("unknown benchmark: {other}");
            }
        }
    }
}

// ── reporting ──────────────────────────────────────────────────

fn report(name: &str, elapsed: Duration, ops: u64, bytes: u64) {
    let secs = elapsed.as_secs_f64();
    let micros_per_op = if ops > 0 {
        (secs * 1_000_000.0) / ops as f64
    } else {
        0.0
    };
    let ops_per_sec = if secs > 0.0 { ops as f64 / secs } else { 0.0 };
    let mb_per_sec = if secs > 0.0 {
        bytes as f64 / secs / (1024.0 * 1024.0)
    } else {
        0.0
    };
    println!(
        "{:<20} : {:>10.3} micros/op {:>10.0} ops/sec; {:>8.1} MB/s",
        name, micros_per_op, ops_per_sec, mb_per_sec,
    );
}

// ── key generation ─────────────────────────────────────────────

fn sequential_key(i: u64, key_size: usize) -> Vec<u8> {
    let s = format!("{:0>width$}", i, width = key_size);
    s.into_bytes()[..key_size].to_vec()
}

fn random_key(rng: &mut SmallRng, key_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; key_size];
    rng.fill(&mut buf[..]);
    buf
}

fn random_value(rng: &mut SmallRng, value_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; value_size];
    rng.fill(&mut buf[..]);
    buf
}

// ── workloads ──────────────────────────────────────────────────

fn run_fillseq(db: &Db, args: &Args, _seed: u64) {
    let mut rng = SmallRng::seed_from_u64(0);
    let val = random_value(&mut rng, args.value_size);
    let start = Instant::now();
    for i in 0..args.num {
        let key = sequential_key(i, args.key_size);
        db.put(&key, &val).unwrap();
    }
    let elapsed = start.elapsed();
    let bytes = args.num * (args.key_size + args.value_size) as u64;
    report("fillseq", elapsed, args.num, bytes);
}

fn run_fillrandom(db: &Db, args: &Args, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let start = Instant::now();
    for _ in 0..args.num {
        let key = random_key(&mut rng, args.key_size);
        let val = random_value(&mut rng, args.value_size);
        db.put(&key, &val).unwrap();
    }
    let elapsed = start.elapsed();
    let bytes = args.num * (args.key_size + args.value_size) as u64;
    report("fillrandom", elapsed, args.num, bytes);
}

fn run_overwrite(db: &Db, args: &Args, seed: u64) {
    // Pre-fill, then overwrite the same keys.
    let mut rng = SmallRng::seed_from_u64(seed);
    let keys: Vec<Vec<u8>> = (0..args.num)
        .map(|_| random_key(&mut rng, args.key_size))
        .collect();
    for k in &keys {
        db.put(k, &vec![0u8; args.value_size]).unwrap();
    }
    let mut rng2 = SmallRng::seed_from_u64(seed + 1);
    let start = Instant::now();
    for k in &keys {
        let val = random_value(&mut rng2, args.value_size);
        db.put(k, &val).unwrap();
    }
    let elapsed = start.elapsed();
    let bytes = args.num * (args.key_size + args.value_size) as u64;
    report("overwrite", elapsed, args.num, bytes);
}

fn run_readseq(db: &Db, args: &Args, seed: u64) {
    // Pre-fill with sequential keys.
    prefill_sequential(db, args);

    let start = Instant::now();
    let mut it = db.iter();
    it.seek_to_first();
    let mut count = 0u64;
    while it.valid() && count < args.num {
        count += 1;
        it.next();
    }
    let elapsed = start.elapsed();
    let bytes = count * (args.key_size + args.value_size) as u64;
    report("readseq", elapsed, count, bytes);
    let _ = seed;
}

fn run_readrandom(db: &Db, args: &Args, seed: u64) {
    // Pre-fill with sequential keys, then read random ones.
    prefill_sequential(db, args);

    let mut rng = SmallRng::seed_from_u64(seed);
    let start = Instant::now();
    let mut found = 0u64;
    for _ in 0..args.num {
        let i = rng.random_range(0..args.num);
        let key = sequential_key(i, args.key_size);
        if db.get(&key).unwrap().is_some() {
            found += 1;
        }
    }
    let elapsed = start.elapsed();
    let bytes = found * (args.key_size + args.value_size) as u64;
    report("readrandom", elapsed, args.num, bytes);
}

fn run_readmissing(db: &Db, args: &Args, seed: u64) {
    prefill_sequential(db, args);

    let mut rng = SmallRng::seed_from_u64(seed);
    let start = Instant::now();
    for _ in 0..args.num {
        // Keys that don't exist — random bytes unlikely to match
        // the zero-padded sequential format.
        let key = random_key(&mut rng, args.key_size);
        let _ = db.get(&key);
    }
    let elapsed = start.elapsed();
    report("readmissing", elapsed, args.num, 0);
}

fn run_deleterandom(db: &Db, args: &Args, seed: u64) {
    prefill_sequential(db, args);

    let mut rng = SmallRng::seed_from_u64(seed);
    let start = Instant::now();
    for _ in 0..args.num {
        let i = rng.random_range(0..args.num);
        let key = sequential_key(i, args.key_size);
        let _ = db.delete(&key);
    }
    let elapsed = start.elapsed();
    report("deleterandom", elapsed, args.num, 0);
}

fn run_seekrandom(db: &Db, args: &Args, seed: u64) {
    prefill_sequential(db, args);

    let mut rng = SmallRng::seed_from_u64(seed);
    let start = Instant::now();
    for _ in 0..args.num {
        let i = rng.random_range(0..args.num);
        let key = sequential_key(i, args.key_size);
        let mut it = db.iter();
        it.seek(&key);
    }
    let elapsed = start.elapsed();
    report("seekrandom", elapsed, args.num, 0);
}

fn run_compact(db: &Db, args: &Args, _seed: u64) {
    prefill_sequential(db, args);

    let start = Instant::now();
    db.compact_range(None, None).unwrap();
    let elapsed = start.elapsed();
    report("compact", elapsed, 1, 0);
}

fn prefill_sequential(db: &Db, args: &Args) {
    let mut rng = SmallRng::seed_from_u64(0);
    let val = random_value(&mut rng, args.value_size);
    for i in 0..args.num {
        let key = sequential_key(i, args.key_size);
        db.put(&key, &val).unwrap();
    }
}
