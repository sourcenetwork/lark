//! probe

use std::fs;
use std::path::Path;
use std::time::Instant;

use lark_kv::{Db, Options};
use tempfile::TempDir;

mod common;

fn list(dir: &Path, prefix: &str) {
    for e in fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            list(&p, &format!("{prefix}{}/", e.file_name().to_string_lossy()));
        } else {
            println!(
                "{prefix}{} = {} bytes",
                e.file_name().to_string_lossy(),
                fs::metadata(&p).unwrap().len()
            );
        }
    }
}

#[test]
fn probe_wal() {
    let dir = TempDir::new().unwrap();
    let db = Db::open(
        dir.path(),
        Options {
            write_buffer_size: 64 << 20,
            ..Options::default()
        },
    )
    .unwrap();
    for i in 0..8 {
        db.put(format!("k{i:02}").as_bytes(), format!("v{i:02}").as_bytes())
            .unwrap();
    }
    db.close().unwrap();
    drop(db);
    println!("--- wal fixture ---");
    list(dir.path(), "");

    let t = Instant::now();
    for _ in 0..50 {
        let db = Db::open(
            dir.path(),
            Options {
                write_buffer_size: 64 << 20,
                ..Options::default()
            },
        )
        .unwrap();
        db.close().unwrap();
        drop(db);
    }
    println!("open+close: {:?} each", t.elapsed() / 50);
}

#[test]
fn probe_sst() {
    let dir = TempDir::new().unwrap();
    let db = Db::open(
        dir.path(),
        Options {
            write_buffer_size: 4096,
            ..Options::default()
        },
    )
    .unwrap();
    for i in 0..24 {
        db.put(format!("k{i:02}").as_bytes(), format!("v{i:02}").as_bytes())
            .unwrap();
    }
    db.compact_range(None, None).unwrap();
    db.close().unwrap();
    drop(db);
    println!("--- sst fixture ---");
    list(dir.path(), "");
    for p in fs::read_dir(dir.path().join("sst")).unwrap().flatten() {
        let b = fs::read(p.path()).unwrap();
        let n = b.len();
        let f = &b[n - 64..];
        let g = |i: usize| u64::from_le_bytes(f[i * 8..i * 8 + 8].try_into().unwrap());
        println!(
            "{:?} len={n} rt_off={} rt_size={} bloom_off={} bloom_size={} idx_off={} idx_size={} num={} magic={:#x}",
            p.file_name(), g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7)
        );
    }
}
