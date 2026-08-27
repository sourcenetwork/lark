//! Independent adversarial review of the G24 SSTable metadata checksums.
//!
//! Two questions, and the second matters as much as the first.
//!
//! 1. Is every bit flip in the footer, the index block and the bloom
//!    region either caught or harmless? The oracle is not "an error was
//!    returned": it is that `get` and a full scan agree with each other
//!    and with the truth. A flip that makes the two surfaces disagree is
//!    silently wrong data, which is the defect G24 names.
//! 2. Do tables written by the *previous* format still read? A
//!    compatibility break is itself a data-loss bug. The fixtures for
//!    that question are produced by building the pre-change tree and
//!    running it, so the bytes under test were never written by the code
//!    under test. Their path is passed in `LARK_LEGACY_FIXTURES`; the
//!    test skips loudly rather than passing vacuously when it is unset.
//!
//! The footer layout is parsed here from the on-disk format rather than
//! imported, so the sweep cannot inherit a wrong region boundary from the
//! code it is checking:
//!
//! ```text
//! [rt_off u64][rt_size u64][bloom_off u64][bloom_size u64]
//! [index_off u64][index_size u64][num_entries u64]
//! (V3/V4 only: [footer checksum u64])
//! [magic u64]
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use lark_kv::{Db, DurabilityMode, IngestOptions, Options};
use tempfile::TempDir;

const KEYS: u32 = 600;

fn opts() -> Options {
    Options {
        write_buffer_size: 64 * 1024 * 1024,
        durability: DurabilityMode::Eventual,
        ..Options::default()
    }
}

fn key(i: u32) -> Vec<u8> {
    format!("key_{i:06}").into_bytes()
}

fn val(i: u32) -> Vec<u8> {
    format!("value_{i:06}").into_bytes()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create_dir_all");
    for entry in fs::read_dir(from).expect("read_dir").flatten() {
        let name = entry.file_name();
        if name == "LOCK" {
            continue;
        }
        let src = entry.path();
        let dst = to.join(&name);
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            fs::copy(&src, &dst).expect("copy");
        }
    }
}

/// A closed database holding `KEYS` keys in exactly one SSTable.
fn one_table_db() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path(), opts()).expect("open");
    for i in 0..KEYS {
        db.put(&key(i), &val(i)).expect("put");
    }
    db.compact_range(None, None).expect("compact");
    db.close().expect("close");
    drop(db);
    let mut ssts: Vec<PathBuf> = fs::read_dir(dir.path().join("sst"))
        .expect("sst dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "sst"))
        .collect();
    ssts.sort();
    assert_eq!(ssts.len(), 1, "the fixture must hold one table: {ssts:?}");
    (dir, ssts.pop().expect("one table"))
}

/// The regions of an SSTable, parsed from its own footer.
struct Regions {
    footer: (usize, usize),
    index: (usize, usize),
    bloom: (usize, usize),
}

fn regions(bytes: &[u8]) -> Regions {
    let n = bytes.len();
    let magic = u64::from_le_bytes(bytes[n - 8..].try_into().expect("8"));
    // v1 and v2 are the unchecksummed LARKSST footers, v3 and v4 the
    // checksummed ones, v5 and v6 the stamped REGOSST flat and
    // partitioned ones. v3 onward share a 72-byte layout, so the field
    // offsets below hold for all of them.
    let footer_size = match magic & 0xff {
        0x01 | 0x02 => 64usize,
        0x03..=0x06 => 72,
        v => panic!("unknown format version {v:#04x} (magic {magic:#018x})"),
    };
    let f = n - footer_size;
    let u64_at = |i: usize| u64::from_le_bytes(bytes[f + i..f + i + 8].try_into().expect("8"));
    Regions {
        footer: (f, footer_size),
        bloom: (u64_at(16) as usize, u64_at(24) as usize),
        index: (u64_at(32) as usize, u64_at(40) as usize),
    }
}

/// What a database served after a flip.
#[derive(PartialEq, Eq, Debug)]
enum Served {
    /// The damage was reported: open, a point read or a scan errored.
    Caught,
    /// Every surface agreed with the truth.
    Harmless,
    /// `get` and the scan disagreed, or one of them served wrong data.
    Wrong(String),
}

/// Open the planted directory and cross-check `get` against a full scan.
fn probe(db_dir: &Path) -> Served {
    let db = match Db::open(db_dir, opts()) {
        Ok(db) => db,
        Err(_) => return Served::Caught,
    };
    let scanned: BTreeMap<Vec<u8>, Vec<u8>> = match db.scan(None, None) {
        Ok(pairs) => pairs.into_iter().collect(),
        Err(_) => return Served::Caught,
    };
    for i in 0..KEYS {
        let k = key(i);
        let got = match db.get(&k) {
            Ok(v) => v,
            Err(_) => return Served::Caught,
        };
        let from_scan = scanned.get(&k).cloned();
        if got != from_scan {
            return Served::Wrong(format!(
                "get and scan disagree on {}: get={:?} scan={:?}",
                String::from_utf8_lossy(&k),
                got.as_deref().map(String::from_utf8_lossy),
                from_scan.as_deref().map(String::from_utf8_lossy),
            ));
        }
        match got {
            Some(v) if v == val(i) => {}
            Some(v) => {
                return Served::Wrong(format!(
                    "{} served {:?}, truth is {:?}",
                    String::from_utf8_lossy(&k),
                    String::from_utf8_lossy(&v),
                    String::from_utf8_lossy(&val(i)),
                ));
            }
            None => {
                return Served::Wrong(format!(
                    "{} vanished but no error was reported",
                    String::from_utf8_lossy(&k),
                ));
            }
        }
    }
    drop(db);
    Served::Harmless
}

/// Flip every bit of `region`, up to `cap` evenly spread positions, and
/// classify each outcome.
fn sweep(label: &str, region: (usize, usize), cap: usize) {
    let (src, table) = one_table_db();
    let original = fs::read(&table).expect("read table");
    let rel = table
        .strip_prefix(src.path())
        .expect("under root")
        .to_path_buf();

    let (start, len) = region;
    assert!(len > 0, "[{label}] the region must be non-empty");
    assert!(
        start + len <= original.len(),
        "[{label}] region {start}+{len} runs past the {} byte file",
        original.len(),
    );

    let step = (len / cap).max(1);
    let work = TempDir::new().expect("tempdir");
    let db_dir = work.path().join("db");

    let (mut caught, mut harmless, mut trials) = (0usize, 0usize, 0usize);
    let mut wrong = Vec::new();
    for off in (start..start + len).step_by(step) {
        for bit in 0..8u8 {
            let mut damaged = original.clone();
            damaged[off] ^= 1 << bit;
            let _ = fs::remove_dir_all(&db_dir);
            copy_tree(src.path(), &db_dir);
            fs::write(db_dir.join(&rel), &damaged).expect("plant");
            trials += 1;
            match probe(&db_dir) {
                Served::Caught => caught += 1,
                Served::Harmless => harmless += 1,
                Served::Wrong(why) => wrong.push(format!("byte {off} bit {bit}: {why}")),
            }
        }
    }
    println!(
        "[{label}] {trials} flips over {len} bytes: {caught} caught, {harmless} harmless, \
         {} silently wrong",
        wrong.len(),
    );
    assert!(
        wrong.is_empty(),
        "[{label}] {} flip(s) served silently wrong data:\n  {}",
        wrong.len(),
        wrong
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

#[test]
fn every_probed_bit_flip_in_the_footer_is_caught_or_harmless() {
    let (src, table) = one_table_db();
    let bytes = fs::read(&table).expect("read");
    let r = regions(&bytes);
    drop(src);
    sweep("footer", r.footer, 72);
}

#[test]
fn every_probed_bit_flip_in_the_index_block_is_caught_or_harmless() {
    let (src, table) = one_table_db();
    let bytes = fs::read(&table).expect("read");
    let r = regions(&bytes);
    drop(src);
    sweep("index", r.index, 120);
}

#[test]
fn every_probed_bit_flip_in_the_bloom_region_is_caught_or_harmless() {
    let (src, table) = one_table_db();
    let bytes = fs::read(&table).expect("read");
    let r = regions(&bytes);
    drop(src);
    sweep("bloom", r.bloom, 120);
}

/// The whole metadata tail in one sweep, so a region the three probes
/// above do not name (the range-tombstone block, any padding between
/// regions) is covered too.
#[test]
fn every_probed_bit_flip_in_the_whole_metadata_tail_is_caught_or_harmless() {
    let (src, table) = one_table_db();
    let bytes = fs::read(&table).expect("read");
    let r = regions(&bytes);
    let tail_start = r.bloom.0.min(r.index.0);
    drop(src);
    sweep("metadata tail", (tail_start, bytes.len() - tail_start), 200);
}

/// Where the legacy fixtures live.
///
/// `tests/fixtures/adv_review_legacy` holds V1 and V2 tables and a whole
/// V1 database directory produced by *building and running the
/// pre-change tree*, so the bytes under test were never written by the
/// code under test. `LARK_LEGACY_FIXTURES` overrides the location for a
/// freshly regenerated set.
fn legacy_dir() -> Option<PathBuf> {
    if let Ok(from_env) = std::env::var("LARK_LEGACY_FIXTURES") {
        return Some(PathBuf::from(from_env));
    }
    let checked_in = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("adv_review_legacy");
    checked_in.join("flat.sst").exists().then_some(checked_in)
}

/// Tables written by the pre-change tree must still read, exactly.
///
/// The trailing magic of each fixture is asserted first, so a fixture
/// accidentally regenerated by the current tree fails the test instead of
/// making it vacuous.
#[test]
fn tables_written_by_the_previous_format_still_read_exactly() {
    let dir = legacy_dir().expect(
        "the checked-in legacy fixtures are missing and LARK_LEGACY_FIXTURES is unset, so \
         backward compatibility would go unproven; this must fail rather than skip",
    );

    let magic_of = |p: &Path| -> u64 {
        let b = fs::read(p).expect("read fixture");
        u64::from_le_bytes(b[b.len() - 8..].try_into().expect("8"))
    };

    // A flat V1 table and a partitioned V2 table, ingested into a fresh
    // database built by the current tree.
    for (name, want) in [
        ("flat.sst", 0x4c41_524b_5353_5401u64),
        ("partitioned.sst", 0x4c41_524b_5353_5402u64),
    ] {
        let src = dir.join(name);
        assert_eq!(
            magic_of(&src),
            want,
            "{name} is not the legacy format any more; the fixture was regenerated",
        );
    }

    // Both legacy external tables must actually be readable, not just
    // carry a legacy magic: ingest each one and read every entry back.
    // The partitioned V2 table is the one that exercises the legacy
    // partitioned-index leaf path.
    //
    // One table per database on purpose. Ingesting several external
    // tables in a single database hits an unrelated, pre-existing bug
    // that silently drops every source after the first (see
    // `tests/adv_review_ingest_cache.rs`), which would contaminate this
    // compatibility result with a defect that has nothing to do with the
    // table format.
    for (name, count, key_of, val_of) in [
        (
            "flat.sst",
            500u32,
            (|i: u32| format!("ext_{i:06}")) as fn(u32) -> String,
            (|i: u32| format!("extval_{i:06}")) as fn(u32) -> String,
        ),
        (
            "partitioned.sst",
            2000,
            |i: u32| format!("p_{i:08}"),
            |i: u32| format!("pv_{i:08}"),
        ),
    ] {
        let ing = TempDir::new().expect("tempdir");
        let idb = Db::open(ing.path(), opts()).expect("open");
        let staged = ing.path().join(name);
        fs::copy(dir.join(name), &staged).expect("stage legacy table");
        idb.ingest_external_files(&[staged], IngestOptions::default())
            .unwrap_or_else(|e| panic!("the current tree must ingest {name}: {e}"));
        for i in 0..count {
            assert_eq!(
                idb.get(key_of(i).as_bytes()).expect("get"),
                Some(val_of(i).into_bytes()),
                "the legacy table {name} lost {}",
                key_of(i),
            );
        }
        idb.close().expect("close");
        println!("legacy {name}: {count} entries ingested and read back exactly");
    }

    // The legacy database directory must open and serve every key,
    // including through the range delete it carries.
    let src_db = dir.join("legacy_db");
    let mut legacy_magics = Vec::new();
    for e in fs::read_dir(src_db.join("sst"))
        .expect("legacy sst dir")
        .flatten()
    {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "sst") {
            legacy_magics.push(magic_of(&p));
        }
    }
    assert!(
        !legacy_magics.is_empty(),
        "the legacy database has no tables"
    );
    for m in &legacy_magics {
        assert!(
            matches!(m & 0xff, 0x01 | 0x02),
            "a legacy table carries {m:#018x}, which is not a pre-change magic",
        );
    }

    let work = TempDir::new().expect("tempdir");
    let db_dir = work.path().join("legacy_db");
    copy_tree(&src_db, &db_dir);
    let db = Db::open(&db_dir, opts()).expect("the current tree must open a legacy database");
    let mut missing = Vec::new();
    for i in 0..5000u32 {
        let k = format!("k_{i:08}");
        let deleted = (100..200).contains(&i);
        let got = db.get(k.as_bytes()).expect("get");
        let want = if deleted {
            None
        } else {
            Some(format!("v_{i:08}").into_bytes())
        };
        if got != want {
            missing.push(format!("{k}: got {got:?} want {want:?}"));
        }
    }
    assert!(
        missing.is_empty(),
        "the current tree lost {} key(s) from a legacy database:\n  {}",
        missing.len(),
        missing
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  "),
    );

    // Writing into it and compacting must mix legacy and current tables
    // without loss.
    for i in 5000..5500u32 {
        db.put(
            format!("k_{i:08}").as_bytes(),
            format!("v_{i:08}").as_bytes(),
        )
        .expect("put into a legacy database");
    }
    db.compact_range(None, None)
        .expect("compact a legacy database");
    for i in 0..5500u32 {
        let k = format!("k_{i:08}");
        let deleted = (100..200).contains(&i);
        let want = if deleted {
            None
        } else {
            Some(format!("v_{i:08}").into_bytes())
        };
        assert_eq!(
            db.get(k.as_bytes()).expect("get"),
            want,
            "{k} was lost by compacting a legacy database",
        );
    }
    db.close().expect("close");
    println!(
        "legacy fixtures verified: {} legacy tables ({} magics), 5500 keys after a \
         mixed-format compaction",
        legacy_magics.len(),
        legacy_magics
            .iter()
            .map(|m| format!("{:#04x}", m & 0xff))
            .collect::<Vec<_>>()
            .join(","),
    );
}
