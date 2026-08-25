//! Exhaustive on-disk corruption: truncation at every offset, bit rot in
//! every region, torn record headers, and structurally impossible table
//! sets.
//!
//! `tests/corruption.rs` and `tests/open_and_corruption.rs` already poke a
//! handful of hand-picked offsets. This file differs from them in two
//! ways that matter.
//!
//! **Coverage.** Every byte offset of the WAL, of an SSTable and of the
//! MANIFEST is truncated, and every bit of every byte is flipped, by the
//! `#[ignore]`d sweeps. The default run takes a seeded, evenly spread
//! sample of the same offsets so `cargo test` stays fast. The sample is
//! deterministic, so an offset that fails once fails every time.
//!
//! **Correctness, not merely liveness.** "did not panic" is a weak bar:
//! an engine that silently serves half a table clears it. Every trial
//! here reads the whole database back and requires one of exactly two
//! outcomes:
//!
//! * a loud refusal or a loud read error, or
//! * data that matches what was actually written.
//!
//! Wrong values, invented keys, a forward scan that disagrees with a
//! point lookup, a reverse scan that disagrees with the forward one, and
//! a scan that stops early while `status()` reports success are all
//! failures, not passes. Those are the ways an LSM engine returns
//! silently wrong data, and none of them are visible to a test that only
//! checks for a panic.
//!
//! A panic raised on a *background* thread would be invisible to the test
//! harness, so `harness::watch_engine_panics` installs a hook that records
//! any panic originating outside `tests/`, and every sweep asserts that none
//! were recorded. A hang would be equally invisible, so every test body
//! runs on a worker thread whose progress counter is watched from the
//! test thread with a deadline plus bounded backoff.

use std::fs;
use std::panic;
use std::sync::atomic::{AtomicU64, Ordering};

use lark_kv::Db;
use tempfile::TempDir;

mod common;
// The crate root of an integration test sits in `tests/`, so the usual
// `file.rs` + `file/` nesting needs an explicit path.
#[path = "corruption_exhaustive/harness.rs"]
mod harness;

use common::fault::{
    builtin_workload, child_entrypoint, flip_bit, garbage, overwrite_range, truncate_at,
    validate_prefix_of_state,
};
use harness::{
    assert_engine_never_panicked, batch_fixture, every, exactly, manifest_frames, never_invents,
    read_state, region, sample, sst_regions, table_fixture, trial, valid_prefix, wal_fixture,
    wal_frames, watch, Recovered, Tally, BLOOM, DATA, FOOTER, INDEX, SAMPLE, SEED, TAG_ADD_FILE,
};

// ─── WAL: truncation ────────────────────────────────────────────────────

/// Truncating the WAL must leave the database holding the state after
/// some whole number of the intended writes, or refuse to open. The exact
/// boundary is asserted, not merely the absence of a panic: a cut on a
/// record boundary must replay exactly the records before it, and a cut
/// anywhere else must be refused, because a partial record cannot be
/// proven to be anything. Catches a replay that swallows a torn tail as
/// if it were data, and one that drops an intact earlier record.
fn wal_truncation_sweep(cuts: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = wal_fixture();
    let rel = fixture.only(".log");
    let frames = wal_frames(fixture.bytes(&rel));
    let boundaries: Vec<u64> = std::iter::once(0)
        .chain(frames.iter().map(|f| f.end))
        .collect();
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    assert_eq!(
        frames.len(),
        fixture.history.len(),
        "this sweep maps one WAL record to one intended write; the fixture no longer does",
    );

    for &cut in cuts {
        progress.fetch_add(1, Ordering::Relaxed);
        let label = format!("WAL truncated to {cut} of {}", fixture.bytes(&rel).len());
        let outcome = trial(fixture, &db, |db| truncate_at(&db.join(&rel), cut));
        let on_boundary = boundaries.contains(&cut);
        match (&outcome, on_boundary) {
            (Recovered::Opened(state), true) => {
                let records = boundaries.iter().filter(|b| **b <= cut).count() - 1;
                match validate_prefix_of_state(state, &fixture.history) {
                    Ok(report) if report.valid_ks.contains(&records) => tally.matched += 1,
                    Ok(report) => tally.violation(format!(
                        "{label}: {records} whole record(s) survived the cut but the state \
                         matches prefix lengths {:?}",
                        report.valid_ks
                    )),
                    Err(e) => tally.violation(format!("{label}: {e}")),
                }
            }
            (Recovered::Refused(_), false) => tally.refused += 1,
            (Recovered::Refused(e), true) => tally.violation(format!(
                "{label}: the cut is on a record boundary, so every earlier record is intact, \
                 yet open refused: {e}"
            )),
            (_, on_boundary) => tally.violation(format!(
                "{label}: on a record boundary: {on_boundary}, outcome: {}",
                outcome.why()
            )),
        }
    }
    tally.finish(what);
}

#[test]
fn a_wal_truncated_at_a_sampled_offset_replays_whole_records_or_refuses() {
    watch("wal truncation sample", |progress| {
        let fixture = wal_fixture();
        let len = fixture.bytes(&fixture.only(".log")).len() as u64;
        wal_truncation_sweep(
            &sample(0, len, SAMPLE, SEED),
            "wal truncation sample",
            progress,
        );
    });
}

/// The exhaustive twin of the sampled sweep: every byte offset of the
/// WAL, with no sampling to hide behind.
#[test]
#[ignore = "exhaustive sweep, measured at 0.04s: run `just test-corruption-slow`"]
fn a_wal_truncated_at_every_offset_replays_whole_records_or_refuses() {
    watch("wal truncation exhaustive", |progress| {
        let fixture = wal_fixture();
        let len = fixture.bytes(&fixture.only(".log")).len() as u64;
        wal_truncation_sweep(&every(0, len), "wal truncation exhaustive", progress);
    });
}

/// A `WriteBatch` is atomic, so a WAL cut inside the record that carries
/// one must leave none of it applied, in every durability mode. Cutting
/// at every offset inside the final batch record catches a replay that
/// applies the operations it managed to parse before hitting the damage.
#[test]
fn a_write_batch_record_cut_in_half_is_never_half_applied() {
    watch("wal batch atomicity", |progress| {
        let fixture = batch_fixture();
        let rel = fixture.only(".log");
        let frames = wal_frames(fixture.bytes(&rel));
        let last = *frames.last().expect("at least one record");
        assert!(
            frames.len() >= 2,
            "the batch fixture must hold more than one record",
        );
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        let expect = valid_prefix(&fixture.history);
        for cut in last.start + 1..last.end {
            progress.fetch_add(1, Ordering::Relaxed);
            let outcome = trial(fixture, &db, |db| truncate_at(&db.join(&rel), cut));
            tally.record(&format!("batch record cut at {cut}"), outcome, &expect);
        }
        tally.finish("wal batch atomicity");
    });
}

// ─── WAL: bit rot ───────────────────────────────────────────────────────

/// Every byte of a WAL record is covered by its checksum: the length, the
/// type byte and the payload all feed `checksum::wal_record`, and the
/// stored checksum is the last four bytes. So every single-bit flip must
/// be caught and no flipped WAL may ever be replayed as data. A flip that
/// opens is a hole in the checksum's coverage and is reported as one.
fn wal_flip_sweep(positions: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = wal_fixture();
    let rel = fixture.only(".log");
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    for &offset in positions {
        for bit in 0..8u8 {
            progress.fetch_add(1, Ordering::Relaxed);
            let outcome = trial(fixture, &db, |db| flip_bit(&db.join(&rel), offset, bit));
            match outcome {
                Recovered::Refused(_) => tally.refused += 1,
                other => tally.violation(format!(
                    "WAL byte {offset} bit {bit}: the record checksum did not catch the flip, \
                     the database {}",
                    other.why()
                )),
            }
        }
    }
    tally.finish(what);
}

#[test]
fn a_sampled_bit_flip_anywhere_in_the_wal_is_caught_by_the_record_checksum() {
    watch("wal bit flips sample", |progress| {
        let fixture = wal_fixture();
        let len = fixture.bytes(&fixture.only(".log")).len() as u64;
        wal_flip_sweep(
            &sample(0, len, SAMPLE, SEED),
            "wal bit flips sample",
            progress,
        );
    });
}

/// The exhaustive twin: every bit of every byte of the WAL.
#[test]
#[ignore = "exhaustive sweep, measured at 0.23s: run `just test-corruption-slow`"]
fn every_bit_flip_in_the_wal_is_caught_by_the_record_checksum() {
    watch("wal bit flips exhaustive", |progress| {
        let fixture = wal_fixture();
        let len = fixture.bytes(&fixture.only(".log")).len() as u64;
        wal_flip_sweep(&every(0, len), "wal bit flips exhaustive", progress);
    });
}

// ─── WAL: torn and trailing bytes ───────────────────────────────────────

/// A record header that promises more payload than the file holds is the
/// signature of a write torn by a crash. Replay must reject it against
/// the bytes actually present rather than trusting the length, and must
/// leave the WAL on disk for repair. Catches both a replay that reads
/// past the record and one that quietly treats the tail as absent.
#[test]
fn a_wal_record_header_promising_more_bytes_than_exist_is_refused() {
    watch("wal torn header", |progress| {
        let fixture = wal_fixture();
        let rel = fixture.only(".log");
        let bytes = fixture.bytes(&rel).to_vec();
        let frames = wal_frames(&bytes);
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        for frame in [frames[0], *frames.last().expect("non-empty")] {
            let real = (frame.end - frame.start - 9) as u32;
            for extra in [1u32, 7, 64, 1 << 20, u32::MAX - real] {
                progress.fetch_add(1, Ordering::Relaxed);
                let claimed = real + extra;
                let outcome = trial(fixture, &db, |db| {
                    overwrite_range(&db.join(&rel), frame.start, &claimed.to_le_bytes())
                });
                let kept = fs::metadata(db.join(&rel)).map(|m| m.len()).unwrap_or(0);
                match outcome {
                    Recovered::Refused(_) => {
                        assert_eq!(
                            kept,
                            bytes.len() as u64,
                            "a refused open must leave the WAL untouched for repair",
                        );
                        tally.refused += 1;
                    }
                    other => tally.violation(format!(
                        "record at {} claiming {claimed} bytes instead of {real}: {}",
                        frame.start,
                        other.why()
                    )),
                }
            }
        }
        tally.finish("wal torn header");
    });
}

/// Garbage appended after the last valid record is what a crash mid-write
/// plus a filesystem that pads with whatever was in the block leaves
/// behind. It must never be replayed as data. Catches a replay loop that
/// trusts a length field it has not checksummed.
#[test]
fn garbage_appended_after_a_valid_wal_is_never_replayed_as_data() {
    watch("wal trailing garbage", |progress| {
        let fixture = wal_fixture();
        let rel = fixture.only(".log");
        let len = fixture.bytes(&rel).len() as u64;
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        let expect = valid_prefix(&fixture.history);
        for n in [1usize, 2, 4, 5, 8, 9, 16, 33, 64, 257] {
            progress.fetch_add(1, Ordering::Relaxed);
            let junk = garbage(SEED ^ n as u64, n);
            let outcome = trial(fixture, &db, |db| {
                overwrite_range(&db.join(&rel), len, &junk)
            });
            tally.record(&format!("{n} garbage bytes appended"), outcome, &expect);
        }
        tally.finish("wal trailing garbage");
    });
}

/// A WAL truncated to nothing is a legal state: the writes it held are
/// lost, but the database must open, serve whatever the tables hold and
/// keep working. Catches an open path that treats a zero-length WAL as
/// corruption, and one that leaves the reopened database unable to write.
#[test]
fn a_zero_length_wal_opens_and_the_database_stays_usable() {
    watch("zero-length wal", |_progress| {
        let fixture = wal_fixture();
        let rel = fixture.only(".log");
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        fixture.plant(&db);
        truncate_at(&db.join(&rel), 0);

        let handle = Db::open(&db, fixture.opts()).expect("a zero-length WAL must still open");
        assert!(
            read_state(&handle).map(|s| s.is_empty()).unwrap_or(false),
            "every write lived in the WAL, so an emptied WAL must leave no data",
        );
        handle
            .put(b"after", b"recovery")
            .expect("put after recovery");
        assert_eq!(
            handle.get(b"after").expect("get after recovery"),
            Some(b"recovery".to_vec()),
        );
        handle.close().expect("close");
        assert_engine_never_panicked("zero-length wal");
    });
}

// ─── SSTable: truncation ────────────────────────────────────────────────

/// An SSTable carries its footer at the end of the file, so any
/// truncation destroys the only map of the file. The engine must refuse
/// to open, name the file, and leave every table on disk: a truncated
/// table is repairable, and an open that silently drops it is not.
/// Catches an open path that treats an unreadable table as an empty one.
fn sst_truncation_sweep(cuts: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = table_fixture();
    let rel = fixture.only(".sst");
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    for &cut in cuts {
        progress.fetch_add(1, Ordering::Relaxed);
        let outcome = trial(fixture, &db, |db| truncate_at(&db.join(&rel), cut));
        match outcome {
            Recovered::Refused(e) => {
                assert!(
                    db.join(&rel).exists(),
                    "cut at {cut}: a refused open must leave the table on disk",
                );
                assert!(
                    !e.trim().is_empty(),
                    "cut at {cut}: the refusal must say something actionable",
                );
                tally.refused += 1;
            }
            other => tally.violation(format!(
                "table truncated to {cut} of {}: {}",
                fixture.bytes(&rel).len(),
                other.why()
            )),
        }
    }
    tally.finish(what);
}

#[test]
fn an_sstable_truncated_at_a_sampled_offset_refuses_to_open_and_keeps_the_file() {
    watch("sst truncation sample", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes(&fixture.only(".sst")).len() as u64;
        sst_truncation_sweep(
            &sample(0, len, SAMPLE, SEED),
            "sst truncation sample",
            progress,
        );
    });
}

/// The exhaustive twin: every byte offset of the table.
#[test]
#[ignore = "exhaustive sweep, measured at 0.04s: run `just test-corruption-slow`"]
fn an_sstable_truncated_at_every_offset_refuses_to_open_and_keeps_the_file() {
    watch("sst truncation exhaustive", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes(&fixture.only(".sst")).len() as u64;
        sst_truncation_sweep(&every(0, len), "sst truncation exhaustive", progress);
    });
}

/// A table file that a crash left at zero bytes still has a MANIFEST
/// entry pointing at it. The open must fail loudly and name the path, so
/// an operator knows which file to restore. Catches an open that reports
/// a bare "invalid data" with no way to act on it.
#[test]
fn a_zero_length_sstable_refuses_to_open_and_names_the_file() {
    watch("zero-length sst", |_progress| {
        let fixture = table_fixture();
        let rel = fixture.only(".sst");
        let leaf = rel.rsplit('/').next().expect("a file name").to_string();
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        fixture.plant(&db);
        truncate_at(&db.join(&rel), 0);
        match Db::open(&db, fixture.opts()) {
            Ok(_) => panic!("a zero-length table must not open"),
            Err(e) => assert!(
                e.to_string().contains(&leaf),
                "the error must name {leaf}, got: {e}",
            ),
        }
        assert_engine_never_panicked("zero-length sst");
    });
}

// ─── SSTable: bit rot, region by region ─────────────────────────────────

/// Flip every bit of the sampled positions in one region and require each
/// trial to be caught or to be provably harmless. "Harmless" means the
/// database still reads back byte-for-byte what was written; anything
/// else is silently wrong data.
fn sst_region_sweep(name: &'static str, positions: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = table_fixture();
    let rel = fixture.only(".sst");
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    let expect = exactly(&fixture.state);
    for &offset in positions {
        for bit in 0..8u8 {
            progress.fetch_add(1, Ordering::Relaxed);
            let outcome = trial(fixture, &db, |db| flip_bit(&db.join(&rel), offset, bit));
            tally.record(
                &format!("{name}: byte {offset} bit {bit} flipped"),
                outcome,
                &expect,
            );
        }
    }
    tally.finish(what);
}

fn sst_region_positions(name: &'static str, count: usize) -> Vec<u64> {
    let fixture = table_fixture();
    let r = region(fixture.bytes(&fixture.only(".sst")), name);
    sample(r.start, r.end, count, SEED)
}

/// Data blocks are framed `[compression: u8][payload][crc: u32]` and the
/// checksum covers the compression byte as well as the payload, so every
/// bit in the region is protected. A flip must surface as an error, at
/// open or at the read that touches the block; it must never come back
/// as a value the caller believes.
#[test]
fn a_bit_flip_in_an_sstable_data_block_is_caught_by_the_block_checksum() {
    watch("sst data block flips", |progress| {
        sst_region_sweep(
            DATA,
            &sst_region_positions(DATA, SAMPLE),
            "sst data block flips",
            progress,
        );
    });
}

/// The index block is not checksummed, so a flip in it is caught only
/// indirectly: a damaged block handle sends the reader to bytes whose own
/// checksum then fails, and a damaged separator key sends a lookup to the
/// wrong block. The first is loud; the second would silently answer "not
/// found" for a key that is on disk. This test proves which of the two
/// happens.
#[test]
fn a_bit_flip_in_the_sstable_index_block_is_caught_or_harmless() {
    watch("sst index flips", |progress| {
        sst_region_sweep(
            INDEX,
            &sst_region_positions(INDEX, SAMPLE),
            "sst index flips",
            progress,
        );
    });
}

/// The bloom region is not checksummed either, and a point lookup trusts
/// it: `may_contain` returning false short-circuits the read. Clearing a
/// set bit therefore hides a key that is physically present, with no
/// error anywhere. This test asserts the invariant that must hold, that a
/// corrupt table is either detected or serves the data it was given, and
/// so reports the hole if it is there.
#[test]
fn a_bit_flip_in_the_sstable_bloom_region_is_caught_or_harmless() {
    watch("sst bloom flips", |progress| {
        sst_region_sweep(
            BLOOM,
            &sst_region_positions(BLOOM, SAMPLE),
            "sst bloom flips",
            progress,
        );
    });
}

/// The 64-byte footer is the map of the file and is not checksummed; the
/// magic number and the region bounds are validated against the file
/// size instead. A flip must either fail that validation or leave every
/// read correct. Catches a footer field that is trusted without being
/// checked.
#[test]
fn a_bit_flip_in_the_sstable_footer_is_caught_or_harmless() {
    watch("sst footer flips", |progress| {
        sst_region_sweep(
            FOOTER,
            &sst_region_positions(FOOTER, SAMPLE),
            "sst footer flips",
            progress,
        );
    });
}

/// The exhaustive twin: every bit of every byte of the table, tallied by
/// region so an unprotected region is named rather than merely counted.
#[test]
#[ignore = "exhaustive sweep, measured at 1.3s: run `just test-corruption-slow`"]
fn every_bit_flip_in_an_sstable_is_caught_or_harmless() {
    watch("sst flips exhaustive", |progress| {
        let fixture = table_fixture();
        let regions = sst_regions(fixture.bytes(&fixture.only(".sst")));
        let mut failed = Vec::new();
        for r in regions {
            let result = panic::catch_unwind(|| {
                sst_region_sweep(
                    r.name,
                    &every(r.start, r.end),
                    &format!("sst {} exhaustive", r.name),
                    progress,
                );
            });
            if result.is_err() {
                failed.push(r.name);
            }
        }
        assert!(
            failed.is_empty(),
            "unprotected SSTable region(s): {failed:?} (details above)",
        );
    });
}

// ─── SSTable: files that are not there ──────────────────────────────────

/// A MANIFEST that references a table which is not on disk is what an
/// out-of-band delete, a partial restore or a half-finished copy leaves.
/// The open must fail and name the missing path; serving the database
/// without the table would silently drop every key it held. Catches an
/// open path that skips a table it cannot find.
#[test]
fn a_manifest_referencing_a_missing_sstable_refuses_to_open_and_names_it() {
    watch("missing sst", |_progress| {
        let fixture = table_fixture();
        let rel = fixture.only(".sst");
        let leaf = rel.rsplit('/').next().expect("a file name").to_string();
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        fixture.plant(&db);
        fs::remove_file(db.join(&rel)).expect("remove the table");
        match Db::open(&db, fixture.opts()) {
            Ok(handle) => {
                let served = read_state(&handle).map(|s| s.len()).unwrap_or(0);
                panic!(
                    "opening without {leaf} served {served} of {} keys instead of failing",
                    fixture.state.len()
                );
            }
            Err(e) => assert!(
                e.to_string().contains(&leaf),
                "the error must name the missing file {leaf}, got: {e}",
            ),
        }
        assert_engine_never_panicked("missing sst");
    });
}

// ─── MANIFEST ───────────────────────────────────────────────────────────

/// A MANIFEST truncated by a crash loses its tail, which legitimately
/// costs whole tables. What it must never do is serve a key that was
/// never written or a value that was never stored under its key, and it
/// must never serve an inconsistent view. Catches a replay that resumes
/// after damaged bytes and reads a later record as if the record before
/// it had been intact.
fn manifest_truncation_sweep(cuts: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = table_fixture();
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    let expect = never_invents(&fixture.state);
    for &cut in cuts {
        progress.fetch_add(1, Ordering::Relaxed);
        let outcome = trial(fixture, &db, |db| truncate_at(&db.join("MANIFEST"), cut));
        tally.record(&format!("MANIFEST truncated to {cut}"), outcome, &expect);
    }
    tally.finish(what);
}

#[test]
fn a_manifest_truncated_at_a_sampled_offset_never_serves_invented_data() {
    watch("manifest truncation sample", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes("MANIFEST").len() as u64;
        manifest_truncation_sweep(
            &sample(0, len, SAMPLE, SEED),
            "manifest truncation sample",
            progress,
        );
    });
}

/// The exhaustive twin: every byte offset of the MANIFEST.
#[test]
#[ignore = "exhaustive sweep, measured at 0.06s: run `just test-corruption-slow`"]
fn a_manifest_truncated_at_every_offset_never_serves_invented_data() {
    watch("manifest truncation exhaustive", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes("MANIFEST").len() as u64;
        manifest_truncation_sweep(&every(0, len), "manifest truncation exhaustive", progress);
    });
}

/// A MANIFEST record checksum covers its length header and its payload,
/// so a flip anywhere in a record must stop replay at that record. The
/// records before it stay valid, so the database may still open with less
/// data; what it may not do is act on the damaged record or on anything
/// after it. Catches a replay that skips a bad record and keeps going.
fn manifest_flip_sweep(positions: &[u64], what: &str, progress: &AtomicU64) {
    let fixture = table_fixture();
    let root = TempDir::new().expect("tempdir");
    let db = root.path().join("db");
    let mut tally = Tally::default();
    let expect = never_invents(&fixture.state);
    for &offset in positions {
        for bit in 0..8u8 {
            progress.fetch_add(1, Ordering::Relaxed);
            let outcome = trial(fixture, &db, |db| {
                flip_bit(&db.join("MANIFEST"), offset, bit)
            });
            tally.record(
                &format!("MANIFEST byte {offset} bit {bit} flipped"),
                outcome,
                &expect,
            );
        }
    }
    tally.finish(what);
}

#[test]
fn a_sampled_bit_flip_in_the_manifest_stops_replay_without_inventing_data() {
    watch("manifest flips sample", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes("MANIFEST").len() as u64;
        manifest_flip_sweep(
            &sample(0, len, SAMPLE, SEED),
            "manifest flips sample",
            progress,
        );
    });
}

/// The exhaustive twin: every bit of every byte of the MANIFEST.
#[test]
#[ignore = "exhaustive sweep, measured at 0.45s: run `just test-corruption-slow`"]
fn every_bit_flip_in_the_manifest_stops_replay_without_inventing_data() {
    watch("manifest flips exhaustive", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes("MANIFEST").len() as u64;
        manifest_flip_sweep(&every(0, len), "manifest flips exhaustive", progress);
    });
}

/// The MANIFEST is appended to without an fsync for most edits, so a
/// crash can leave a record header promising more bytes than were ever
/// written. Replay must stop there against the file size rather than
/// trusting the length. Catches a replay that indexes past the end of the
/// buffer or invents a record out of whatever follows.
#[test]
fn a_manifest_record_header_promising_more_bytes_than_exist_stops_replay() {
    watch("manifest torn header", |progress| {
        let fixture = table_fixture();
        let bytes = fixture.bytes("MANIFEST").to_vec();
        let frames = manifest_frames(&bytes);
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        let expect = never_invents(&fixture.state);
        for frame in [frames[0], *frames.last().expect("non-empty")] {
            let real = (frame.end - frame.start - 8) as u32;
            for extra in [1u32, 9, 4096, 1 << 20, u32::MAX - real] {
                progress.fetch_add(1, Ordering::Relaxed);
                let claimed = real + extra;
                let outcome = trial(fixture, &db, |db| {
                    overwrite_range(&db.join("MANIFEST"), frame.start, &claimed.to_le_bytes())
                });
                tally.record(
                    &format!("record at {} claiming {claimed} bytes", frame.start),
                    outcome,
                    &expect,
                );
            }
        }
        tally.finish("manifest torn header");
    });
}

/// Garbage after the last committed record must cost the tail and nothing
/// else: every table committed before it is still described by intact
/// records, so all of that data must still be there. Catches a replay
/// that abandons the whole file when it meets a bad record, which would
/// turn a scribble on the tail into total data loss.
#[test]
fn garbage_appended_after_a_valid_manifest_keeps_every_committed_table() {
    watch("manifest trailing garbage", |progress| {
        let fixture = table_fixture();
        let len = fixture.bytes("MANIFEST").len() as u64;
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        let expect = exactly(&fixture.state);
        for n in [1usize, 3, 4, 5, 8, 16, 64, 129, 512] {
            progress.fetch_add(1, Ordering::Relaxed);
            let junk = garbage(SEED ^ (n as u64) << 8, n);
            let outcome = trial(fixture, &db, |db| {
                overwrite_range(&db.join("MANIFEST"), len, &junk)
            });
            tally.record(&format!("{n} garbage bytes appended"), outcome, &expect);
        }
        tally.finish("manifest trailing garbage");
    });
}

/// Appending a copy of an existing record leaves a syntactically perfect
/// MANIFEST that says something impossible: for an `AddFile` record, that
/// one file id is live twice at one level. Levels above L0 are assumed
/// sorted and non-overlapping and are binary searched, so a duplicate is
/// a chance to return a key twice, to break iteration order, or to make a
/// point lookup and a scan disagree. Whatever the engine does with it, it
/// must not do that.
#[test]
fn a_manifest_that_names_the_same_table_twice_never_serves_an_inconsistent_view() {
    watch("duplicate manifest record", |progress| {
        let fixture = table_fixture();
        let bytes = fixture.bytes("MANIFEST").to_vec();
        let frames = manifest_frames(&bytes);
        let add_files = frames.iter().filter(|f| f.kind == TAG_ADD_FILE).count();
        assert!(
            add_files > 0,
            "the fixture MANIFEST must contain an AddFile record to duplicate",
        );
        let root = TempDir::new().expect("tempdir");
        let db = root.path().join("db");
        let mut tally = Tally::default();
        let expect = never_invents(&fixture.state);
        for frame in frames {
            progress.fetch_add(1, Ordering::Relaxed);
            let copy = bytes[frame.start as usize..frame.end as usize].to_vec();
            let outcome = trial(fixture, &db, |db| {
                overwrite_range(&db.join("MANIFEST"), bytes.len() as u64, &copy)
            });
            tally.record(
                &format!("record tag {} at {} duplicated", frame.kind, frame.start),
                outcome,
                &expect,
            );
        }
        tally.finish("duplicate manifest record");
    });
}

// ─── child entry point ──────────────────────────────────────────────────

#[test]
#[ignore = "child process entry point, re-executed by the crash harness"]
fn crash_child() {
    child_entrypoint(builtin_workload);
}
