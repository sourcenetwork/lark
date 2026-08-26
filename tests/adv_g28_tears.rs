//! Adversarial probe for the discarded-table open guard (G28) under the
//! tear modes a real filesystem produces that are not `Truncate`.
//!
//! The shipped fix dismisses a *zero-length* orphan SSTable. A power cut
//! inside the first flush does not always leave a zero-length file: ext4
//! with blocks already allocated leaves zeros at full length, and a device
//! that tears at sector granularity leaves a prefix followed by garbage.
//! Both keep every acknowledged write in the fsynced WAL, so both must
//! open.

mod common;

use common::fault::{
    self, ChildOutcome, ChildSpec, CrashRun, CutPoint, Phase, PowerLossOptions, Recovery, TearMode,
    Trigger,
};
use lark_kv::{DurabilityMode, Options};
use std::time::Duration;
use tempfile::TempDir;

#[test]
#[ignore = "child process entry point, re-executed by the crash harness"]
fn crash_child() {
    fault::child_entrypoint(fault::builtin_workload);
}

fn opts(write_buffer_size: usize) -> Options {
    Options {
        write_buffer_size,
        ..Options::default()
    }
}

fn cut_first_flush(db: &std::path::Path, tear: TearMode) -> ChildOutcome {
    let spec = ChildSpec::new(Phase::DuringFlush, db).durability(DurabilityMode::Immediate);
    let out = CrashRun::new(spec)
        .trigger(Trigger::sst_write(1))
        .timeout(Duration::from_secs(180))
        .run();
    out.assert_killed();
    let popts = PowerLossOptions::default().tear(tear);
    fault::simulate_power_loss_with(&out.spec.db_path, &out.journal, CutPoint::End, &popts);
    out
}

fn probe(tear: TearMode) {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("db");
    let out = cut_first_flush(&db, tear);
    let ssts = fault::find_ssts(&db);
    let lens: Vec<u64> = ssts.iter().map(|p| fault::file_len(p)).collect();
    println!(
        "{tear:?}: orphan SSTables {ssts:?} lengths {lens:?}, {} acked",
        out.acked_count()
    );
    assert!(out.acked_count() > 0, "no write was acknowledged");

    match fault::recover_and_validate(&db, opts(8 * 1024), &out.history) {
        Recovery::Recovered(r) => {
            fault::assert_acked_survived(&r, &out.acked);
            println!("{tear:?}: opened, {} writes recovered", r.k);
        }
        Recovery::RefusedToOpen(e) => {
            // Prove the data is durable by removing the orphan and retrying.
            let salvage = tmp.path().join("salvage");
            fault::copy_tree(&db, &salvage);
            let mut removed = Vec::new();
            for sst in fault::find_ssts(&salvage) {
                std::fs::remove_file(&sst).unwrap();
                removed.push(sst);
            }
            let salvaged = fault::recover_and_validate(&salvage, opts(8 * 1024), &out.history);
            let k = match &salvaged {
                Recovery::Recovered(r) => r.k,
                Recovery::RefusedToOpen(e2) => panic!("salvage also refused: {e2}"),
            };
            panic!(
                "{tear:?}: {} acknowledged Immediate-durability writes lost. The database \
                 refuses to open: {e}\nAfter removing the orphan(s) {removed:?} the same \
                 directory opens and recovers {k} writes, so the WAL held them all along.",
                out.acked_count(),
            );
        }
    }
}

#[test]
fn a_first_flush_cut_that_zeroes_the_orphan_keeps_every_acknowledged_write() {
    probe(TearMode::Zero);
}

#[test]
fn a_first_flush_cut_that_tears_a_sector_keeps_every_acknowledged_write() {
    probe(TearMode::TornSector);
}

#[test]
fn a_first_flush_cut_that_garbles_the_orphan_keeps_every_acknowledged_write() {
    probe(TearMode::Garbage);
}

/// Convergence: after a zero-length orphan lets the open through, the
/// database must reach a steady state. Reopening repeatedly, writing
/// through, and closing must keep every write and must never start
/// refusing again.
#[test]
fn an_open_that_dismissed_a_zero_length_orphan_converges_over_repeated_reopens() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("db");
    let out = cut_first_flush(&db, TearMode::Truncate);
    let orphans = fault::find_ssts(&db);
    assert!(
        orphans.iter().all(|p| fault::file_len(p) == 0),
        "this probe needs the zero-length shape, got {orphans:?}",
    );

    // The pre-crash acknowledged writes must come back before anything
    // else touches the directory; the reopen cycles below then add keys
    // of their own, which the prefix validator would reject.
    let baseline = tmp.path().join("baseline");
    fault::copy_tree(&db, &baseline);
    match fault::recover_and_validate(&baseline, opts(8 * 1024), &out.history) {
        Recovery::Recovered(r) => fault::assert_acked_survived(&r, &out.acked),
        Recovery::RefusedToOpen(e) => panic!("the dismissed orphan still refused the open: {e}"),
    }

    let mut extra: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for cycle in 0..6u32 {
        let d = lark_kv::Db::open(&db, opts(8 * 1024)).unwrap_or_else(|e| {
            panic!("cycle {cycle}: reopen refused after the orphan was dismissed: {e}")
        });
        for (k, v) in &extra {
            assert_eq!(
                d.get(k).expect("get"),
                Some(v.clone()),
                "cycle {cycle}: a write from an earlier cycle is gone",
            );
        }
        let k = format!("cycle_{cycle:03}").into_bytes();
        let v = format!("value_{cycle:03}").into_bytes();
        d.put(&k, &v).expect("put");
        extra.push((k, v));
        d.close().expect("close");
        drop(d);
    }

    let survivors = fault::find_ssts(&db);
    println!(
        "convergence: 6 reopen cycles, {} sst file(s) left, orphan(s) {:?}",
        survivors.len(),
        survivors
            .iter()
            .map(|p| (
                p.file_name().map(|n| n.to_string_lossy().into_owned()),
                fault::file_len(p)
            ))
            .collect::<Vec<_>>(),
    );
}
