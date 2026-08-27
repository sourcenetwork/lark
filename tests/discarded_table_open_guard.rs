//! Adversarial probe for the discarded-table open guard (the discarded-table open guard) under the
//! tear modes a real filesystem produces that are not `Truncate`.
//!
//! The guard dismisses an unreferenced table only when the file proves
//! it holds nothing: it is zero bytes long, or its footer parses and its
//! index block is empty. Everything else counts, and the open refuses.
//!
//! A power cut inside the first flush does not always leave a
//! zero-length file. ext4 with the blocks already allocated leaves zeros
//! at full length, and a device that tears at sector granularity leaves
//! a prefix followed by unrelated bytes. Both keep every acknowledged
//! write in the fsynced WAL, so ideally both would open. Neither can be
//! told apart from a *complete* table whose tail a lost or misdirected
//! write zeroed or garbled, and the manifest that would have said which
//! is the thing that is damaged.
//!
//! So the two sides of the trade are:
//!
//! * dismiss the file and the open silently serves a database missing
//!   everything a real table held;
//! * count it and the open refuses loudly, names the file, deletes
//!   nothing, and the operator moves it aside and recovers every
//!   acknowledged write from the WAL.
//!
//! The second is the standing choice, because it is the recoverable one.
//! `a_first_flush_cut_that_leaves_an_unprovable_orphan_refuses_and_salvages`
//! measures its cost rather than hiding it, and is the regression gate
//! for that contract.

mod common;

use common::fault::{
    self, ChildOutcome, ChildSpec, CrashRun, CutPoint, Phase, PowerLossOptions, Recovery, TearMode,
    Trigger,
};
use lark_kv::{DurabilityMode, Options};
use std::time::Duration;
use tempfile::TempDir;

#[test]
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

/// A tear that leaves nothing readable behind must still open, with
/// every acknowledged write intact.
fn probe_opens(tear: TearMode) {
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
        Recovery::RefusedToOpen(e) => panic!("{tear:?} must not block the open: {e}"),
    }
}

/// A tear that leaves bytes the guard cannot prove empty must refuse,
/// and the refusal must be recoverable: the error names the orphan, the
/// orphan is left byte-for-byte on disk, and moving it aside brings back
/// every acknowledged write.
fn probe_refuses_and_salvages(tear: TearMode) {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().join("db");
    let out = cut_first_flush(&db, tear);
    let orphans = fault::find_ssts(&db);
    let before: Vec<u64> = orphans.iter().map(|p| fault::file_len(p)).collect();
    assert!(
        !orphans.is_empty() && before.iter().all(|n| *n > 0),
        "{tear:?} must leave a non-empty orphan for this probe, got {orphans:?} {before:?}",
    );
    assert!(out.acked_count() > 0, "no write was acknowledged");

    let err = match fault::recover_and_validate(&db, opts(8 * 1024), &out.history) {
        Recovery::RefusedToOpen(e) => e.to_string(),
        Recovery::Recovered(_) => panic!(
            "{tear:?}: an orphan that cannot be proved empty must not be dismissed; \
             dismissing it opens the database without whatever the table held",
        ),
    };
    for orphan in &orphans {
        let name = orphan.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            err.contains(&name),
            "{tear:?}: the refusal must name the file it is holding the database for: {err}",
        );
    }

    let after: Vec<u64> = orphans.iter().map(|p| fault::file_len(p)).collect();
    assert_eq!(
        before, after,
        "{tear:?}: a refused open must leave every orphan exactly as it found it",
    );

    let salvage = tmp.path().join("salvage");
    fault::copy_tree(&db, &salvage);
    for sst in fault::find_ssts(&salvage) {
        std::fs::remove_file(&sst).unwrap();
    }
    match fault::recover_and_validate(&salvage, opts(8 * 1024), &out.history) {
        Recovery::Recovered(r) => {
            fault::assert_acked_survived(&r, &out.acked);
            println!(
                "{tear:?}: refused with {} acked write(s) behind it, all {} recovered after \
                 moving the orphan aside",
                out.acked_count(),
                r.k,
            );
        }
        Recovery::RefusedToOpen(e) => panic!("{tear:?}: the salvage path also refused: {e}"),
    }
}

#[test]
fn a_first_flush_cut_that_tears_a_sector_keeps_every_acknowledged_write() {
    probe_opens(TearMode::TornSector);
}

/// The measured cost of the standing choice, recorded rather than
/// hidden.
///
/// `TearMode::Zero` is the ext4 delayed-allocation shape and
/// `TearMode::Garbage` is the harness's harshest synthetic one. Both
/// leave an orphan the guard cannot prove empty, so both refuse, and
/// every acknowledged write sits intact in the fsynced WAL behind a
/// database that will not open until an operator moves the file aside.
///
/// That cost is real and this test measures it. It is taken because the
/// alternative is worse and silent: a rule that dismissed either shape
/// would also dismiss a *complete* table whose final block a lost write
/// zeroed, and a wiped manifest next to one of those would open and
/// serve a database missing everything that table held, with no error.
/// `adv_review_g28_guard::a_real_table_whose_tail_block_was_zeroed_is_dismissed_at_sector_and_block_sizes`
/// is the probe that holds that door shut, and
/// `open_and_corruption::a_wiped_manifest_next_to_an_unreadable_orphan_table_refuses_to_open`
/// and
/// `adversarial_open_guard::a_truncated_but_non_empty_orphan_table_refuses_at_every_length`
/// are the two that hold it shut for the truncated and unreadable
/// shapes.
#[test]
fn a_first_flush_cut_that_leaves_an_unprovable_orphan_refuses_and_salvages() {
    probe_refuses_and_salvages(TearMode::Zero);
    probe_refuses_and_salvages(TearMode::Garbage);
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
