//! The phases that each need their own process: reopen, the kill, the
//! recovery, the forced compaction, and the last reopen after it.

use regolith::{Db, Options, WriteOptions};

use crate::data;
use crate::report::Report;
use crate::verify;

/// Exit status the crash phase leaves behind. Anything else from that
/// phase means it returned normally, which would defeat the test.
pub const CRASH_EXIT_CODE: i32 = 9;

/// Phase `reopen`: a fresh process reads the database the lifecycle
/// phase closed and proves every surviving byte round-tripped through
/// the on-disk format.
pub fn reopen(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before reopen");
    let db = Db::open(dir, opts).map_err(|e| format!("reopen failed: {e}"))?;
    report.stage("after reopen");
    verify::verify_final_state(&db, report, "reopen")?;
    report.stage("after full verification");
    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    report.stage("after close");
    Ok(report.finish("reopen"))
}

/// Phase `crash`: write, acknowledge, and vanish.
///
/// Never returns. The database is still open and no `close` and no
/// destructor runs, which is the point: whatever the next process can
/// read had to come out of the write-ahead log.
pub fn crash(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before crash-phase open");
    let db = Db::open(dir, opts).map_err(|e| format!("open failed: {e}"))?;
    report.stage("after open");

    let sync = WriteOptions::sync();
    for i in 0..data::CRASH_SYNC {
        let key = data::crash_sync_key(i);
        db.put_opt(&sync, &key, &data::value(i, data::GEN_CRASH))
            .map_err(|e| format!("sync put failed: {e}"))?;
    }
    report.check_u64("crash.acked_sync", data::CRASH_SYNC);
    report.stage("after fsynced writes");

    for i in 0..data::CRASH_ASYNC {
        let key = data::crash_async_key(i);
        db.put(&key, &data::value(i, data::GEN_CRASH))
            .map_err(|e| format!("async put failed: {e}"))?;
    }
    report.check_u64("crash.acked_async", data::CRASH_ASYNC);
    report.stage("after unsynced writes");
    report.finish("crash");
    println!("NOTE  exiting {CRASH_EXIT_CODE} with the database still open");

    // The whole point of the phase. No close, no unwinding, no
    // destructor: the next process must rebuild from the log alone.
    std::process::exit(CRASH_EXIT_CODE);
}

/// Phase `recover`: reopen after the kill and account for every write
/// the crashed process had acknowledged.
pub fn recover(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before recovery open");
    let db = Db::open(dir, opts).map_err(|e| format!("recovery open failed: {e}"))?;
    report.stage("after recovery open");

    verify::verify_final_state(&db, report, "recover")?;
    verify_crash_keys(&db, report, "recover")?;
    report.stage("after full verification");

    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    report.stage("after close");
    Ok(report.finish("recover"))
}

/// Phase `compact`: force every pending compaction, then prove the
/// database still holds exactly what it held before.
pub fn compact(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before compaction open");
    let db = Db::open(dir, opts).map_err(|e| format!("open failed: {e}"))?;
    report.stage("after open");

    if let Some(levels) = db.get_property("regolith.levelstats") {
        report.property("levelstats.before", &levels);
    }

    db.flush().map_err(|e| format!("flush failed: {e}"))?;
    report.stage("after flush");

    db.compact_range(None, None)
        .map_err(|e| format!("compact_range failed: {e}"))?;
    report.stage("after compact_range");

    let mut steps = 0u64;
    // Bounded so a scheduler that always reports work cannot spin
    // forever; a healthy database drains long before this.
    while steps < 256 {
        match db
            .compact_step()
            .map_err(|e| format!("compact_step failed: {e}"))?
        {
            true => steps += 1,
            false => break,
        }
    }
    report.check_u64("compact.extra_steps", steps);
    report.stage("after compact_step drain");

    if let Some(levels) = db.get_property("regolith.levelstats") {
        report.property("levelstats.after", &levels);
    }

    verify::verify_final_state(&db, report, "compact")?;
    verify_crash_keys(&db, report, "compact")?;
    report.stage("after full verification");

    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    report.stage("after close");
    Ok(report.finish("compact"))
}

/// Phase `final`: one more fresh process, after the compaction
/// rewrote every file, reading the whole database back.
pub fn final_check(
    dir: &std::path::Path,
    opts: Options,
    report: &mut Report,
) -> Result<bool, String> {
    report.stage("baseline, before final open");
    let db = Db::open(dir, opts).map_err(|e| format!("final open failed: {e}"))?;
    report.stage("after final open");
    verify::verify_final_state(&db, report, "final")?;
    verify_crash_keys(&db, report, "final")?;
    report.stage("after full verification");
    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    report.stage("after close");
    Ok(report.finish("final"))
}

/// Account for the records the crashed process had acknowledged.
///
/// The fsynced records are a hard contract and a shortfall is a
/// failure. The unsynced ones are recorded exactly as observed too:
/// `DurabilityMode::Eventual` documents a process crash as survivable
/// because the record reached the log, so a shortfall there is a real
/// finding rather than a tolerance.
fn verify_crash_keys(db: &Db, report: &mut Report, label: &str) -> Result<(), String> {
    let mut sync_ok = 0u64;
    let mut sync_wrong = 0u64;
    for i in 0..data::CRASH_SYNC {
        let key = data::crash_sync_key(i);
        match db
            .get(&key)
            .map_err(|e| format!("get {} failed: {e}", String::from_utf8_lossy(&key)))?
        {
            Some(v) if v == data::value(i, data::GEN_CRASH) => sync_ok += 1,
            Some(_) => sync_wrong += 1,
            None => {}
        }
    }
    report.expect_u64(
        &format!("{label}.crash_sync.recovered"),
        sync_ok,
        data::CRASH_SYNC,
    );
    report.expect_u64(&format!("{label}.crash_sync.corrupt"), sync_wrong, 0);

    let mut async_ok = 0u64;
    let mut async_wrong = 0u64;
    for i in 0..data::CRASH_ASYNC {
        let key = data::crash_async_key(i);
        match db
            .get(&key)
            .map_err(|e| format!("get {} failed: {e}", String::from_utf8_lossy(&key)))?
        {
            Some(v) if v == data::value(i, data::GEN_CRASH) => async_ok += 1,
            Some(_) => async_wrong += 1,
            None => {}
        }
    }
    report.expect_u64(
        &format!("{label}.crash_async.recovered"),
        async_ok,
        data::CRASH_ASYNC,
    );
    report.expect_u64(&format!("{label}.crash_async.corrupt"), async_wrong, 0);

    let mut iter = db.iter();
    let walk = verify::walk_forward(&mut iter, b"crash/")?;
    report.expect_u64(
        &format!("{label}.crash_walk.count"),
        walk.count,
        data::CRASH_SYNC + data::CRASH_ASYNC,
    );
    report.expect_u64(
        &format!("{label}.crash_walk.order_violations"),
        walk.order_violations,
        0,
    );
    Ok(())
}
