//! Phases that deliberately leave the database in a state nobody
//! designed for, and a report-only phase to describe what came back.
//!
//! `survey` asserts nothing. That is the point: after a torn log or a
//! kill in the middle of a compaction there is no single right answer
//! to assert, but there is exactly one answer a given build should
//! give, and the native and the wasm build must give the same one.
//! Any difference between the two transcripts is a platform bug.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lark_kv::{CompactionDecision, CompactionFilter, Db, Options};

use crate::data;
use crate::report::Report;
use crate::restart::CRASH_EXIT_CODE;
use crate::verify;

/// Phase `survey`: open the database, describe everything readable,
/// assert nothing.
///
/// A failed open is described too, rather than aborting, so that a
/// database only one of the two platforms can open shows up as a
/// difference in the transcript instead of as two dissimilar errors.
pub fn survey(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before survey open");
    let db = match Db::open(dir, opts) {
        Ok(db) => {
            report.check("survey.open", "ok");
            db
        }
        Err(err) => {
            report.check("survey.open", &format!("error: {err}"));
            report.stage("after failed open");
            return Ok(report.finish("survey"));
        }
    };
    report.stage("after survey open");

    let mut present = 0u64;
    let mut wrong = 0u64;
    for i in 0..data::RECORDS {
        if let Some(v) = db
            .get(&data::bulk_key(i))
            .map_err(|e| format!("get failed: {e}"))?
        {
            match data::expect_final(i) {
                data::Expect::Present(idx, gen) if v == data::value(idx, gen) => present += 1,
                _ => wrong += 1,
            }
        }
    }
    report.check_u64("survey.bulk.matching", present);
    report.check_u64("survey.bulk.unexpected", wrong);

    let mut late = 0u64;
    for j in 0..data::LATE_RECORDS {
        if db
            .get(&data::late_key(j))
            .map_err(|e| format!("get failed: {e}"))?
            .as_deref()
            == Some(data::value(j, data::GEN_LATE).as_slice())
        {
            late += 1;
        }
    }
    report.check_u64("survey.late.matching", late);

    let mut sync_ok = 0u64;
    let mut sync_first_missing = data::CRASH_SYNC;
    for i in 0..data::CRASH_SYNC {
        if db
            .get(&data::crash_sync_key(i))
            .map_err(|e| format!("get failed: {e}"))?
            .as_deref()
            == Some(data::value(i, data::GEN_CRASH).as_slice())
        {
            sync_ok += 1;
        } else if sync_first_missing == data::CRASH_SYNC {
            sync_first_missing = i;
        }
    }
    report.check_u64("survey.crash_sync.matching", sync_ok);
    report.check_u64("survey.crash_sync.first_missing", sync_first_missing);

    let mut async_ok = 0u64;
    let mut async_first_missing = data::CRASH_ASYNC;
    for i in 0..data::CRASH_ASYNC {
        if db
            .get(&data::crash_async_key(i))
            .map_err(|e| format!("get failed: {e}"))?
            .as_deref()
            == Some(data::value(i, data::GEN_CRASH).as_slice())
        {
            async_ok += 1;
        } else if async_first_missing == data::CRASH_ASYNC {
            async_first_missing = i;
        }
    }
    report.check_u64("survey.crash_async.matching", async_ok);
    report.check_u64("survey.crash_async.first_missing", async_first_missing);

    let mut iter = db.iter();
    let walk = verify::walk_forward(&mut iter, b"")?;
    report.check_u64("survey.walk.count", walk.count);
    report.check_digest("survey.walk.digest", walk.digest);
    report.check_u64("survey.walk.order_violations", walk.order_violations);

    if let Some(levels) = db.get_property("lark.levelstats") {
        report.property("survey.levelstats", &levels);
    }
    report.stage("after survey");

    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    Ok(report.finish("survey"))
}

/// Phase `crash-compact`: exit from inside a compaction.
///
/// A compaction filter counts the entries it is shown and takes the
/// process out at a fixed count, so the kill lands while a compaction
/// output file is half written and the manifest still describes the
/// inputs. The next open has to ignore the orphan and read the old
/// files. Never returns.
pub fn crash_compact(
    dir: &std::path::Path,
    mut opts: Options,
    budget: u64,
    report: &mut Report,
) -> Result<bool, String> {
    report.stage("baseline, before compaction-kill open");
    opts.compaction_filter = Some(Arc::new(ExitAfter::new(budget)));
    let db = Db::open(dir, opts).map_err(|e| format!("open failed: {e}"))?;
    report.stage("after open");
    report.check_u64("crash_compact.budget", budget);

    db.flush().map_err(|e| format!("flush failed: {e}"))?;
    let outcome = match db.compact_range(None, None) {
        Ok(()) => "compaction finished without reaching the budget".to_string(),
        Err(err) => format!("compaction returned an error: {err}"),
    };
    // Reaching here means the filter never fired, which makes the
    // phase a no-op rather than a crash. Say so instead of exiting 9
    // and letting a driver read it as a successful kill.
    report.check("crash_compact.outcome", &outcome);
    report.stage("after compaction");
    Ok(report.finish("crash-compact"))
}

/// A compaction filter that takes the process out after it has been
/// shown `budget` entries.
struct ExitAfter {
    remaining: AtomicU64,
}

impl ExitAfter {
    fn new(budget: u64) -> Self {
        Self {
            remaining: AtomicU64::new(budget),
        }
    }
}

impl CompactionFilter for ExitAfter {
    fn filter(&self, _level: usize, _key: &[u8], _value: &[u8]) -> CompactionDecision {
        let left = self.remaining.load(Ordering::Relaxed);
        if left == 0 {
            return CompactionDecision::Keep;
        }
        self.remaining.store(left - 1, Ordering::Relaxed);
        if left == 1 {
            println!("NOTE  compaction filter exiting {CRASH_EXIT_CODE} mid-compaction");
            std::process::exit(CRASH_EXIT_CODE);
        }
        CompactionDecision::Keep
    }

    fn name(&self) -> &'static str {
        "conformance-exit-after"
    }
}
