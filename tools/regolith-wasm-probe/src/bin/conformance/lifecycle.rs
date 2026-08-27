//! Phase `write`: the single-process lifecycle.
//!
//! Open, bulk load, read every record back, delete a third, prove the
//! survivors are untouched, walk ascending and descending, page the
//! same range through two other read APIs, then hold a snapshot open
//! while overwrites and deletes land on top of it and prove the
//! snapshot did not move. Ends with a clean `close`, so whatever the
//! next phase reads came off disk.

use regolith::{Db, Options, WriteBatch};

use crate::data::{self, Expect};
use crate::report::Report;
use crate::verify;

/// Entries materialized per `scan_page` call.
const PAGE: usize = 256;

/// Run the lifecycle against a fresh database at `dir`.
pub fn run(dir: &std::path::Path, opts: Options, report: &mut Report) -> Result<bool, String> {
    report.stage("baseline, before open");

    let db = Db::open(dir, opts).map_err(|e| format!("open failed: {e}"))?;
    report.stage("after open");
    report.check("capabilities", &format!("{:?}", db.capabilities()));

    bulk_load(&db, report)?;
    report.stage("after 5000 puts");
    verify::point_reads_bulk(&db, report, data::expect_after_load, "load.readback")?;
    report.stage("after read-back of every record");

    delete_every_third(&db, report)?;
    report.stage("after deleting every third key");
    verify::point_reads_bulk(&db, report, data::expect_after_delete, "delete.readback")?;

    ordered_walks(&db, report)?;
    report.stage("after ascending and descending walks");

    bounded_scan(&db, report)?;
    paged_scan(&db, report)?;
    report.stage("after scan and scan_page");

    snapshot_isolation(&db, report)?;
    report.stage("after snapshot isolation");

    verify::point_reads_bulk(&db, report, data::expect_final, "final.bulk")?;
    verify::point_reads_late(&db, report, "final.late")?;

    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    report.stage("after close");
    Ok(report.finish("write"))
}

fn bulk_load(db: &Db, report: &mut Report) -> Result<(), String> {
    let mut written = 0u64;
    let mut bytes = 0u64;
    for i in 0..data::RECORDS {
        let key = data::bulk_key(i);
        let value = data::value(i, data::GEN_BULK);
        bytes += value.len() as u64;
        db.put(&key, &value)
            .map_err(|e| format!("put {} failed: {e}", String::from_utf8_lossy(&key)))?;
        written += 1;
    }
    report.expect_u64("load.written", written, data::RECORDS);
    report.check_u64("load.value_bytes", bytes);
    Ok(())
}

fn delete_every_third(db: &Db, report: &mut Report) -> Result<(), String> {
    let mut batch = WriteBatch::new();
    let mut deleted = 0u64;
    for i in (0..data::RECORDS).step_by(3) {
        batch.delete(&data::bulk_key(i));
        deleted += 1;
    }
    db.write(batch)
        .map_err(|e| format!("delete batch failed: {e}"))?;
    report.check_u64("delete.issued", deleted);
    Ok(())
}

fn ordered_walks(db: &Db, report: &mut Report) -> Result<(), String> {
    let (fwd_digest, fwd_count) = data::expected_bulk_digest(data::expect_after_delete);
    let mut iter = db.iter();
    let forward = verify::walk_forward(&mut iter, b"key/")?;
    forward.record(report, "walk.forward", fwd_count, fwd_digest);

    let (rev_digest, rev_count) = data::expected_bulk_digest_reverse(data::expect_after_delete);
    let mut iter = db.iter();
    let backward = verify::walk_backward(&mut iter, b"key/")?;
    backward.record(report, "walk.backward", rev_count, rev_digest);
    Ok(())
}

fn bounded_scan(db: &Db, report: &mut Report) -> Result<(), String> {
    let start = data::bulk_key(1_000);
    let end = data::bulk_key(2_000);
    let entries = db
        .scan(Some(&start), Some(&end))
        .map_err(|e| format!("scan failed: {e}"))?;

    let mut expected = data::Digest::new();
    let mut expected_count = 0u64;
    for i in 1_000..2_000 {
        if let Expect::Present(idx, gen) = data::expect_after_delete(i) {
            expected.entry(&data::bulk_key(i), &data::value(idx, gen));
            expected_count += 1;
        }
    }

    let mut actual = data::Digest::new();
    let mut violations = 0u64;
    let mut previous: Option<&[u8]> = None;
    for (key, value) in &entries {
        if let Some(prev) = previous {
            if prev >= key.as_slice() {
                violations += 1;
            }
        }
        previous = Some(key);
        actual.entry(key, value);
    }
    report.expect_u64("scan.count", entries.len() as u64, expected_count);
    report.expect_digest("scan.digest", actual.finish(), expected.finish());
    report.expect_u64("scan.order_violations", violations, 0);
    Ok(())
}

fn paged_scan(db: &Db, report: &mut Report) -> Result<(), String> {
    let lower = b"key/".to_vec();
    let upper = b"key0".to_vec();
    let mut cursor = Some(lower);
    let mut pages = 0u64;
    let mut count = 0u64;
    let mut digest = data::Digest::new();
    let mut violations = 0u64;
    let mut previous: Option<Vec<u8>> = None;

    while let Some(start) = cursor {
        let page = db
            .scan_page(Some(&start), Some(&upper), PAGE)
            .map_err(|e| format!("scan_page failed: {e}"))?;
        pages += 1;
        for (key, value) in &page.entries {
            if let Some(prev) = &previous {
                if prev.as_slice() >= key.as_slice() {
                    violations += 1;
                }
            }
            previous = Some(key.clone());
            digest.entry(key, value);
            count += 1;
        }
        cursor = page.next_start;
    }

    let (expected_digest, expected_count) = data::expected_bulk_digest(data::expect_after_delete);
    report.expect_u64("scan_page.count", count, expected_count);
    report.expect_digest("scan_page.digest", digest.finish(), expected_digest);
    report.expect_u64("scan_page.order_violations", violations, 0);
    report.check_u64("scan_page.pages", pages);
    Ok(())
}

fn snapshot_isolation(db: &Db, report: &mut Report) -> Result<(), String> {
    let snapshot = db.snapshot();
    let (pinned_digest, pinned_count) = data::expected_bulk_digest(data::expect_after_delete);

    for j in 0..data::LATE_RECORDS {
        let key = data::late_key(j);
        db.put(&key, &data::value(j, data::GEN_LATE))
            .map_err(|e| format!("late put failed: {e}"))?;
    }
    for i in data::OVERWRITE_LO..data::OVERWRITE_HI {
        if i % 3 == 0 {
            continue;
        }
        db.put(&data::bulk_key(i), &data::value(i, data::GEN_OVERWRITE))
            .map_err(|e| format!("overwrite failed: {e}"))?;
    }
    for i in data::LATE_DELETE_LO..data::LATE_DELETE_HI {
        if i % 3 == 0 {
            continue;
        }
        db.delete(&data::bulk_key(i))
            .map_err(|e| format!("late delete failed: {e}"))?;
    }

    let mut stale_late = 0u64;
    for j in (0..data::LATE_RECORDS).step_by(7) {
        if snapshot
            .get(&data::late_key(j))
            .map_err(|e| format!("snapshot get failed: {e}"))?
            .is_some()
        {
            stale_late += 1;
        }
    }
    report.expect_u64("snapshot.sees_late_writes", stale_late, 0);

    let mut held_overwrite = 0u64;
    for i in data::OVERWRITE_LO..data::OVERWRITE_HI {
        if i % 3 == 0 {
            continue;
        }
        let got = snapshot
            .get(&data::bulk_key(i))
            .map_err(|e| format!("snapshot get failed: {e}"))?;
        if got.as_deref() == Some(data::value(i, data::GEN_BULK).as_slice()) {
            held_overwrite += 1;
        }
    }
    report.check_u64("snapshot.holds_pre_overwrite_value", held_overwrite);

    let mut held_delete = 0u64;
    for i in data::LATE_DELETE_LO..data::LATE_DELETE_HI {
        if i % 3 == 0 {
            continue;
        }
        let got = snapshot
            .get(&data::bulk_key(i))
            .map_err(|e| format!("snapshot get failed: {e}"))?;
        if got.as_deref() == Some(data::value(i, data::GEN_BULK).as_slice()) {
            held_delete += 1;
        }
    }
    report.check_u64("snapshot.holds_deleted_value", held_delete);

    let mut iter = snapshot.iter();
    let walk = verify::walk_forward(&mut iter, b"key/")?;
    walk.record(report, "snapshot.walk", pinned_count, pinned_digest);

    let mut iter = snapshot.iter();
    let late_walk = verify::walk_forward(&mut iter, b"late/")?;
    report.expect_u64("snapshot.late_walk.count", late_walk.count, 0);

    drop(snapshot);

    let mut live_overwrite = 0u64;
    for i in data::OVERWRITE_LO..data::OVERWRITE_HI {
        if i % 3 == 0 {
            continue;
        }
        let got = db
            .get(&data::bulk_key(i))
            .map_err(|e| format!("get failed: {e}"))?;
        if got.as_deref() == Some(data::value(i, data::GEN_OVERWRITE).as_slice()) {
            live_overwrite += 1;
        }
    }
    report.check_u64("live.sees_overwrite", live_overwrite);
    Ok(())
}
