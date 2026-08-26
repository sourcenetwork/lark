//! The full database lifecycle, phase by phase.
//!
//! The point of this file is that it does not stop at "did not trap".
//! Every phase asserts the bytes it reads, the final phase reopens the
//! database from disk and verifies the entire dataset against
//! [`crate::dataset`], and any mismatch is an error the caller turns
//! into a non-zero exit.

use std::path::Path;

use lark_kv::{Db, Options, WriteBatch};

use crate::capabilities;
use crate::check::{expect_absent, expect_value, show};
use crate::dataset::{self, BATCH_RECORDS, OVERWRITE_INDEX, OVERWRITE_TAG, RANGE_DELETE_LEN};
use crate::host::Finding;
use crate::queries;
use crate::report::Reporter;

/// Smoke keys written and then deleted in the point-operation phase.
const SMOKE_KEYS: u64 = 3;

/// Run open, write, read, scan, snapshot, iterate, compact, close,
/// reopen, and full read-back against `path`.
///
/// `options` is called once per open so the reopen uses an identical
/// configuration built from scratch, which is what a host process
/// would do. `host` carries what [`crate::host::probe`] measured, so
/// the open phase can check it against what the database claims.
pub fn run(
    path: &Path,
    options: &dyn Fn() -> Options,
    records: u64,
    host: &[Finding],
    reporter: &mut Reporter,
) -> Result<(), String> {
    if records < dataset::MIN_RECORDS {
        return Err(format!(
            "--records must be at least {}; below that the deleted, \
             range-deleted, and overwritten indices collide",
            dataset::MIN_RECORDS
        ));
    }

    let db = open(path, options())?;
    reporter.pass("open");

    reporter.note(&capabilities::describe(db.capabilities()));
    capabilities::check(&db, host)?;
    reporter.pass("capabilities match the host");

    point_ops(&db)?;
    reporter.pass("put/get/delete");

    write_batch(&db)?;
    reporter.pass("write batch");

    bulk_write(&db, records)?;
    reporter.pass("bulk write");

    mutate(&db, records)?;
    reporter.pass("overwrite/delete/delete_range");

    point_reads(&db, records)?;
    reporter.pass("point reads");

    queries::scan(&db, records)?;
    reporter.pass("scan");

    queries::scan_pages(&db, records)?;
    reporter.pass("scan_page");

    queries::snapshot(&db, records)?;
    reporter.pass("snapshot isolation");

    queries::iterate(&db, records)?;
    reporter.pass("iterator seek + walk");

    db.compact_range(None, None)
        .map_err(|e| format!("compact_range failed: {e}"))?;
    reporter.pass("flush + compact_range");

    point_reads(&db, records)?;
    reporter.pass("point reads after compaction");

    db.close().map_err(|e| format!("close failed: {e}"))?;
    drop(db);
    reporter.pass("close");

    let db = open(path, options())?;
    reporter.pass("reopen");

    read_back(&db, records)?;
    reporter.pass("read back after reopen");

    db.close()
        .map_err(|e| format!("close after reopen failed: {e}"))?;
    drop(db);
    reporter.pass("close after reopen");

    Ok(())
}

fn open(path: &Path, opts: Options) -> Result<Db, String> {
    Db::open(path, opts).map_err(|e| format!("open {} failed: {e}", path.display()))
}

fn point_ops(db: &Db) -> Result<(), String> {
    for i in 0..SMOKE_KEYS {
        let key = dataset::smoke_key(i);
        db.put(&key, &dataset::value(i))
            .map_err(|e| format!("put {} failed: {e}", show(&key)))?;
        expect_value(db, &key, &dataset::value(i))?;
    }

    // Overwrite in place, then delete: the tombstone must shadow both
    // versions, not just the newer one.
    let key = dataset::smoke_key(0);
    db.put(&key, &dataset::value(OVERWRITE_TAG))
        .map_err(|e| format!("overwrite {} failed: {e}", show(&key)))?;
    expect_value(db, &key, &dataset::value(OVERWRITE_TAG))?;

    for i in 0..SMOKE_KEYS {
        let key = dataset::smoke_key(i);
        db.delete(&key)
            .map_err(|e| format!("delete {} failed: {e}", show(&key)))?;
        expect_absent(db, &key, "deleted in the point-operation phase")?;
    }
    Ok(())
}

fn write_batch(db: &Db) -> Result<(), String> {
    let mut batch = WriteBatch::new();
    for i in 0..BATCH_RECORDS {
        batch.put(&dataset::batch_key(i), &dataset::value(i));
    }
    batch.delete(&dataset::batch_key(dataset::BATCH_DELETED));
    db.write(batch)
        .map_err(|e| format!("write batch failed: {e}"))?;

    for i in 0..BATCH_RECORDS {
        let key = dataset::batch_key(i);
        match dataset::batch_expected(i) {
            Some(want) => expect_value(db, &key, &want)?,
            None => expect_absent(db, &key, "deleted by the same batch that wrote it")?,
        }
    }
    Ok(())
}

fn bulk_write(db: &Db, records: u64) -> Result<(), String> {
    for i in 0..records {
        let key = dataset::bulk_key(i);
        db.put(&key, &dataset::value(i))
            .map_err(|e| format!("bulk put {} failed: {e}", show(&key)))?;
    }
    Ok(())
}

fn mutate(db: &Db, records: u64) -> Result<(), String> {
    let overwritten = dataset::bulk_key(OVERWRITE_INDEX);
    db.put(
        &overwritten,
        &dataset::value(OVERWRITE_INDEX + OVERWRITE_TAG),
    )
    .map_err(|e| format!("overwrite {} failed: {e}", show(&overwritten)))?;

    let deleted = dataset::bulk_key(records / 2);
    db.delete(&deleted)
        .map_err(|e| format!("delete {} failed: {e}", show(&deleted)))?;

    let start = dataset::bulk_key(records / 4);
    let end = dataset::bulk_key(records / 4 + RANGE_DELETE_LEN);
    db.delete_range(&start, &end)
        .map_err(|e| format!("delete_range failed: {e}"))?;
    Ok(())
}

fn point_reads(db: &Db, records: u64) -> Result<(), String> {
    for i in 0..records {
        let key = dataset::bulk_key(i);
        match dataset::bulk_expected(i, records) {
            Some(want) => expect_value(db, &key, &want)?,
            None => expect_absent(db, &key, "removed by delete or delete_range")?,
        }
    }
    Ok(())
}

fn read_back(db: &Db, records: u64) -> Result<(), String> {
    point_reads(db, records)?;

    for i in 0..BATCH_RECORDS {
        let key = dataset::batch_key(i);
        match dataset::batch_expected(i) {
            Some(want) => expect_value(db, &key, &want)?,
            None => expect_absent(db, &key, "deleted by the batch before close")?,
        }
    }

    for i in 0..SMOKE_KEYS {
        expect_absent(db, &dataset::smoke_key(i), "deleted before close")?;
    }

    expect_value(
        db,
        &dataset::post_snapshot_key(),
        &dataset::value(OVERWRITE_TAG),
    )?;

    queries::scan(db, records)?;
    queries::scan_pages(db, records)?;
    Ok(())
}
