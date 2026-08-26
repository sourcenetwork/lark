//! The read-shaped phases: range scan, pagination, snapshot
//! isolation, and iterator navigation.
//!
//! Split out of `lifecycle.rs` to keep both files inside the project's
//! file-size guideline; `lifecycle::run` calls straight into these.

use lark_kv::Db;

use crate::check::{expect_value, show};
use crate::dataset::{self, OVERWRITE_TAG};

/// Entries requested per page in the pagination phase.
const PAGE_LIMIT: usize = 64;

/// Bulk records the iterator phase walks forward from the first key.
const ITER_WALK: usize = 50;

pub fn scan(db: &Db, records: u64) -> Result<(), String> {
    // A band that straddles the range-deleted hole, so the scan has to
    // skip it rather than merely return a contiguous run.
    let lo_index = records / 8;
    let hi_index = (records / 2).min(records);
    let lo = dataset::bulk_key(lo_index);
    let hi = dataset::bulk_key(hi_index);

    let got = db
        .scan(Some(&lo), Some(&hi))
        .map_err(|e| format!("scan failed: {e}"))?;

    let want: Vec<(Vec<u8>, Vec<u8>)> = (lo_index..hi_index)
        .filter_map(|i| dataset::bulk_expected(i, records).map(|v| (dataset::bulk_key(i), v)))
        .collect();

    if got.len() != want.len() {
        return Err(format!(
            "scan [{}, {}) returned {} entries, expected {}",
            show(&lo),
            show(&hi),
            got.len(),
            want.len()
        ));
    }
    for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.0 != w.0 {
            return Err(format!(
                "scan entry {index}: key {}, expected {}",
                show(&g.0),
                show(&w.0)
            ));
        }
        dataset::check(&show(&w.0), &w.1, &g.1)?;
    }
    Ok(())
}

pub fn scan_pages(db: &Db, records: u64) -> Result<(), String> {
    let lo = dataset::bulk_key(0);
    let hi = dataset::bulk_key(records);
    let mut start = Some(lo.clone());
    let mut seen: u64 = 0;
    let mut pages = 0u64;

    while let Some(from) = start {
        let page = db
            .scan_page(Some(&from), Some(&hi), PAGE_LIMIT)
            .map_err(|e| format!("scan_page failed: {e}"))?;
        if page.entries.is_empty() && page.next_start.is_some() {
            return Err("scan_page returned an empty page with more to come".to_string());
        }
        seen += page.entries.len() as u64;
        pages += 1;
        if pages > (records / PAGE_LIMIT as u64) + 4 {
            return Err("scan_page did not terminate within the expected page count".to_string());
        }
        start = page.next_start;
    }

    let want = (0..records)
        .filter(|&i| dataset::bulk_expected(i, records).is_some())
        .count() as u64;
    if seen != want {
        return Err(format!("scan_page walked {seen} entries, expected {want}"));
    }
    Ok(())
}

pub fn snapshot(db: &Db, records: u64) -> Result<(), String> {
    let snap = db.snapshot();
    let key = dataset::post_snapshot_key();
    db.put(&key, &dataset::value(OVERWRITE_TAG))
        .map_err(|e| format!("post-snapshot put failed: {e}"))?;

    match snap
        .get(&key)
        .map_err(|e| format!("snapshot get failed: {e}"))?
    {
        None => {}
        Some(v) => {
            return Err(format!(
                "snapshot saw a key written after it was taken ({} bytes)",
                v.len()
            ))
        }
    }
    expect_value(db, &key, &dataset::value(OVERWRITE_TAG))?;

    // The snapshot must still serve data written before it was taken.
    let live = dataset::first_live_at_or_after(0, records)
        .ok_or_else(|| "no live bulk record to read through the snapshot".to_string())?;
    let want = dataset::bulk_expected(live, records)
        .ok_or_else(|| format!("bulk record {live} is reported live but has no expected value"))?;
    let label = show(&dataset::bulk_key(live));
    match snap
        .get(&dataset::bulk_key(live))
        .map_err(|e| format!("snapshot get {label} failed: {e}"))?
    {
        Some(got) => dataset::check(&label, &want, &got)?,
        None => return Err(format!("snapshot lost pre-snapshot record {label}")),
    }
    Ok(())
}

pub fn iterate(db: &Db, records: u64) -> Result<(), String> {
    let mut it = db.iter();

    it.seek_to_first();
    let first = it
        .key()
        .ok_or_else(|| "iterator invalid after seek_to_first".to_string())?
        .to_vec();
    let want_first = dataset::batch_key(0);
    if first != want_first {
        return Err(format!(
            "seek_to_first gave {}, expected {}",
            show(&first),
            show(&want_first)
        ));
    }

    let mut previous = first;
    for step in 0..ITER_WALK {
        it.next();
        if !it.valid() {
            return Err(format!("iterator went invalid after {step} forward steps"));
        }
        let key = it
            .key()
            .ok_or_else(|| format!("iterator valid but keyless at step {step}"))?
            .to_vec();
        if key <= previous {
            return Err(format!(
                "iterator went backwards: {} followed {}",
                show(&key),
                show(&previous)
            ));
        }
        previous = key;
    }

    let target = dataset::first_live_at_or_after(records / 2, records)
        .ok_or_else(|| "no live bulk record at or after the midpoint".to_string())?;
    let target_key = dataset::bulk_key(target);
    it.seek(&target_key);
    let landed = it
        .key()
        .ok_or_else(|| format!("iterator invalid after seek to {}", show(&target_key)))?;
    if landed != target_key.as_slice() {
        return Err(format!(
            "seek({}) landed on {}",
            show(&target_key),
            show(landed)
        ));
    }
    let want = dataset::bulk_expected(target, records).ok_or_else(|| {
        format!("bulk record {target} is reported live but has no expected value")
    })?;
    let got = it
        .value()
        .ok_or_else(|| format!("iterator has no value at {}", show(&target_key)))?
        .to_vec();
    dataset::check(&show(&target_key), &want, &got)?;

    // Walk back to the previous live record and check the order holds
    // in reverse too.
    it.prev();
    let before = it
        .key()
        .ok_or_else(|| "iterator invalid after prev".to_string())?;
    if before >= target_key.as_slice() {
        return Err(format!(
            "prev from {} gave {}, which is not smaller",
            show(&target_key),
            show(before)
        ));
    }

    it.seek_to_last();
    let last = it
        .key()
        .ok_or_else(|| "iterator invalid after seek_to_last".to_string())?
        .to_vec();
    let want_last = dataset::post_snapshot_key();
    if last != want_last {
        return Err(format!(
            "seek_to_last gave {}, expected {}",
            show(&last),
            show(&want_last)
        ));
    }

    it.status().map_err(|e| format!("iterator status: {e}"))
}
