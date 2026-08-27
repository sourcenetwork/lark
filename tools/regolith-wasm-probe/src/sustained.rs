//! The sustained-write phase.
//!
//! The lifecycle phases never fill a memtable more than a handful of
//! times, so they pass even on a build where L0 grows without bound.
//! This phase is the one that catches that: a small write buffer, many
//! writes, and no explicit `compact_range` anywhere. With no
//! background worker and no inline compaction the writer stalls at
//! `level0_stop_writes_trigger` and never resumes, so a build that
//! wedges hangs here instead of reporting a false pass.

use std::path::Path;

use regolith::{Db, Options};

use crate::dataset;
use crate::report::Reporter;

/// Bulk records re-read at the end to prove the data survived.
const SAMPLE_STRIDE: u64 = 97;

/// Write `count` records into a database at `path` with `write_buffer`
/// bytes of memtable, never calling `compact_range`, then verify a
/// strided sample.
pub fn run(
    path: &Path,
    mut opts: Options,
    write_buffer: usize,
    count: u64,
    reporter: &mut Reporter,
) -> Result<(), String> {
    opts.write_buffer_size = write_buffer;
    let db = Db::open(path, opts)
        .map_err(|e| format!("sustained open {} failed: {e}", path.display()))?;
    reporter.pass("sustained open");

    for i in 0..count {
        let key = dataset::bulk_key(i);
        db.put(&key, &dataset::value(i)).map_err(|e| {
            format!(
                "sustained put {} failed at record {i}: {e}",
                String::from_utf8_lossy(&key)
            )
        })?;
    }
    reporter.pass("sustained writes");

    let mut checked = 0u64;
    let mut i = 0u64;
    while i < count {
        let key = dataset::bulk_key(i);
        let label = String::from_utf8_lossy(&key).into_owned();
        match db
            .get(&key)
            .map_err(|e| format!("sustained get {label} failed: {e}"))?
        {
            Some(got) => dataset::check(&label, &dataset::value(i), &got)?,
            None => return Err(format!("sustained record {label} is missing")),
        }
        checked += 1;
        i += SAMPLE_STRIDE;
    }
    reporter.note(&format!("sustained sample verified {checked} records"));
    reporter.pass("sustained read back");

    db.close()
        .map_err(|e| format!("sustained close failed: {e}"))?;
    Ok(())
}
