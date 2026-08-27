//! Ordered walks and whole-keyspace verification, shared by every
//! phase so the native and the wasm run execute the same code and a
//! divergence can only come from the platform underneath.

use lark_kv::{CfIter, Db};

use crate::data::{self, Expect};
use crate::report::Report;

/// What an ordered walk observed.
pub struct WalkResult {
    /// Entries visited.
    pub count: u64,
    /// Digest folded in traversal order.
    pub digest: u64,
    /// First key visited, if any.
    pub first: Option<Vec<u8>>,
    /// Last key visited, if any.
    pub last: Option<Vec<u8>>,
    /// Adjacent pairs that were not strictly ordered in the direction
    /// of travel. Any non-zero value is a correctness failure.
    pub order_violations: u64,
}

impl WalkResult {
    /// Record every field of the walk under `name`, failing when the
    /// count, the digest, or the ordering disagrees with the model.
    pub fn record(&self, report: &mut Report, name: &str, expect_count: u64, expect_digest: u64) {
        report.expect_u64(&format!("{name}.count"), self.count, expect_count);
        report.expect_digest(&format!("{name}.digest"), self.digest, expect_digest);
        report.expect_u64(
            &format!("{name}.order_violations"),
            self.order_violations,
            0,
        );
        report.check(&format!("{name}.first"), &render(self.first.as_deref()));
        report.check(&format!("{name}.last"), &render(self.last.as_deref()));
    }
}

fn render(key: Option<&[u8]>) -> String {
    match key {
        Some(k) => String::from_utf8_lossy(k).into_owned(),
        None => "none".to_string(),
    }
}

/// Walk ascending from `prefix` (or from the first key when `prefix`
/// is empty), stopping at the first key outside the prefix.
pub fn walk_forward(iter: &mut CfIter<'_>, prefix: &[u8]) -> Result<WalkResult, String> {
    if prefix.is_empty() {
        iter.seek_to_first();
    } else {
        iter.seek(prefix);
    }
    walk(iter, prefix, Direction::Forward)
}

/// Walk descending over `prefix` (or from the last key when `prefix`
/// is empty), stopping at the first key outside the prefix.
pub fn walk_backward(iter: &mut CfIter<'_>, prefix: &[u8]) -> Result<WalkResult, String> {
    if prefix.is_empty() {
        iter.seek_to_last();
    } else {
        iter.seek_for_prev(&upper_bound(prefix));
    }
    walk(iter, prefix, Direction::Backward)
}

enum Direction {
    Forward,
    Backward,
}

/// The smallest key that sorts after every key carrying `prefix`.
/// Incrementing the final byte is enough for the ASCII prefixes this
/// harness uses; a prefix ending in `0xff` would need a carry, and
/// none does.
fn upper_bound(prefix: &[u8]) -> Vec<u8> {
    let mut bound = prefix.to_vec();
    if let Some(last) = bound.last_mut() {
        *last = last.saturating_add(1);
    }
    bound
}

fn walk(iter: &mut CfIter<'_>, prefix: &[u8], dir: Direction) -> Result<WalkResult, String> {
    let mut result = WalkResult {
        count: 0,
        digest: data::Digest::new().finish(),
        first: None,
        last: None,
        order_violations: 0,
    };
    let mut digest = data::Digest::new();
    let mut previous: Option<Vec<u8>> = None;

    while iter.valid() {
        let (key, value) = match (iter.key(), iter.value()) {
            (Some(k), Some(v)) => (k.to_vec(), v.to_vec()),
            _ => return Err("iterator reported valid with no key or value".to_string()),
        };
        if !key.starts_with(prefix) {
            break;
        }
        if let Some(prev) = &previous {
            let ordered = match dir {
                Direction::Forward => prev.as_slice() < key.as_slice(),
                Direction::Backward => prev.as_slice() > key.as_slice(),
            };
            if !ordered {
                result.order_violations += 1;
            }
        } else {
            result.first = Some(key.clone());
        }
        digest.entry(&key, &value);
        result.count += 1;
        previous = Some(key);
        match dir {
            Direction::Forward => iter.next(),
            Direction::Backward => iter.prev(),
        }
    }
    iter.status().map_err(|e| format!("iterator status: {e}"))?;
    result.digest = digest.finish();
    result.last = previous;
    Ok(result)
}

/// Read every bulk key back with a point read and compare byte for
/// byte against the model.
pub fn point_reads_bulk(
    db: &Db,
    report: &mut Report,
    expect: fn(u64) -> Expect,
    name: &str,
) -> Result<(), String> {
    let mut present = 0u64;
    let mut absent = 0u64;
    let mut expected_present = 0u64;
    let mut mismatched_bytes = 0u64;
    for i in 0..data::RECORDS {
        let key = data::bulk_key(i);
        let got = db
            .get(&key)
            .map_err(|e| format!("get {} failed: {e}", String::from_utf8_lossy(&key)))?;
        match (expect(i), got) {
            (Expect::Absent, None) => absent += 1,
            (Expect::Absent, Some(v)) => {
                report.fail(
                    name,
                    &format!(
                        "{} should be deleted but returned {} bytes",
                        String::from_utf8_lossy(&key),
                        v.len()
                    ),
                );
            }
            (Expect::Present(idx, gen), None) => {
                expected_present += 1;
                report.fail(name, &format!("{} missing", String::from_utf8_lossy(&key)));
                let _ = (idx, gen);
            }
            (Expect::Present(idx, gen), Some(v)) => {
                expected_present += 1;
                let want = data::value(idx, gen);
                if v == want {
                    present += 1;
                } else {
                    mismatched_bytes += 1;
                    report.fail(
                        name,
                        &format!(
                            "{} differs: expected {} bytes, got {} bytes, first difference at {}",
                            String::from_utf8_lossy(&key),
                            want.len(),
                            v.len(),
                            first_difference(&want, &v)
                        ),
                    );
                }
            }
        }
    }
    report.expect_u64(&format!("{name}.present"), present, expected_present);
    report.expect_u64(
        &format!("{name}.absent"),
        absent,
        data::RECORDS - expected_present,
    );
    report.expect_u64(&format!("{name}.byte_mismatches"), mismatched_bytes, 0);
    Ok(())
}

/// Index of the first differing byte, or the shorter length when one
/// is a prefix of the other.
fn first_difference(want: &[u8], got: &[u8]) -> usize {
    want.iter()
        .zip(got.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| want.len().min(got.len()))
}

/// Read every `late/` key back and compare byte for byte.
pub fn point_reads_late(db: &Db, report: &mut Report, name: &str) -> Result<(), String> {
    let mut matched = 0u64;
    for j in 0..data::LATE_RECORDS {
        let key = data::late_key(j);
        let got = db
            .get(&key)
            .map_err(|e| format!("get {} failed: {e}", String::from_utf8_lossy(&key)))?;
        match got {
            Some(v) if v == data::value(j, data::GEN_LATE) => matched += 1,
            Some(v) => report.fail(
                name,
                &format!(
                    "{} differs: {} bytes returned",
                    String::from_utf8_lossy(&key),
                    v.len()
                ),
            ),
            None => report.fail(name, &format!("{} missing", String::from_utf8_lossy(&key))),
        }
    }
    report.expect_u64(&format!("{name}.matched"), matched, data::LATE_RECORDS);
    Ok(())
}

/// The whole-database check a fresh process runs: every bulk key,
/// every late key, an ascending walk, and a descending walk.
pub fn verify_final_state(db: &Db, report: &mut Report, label: &str) -> Result<(), String> {
    point_reads_bulk(db, report, data::expect_final, &format!("{label}.bulk"))?;
    point_reads_late(db, report, &format!("{label}.late"))?;

    let (fwd_digest, fwd_count) = data::expected_bulk_digest(data::expect_final);
    let mut iter = db.iter();
    let forward = walk_forward(&mut iter, b"key/")?;
    forward.record(report, &format!("{label}.forward"), fwd_count, fwd_digest);

    let (rev_digest, rev_count) = data::expected_bulk_digest_reverse(data::expect_final);
    let mut iter = db.iter();
    let backward = walk_backward(&mut iter, b"key/")?;
    backward.record(report, &format!("{label}.backward"), rev_count, rev_digest);

    let (late_digest, late_count) = data::expected_late_digest();
    let mut iter = db.iter();
    let late = walk_forward(&mut iter, b"late/")?;
    late.record(
        report,
        &format!("{label}.late_walk"),
        late_count,
        late_digest,
    );
    Ok(())
}
