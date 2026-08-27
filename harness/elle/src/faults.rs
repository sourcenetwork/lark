//! Write-ahead-log damage used to force recovery paths into the history.
//!
//! lark keeps its write-ahead log at `<db>/wal/wal_<id>.log`. Both
//! faults here damage only bytes written after a recorded high-water
//! mark, which is taken immediately before the doomed child process
//! writes anything. Everything the history reports as committed lives
//! below that mark, so recovery is required to preserve it: a checker
//! failure after one of these faults is a real durability bug, never an
//! artifact of the fault.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Newest write-ahead log file in a lark database directory.
pub fn newest_wal(db_dir: &Path) -> std::io::Result<Option<PathBuf>> {
    let wal_dir = db_dir.join("wal");
    let mut newest: Option<(String, PathBuf)> = None;
    let entries = match std::fs::read_dir(&wal_dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    for entry in entries {
        let path = entry?.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) if name.starts_with("wal_") && name.ends_with(".log") => name.to_string(),
            _ => continue,
        };
        // Fixed-width ids, so lexicographic order is numeric order.
        if newest.as_ref().is_none_or(|(best, _)| name > *best) {
            newest = Some((name, path));
        }
    }
    Ok(newest.map(|(_, path)| path))
}

/// High-water mark of the write-ahead log: the file that is live now
/// and how many bytes of it are already committed.
pub struct WalMark {
    pub path: PathBuf,
    pub len: u64,
}

impl WalMark {
    pub fn capture(db_dir: &Path) -> std::io::Result<Option<Self>> {
        let path = match newest_wal(db_dir)? {
            Some(path) => path,
            None => return Ok(None),
        };
        let len = std::fs::metadata(&path)?.len();
        Ok(Some(Self { path, len }))
    }

    pub fn encode(&self) -> String {
        format!("WALMARK\t{}\t{}", self.path.display(), self.len)
    }

    pub fn decode(line: &str) -> Option<Self> {
        let mut parts = line.trim_end().split('\t');
        if parts.next()? != "WALMARK" {
            return None;
        }
        let path = PathBuf::from(parts.next()?);
        let len = parts.next()?.parse::<u64>().ok()?;
        Some(Self { path, len })
    }

    /// Reject a mark whose log rotated or did not grow. Damaging a
    /// rotated log would destroy data the history reports as committed,
    /// so the caller skips the fault instead of lying about it.
    fn damageable_len(&self) -> std::io::Result<Option<u64>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let now = std::fs::metadata(&self.path)?.len();
        if now <= self.len {
            return Ok(None);
        }
        Ok(Some(now))
    }
}

/// Truncate the log inside the doomed region, leaving a partial record
/// at the tail. Recovery has to discard the partial record and keep
/// everything below the mark.
pub fn truncate_wal_tail(mark: &WalMark) -> std::io::Result<Option<String>> {
    let now = match mark.damageable_len()? {
        Some(now) => now,
        None => return Ok(None),
    };
    let cut = mark.len + (now - mark.len) / 2;
    let file = std::fs::OpenOptions::new().write(true).open(&mark.path)?;
    file.set_len(cut)?;
    file.sync_all()?;
    Ok(Some(format!(
        "truncated {} from {} to {} bytes (mark {})",
        mark.path.display(),
        now,
        cut,
        mark.len
    )))
}

/// Overwrite the final bytes of the log with a pattern no checksum will
/// accept, simulating the last record being half-written when the
/// process died. The damage stays inside the doomed region, so every
/// record the history reports as committed is left intact.
pub fn tear_wal_write(mark: &WalMark) -> std::io::Result<Option<String>> {
    let now = match mark.damageable_len()? {
        Some(now) => now,
        None => return Ok(None),
    };
    let span = std::cmp::min(32, now - mark.len);
    if span == 0 {
        return Ok(None);
    }
    let start = now - span;
    let mut file = std::fs::OpenOptions::new().write(true).open(&mark.path)?;
    file.seek(SeekFrom::Start(start))?;
    file.write_all(&vec![0x5A; span as usize])?;
    file.sync_all()?;
    Ok(Some(format!(
        "tore the final {} bytes of {} at offset {} (mark {})",
        span,
        mark.path.display(),
        start,
        mark.len
    )))
}
