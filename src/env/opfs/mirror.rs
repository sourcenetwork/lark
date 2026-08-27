//! Mirror mode: the database resident in linear memory, written back to
//! OPFS whole-file.
//!
//! This is the strategy defradb's OPFS backend uses, and it exists here
//! for the same reason: `createSyncAccessHandle` is worker-only, so a
//! database opened on the main thread has no synchronous path to storage
//! at all. `FileSystemWritableFileStream` is asynchronous but available
//! everywhere, so the engine runs against RAM and
//! [`super::OpfsEnv::persist`] pushes dirty files out.
//!
//! Two ceilings follow, and both are reported rather than hidden. The
//! whole database is resident, so [`super::OpfsOptions::max_resident_bytes`]
//! fails a write loudly instead of growing linear memory until the tab
//! dies (wasm pages are never returned to the host). And nothing is
//! durable until `persist()` resolves, so
//! [`crate::env::Capabilities::durable_sync`] is `false` here.
//!
//! # Physical naming
//!
//! OPFS directories are flat for lark's purposes, so a logical path is
//! escaped into one filename: `%` becomes `%25`, then `/` becomes `%2F`
//! and `\` becomes `%5C`. Every mirror file carries the `.lark-file-`
//! prefix so a mirror database and a slot pool can share one directory.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use wasm_bindgen::JsValue;

use super::js;

const FILE_PREFIX: &str = ".lark-file-";

/// Escape a logical path into a single OPFS entry name.
fn encode_name(path: &Path) -> String {
    let mut out = String::from(FILE_PREFIX);
    for ch in path.to_string_lossy().chars() {
        match ch {
            '%' => out.push_str("%25"),
            '/' => out.push_str("%2F"),
            '\\' => out.push_str("%5C"),
            other => out.push(other),
        }
    }
    out
}

/// Reverse [`encode_name`]. Returns `None` for an entry that is not a
/// mirror file or whose escapes are malformed.
fn decode_name(name: &str) -> Option<PathBuf> {
    let body = name.strip_prefix(FILE_PREFIX)?;
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let code = body.get(i + 1..i + 3)?;
            match code {
                "25" => out.push('%'),
                "2F" => out.push('/'),
                "5C" => out.push('\\'),
                _ => return None,
            }
            i += 3;
        } else {
            let ch = body[i..].chars().next()?;
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    Some(PathBuf::from(out))
}

/// One dirty file as `persist` will write it: the logical path, the
/// version that made it dirty, and the bytes to send.
type PendingWrite = (PathBuf, u64, Vec<u8>);

struct Entry {
    data: Vec<u8>,
    /// Bumped on every mutation so [`MirrorFs::take_persist_batch`] can
    /// tell a file that was re-dirtied during the await from one that
    /// was not.
    version: u64,
}

struct State {
    files: HashMap<PathBuf, Entry>,
    dirs: BTreeSet<PathBuf>,
    dirty: HashSet<PathBuf>,
    deleted: HashSet<PathBuf>,
    resident: usize,
    next_version: u64,
}

/// The in-memory mirror of an OPFS-backed database.
pub(super) struct MirrorFs {
    state: Mutex<State>,
    max_resident: usize,
    mount: super::sah::MountId,
}

impl Drop for MirrorFs {
    fn drop(&mut self) {
        super::sah::release_mount(self.mount);
    }
}

impl std::fmt::Debug for MirrorFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `try_lock` so formatting an error while the state is held
        // cannot deadlock a single-threaded module.
        let mut out = f.debug_struct("MirrorFs");
        match self.state.try_lock() {
            Some(state) => out
                .field("files", &state.files.len())
                .field("resident_bytes", &state.resident),
            None => out.field("state", &"locked"),
        }
        .field("max_resident_bytes", &self.max_resident)
        .finish()
    }
}

impl MirrorFs {
    /// Build a mirror from files already read out of OPFS.
    pub(super) fn new(
        mount: super::sah::MountId,
        loaded: Vec<(PathBuf, Vec<u8>)>,
        max_resident: usize,
    ) -> Self {
        let mut state = State {
            files: HashMap::with_capacity(loaded.len()),
            dirs: BTreeSet::new(),
            dirty: HashSet::new(),
            deleted: HashSet::new(),
            resident: 0,
            next_version: 1,
        };
        for (path, data) in loaded {
            state.resident += data.len();
            super::register_ancestors(&mut state.dirs, &path);
            state.files.insert(path, Entry { data, version: 0 });
        }
        Self {
            state: Mutex::new(state),
            max_resident,
            mount,
        }
    }

    /// Read every mirror file out of an OPFS directory.
    pub(super) async fn load(directory: &JsValue) -> Result<Vec<(PathBuf, Vec<u8>)>, JsValue> {
        let entries = js::list_files(directory).await?;
        let mut loaded = Vec::new();
        for (name, handle) in entries {
            let Some(path) = decode_name(&name) else {
                continue;
            };
            loaded.push((path, js::read_whole_file(&handle).await?));
        }
        Ok(loaded)
    }

    pub(super) fn resident_bytes(&self) -> usize {
        self.state.lock().resident
    }

    pub(super) fn pending_bytes(&self) -> usize {
        let state = self.state.lock();
        state
            .dirty
            .iter()
            .filter_map(|path| state.files.get(path))
            .map(|entry| entry.data.len())
            .sum()
    }

    /// Snapshot the work `persist` has to do, releasing the lock before
    /// any `await`.
    fn take_persist_batch(&self) -> (Vec<PendingWrite>, Vec<PathBuf>) {
        let state = self.state.lock();
        let writes = state
            .dirty
            .iter()
            .filter_map(|path| {
                state
                    .files
                    .get(path)
                    .map(|entry| (path.clone(), entry.version, entry.data.clone()))
            })
            .collect();
        let deletes = state.deleted.iter().cloned().collect();
        (writes, deletes)
    }

    /// Clear the tracking entries that `persist` actually wrote out. A
    /// file mutated while the write was in flight keeps its dirty mark.
    fn settle(&self, written: &[(PathBuf, u64)], deleted: &[PathBuf]) {
        let mut state = self.state.lock();
        for (path, version) in written {
            if state.files.get(path).map(|e| e.version) == Some(*version) {
                state.dirty.remove(path);
            }
        }
        for path in deleted {
            state.deleted.remove(path);
        }
    }

    /// Write every dirty file back to OPFS and drop the deleted ones.
    pub(super) async fn persist(&self) -> Result<(), JsValue> {
        let (writes, deletes) = self.take_persist_batch();
        if writes.is_empty() && deletes.is_empty() {
            return Ok(());
        }
        let directory = super::sah::mount_directory(self.mount)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut written = Vec::with_capacity(writes.len());
        for (path, version, data) in &writes {
            js::write_whole_file(&directory, &encode_name(path), data).await?;
            written.push((path.clone(), *version));
        }
        for path in &deletes {
            // A file deleted before it was ever persisted has no OPFS
            // entry; that is not an error.
            let _ = js::remove_entry(&directory, &encode_name(path)).await;
        }

        self.settle(&written, &deletes);
        Ok(())
    }

    fn quota_error(&self, want: usize) -> io::Error {
        io::Error::other(format!(
            "OPFS mirror mode holds the whole database in memory and this write \
             would reach {want} bytes, over the {} byte limit; raise \
             OpfsOptions::max_resident_bytes or open the database in a worker, \
             where OpfsMode::Sah streams to storage instead",
            self.max_resident
        ))
    }

    pub(super) fn write_at(&self, path: &Path, at: u64, buf: &[u8]) -> io::Result<()> {
        let mut state = self.state.lock();
        let at = at as usize;
        let entry = state.files.get(path).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "file was removed while open")
        })?;
        let old_len = entry.data.len();
        let new_len = old_len.max(at.saturating_add(buf.len()));
        let projected = state.resident.saturating_sub(old_len) + new_len;
        if projected > self.max_resident {
            return Err(self.quota_error(projected));
        }

        let version = state.next_version;
        state.next_version += 1;
        let entry = state.files.get_mut(path).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "file was removed while open")
        })?;
        if entry.data.len() < new_len {
            entry.data.resize(new_len, 0);
        }
        entry.data[at..at + buf.len()].copy_from_slice(buf);
        entry.version = version;
        state.resident = projected;
        state.dirty.insert(path.to_path_buf());
        Ok(())
    }

    pub(super) fn read_at(&self, path: &Path, at: u64, buf: &mut [u8]) -> io::Result<usize> {
        let state = self.state.lock();
        let entry = state.files.get(path).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "file was removed while open")
        })?;
        let at = at as usize;
        if at >= entry.data.len() {
            return Ok(0);
        }
        let n = buf.len().min(entry.data.len() - at);
        buf[..n].copy_from_slice(&entry.data[at..at + n]);
        Ok(n)
    }

    pub(super) fn file_len(&self, path: &Path) -> io::Result<u64> {
        let state = self.state.lock();
        state
            .files
            .get(path)
            .map(|entry| entry.data.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "file was removed while open"))
    }

    pub(super) fn set_len(&self, path: &Path, len: u64) -> io::Result<()> {
        let mut state = self.state.lock();
        let version = state.next_version;
        state.next_version += 1;
        let entry = state.files.get_mut(path).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "file was removed while open")
        })?;
        let old_len = entry.data.len();
        let len = len as usize;
        entry.data.resize(len, 0);
        entry.version = version;
        state.resident = state.resident.saturating_sub(old_len) + len;
        state.dirty.insert(path.to_path_buf());
        Ok(())
    }

    /// Create the file when absent, optionally emptying it first.
    /// Returns the resulting length.
    pub(super) fn create(&self, path: &Path, truncate: bool) -> io::Result<u64> {
        let mut state = self.state.lock();
        let version = state.next_version;
        state.next_version += 1;
        let old_len = state.files.get(path).map(|entry| entry.data.len());
        match state.files.get_mut(path) {
            Some(entry) if truncate => {
                entry.data.clear();
                entry.version = version;
            }
            Some(_) => {}
            None => {
                state.files.insert(
                    path.to_path_buf(),
                    Entry {
                        data: Vec::new(),
                        version,
                    },
                );
            }
        }
        let len = match (old_len, truncate) {
            (Some(old), true) => {
                state.resident = state.resident.saturating_sub(old);
                0
            }
            (Some(old), false) => old,
            (None, _) => 0,
        };
        state.dirty.insert(path.to_path_buf());
        state.deleted.remove(path);
        super::register_ancestors(&mut state.dirs, path);
        Ok(len as u64)
    }

    pub(super) fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut state = self.state.lock();
        state.dirs.insert(path.to_path_buf());
        super::register_ancestors(&mut state.dirs, path);
        Ok(())
    }

    pub(super) fn exists(&self, path: &Path) -> bool {
        let state = self.state.lock();
        state.files.contains_key(path) || state.dirs.contains(path)
    }

    pub(super) fn metadata(&self, path: &Path) -> io::Result<(u64, bool)> {
        let state = self.state.lock();
        if let Some(entry) = state.files.get(path) {
            return Ok((entry.data.len() as u64, false));
        }
        if state.dirs.contains(path) {
            return Ok((0, true));
        }
        Err(super::not_found(path))
    }

    pub(super) fn read_dir(&self, path: &Path) -> io::Result<Vec<(PathBuf, bool)>> {
        let state = self.state.lock();
        if !state.dirs.contains(path) {
            return Err(super::not_found(path));
        }
        Ok(super::children(path, state.files.keys(), state.dirs.iter()))
    }

    pub(super) fn remove_file(&self, path: &Path) -> io::Result<()> {
        let mut state = self.state.lock();
        match state.files.remove(path) {
            Some(entry) => {
                state.resident = state.resident.saturating_sub(entry.data.len());
                state.dirty.remove(path);
                state.deleted.insert(path.to_path_buf());
                Ok(())
            }
            None => Err(super::not_found(path)),
        }
    }

    pub(super) fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut state = self.state.lock();
        let entry = state
            .files
            .remove(from)
            .ok_or_else(|| super::not_found(from))?;
        if let Some(replaced) = state.files.insert(to.to_path_buf(), entry) {
            state.resident = state.resident.saturating_sub(replaced.data.len());
        }
        state.dirty.remove(from);
        state.deleted.insert(from.to_path_buf());
        state.dirty.insert(to.to_path_buf());
        state.deleted.remove(to);
        super::register_ancestors(&mut state.dirs, to);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn names_round_trip_through_the_escape() {
        for original in ["db/MANIFEST", "db/sst/000001.sst", "a%b/c\\d", "plain"] {
            let encoded = encode_name(Path::new(original));
            assert!(encoded.starts_with(FILE_PREFIX));
            assert!(!encoded[FILE_PREFIX.len()..].contains('/'));
            assert_eq!(decode_name(&encoded), Some(PathBuf::from(original)));
        }
    }

    #[wasm_bindgen_test]
    fn a_foreign_entry_is_not_decoded() {
        assert_eq!(decode_name(".lark-sah-0000"), None);
        assert_eq!(decode_name("something-else"), None);
    }
}
