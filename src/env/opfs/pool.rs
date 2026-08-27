//! The Rust-side bookkeeping for a slot pool.
//!
//! [`super::sah`] owns the on-disk slot format and the JS handles; this
//! module owns the map from lark's logical paths to slot indices, the
//! free list, and the virtual directory set. Every operation here is
//! synchronous, which is the whole point: `Db::open` lists directories and
//! the write path creates files, both from code that cannot `await`.

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::{Path, PathBuf};

use crate::sync::Mutex;

use super::sah::{self, MountId, Slot, SlotHeader};
use super::{children, not_found, register_ancestors};

struct PoolState {
    slots: Vec<Slot>,
    by_path: HashMap<PathBuf, usize>,
    free: Vec<usize>,
    dirs: BTreeSet<PathBuf>,
    /// One counter for the whole pool, so the newest path assignment
    /// always carries the highest generation. Mount uses that to settle a
    /// rename that a crash interrupted.
    next_generation: u64,
}

/// A pool of pre-opened OPFS sync access handles.
pub(super) struct SahPool {
    mount: MountId,
    state: Mutex<PoolState>,
}

impl std::fmt::Debug for SahPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("SahPool");
        match self.state.try_lock() {
            Some(state) => out
                .field("slots", &state.slots.len())
                .field("free", &state.free.len())
                .field("files", &state.by_path.len()),
            None => out.field("state", &"locked"),
        }
        .finish()
    }
}

impl Drop for SahPool {
    fn drop(&mut self) {
        sah::release_mount(self.mount);
    }
}

impl SahPool {
    /// Build a pool from the headers read at mount.
    pub(super) fn new(mount: MountId, headers: Vec<Option<SlotHeader>>) -> Self {
        let mut state = PoolState {
            slots: Vec::with_capacity(headers.len()),
            by_path: HashMap::new(),
            free: Vec::new(),
            dirs: BTreeSet::new(),
            next_generation: 1,
        };
        install(mount, &mut state, headers);
        Self {
            mount,
            state: Mutex::new(state),
        }
    }

    /// Take slots opened after mount into the pool, keeping whatever
    /// logical files their headers already claim.
    pub(super) fn adopt_slots(&self, headers: Vec<Option<SlotHeader>>) {
        let mut state = self.state.lock();
        install(self.mount, &mut state, headers);
    }

    /// The handle-registry mount this pool draws its slots from.
    pub(super) fn mount_id(&self) -> MountId {
        self.mount
    }

    /// Slots with no logical file assigned.
    pub(super) fn free_slots(&self) -> usize {
        self.state.lock().free.len()
    }

    /// Total slots in the pool.
    pub(super) fn slot_count(&self) -> usize {
        self.state.lock().slots.len()
    }

    pub(super) fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let mut state = self.state.lock();
        state.dirs.insert(path.to_path_buf());
        register_ancestors(&mut state.dirs, path);
        Ok(())
    }

    pub(super) fn read_dir(&self, path: &Path) -> io::Result<Vec<(PathBuf, bool)>> {
        let state = self.state.lock();
        if !state.dirs.contains(path) {
            return Err(not_found(path));
        }
        Ok(children(path, state.by_path.keys(), state.dirs.iter()))
    }

    pub(super) fn exists(&self, path: &Path) -> bool {
        let state = self.state.lock();
        state.by_path.contains_key(path) || state.dirs.contains(path)
    }

    pub(super) fn metadata(&self, path: &Path) -> io::Result<(u64, bool)> {
        let state = self.state.lock();
        if let Some(&slot) = state.by_path.get(path) {
            return Ok((state.slots[slot].len, false));
        }
        if state.dirs.contains(path) {
            return Ok((0, true));
        }
        Err(not_found(path))
    }

    /// Resolve a path for reading, returning the slot and the generation
    /// a later read must still match.
    pub(super) fn open_read(&self, path: &Path) -> io::Result<(usize, u64)> {
        let state = self.state.lock();
        let slot = *state.by_path.get(path).ok_or_else(|| not_found(path))?;
        Ok((slot, state.slots[slot].generation))
    }

    /// Resolve or create a path for writing, returning the slot, its
    /// generation, and the offset the first write lands at.
    pub(super) fn open_write(&self, path: &Path, append: bool) -> io::Result<(usize, u64, u64)> {
        sah::check_path_fits(path)?;
        let mut state = self.state.lock();

        if let Some(&slot) = state.by_path.get(path) {
            if append {
                let entry = &state.slots[slot];
                return Ok((slot, entry.generation, entry.len));
            }
            sah::truncate_slot(self.mount, slot, 0)?;
            let entry = &mut state.slots[slot];
            entry.len = 0;
            entry.header_dirty = true;
            return Ok((slot, entry.generation, 0));
        }

        let slot = state.free.pop().ok_or_else(|| {
            io::Error::other(format!(
                "OPFS handle pool is full ({} slots, all assigned); call \
                 OpfsEnv::grow_pool from an async context to add more",
                state.slots.len()
            ))
        })?;
        let generation = state.next_generation;
        state.next_generation += 1;

        if let Err(e) = sah::truncate_slot(self.mount, slot, 0) {
            state.free.push(slot);
            return Err(e);
        }
        let entry = &mut state.slots[slot];
        entry.path = Some(path.to_path_buf());
        entry.len = 0;
        entry.generation = generation;
        entry.header_dirty = true;

        state.by_path.insert(path.to_path_buf(), slot);
        register_ancestors(&mut state.dirs, path);
        Ok((slot, generation, 0))
    }

    pub(super) fn read_at(
        &self,
        slot: usize,
        generation: u64,
        at: u64,
        buf: &mut [u8],
    ) -> io::Result<usize> {
        let len = {
            let state = self.state.lock();
            live(&state, slot, generation)?.len
        };
        if at >= len {
            return Ok(0);
        }
        let want = buf.len().min((len - at) as usize);
        sah::read_slot(self.mount, slot, at, &mut buf[..want])
    }

    pub(super) fn write_at(
        &self,
        slot: usize,
        generation: u64,
        at: u64,
        buf: &[u8],
    ) -> io::Result<()> {
        {
            let state = self.state.lock();
            live(&state, slot, generation)?;
        }
        // Contents first, header on sync: a crash in between loses the
        // tail, which WAL replay and orphan SSTables already tolerate.
        sah::write_slot(self.mount, slot, at, buf)?;
        let mut state = self.state.lock();
        let entry = live_mut(&mut state, slot, generation)?;
        entry.len = entry.len.max(at + buf.len() as u64);
        entry.header_dirty = true;
        Ok(())
    }

    pub(super) fn file_len(&self, slot: usize, generation: u64) -> io::Result<u64> {
        let state = self.state.lock();
        Ok(live(&state, slot, generation)?.len)
    }

    pub(super) fn set_len(&self, slot: usize, generation: u64, len: u64) -> io::Result<()> {
        {
            let state = self.state.lock();
            live(&state, slot, generation)?;
        }
        sah::truncate_slot(self.mount, slot, len)?;
        let mut state = self.state.lock();
        let entry = live_mut(&mut state, slot, generation)?;
        entry.len = len;
        entry.header_dirty = true;
        Ok(())
    }

    /// Make a slot's contents and its name binding durable.
    pub(super) fn sync(&self, slot: usize, generation: u64) -> io::Result<()> {
        let snapshot = {
            let state = self.state.lock();
            live(&state, slot, generation)?.clone()
        };
        if !snapshot.header_dirty {
            return sah::flush_slot(self.mount, slot);
        }
        sah::sync_slot(self.mount, slot, &snapshot)?;
        let mut state = self.state.lock();
        if let Ok(entry) = live_mut(&mut state, slot, generation) {
            entry.header_dirty = false;
        }
        Ok(())
    }

    /// Make the name bindings of a directory's files durable.
    ///
    /// OPFS has no directory object to fsync; the binding from a logical
    /// path to a slot lives in that slot's header, so flushing the dirty
    /// headers under `path` is the exact equivalent.
    pub(super) fn sync_dir(&self, path: &Path) -> io::Result<()> {
        let pending: Vec<(usize, Slot)> = {
            let state = self.state.lock();
            state
                .by_path
                .iter()
                .filter(|(file, _)| file.parent() == Some(path))
                .map(|(_, slot)| *slot)
                .filter(|slot| state.slots[*slot].header_dirty)
                .map(|slot| (slot, state.slots[slot].clone()))
                .collect()
        };
        for (slot, snapshot) in pending {
            sah::sync_slot(self.mount, slot, &snapshot)?;
            let mut state = self.state.lock();
            if state.slots[slot].generation == snapshot.generation {
                state.slots[slot].header_dirty = false;
            }
        }
        Ok(())
    }

    pub(super) fn remove_file(&self, path: &Path) -> io::Result<()> {
        let mut state = self.state.lock();
        let slot = state.by_path.remove(path).ok_or_else(|| not_found(path))?;
        release_slot(self.mount, &mut state, slot)
    }

    /// Rebind `from`'s slot to `to`, then release whatever held `to`.
    ///
    /// The new binding is made durable before the old one is dropped, so
    /// a crash in the window leaves two slots claiming `to` and mount
    /// keeps the one with the higher generation. That is what makes
    /// rename atomic on a filesystem with no atomic rename.
    pub(super) fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        sah::check_path_fits(to)?;
        let mut state = self.state.lock();
        let slot = *state.by_path.get(from).ok_or_else(|| not_found(from))?;

        let previous_path = state.slots[slot].path.clone();
        let previous_generation = state.slots[slot].generation;
        let generation = state.next_generation;
        state.next_generation += 1;

        state.slots[slot].path = Some(to.to_path_buf());
        state.slots[slot].generation = generation;
        let snapshot = state.slots[slot].clone();

        if let Err(e) = sah::sync_slot(self.mount, slot, &snapshot) {
            state.slots[slot].path = previous_path;
            state.slots[slot].generation = previous_generation;
            return Err(e);
        }
        state.slots[slot].header_dirty = false;

        state.by_path.remove(from);
        if let Some(replaced) = state.by_path.insert(to.to_path_buf(), slot) {
            release_slot(self.mount, &mut state, replaced)?;
        }
        register_ancestors(&mut state.dirs, to);
        Ok(())
    }
}

/// Append `headers` as new slots.
///
/// A slot whose header is absent, torn, or marked free joins the free
/// list. When two slots claim one logical path - the window a crash
/// during rename leaves open - the higher generation wins and the loser
/// is released.
fn install(mount: MountId, state: &mut PoolState, headers: Vec<Option<SlotHeader>>) {
    let mut superseded = Vec::new();

    for header in headers {
        let index = state.slots.len();
        let claimed = match header {
            Some(header) => {
                state.next_generation = state.next_generation.max(header.generation + 1);
                if header.in_use && !header.path.is_empty() {
                    Some(header)
                } else {
                    None
                }
            }
            None => None,
        };

        match claimed {
            Some(header) => {
                let path = PathBuf::from(&header.path);
                state.slots.push(Slot {
                    path: Some(path.clone()),
                    len: header.len,
                    generation: header.generation,
                    header_dirty: false,
                });
                match state.by_path.get(&path).copied() {
                    Some(rival) if state.slots[rival].generation >= header.generation => {
                        superseded.push(index);
                    }
                    Some(rival) => {
                        superseded.push(rival);
                        state.by_path.insert(path.clone(), index);
                    }
                    None => {
                        state.by_path.insert(path.clone(), index);
                    }
                }
                register_ancestors(&mut state.dirs, &path);
            }
            None => {
                state.slots.push(Slot {
                    path: None,
                    len: 0,
                    generation: 0,
                    header_dirty: false,
                });
                state.free.push(index);
            }
        }
    }

    for index in superseded {
        // A failure here only means the duplicate survives to the next
        // mount, which resolves it the same deterministic way.
        if let Err(e) = release_slot(mount, state, index) {
            tracing::warn!(slot = index, error = %e, "could not release a superseded OPFS slot");
        }
    }
}

fn stale() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        "the file behind this handle was removed or renamed",
    )
}

fn live(state: &PoolState, slot: usize, generation: u64) -> io::Result<&Slot> {
    match state.slots.get(slot) {
        Some(entry) if entry.path.is_some() && entry.generation == generation => Ok(entry),
        _ => Err(stale()),
    }
}

fn live_mut(state: &mut PoolState, slot: usize, generation: u64) -> io::Result<&mut Slot> {
    match state.slots.get_mut(slot) {
        Some(entry) if entry.path.is_some() && entry.generation == generation => Ok(entry),
        _ => Err(stale()),
    }
}

/// Mark a slot free on disk and in memory, and give its bytes back to the
/// origin's quota.
fn release_slot(mount: MountId, state: &mut PoolState, slot: usize) -> io::Result<()> {
    let generation = state.next_generation;
    state.next_generation += 1;
    let Some(entry) = state.slots.get_mut(slot) else {
        return Ok(());
    };
    entry.path = None;
    entry.len = 0;
    entry.generation = generation;
    entry.header_dirty = false;
    let snapshot = entry.clone();

    sah::sync_slot(mount, slot, &snapshot)?;
    sah::truncate_slot(mount, slot, 0)?;
    if !state.free.contains(&slot) {
        state.free.push(slot);
    }
    Ok(())
}
