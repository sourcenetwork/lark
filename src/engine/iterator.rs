//! Streaming iterator: merges memtable + frozen memtables + all SSTable
//! levels into a single forward cursor honoring MVCC snapshot visibility
//! and tombstone suppression.
//!
//! # Sources
//!
//! The iterator is built from a priority-ordered list of **level iterators**:
//!
//! 1. The active memtable (newest data)
//! 2. Frozen memtables (newest first)
//! 3. L0 SSTables (newest first; L0 files may overlap)
//! 4. L1..Ln SSTables (one level at a time; within a level files are
//!    non-overlapping so their order is irrelevant)
//!
//! Each level iterator yields `(internal_key, value)` pairs in sorted
//! internal-key order. Because the internal key encodes `!seq`, newer
//! versions of the same user key precede older ones in the natural ordering.
//!
//! # Merging
//!
//! [`MergingIter`] holds one [`LevelIter`] per source and advances by
//! picking the smallest internal key across all valid sources on every
//! step. The top-level [`LarkIterator`] then:
//!
//! - Drops entries with `seq > snapshot_seq` (not visible at the captured
//!   snapshot).
//! - Deduplicates by user key, keeping only the newest visible version.
//! - Treats a tombstone as "this user key is deleted" — the whole key is
//!   skipped, and any older versions in lower levels are suppressed.
//!
//! # Safety against concurrent compaction
//!
//! The iterator holds an [`Arc<Version>`](super::manifest::Version) and an
//! `Arc<SsTableReader>` per file. Each reader keeps its SSTable file
//! handle open for the reader's lifetime, so even if compaction unlinks
//! the path the OS preserves the bytes via file-descriptor refcounting.

use std::io;
use std::ops::Bound;
use std::sync::Arc;

use super::block_cache::BlockCache;
use super::internal_key::{decode_internal_key, lookup_key, VALUE_TYPE_DELETION};
use super::manifest::Version;
use super::memtable::MemTable;
use super::sstable::SsTableReader;

/// A cursor into one ordered source that yields `(internal_key, value)`
/// pairs. Used by the merging iterator.
enum LevelIter {
    Memtable(MemtableLevelIter),
    SsTable(SsTableLevelIter),
}

impl LevelIter {
    fn seek_to_first(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_to_first();
                Ok(())
            }
            Self::SsTable(it) => it.seek_to_first(),
        }
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek(target);
                Ok(())
            }
            Self::SsTable(it) => it.seek(target),
        }
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_for_prev(target);
                Ok(())
            }
            Self::SsTable(it) => it.seek_for_prev(target),
        }
    }

    fn advance(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.advance();
                Ok(())
            }
            Self::SsTable(it) => it.advance(),
        }
    }

    fn key(&self) -> Option<&[u8]> {
        match self {
            Self::Memtable(it) => it.curr.as_ref().map(|(k, _)| k.as_slice()),
            Self::SsTable(it) => it.curr.as_ref().map(|(k, _)| k.as_slice()),
        }
    }

    fn value(&self) -> Option<&[u8]> {
        match self {
            Self::Memtable(it) => it.curr.as_ref().map(|(_, v)| v.as_slice()),
            Self::SsTable(it) => it.curr.as_ref().map(|(_, v)| v.as_slice()),
        }
    }
}

struct MemtableLevelIter {
    mt: Arc<MemTable>,
    curr: Option<(Vec<u8>, Vec<u8>)>,
}

impl MemtableLevelIter {
    fn new(mt: Arc<MemTable>) -> Self {
        Self { mt, curr: None }
    }

    fn seek_to_first(&mut self) {
        self.curr = self.mt.first_entry_from(Bound::Unbounded);
    }

    fn seek(&mut self, target: &[u8]) {
        self.curr = self.mt.first_entry_from(Bound::Included(target));
    }

    fn seek_for_prev(&mut self, target: &[u8]) {
        self.curr = self.mt.last_entry_before(Bound::Included(target));
    }

    fn advance(&mut self) {
        if let Some((k, _)) = self.curr.take() {
            self.curr = self.mt.first_entry_from(Bound::Excluded(k.as_slice()));
        }
    }
}

struct SsTableLevelIter {
    reader: Arc<SsTableReader>,
    cache: Arc<BlockCache>,
    block_idx: usize,
    block_entries: Vec<(Vec<u8>, Vec<u8>)>,
    entry_pos: usize,
    curr: Option<(Vec<u8>, Vec<u8>)>,
}

impl SsTableLevelIter {
    fn new(reader: Arc<SsTableReader>, cache: Arc<BlockCache>) -> Self {
        Self {
            reader,
            cache,
            block_idx: 0,
            block_entries: Vec::new(),
            entry_pos: 0,
            curr: None,
        }
    }

    fn seek_to_first(&mut self) -> io::Result<()> {
        if self.reader.num_blocks() == 0 {
            self.curr = None;
            return Ok(());
        }
        self.block_idx = 0;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        self.entry_pos = 0;
        self.curr = self.block_entries.first().cloned();
        Ok(())
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        let block_idx = match self.reader.seek_block(target) {
            Some(i) => i,
            None => {
                self.curr = None;
                return Ok(());
            }
        };
        self.block_idx = block_idx;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        // Within the block, find the first entry >= target.
        self.entry_pos = self
            .block_entries
            .iter()
            .position(|(k, _)| k.as_slice() >= target)
            .unwrap_or(self.block_entries.len());
        // If the target falls past the end of this block, advance to the
        // next block — the block the binary-search landed on has
        // `last_key >= target` but its contents may all be < target if the
        // last key exactly equals target. In practice seek_block returns
        // the right block, but handle the edge defensively.
        if self.entry_pos >= self.block_entries.len() {
            self.block_idx += 1;
            if self.block_idx >= self.reader.num_blocks() {
                self.curr = None;
                return Ok(());
            }
            self.block_entries = self
                .reader
                .load_block_entries(self.block_idx, &self.cache)?;
            self.entry_pos = 0;
        }
        self.curr = self.block_entries.get(self.entry_pos).cloned();
        Ok(())
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        // Find the block that could contain the largest key <= target.
        // `seek_block` returns the first block whose last-key is >= target,
        // which is where such a key would live. If no block qualifies (all
        // last-keys are > target), the target could still live in the very
        // first block if its first key is <= target.
        let candidate_block = self.reader.seek_block(target).or_else(|| {
            if self.reader.num_blocks() > 0 {
                Some(0)
            } else {
                None
            }
        });
        let block_idx = match candidate_block {
            Some(i) => i,
            None => {
                self.curr = None;
                return Ok(());
            }
        };
        self.block_idx = block_idx;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        // Within the block, find the largest entry <= target by a linear
        // walk. This is O(block_size / entry_size), which is fine — one
        // block is ~hundreds of entries.
        let mut best: Option<usize> = None;
        for (i, (k, _)) in self.block_entries.iter().enumerate() {
            if k.as_slice() <= target {
                best = Some(i);
            } else {
                break;
            }
        }
        match best {
            Some(i) => {
                self.entry_pos = i;
                self.curr = self.block_entries.get(i).cloned();
            }
            None => {
                // No entry in this block is <= target. Try the previous
                // block if any — its last entry is necessarily < target.
                if self.block_idx == 0 {
                    self.curr = None;
                    return Ok(());
                }
                self.block_idx -= 1;
                self.block_entries = self
                    .reader
                    .load_block_entries(self.block_idx, &self.cache)?;
                self.entry_pos = self.block_entries.len().saturating_sub(1);
                self.curr = self.block_entries.last().cloned();
            }
        }
        Ok(())
    }

    fn advance(&mut self) -> io::Result<()> {
        self.entry_pos += 1;
        if self.entry_pos < self.block_entries.len() {
            self.curr = self.block_entries.get(self.entry_pos).cloned();
            return Ok(());
        }
        // Move to the next block.
        self.block_idx += 1;
        if self.block_idx >= self.reader.num_blocks() {
            self.curr = None;
            return Ok(());
        }
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        self.entry_pos = 0;
        self.curr = self.block_entries.first().cloned();
        Ok(())
    }
}

/// Merges multiple `LevelIter`s into a single stream of internal-key /
/// value pairs in ascending internal-key order. Picks the winning source
/// on every step via linear scan — for the small number of levels lark
/// produces (≤ 30 in worst case) this beats a heap's overhead.
struct MergingIter {
    levels: Vec<LevelIter>,
    current_idx: Option<usize>,
}

impl MergingIter {
    fn new(levels: Vec<LevelIter>) -> Self {
        Self {
            levels,
            current_idx: None,
        }
    }

    fn seek_to_first(&mut self) -> io::Result<()> {
        for lvl in &mut self.levels {
            lvl.seek_to_first()?;
        }
        self.pick_smallest();
        Ok(())
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        for lvl in &mut self.levels {
            lvl.seek(target)?;
        }
        self.pick_smallest();
        Ok(())
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        // For seek_for_prev, each level positions at the largest key ≤
        // target, and the merging step picks the *largest* across all
        // levels (because ≤ target means "closest to target from below").
        for lvl in &mut self.levels {
            lvl.seek_for_prev(target)?;
        }
        self.pick_largest();
        Ok(())
    }

    fn advance(&mut self) -> io::Result<()> {
        if let Some(idx) = self.current_idx {
            self.levels[idx].advance()?;
        }
        self.pick_smallest();
        Ok(())
    }

    fn key(&self) -> Option<&[u8]> {
        self.current_idx.and_then(|i| self.levels[i].key())
    }

    fn value(&self) -> Option<&[u8]> {
        self.current_idx.and_then(|i| self.levels[i].value())
    }

    fn pick_smallest(&mut self) {
        let mut best: Option<usize> = None;
        for (i, lvl) in self.levels.iter().enumerate() {
            let Some(k) = lvl.key() else { continue };
            match best {
                None => best = Some(i),
                Some(bi) => {
                    // Safe: best is set so bi has Some key too.
                    let bk = self.levels[bi].key().unwrap();
                    if k < bk {
                        best = Some(i);
                    }
                }
            }
        }
        self.current_idx = best;
    }

    fn pick_largest(&mut self) {
        let mut best: Option<usize> = None;
        for (i, lvl) in self.levels.iter().enumerate() {
            let Some(k) = lvl.key() else { continue };
            match best {
                None => best = Some(i),
                Some(bi) => {
                    let bk = self.levels[bi].key().unwrap();
                    if k > bk {
                        best = Some(i);
                    }
                }
            }
        }
        self.current_idx = best;
    }
}

/// Streaming iterator over a consistent view of the database at a given
/// snapshot sequence. Wraps [`MergingIter`] and adds user-key
/// deduplication, snapshot visibility filtering, and tombstone suppression.
pub(crate) struct LarkIterator {
    inner: MergingIter,
    snapshot_seq: u64,
    /// Most recently produced `(user_key, value)` pair, if any.
    curr_user: Option<(Vec<u8>, Vec<u8>)>,
    /// Pinning handle — keeping this alive guarantees compaction cannot
    /// drop SSTable metadata out from under us. The SsTableReaders owned
    /// by the level iterators similarly keep file handles live.
    _version: Arc<Version>,
    /// Sticky error from the most recent I/O attempt, if any. Cleared on
    /// the next successful seek.
    error: Option<io::Error>,
}

impl LarkIterator {
    /// Build an iterator over `(active_memtable, frozen_memtables, version)`
    /// at the given snapshot sequence. SSTable readers come directly from
    /// the pinned `Version`, which holds `Arc<LiveSst>`s whose file
    /// descriptors are guaranteed to stay open for the lifetime of any
    /// version that still references them — so the iterator is immune to
    /// concurrent compaction unlinking files.
    pub(crate) fn new(
        active: Arc<MemTable>,
        frozen: Vec<Arc<MemTable>>,
        version: Arc<Version>,
        cache: Arc<BlockCache>,
        snapshot_seq: u64,
    ) -> Self {
        let mut levels: Vec<LevelIter> = Vec::new();
        levels.push(LevelIter::Memtable(MemtableLevelIter::new(active)));
        for mt in frozen.iter().rev() {
            levels.push(LevelIter::Memtable(MemtableLevelIter::new(Arc::clone(mt))));
        }
        // L0: newest first.
        for file in version.levels[0].iter().rev() {
            levels.push(LevelIter::SsTable(SsTableLevelIter::new(
                Arc::clone(&file.reader),
                Arc::clone(&cache),
            )));
        }
        // L1+: within a level file order doesn't matter (non-overlapping).
        for level in 1..version.levels.len() {
            for file in &version.levels[level] {
                levels.push(LevelIter::SsTable(SsTableLevelIter::new(
                    Arc::clone(&file.reader),
                    Arc::clone(&cache),
                )));
            }
        }

        Self {
            inner: MergingIter::new(levels),
            snapshot_seq,
            curr_user: None,
            _version: version,
            error: None,
        }
    }

    pub(crate) fn seek_to_first(&mut self) {
        self.error = None;
        self.curr_user = None;
        if let Err(e) = self.inner.seek_to_first() {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn seek(&mut self, target: &[u8]) {
        self.error = None;
        self.curr_user = None;
        // Smallest internal key for `target` at any seq: `target || !u64::MAX || 0`.
        // This positions the merging iterator at the newest version of the
        // target user key, or the first user key > target if none exists.
        let search_key = lookup_key(target, u64::MAX);
        if let Err(e) = self.inner.seek(&search_key) {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn seek_for_prev(&mut self, target: &[u8]) {
        self.error = None;
        self.curr_user = None;
        // Largest internal key with user key `target`: entries within a
        // user key sort by `!seq` ascending, so the largest is
        // `target || !0 || 0`. Reverse-seeking to this probe puts the
        // merging iterator at the most recent version of `target` or the
        // last version of the preceding user key.
        let probe = lookup_key(target, 0);
        if let Err(e) = self.inner.seek_for_prev(&probe) {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn next(&mut self) {
        if self.curr_user.is_none() || self.error.is_some() {
            return;
        }
        // `materialize_next_visible` always leaves the inner iterator
        // advanced past the current user key (via `consume_user_key`), so
        // a plain call here picks up the next visible user key.
        self.curr_user = None;
        self.materialize_next_visible();
    }

    pub(crate) fn valid(&self) -> bool {
        self.curr_user.is_some() && self.error.is_none()
    }

    pub(crate) fn key(&self) -> Option<&[u8]> {
        self.curr_user.as_ref().map(|(k, _)| k.as_slice())
    }

    pub(crate) fn value(&self) -> Option<&[u8]> {
        self.curr_user.as_ref().map(|(_, v)| v.as_slice())
    }

    pub(crate) fn status(&self) -> io::Result<()> {
        match &self.error {
            Some(e) => Err(io::Error::new(e.kind(), e.to_string())),
            None => Ok(()),
        }
    }

    /// Walk the merging iterator forward until the first user key whose
    /// most-recent visible version is a live value; set `curr_user` to it
    /// and advance the inner iterator past every remaining entry in that
    /// user-key group. If no live user key remains, `curr_user` stays
    /// `None` and the iterator becomes invalid.
    fn materialize_next_visible(&mut self) {
        loop {
            let Some(ik) = self.inner.key() else {
                self.curr_user = None;
                return;
            };
            let (uk, seq, vt) = decode_internal_key(ik);

            if seq > self.snapshot_seq {
                // Invisible at this snapshot. Skip this single entry and
                // keep looking — a later entry in the same user-key group
                // may be visible, or we may cross into a new user key.
                if let Err(e) = self.inner.advance() {
                    self.error = Some(e);
                    self.curr_user = None;
                    return;
                }
                continue;
            }

            // Newest visible entry for this user key. If it's a tombstone
            // the whole user key is deleted at this snapshot; otherwise
            // it's the value the public iterator should yield.
            let uk_owned = uk.to_vec();
            if vt == VALUE_TYPE_DELETION {
                self.consume_user_key(&uk_owned);
                continue;
            }
            let v = self.inner.value().map(|s| s.to_vec()).unwrap_or_default();
            self.curr_user = Some((uk_owned.clone(), v));
            self.consume_user_key(&uk_owned);
            return;
        }
    }

    /// Advance the merging iterator past every entry whose user key
    /// matches `user_key`. Called after materializing a live value (to
    /// leave the cursor positioned at the next user key) or after
    /// hitting a tombstone (to skip every older version in the group).
    fn consume_user_key(&mut self, user_key: &[u8]) {
        loop {
            let Some(ik) = self.inner.key() else { return };
            let (uk, _, _) = decode_internal_key(ik);
            if uk != user_key {
                return;
            }
            if let Err(e) = self.inner.advance() {
                self.error = Some(e);
                return;
            }
        }
    }
}
