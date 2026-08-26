//! Streaming iterator: merges memtable + frozen memtables + all SSTable
//! levels into a single cursor honoring MVCC snapshot visibility and
//! tombstone suppression, supporting both forward and reverse iteration.
//!
//! # Sources
//!
//! The iterator is built from a priority-ordered list of **level iterators**:
//!
//! 1. The active memtable (newest data)
//! 2. Frozen memtables (newest first)
//! 3. L0 SSTables (newest first; L0 files may overlap)
//! 4. L1..Ln SSTables (one level at a time; point files within a level
//!    are non-overlapping, and RT-only files carry no point entries)
//!
//! Each level iterator yields `(internal_key, value)` pairs in sorted
//! internal-key order. Because the internal key encodes `!seq`, newer
//! versions of the same user key precede older ones in forward order.
//!
//! # Merging
//!
//! [`MergingIter`] holds one [`LevelIter`] per source. In forward mode it
//! picks the smallest internal key across all valid sources; in reverse
//! mode it picks the largest. The top-level [`LarkIterator`] then:
//!
//! - Drops entries with `seq > snapshot_seq` (not visible at the captured
//!   snapshot).
//! - Deduplicates by user key, keeping only the newest visible version.
//! - Treats a tombstone as "this user key is deleted" - the whole group
//!   is skipped, and any older versions in lower levels are suppressed.
//!
//! # Forward vs reverse materialization
//!
//! In **forward** mode, entries within a user-key group are visited newest-
//! seq first, so the first visible entry we see is the answer - the rest
//! of the group can be consumed quickly.
//!
//! In **reverse** mode, entries within a user-key group are visited
//! oldest-seq first (because internal keys are `!seq`-sorted). We must
//! scan the *entire* group before we know the latest visible version, so
//! [`LarkIterator::materialize_prev_visible`] accumulates the most recent
//! visible entry seen and emits it once the group ends.
//!
//! # Direction changes
//!
//! Calling `next()` while in reverse mode (or vice versa) flips the
//! direction. To avoid yielding the already-emitted user key again, the
//! iterator re-seeks every level to the position just past the current
//! user key in the new direction - see [`LarkIterator::flip_to_forward`]
//! and [`LarkIterator::flip_to_reverse`].
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

use super::block::encoded_entry_size;
use super::block::{Block, RESTART_INTERVAL, decode_entry_at};
use super::block_cache::BlockCache;
use super::internal_key::{
    INTERNAL_KEY_SUFFIX_LEN, VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE,
    encode_internal_key,
    compare_internal_keys, decode_internal_key, user_key_of,
};
use super::lookup_key::LookupKey;
use super::manifest::Version;
use super::memtable::MemTable;
use super::range_tombstone::{RangeTombstone, RangeTombstoneSet};
use super::sstable::{LiveSst, SsTableBlockCursor, SsTableReader};
use crate::DbSlice;
use crate::options::MergeOperator;
use crate::options::PrefixExtractor;

/// Scan direction for [`LarkIterator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Reverse,
}

/// Build an internal-key probe strictly greater than any real entry for
/// `user_key`. Used to seek past the last version of a user key when
/// switching from reverse to forward iteration.
fn above_all_versions(user_key: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(user_key.len() + INTERNAL_KEY_SUFFIX_LEN);
    buf.extend_from_slice(user_key);
    buf.extend_from_slice(&[0xff; INTERNAL_KEY_SUFFIX_LEN]);
    buf
}

/// A cursor into one ordered source that yields `(internal_key, value)`
/// pairs. Used by the merging iterator.
enum LevelIter {
    Memtable(MemtableLevelIter),
    SsTable(SsTableLevelIter),
    LevelConcat(LevelConcatIter),
}

impl LevelIter {
    fn seek_to_first(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_to_first();
                Ok(())
            }
            Self::SsTable(it) => it.seek_to_first(),
            Self::LevelConcat(it) => it.seek_to_first(),
        }
    }

    fn seek_to_last(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_to_last();
                Ok(())
            }
            Self::SsTable(it) => it.seek_to_last(),
            Self::LevelConcat(it) => it.seek_to_last(),
        }
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek(target);
                Ok(())
            }
            Self::SsTable(it) => it.seek(target),
            Self::LevelConcat(it) => it.seek(target),
        }
    }

    /// Like [`seek`], but additionally consults the underlying
    /// SSTable's prefix bloom filter (when present) and leaves the
    /// level iterator empty if the file cannot contain any key with
    /// the given prefix. Memtable levels always perform a normal seek.
    fn seek_with_prefix_skip(&mut self, target: &[u8], prefix: &[u8]) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek(target);
                Ok(())
            }
            Self::SsTable(it) => {
                if !it.reader.may_have_prefix(prefix, &it.cache)? {
                    it.valid = false;
                    return Ok(());
                }
                it.seek(target)
            }
            Self::LevelConcat(it) => it.seek(target),
        }
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_for_prev(target);
                Ok(())
            }
            Self::SsTable(it) => it.seek_for_prev(target),
            Self::LevelConcat(it) => it.seek_for_prev(target),
        }
    }

    fn advance(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.advance();
                Ok(())
            }
            Self::SsTable(it) => it.advance(),
            Self::LevelConcat(it) => it.advance(),
        }
    }

    fn advance_backward(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.advance_backward();
                Ok(())
            }
            Self::SsTable(it) => it.advance_backward(),
            Self::LevelConcat(it) => it.advance_backward(),
        }
    }

    fn key(&self) -> Option<&[u8]> {
        match self {
            Self::Memtable(it) => it.curr.as_ref().map(|(k, _)| k.as_slice()),
            Self::SsTable(it) => {
                if it.valid {
                    Some(&it.cached_key[..])
                } else {
                    None
                }
            }
            Self::LevelConcat(it) => it.key(),
        }
    }

    fn value(&self) -> Option<&[u8]> {
        match self {
            Self::Memtable(it) => it.curr.as_ref().map(|(_, v)| v.as_slice()),
            Self::SsTable(it) => it.value(),
            Self::LevelConcat(it) => it.value(),
        }
    }

    /// The current value as an owning view, so it survives the cursor
    /// moving on. Free for an SSTable source (one reference count on
    /// the block already in memory); a copy for a memtable source,
    /// whose values are separately owned buffers.
    fn value_slice(&self) -> Option<DbSlice> {
        match self {
            Self::Memtable(it) => it.curr.as_ref().map(|(_, v)| v.clone()),
            Self::SsTable(it) => it.value_slice(),
            Self::LevelConcat(it) => it.value_slice(),
        }
    }
}

/// Cursor over one memtable.
///
/// The skip list has no back pointers, so every step is a fresh
/// `O(log N)` seek from the current key. Both halves of `curr` are
/// arena-backed views rather than copies, so a step costs two reference
/// counts and no allocation; the previous step's views are dropped only
/// after the new seek has read the key it seeks from.
struct MemtableLevelIter {
    mt: Arc<MemTable>,
    curr: Option<(DbSlice, DbSlice)>,
}

impl MemtableLevelIter {
    fn new(mt: Arc<MemTable>) -> Self {
        Self { mt, curr: None }
    }

    fn seek_to_first(&mut self) {
        self.curr = self.mt.first_slice_from(Bound::Unbounded);
    }

    fn seek_to_last(&mut self) {
        self.curr = self.mt.last_slice_before(Bound::Unbounded);
    }

    fn seek(&mut self, target: &[u8]) {
        self.curr = self.mt.first_slice_from(Bound::Included(target));
    }

    fn seek_for_prev(&mut self, target: &[u8]) {
        self.curr = self.mt.last_slice_before(Bound::Included(target));
    }

    fn advance(&mut self) {
        let Some((key, _)) = self.curr.take() else {
            return;
        };
        self.curr = self.mt.first_slice_from(Bound::Excluded(key.as_slice()));
    }

    fn advance_backward(&mut self) {
        let Some((key, _)) = self.curr.take() else {
            return;
        };
        self.curr = self.mt.last_slice_before(Bound::Excluded(key.as_slice()));
    }
}

struct SsTableLevelIter {
    reader: Arc<SsTableReader>,
    cache: Arc<BlockCache>,
    block: Option<Arc<Block>>,
    block_cursor: Option<SsTableBlockCursor>,
    /// Byte offset of the current entry within `block.entry_data()`.
    entry_pos: usize,
    /// Byte offset just past the current entry (start of the next).
    next_entry_pos: usize,
    cached_key: Vec<u8>,
    cached_value_offset: usize,
    cached_value_len: usize,
    valid: bool,
    current_entry_index: Option<usize>,
    /// Lazily built on the first backward step; maps entry index to
    /// byte offset in entry_data.
    entry_offsets: Option<Vec<usize>>,
}

impl SsTableLevelIter {
    fn new(reader: Arc<SsTableReader>, cache: Arc<BlockCache>) -> Self {
        Self {
            reader,
            cache,
            block: None,
            block_cursor: None,
            entry_pos: 0,
            next_entry_pos: 0,
            cached_key: Vec::new(),
            cached_value_offset: 0,
            cached_value_len: 0,
            valid: false,
            current_entry_index: None,
            entry_offsets: None,
        }
    }

    /// The current value, borrowed from the decoded block holding it.
    fn value(&self) -> Option<&[u8]> {
        if !self.valid {
            return None;
        }
        let data = self.block.as_ref()?.entry_data();
        let end = self
            .cached_value_offset
            .checked_add(self.cached_value_len)?;
        data.get(self.cached_value_offset..end)
    }

    /// The current value as an owning view over the block holding it.
    fn value_slice(&self) -> Option<DbSlice> {
        if !self.valid {
            return None;
        }
        let block = Arc::clone(self.block.as_ref()?);
        DbSlice::from_block(block, self.cached_value_offset, self.cached_value_len)
    }

    fn load_block(&mut self, cursor: SsTableBlockCursor) -> io::Result<()> {
        self.block = Some(self.reader.load_block_at_cursor(&cursor, &self.cache)?);
        self.block_cursor = Some(cursor);
        self.entry_pos = 0;
        self.next_entry_pos = 0;
        self.entry_offsets = None;
        self.current_entry_index = None;
        self.cached_key.clear();
        Ok(())
    }

    fn decode_current(&mut self) {
        let data = self.block.as_ref().unwrap().entry_data();
        if self.entry_pos >= data.len() {
            self.valid = false;
            return;
        }
        let (consumed, val_off, val_len) =
            decode_entry_at(data, self.entry_pos, &mut self.cached_key);
        self.next_entry_pos = self.entry_pos + consumed;
        self.cached_value_offset = val_off;
        self.cached_value_len = val_len;
        self.current_entry_index = self
            .entry_offsets
            .as_ref()
            .and_then(|offsets| offsets.binary_search(&self.entry_pos).ok());
        self.valid = true;
    }

    fn seek_to_first(&mut self) -> io::Result<()> {
        let Some(cursor) = self.reader.first_block_cursor(&self.cache)? else {
            self.valid = false;
            return Ok(());
        };
        self.load_block(cursor)?;
        self.decode_current();
        Ok(())
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        let cursor = match self.reader.seek_block_cursor(target, &self.cache)? {
            Some(cursor) => cursor,
            None => {
                self.valid = false;
                return Ok(());
            }
        };
        self.load_block(cursor)?;
        let data = self.block.as_ref().unwrap().entry_data();
        self.entry_pos = 0;
        self.cached_key.clear();
        while self.entry_pos < data.len() {
            let (consumed, val_off, val_len) =
                decode_entry_at(data, self.entry_pos, &mut self.cached_key);
            self.next_entry_pos = self.entry_pos + consumed;
            self.cached_value_offset = val_off;
            self.cached_value_len = val_len;
            if compare_internal_keys(&self.cached_key, target).is_ge() {
                self.valid = true;
                return Ok(());
            }
            self.entry_pos = self.next_entry_pos;
        }
        // Fell off end of block - try next block.
        let next = self
            .reader
            .next_block_cursor(self.block_cursor.as_ref().unwrap(), &self.cache)?;
        let Some(next) = next else {
            self.valid = false;
            return Ok(());
        };
        self.load_block(next)?;
        self.decode_current();
        Ok(())
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        let cursor = match self.reader.seek_block_cursor(target, &self.cache)? {
            Some(cursor) => cursor,
            None => match self.reader.last_block_cursor(&self.cache)? {
                Some(cursor) => cursor,
                None => {
                    self.valid = false;
                    return Ok(());
                }
            },
        };
        self.load_block(cursor)?;
        self.build_entry_offsets();
        let offsets = self.entry_offsets.as_ref().unwrap();
        let data = self.block.as_ref().unwrap().entry_data();
        let mut best: Option<usize> = None;
        let mut temp_key = Vec::new();
        for (i, &off) in offsets.iter().enumerate() {
            let (_consumed, _vo, _vl) = decode_entry_at(data, off, &mut temp_key);
            if compare_internal_keys(&temp_key, target).is_le() {
                best = Some(i);
            } else {
                break;
            }
        }
        match best {
            Some(idx) => {
                self.replay_key_to_index(idx);
                self.valid = true;
            }
            None => {
                let prev = self
                    .reader
                    .prev_block_cursor(self.block_cursor.as_ref().unwrap(), &self.cache)?;
                let Some(prev) = prev else {
                    self.valid = false;
                    return Ok(());
                };
                self.load_block(prev)?;
                self.build_entry_offsets();
                let Some(last) = self.last_entry_index() else {
                    self.valid = false;
                    return Ok(());
                };
                self.replay_key_to_index(last);
                self.valid = true;
            }
        }
        Ok(())
    }

    fn advance(&mut self) -> io::Result<()> {
        if !self.valid {
            return Ok(());
        }
        self.entry_pos = self.next_entry_pos;
        let data = self.block.as_ref().unwrap().entry_data();
        if self.entry_pos >= data.len() {
            let next = self
                .reader
                .next_block_cursor(self.block_cursor.as_ref().unwrap(), &self.cache)?;
            let Some(next) = next else {
                self.valid = false;
                return Ok(());
            };
            self.load_block(next)?;
        }
        self.decode_current();
        Ok(())
    }

    fn seek_to_last(&mut self) -> io::Result<()> {
        let Some(cursor) = self.reader.last_block_cursor(&self.cache)? else {
            self.valid = false;
            return Ok(());
        };
        self.load_block(cursor)?;
        self.build_entry_offsets();
        let Some(last) = self.last_entry_index() else {
            self.valid = false;
            return Ok(());
        };
        self.replay_key_to_index(last);
        self.valid = true;
        Ok(())
    }

    fn advance_backward(&mut self) -> io::Result<()> {
        if !self.valid {
            return Ok(());
        }
        if self.entry_offsets.is_none() {
            self.build_entry_offsets();
        }
        let offsets = self.entry_offsets.as_ref().unwrap();
        let cur_idx = self.current_entry_index.unwrap_or_else(|| {
            offsets
                .binary_search(&self.entry_pos)
                .unwrap_or_else(|idx| idx.saturating_sub(1))
        });
        if cur_idx == 0 {
            let prev = self
                .reader
                .prev_block_cursor(self.block_cursor.as_ref().unwrap(), &self.cache)?;
            let Some(prev) = prev else {
                self.valid = false;
                return Ok(());
            };
            self.load_block(prev)?;
            self.build_entry_offsets();
            let Some(last) = self.last_entry_index() else {
                self.valid = false;
                return Ok(());
            };
            self.replay_key_to_index(last);
            self.valid = true;
            return Ok(());
        }
        self.replay_key_to_index(cur_idx - 1);
        self.valid = true;
        Ok(())
    }

    fn build_entry_offsets(&mut self) {
        let data = self.block.as_ref().unwrap().entry_data();
        let mut offsets = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            offsets.push(pos);
            pos += encoded_entry_size(data, pos);
        }
        self.entry_offsets = Some(offsets);
    }

    /// Index of the last entry in the currently loaded block, or `None`
    /// when that block holds no entries.
    ///
    /// A range-tombstone-only SSTable has zero data entries, so every
    /// reverse-stepping path has to treat an empty block as "keep
    /// walking back" rather than indexing one before the start.
    fn last_entry_index(&self) -> Option<usize> {
        self.entry_offsets.as_ref()?.len().checked_sub(1)
    }

    /// Replay key reconstruction from the nearest restart point up to
    /// `target_idx` in `entry_offsets`. After this call `cached_key`,
    /// `entry_pos`, `next_entry_pos`, and cached value metadata are
    /// all consistent with the entry at `target_idx`.
    fn replay_key_to_index(&mut self, target_idx: usize) {
        let restart_idx = target_idx / RESTART_INTERVAL;
        let block = self.block.as_ref().unwrap();
        let data = block.entry_data();
        let start_offset = if restart_idx > 0 && restart_idx < block.restart_count() {
            block.restart_offset(restart_idx)
        } else {
            0
        };
        self.cached_key.clear();
        let start_entry_idx = restart_idx * RESTART_INTERVAL;
        let mut pos = start_offset;
        for idx in start_entry_idx..=target_idx {
            let (consumed, val_off, val_len) = decode_entry_at(data, pos, &mut self.cached_key);
            if idx == target_idx {
                self.entry_pos = pos;
                self.next_entry_pos = pos + consumed;
                self.cached_value_offset = val_off;
                self.cached_value_len = val_len;
                self.current_entry_index = Some(target_idx);
            }
            pos += consumed;
        }
    }
}

// ─── LevelConcatIter ───────────────────────────────────────────────────────

/// Concatenation iterator for a sorted L1+ level. Opens one SSTable at a
/// time, skipping RT-only files whose metadata can overlap point files at
/// range boundaries.
struct LevelConcatIter {
    files: Vec<Arc<LiveSst>>,
    cache: Arc<BlockCache>,
    file_idx: usize,
    current: Option<SsTableLevelIter>,
}

impl LevelConcatIter {
    fn new(files: Vec<Arc<LiveSst>>, cache: Arc<BlockCache>) -> Self {
        Self {
            files,
            cache,
            file_idx: 0,
            current: None,
        }
    }

    fn open_current(&mut self) -> io::Result<()> {
        if self.file_idx < self.files.len() {
            self.current = Some(SsTableLevelIter::new(
                Arc::clone(&self.files[self.file_idx].reader),
                Arc::clone(&self.cache),
            ));
        } else {
            self.current = None;
        }
        Ok(())
    }

    fn seek_to_first(&mut self) -> io::Result<()> {
        if self.files.is_empty() {
            self.current = None;
            return Ok(());
        }

        self.file_idx = 0;
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek_to_first()?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            self.file_idx += 1;
            if self.file_idx >= self.files.len() {
                self.current = None;
                return Ok(());
            }
        }
    }

    fn seek_to_last(&mut self) -> io::Result<()> {
        if self.files.is_empty() {
            self.current = None;
            return Ok(());
        }

        self.file_idx = self.files.len() - 1;
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek_to_last()?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            if self.file_idx == 0 {
                self.current = None;
                return Ok(());
            }
            self.file_idx -= 1;
        }
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        if self.files.is_empty() {
            self.current = None;
            return Ok(());
        }
        let uk = user_key_of(target);
        let idx = self
            .files
            .partition_point(|f| f.meta.largest_key.as_slice() < uk);
        if idx >= self.files.len() {
            self.current = None;
            return Ok(());
        }
        self.file_idx = idx;
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek(target)?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            self.file_idx += 1;
            if self.file_idx >= self.files.len() {
                self.current = None;
                return Ok(());
            }
        }
    }

    fn seek_for_prev(&mut self, target: &[u8]) -> io::Result<()> {
        if self.files.is_empty() {
            self.current = None;
            return Ok(());
        }
        let uk = user_key_of(target);
        let idx = self
            .files
            .partition_point(|f| f.meta.largest_key.as_slice() < uk);
        let idx = if idx >= self.files.len() {
            self.files.len() - 1
        } else {
            idx
        };
        self.file_idx = idx;
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek_for_prev(target)?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            if self.file_idx == 0 {
                self.current = None;
                return Ok(());
            }
            self.file_idx -= 1;
        }
    }

    fn advance(&mut self) -> io::Result<()> {
        if let Some(ref mut it) = self.current {
            it.advance()?;
            if it.valid {
                return Ok(());
            }
        }
        self.file_idx += 1;
        if self.file_idx >= self.files.len() {
            self.current = None;
            return Ok(());
        }
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek_to_first()?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            self.file_idx += 1;
            if self.file_idx >= self.files.len() {
                self.current = None;
                return Ok(());
            }
        }
    }

    fn advance_backward(&mut self) -> io::Result<()> {
        if let Some(ref mut it) = self.current {
            it.advance_backward()?;
            if it.valid {
                return Ok(());
            }
        }
        if self.file_idx == 0 {
            self.current = None;
            return Ok(());
        }
        self.file_idx -= 1;
        loop {
            self.open_current()?;
            self.current.as_mut().unwrap().seek_to_last()?;
            if self.current.as_ref().is_some_and(|it| it.valid) {
                return Ok(());
            }
            if self.file_idx == 0 {
                self.current = None;
                return Ok(());
            }
            self.file_idx -= 1;
        }
    }

    fn key(&self) -> Option<&[u8]> {
        self.current.as_ref().and_then(|it| {
            if it.valid {
                Some(it.cached_key.as_slice())
            } else {
                None
            }
        })
    }

    fn value(&self) -> Option<&[u8]> {
        self.current.as_ref().and_then(|it| it.value())
    }

    fn value_slice(&self) -> Option<DbSlice> {
        self.current.as_ref().and_then(|it| it.value_slice())
    }
}

/// Merges multiple `LevelIter`s into a single stream of internal-key /
/// value pairs in ascending internal-key order. Picks the winning source
/// on every step via linear scan - for the small number of levels lark
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

    fn seek_to_last(&mut self) -> io::Result<()> {
        for lvl in &mut self.levels {
            lvl.seek_to_last()?;
        }
        self.pick_largest();
        Ok(())
    }

    fn seek(&mut self, target: &[u8]) -> io::Result<()> {
        for lvl in &mut self.levels {
            lvl.seek(target)?;
        }
        self.pick_smallest();
        Ok(())
    }

    fn seek_with_prefix_skip(&mut self, target: &[u8], prefix: &[u8]) -> io::Result<()> {
        for lvl in &mut self.levels {
            lvl.seek_with_prefix_skip(target, prefix)?;
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

    fn advance_backward(&mut self) -> io::Result<()> {
        if let Some(idx) = self.current_idx {
            self.levels[idx].advance_backward()?;
        }
        self.pick_largest();
        Ok(())
    }

    fn key(&self) -> Option<&[u8]> {
        self.current_idx.and_then(|i| self.levels[i].key())
    }

    fn value(&self) -> Option<&[u8]> {
        self.current_idx.and_then(|i| self.levels[i].value())
    }

    fn value_slice(&self) -> Option<DbSlice> {
        self.current_idx.and_then(|i| self.levels[i].value_slice())
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
                    if compare_internal_keys(k, bk).is_lt() {
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
                    if compare_internal_keys(k, bk).is_gt() {
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
/// deduplication, snapshot visibility filtering, tombstone suppression,
/// and bidirectional iteration.
pub(crate) struct LarkIterator {
    inner: MergingIter,
    snapshot_seq: u64,
    /// Current scan direction. `next()` / `prev()` honor this; calling
    /// the opposite-direction method flips it and re-seeks the merging
    /// iterator so the first move in the new direction lands on the
    /// correct neighbor of `curr_user`.
    direction: Direction,
    /// True when the forward path has positioned the MergingIter on
    /// a visible entry. key()/value() delegate through to the inner
    /// iterator in this state.
    valid_entry: bool,
    /// Reusable buffer holding the user key of the current entry.
    /// Used by consume_curr_user_key_forward and covering_rt_seq
    /// during materialization. NOT used by key()/value().
    curr_user_key: Vec<u8>,
    /// Owned storage used only by the reverse iteration path
    /// (materialize_prev_visible), which must accumulate multiple
    /// entries before deciding which one to yield.
    reverse_curr: Option<(Vec<u8>, Vec<u8>)>,
    /// Owned storage for the rare merge-result case, where the
    /// value is computed rather than borrowed from a block.
    merge_result: Option<(Vec<u8>, Vec<u8>)>,
    /// Snapshot of every range tombstone the iterator must honor, indexed
    /// once at construction so per-key visibility checks do not re-lock
    /// memtables or scan unrelated ranges.
    range_tombstones: RangeTombstoneSet,
    /// Pinning handle - keeping this alive guarantees compaction cannot
    /// drop SSTable metadata out from under us. The SsTableReaders owned
    /// by the level iterators similarly keep file handles live.
    _version: Arc<Version>,
    /// Sticky error from the most recent I/O attempt, if any. Cleared on
    /// the next successful seek.
    error: Option<io::Error>,
    /// True when `error` represents a terminal iterator state, such as
    /// constructing an iterator from a closed engine. Terminal errors
    /// are reported by `status` and are not cleared by later seeks.
    terminal_error: bool,
    /// Exclusive upper bound set by [`LarkIterator::seek_prefix`]. When
    /// `Some`, forward iteration stops as soon as the next visible key
    /// is `>= upper_bound`, confining the scan to the originally seeked
    /// prefix. Cleared by any other seek.
    upper_bound: Option<Vec<u8>>,
    /// Prefix extractor captured at construction. Used by
    /// [`LarkIterator::seek_prefix`] to derive the bloom probe for
    /// the caller's query prefix. If the extractor cannot produce a
    /// prefix from the query itself, the iterator falls back to a
    /// plain upper-bound scan without bloom skipping.
    prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
    /// Optional merge operator. When `Some`, the iterator collapses
    /// merge chains encountered during materialization into a
    /// single final value via [`MergeOperator::full_merge`].
    merge_operator: Option<Arc<dyn MergeOperator>>,
    /// When `true`, the MergingIter is still positioned at the
    /// entry we last yielded (or an older version of the same user
    /// key). The next `next()` must consume the rest of that
    /// user-key group before materializing a new visible entry.
    /// This defers the `consume_curr_user_key_forward` work from
    /// the end of one `next()` to the start of the following one,
    /// halving the number of `advance() + pick_smallest()` cycles
    /// per visible entry in a single-version sequential scan.
    pending_consume: bool,
    /// Last user key handed to the caller during forward iteration.
    /// A sorted merge must yield strictly increasing user keys; a file
    /// whose index has been corrupted into pointing back at a block
    /// already visited would otherwise stream duplicate keys forever,
    /// so a violation is reported as corruption instead of scanned.
    last_forward_user_key: Option<Vec<u8>>,
}

/// Compute the exclusive upper bound of all keys that start with
/// `prefix`: the shortest byte string strictly greater than every
/// `prefix || ...` key. Returns `None` if `prefix` is empty or
/// consists entirely of `0xff` bytes, in which case every key `>=
/// prefix` is in-bounds.
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(last) = out.last_mut() {
        if *last != 0xff {
            *last += 1;
            return Some(out);
        }
        out.pop();
    }
    None
}

impl LarkIterator {
    /// Build an iterator over `(active_memtable, frozen_memtables, version)`
    /// at the given snapshot sequence. SSTable readers come directly from
    /// the pinned `Version`, which holds `Arc<LiveSst>`s whose file
    /// descriptors are guaranteed to stay open for the lifetime of any
    /// version that still references them - so the iterator is immune to
    /// concurrent compaction unlinking files.
    pub(crate) fn new(
        active: Arc<MemTable>,
        frozen: Vec<Arc<MemTable>>,
        version: Arc<Version>,
        cache: Arc<BlockCache>,
        snapshot_seq: u64,
        prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
        merge_operator: Option<Arc<dyn MergeOperator>>,
    ) -> Self {
        let mut levels: Vec<LevelIter> = Vec::new();
        let mut range_tombstones: Vec<RangeTombstone> = Vec::new();

        range_tombstones.extend(active.clone_range_tombstones());
        levels.push(LevelIter::Memtable(MemtableLevelIter::new(active)));
        for mt in frozen.iter().rev() {
            range_tombstones.extend(mt.clone_range_tombstones());
            levels.push(LevelIter::Memtable(MemtableLevelIter::new(Arc::clone(mt))));
        }
        // L0: newest first.
        for file in version.levels[0].iter().rev() {
            range_tombstones.extend(file.reader.range_tombstones().iter().cloned());
            levels.push(LevelIter::SsTable(SsTableLevelIter::new(
                Arc::clone(&file.reader),
                Arc::clone(&cache),
            )));
        }
        // L1+: one concatenation iterator per level rather than one
        // LevelIter per file, which `LevelConcatIter` may do only because
        // the files it is handed are sorted and non-overlapping.
        //
        // Compaction also emits range-tombstone-only files, whose key
        // range is the tombstone's and therefore *does* overlap the data
        // files beside it. They carry zero point entries, and their
        // tombstones are collected below for every file in the level
        // regardless, so they are dropped from the concat list: leaving
        // one in place lets `seek_for_prev` bisect onto an empty file and
        // report "no key <= target" while a live key sits in the data
        // file that sorts after it.
        for level in 1..version.levels.len() {
            if version.levels[level].is_empty() {
                continue;
            }
            for file in &version.levels[level] {
                range_tombstones.extend(file.reader.range_tombstones().iter().cloned());
            }
            let mut sorted: Vec<Arc<LiveSst>> = version.levels[level]
                .iter()
                .filter(|f| f.meta.num_entries > 0)
                .map(Arc::clone)
                .collect();
            if sorted.is_empty() {
                continue;
            }
            sorted.sort_by(|a, b| a.meta.smallest_key.cmp(&b.meta.smallest_key));
            levels.push(LevelIter::LevelConcat(LevelConcatIter::new(
                sorted,
                Arc::clone(&cache),
            )));
        }

        Self {
            inner: MergingIter::new(levels),
            snapshot_seq,
            direction: Direction::Forward,
            valid_entry: false,
            curr_user_key: Vec::new(),
            reverse_curr: None,
            merge_result: None,
            range_tombstones: RangeTombstoneSet::from_vec(range_tombstones),
            _version: version,
            error: None,
            terminal_error: false,
            upper_bound: None,
            prefix_extractor,
            merge_operator,
            pending_consume: false,
            last_forward_user_key: None,
        }
    }

    /// Largest seq of any range tombstone covering `user_key` that is
    /// visible at this iterator's snapshot. Returns 0 when none.
    fn covering_rt_seq(&self, user_key: &[u8]) -> u64 {
        if self.range_tombstones.is_empty() {
            return 0;
        }
        self.range_tombstones
            .max_covering_seq(user_key, self.snapshot_seq)
    }

    pub(crate) fn seek_to_first(&mut self) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.upper_bound = None;
        self.direction = Direction::Forward;
        if let Err(e) = self.inner.seek_to_first() {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn seek_to_last(&mut self) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.upper_bound = None;
        self.direction = Direction::Reverse;
        if let Err(e) = self.inner.seek_to_last() {
            self.error = Some(e);
            return;
        }
        self.materialize_prev_visible();
    }

    pub(crate) fn seek(&mut self, target: &[u8]) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.upper_bound = None;
        self.direction = Direction::Forward;
        // Smallest internal key for `target` at any seq: `target || !u64::MAX || 0`.
        // This positions the merging iterator at the newest version of the
        // target user key, or the first user key > target if none exists.
        let search_key = LookupKey::from_prefixed(target, u64::MAX);
        if let Err(e) = self.inner.seek(search_key.internal()) {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn seek_for_prev(&mut self, target: &[u8]) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.upper_bound = None;
        self.direction = Direction::Reverse;
        // Reverse-seek to the largest internal key ≤ `target`. Probe with
        // `above_all_versions(target)` so `seek_for_prev` lands at the
        // oldest-seq entry of `target` itself (or of the preceding user
        // key if `target` isn't present). Walking reverse through a
        // user-key group visits entries in ascending seq order, which is
        // exactly what `materialize_prev_visible` expects.
        let probe = above_all_versions(target);
        if let Err(e) = self.inner.seek_for_prev(&probe) {
            self.error = Some(e);
            return;
        }
        self.materialize_prev_visible();
    }

    /// Position the iterator at the first user key `>= prefix` and
    /// confine forward iteration to keys that start with `prefix`. When
    /// the underlying SSTables have a matching prefix bloom filter,
    /// files that demonstrably cannot contain `prefix` are skipped
    /// entirely; files built without a prefix bloom are consulted
    /// normally (safe superset).
    /// Position the cursor at the newest visible entry whose user key
    /// is strictly below `exclusive_upper`, and switch to reverse
    /// iteration.
    ///
    /// This is the primitive a bounded reverse scan needs.
    /// [`Self::seek_for_prev`] takes an **inclusive** bound, and a
    /// caller holding an exclusive one cannot build an inclusive probe
    /// from it: byte strings have no predecessor, so no suffix of
    /// `0xff` bytes is an upper bound for user keys of every length.
    pub(crate) fn seek_to_last_before(&mut self, exclusive_upper: &[u8]) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.upper_bound = None;
        self.direction = Direction::Reverse;
        // `lookup_key(bound, u64::MAX)` encodes as `bound || 00*8 ||
        // 00`, the smallest internal key any entry for user key `bound`
        // could carry. Every internal key strictly below it belongs to
        // a user key strictly below `bound`, whatever its length.
        let probe = encode_internal_key(exclusive_upper, u64::MAX, VALUE_TYPE_DELETION);
        if let Err(e) = self.inner.seek_for_prev(&probe) {
            self.error = Some(e);
            return;
        }
        // `seek_for_prev` is inclusive, so step back over the one entry
        // shape that can equal the probe: user key `bound` at seq
        // `u64::MAX`. The engine never allocates that sequence, so this
        // normally runs zero times; the loop makes the bound exact
        // rather than exact in practice.
        while let Some(k) = self.inner.key() {
            if user_key_of(k) < exclusive_upper {
                break;
            }
            if let Err(e) = self.inner.advance_backward() {
                self.error = Some(e);
                return;
            }
        }
        self.materialize_prev_visible();
    }

    pub(crate) fn seek_prefix(&mut self, prefix: &[u8]) {
        if self.terminal_error {
            return;
        }
        self.error = None;
        self.valid_entry = false;
        self.last_forward_user_key = None;
        self.merge_result = None;
        self.reverse_curr = None;
        self.pending_consume = false;
        self.direction = Direction::Forward;
        self.upper_bound = prefix_upper_bound(prefix);

        // Consult per-SSTable prefix blooms when the extractor can
        // derive the same kind of bloom key from the query prefix that
        // was indexed for table keys. This lets a fixed-length
        // extractor skip whole files for longer prefixes that share
        // the indexed stem.
        let bloom_probe = self
            .prefix_extractor
            .as_ref()
            .and_then(|ex| ex.extract_query(prefix).map(|p| p.to_vec()));

        let search_key = LookupKey::from_prefixed(prefix, u64::MAX);
        let res = if let Some(probe) = bloom_probe.as_deref() {
            self.inner
                .seek_with_prefix_skip(search_key.internal(), probe)
        } else {
            self.inner.seek(search_key.internal())
        };
        if let Err(e) = res {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn next(&mut self) {
        if !self.valid() {
            return;
        }
        if self.direction == Direction::Reverse {
            self.flip_to_forward();
        }
        self.merge_result = None;
        if self.pending_consume {
            self.consume_curr_user_key_forward();
            self.pending_consume = false;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn prev(&mut self) {
        if !self.valid() {
            return;
        }
        if self.pending_consume {
            self.consume_curr_user_key_forward();
            self.pending_consume = false;
        }
        self.merge_result = None;
        if self.direction == Direction::Forward {
            self.flip_to_reverse();
        }
        self.reverse_curr = None;
        self.materialize_prev_visible();
    }

    pub(crate) fn valid(&self) -> bool {
        if self.error.is_some() {
            return false;
        }
        match self.direction {
            Direction::Forward => self.valid_entry || self.merge_result.is_some(),
            Direction::Reverse => self.reverse_curr.is_some(),
        }
    }

    pub(crate) fn key(&self) -> Option<&[u8]> {
        match self.direction {
            Direction::Forward => {
                if let Some((k, _)) = &self.merge_result {
                    return Some(k.as_slice());
                }
                if !self.valid_entry {
                    return None;
                }
                self.inner.key().map(user_key_of)
            }
            Direction::Reverse => self.reverse_curr.as_ref().map(|(k, _)| k.as_slice()),
        }
    }

    pub(crate) fn value(&self) -> Option<&[u8]> {
        match self.direction {
            Direction::Forward => {
                if let Some((_, v)) = &self.merge_result {
                    return Some(v.as_slice());
                }
                if !self.valid_entry {
                    return None;
                }
                self.inner.value()
            }
            Direction::Reverse => self.reverse_curr.as_ref().map(|(_, v)| v.as_slice()),
        }
    }

    /// The current value as an owning view over whatever already holds
    /// it. Zero-copy while iterating SSTables in forward order; a copy
    /// for a memtable-resident entry, a merge result, or the reverse
    /// path, all of which own their bytes separately.
    pub(crate) fn value_slice(&self) -> Option<DbSlice> {
        match self.direction {
            Direction::Forward => {
                if let Some((_, v)) = &self.merge_result {
                    return Some(DbSlice::from_vec(v.clone()));
                }
                if !self.valid_entry {
                    return None;
                }
                self.inner.value_slice()
            }
            Direction::Reverse => self
                .reverse_curr
                .as_ref()
                .map(|(_, v)| DbSlice::from_vec(v.clone())),
        }
    }

    pub(crate) fn status(&self) -> io::Result<()> {
        match &self.error {
            Some(e) => Err(io::Error::new(e.kind(), e.to_string())),
            None => Ok(()),
        }
    }

    pub(crate) fn set_error(&mut self, err: io::Error) {
        self.error = Some(err);
        self.terminal_error = true;
        self.valid_entry = false;
        self.merge_result = None;
        self.reverse_curr = None;
    }

    /// Walk the merging iterator forward until the first user key whose
    /// most-recent visible version is a live value; set `curr_user` to it
    /// and advance the inner iterator past every remaining entry in that
    /// user-key group. If no live user key remains, `curr_user` stays
    /// `None` and the iterator becomes invalid.
    /// Consume all remaining entries with the same user key as
    /// `self.curr_user`. Borrows `self.inner` and `self.curr_user`
    /// in non-overlapping scopes so the borrow checker is satisfied.
    fn consume_curr_user_key_forward(&mut self) {
        // The inner iterator is positioned at the entry we just
        // yielded. Advance past it unconditionally - we know it
        // matches curr_user_key, so the first-iteration check is
        // wasted work. This saves one decode_internal_key +
        // one comparison per visible entry in the common case.
        if let Err(e) = self.inner.advance() {
            self.error = Some(e);
            return;
        }
        // If there are older versions of the same user key,
        // advance past them too.
        loop {
            let matches = {
                let Some(ik) = self.inner.key() else {
                    return;
                };
                let (uk, _, _) = decode_internal_key(ik);
                uk == self.curr_user_key.as_slice()
            };
            if !matches {
                return;
            }
            if let Err(e) = self.inner.advance() {
                self.error = Some(e);
                return;
            }
        }
    }

    fn materialize_next_visible(&mut self) {
        self.valid_entry = false;
        loop {
            let Some(ik) = self.inner.key() else {
                return;
            };
            let (uk, seq, vt) = decode_internal_key(ik);

            let went_backwards = self
                .last_forward_user_key
                .as_deref()
                .is_some_and(|last| uk <= last);
            if went_backwards {
                self.error = Some(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "SSTable iteration went backwards: file index or data block is corrupt",
                ));
                return;
            }

            if let Some(ub) = self.upper_bound.as_deref()
                && uk >= ub
            {
                return;
            }

            if seq > self.snapshot_seq {
                if let Err(e) = self.inner.advance() {
                    self.error = Some(e);
                    return;
                }
                continue;
            }

            // Inline the RT check: skip the function call when
            // there are no range tombstones (the common case in
            // sequential scans). This saves ~5ns per entry.
            let rt_seq = if self.range_tombstones.is_empty() {
                0
            } else {
                self.covering_rt_seq(uk)
            };
            if rt_seq > seq {
                self.curr_user_key.clear();
                self.curr_user_key.extend_from_slice(uk);
                self.consume_curr_user_key_forward();
                continue;
            }
            match vt {
                VALUE_TYPE_DELETION => {
                    self.curr_user_key.clear();
                    self.curr_user_key.extend_from_slice(uk);
                    self.consume_curr_user_key_forward();
                    continue;
                }
                VALUE_TYPE_MERGE => {
                    let uk_owned = uk.to_vec();
                    match self.collapse_merge_chain_forward(&uk_owned, rt_seq) {
                        Ok(Some(v)) => {
                            self.curr_user_key.clear();
                            self.curr_user_key.extend_from_slice(&uk_owned);
                            self.last_forward_user_key = Some(uk_owned.clone());
                            self.merge_result = Some((uk_owned, v));
                            self.pending_consume = false;
                            return;
                        }
                        Ok(None) => continue,
                        Err(e) => {
                            self.error = Some(e);
                            return;
                        }
                    }
                }
                _ => {
                    // Zero-copy hot path: record the user key for
                    // consume/dedup logic and mark the entry valid.
                    // key()/value() delegate to self.inner.
                    self.curr_user_key.clear();
                    self.curr_user_key.extend_from_slice(uk);
                    match self.last_forward_user_key.as_mut() {
                        Some(last) => {
                            last.clear();
                            last.extend_from_slice(uk);
                        }
                        None => self.last_forward_user_key = Some(uk.to_vec()),
                    }
                    self.valid_entry = true;
                    self.pending_consume = true;
                    return;
                }
            }
        }
    }

    /// Collect every visible entry for `user_key` starting from the
    /// iterator's current forward position, expecting the newest
    /// entry to be a merge operand. Walks through successive older
    /// entries in the same user-key group (which sort after the
    /// merge in internal-key order) until a terminator (Value or
    /// Deletion) is reached, the user key changes, or we run off
    /// the end. Calls the merge operator to materialize the final
    /// value and advances the inner iterator past the entire group.
    ///
    /// Returns `Ok(Some(value))` on success, `Ok(None)` when the
    /// chain collapses to a deletion (caller should try the next
    /// user key), or `Err` when the merge operator fails.
    ///
    /// `rt_seq` is the covering range-tombstone seq; entries with
    /// `seq <= rt_seq` are hidden by it and terminate the walk as
    /// if a deletion had been reached.
    fn collapse_merge_chain_forward(
        &mut self,
        user_key: &[u8],
        rt_seq: u64,
    ) -> io::Result<Option<Vec<u8>>> {
        let merge_op = match self.merge_operator.clone() {
            Some(op) => op,
            None => {
                // No merge operator - can't collapse. Treat
                // merges as invisible: without an operator there
                // is no way to produce a value from the chain, so
                // reads see the key as missing.
                self.consume_user_key_forward(user_key);
                return Ok(None);
            }
        };

        // Operands collected oldest-first so we can pass them to
        // `full_merge` in the expected order. We build newest-first
        // while walking forward and reverse at the end.
        let mut operands_newest_first: Vec<Vec<u8>> = Vec::new();
        let mut base: Option<Vec<u8>> = None;
        let mut had_terminator = false;

        #[allow(clippy::while_let_loop)]
        loop {
            let Some(ik) = self.inner.key() else { break };
            let (uk, seq, vt) = decode_internal_key(ik);
            if uk != user_key {
                break;
            }
            if seq > self.snapshot_seq {
                // Invisible: skip without adding to chain.
                self.inner.advance()?;
                continue;
            }
            if rt_seq > 0 && seq <= rt_seq {
                // Range tombstone hides this and every older entry
                // for the same user key - synthesize a deletion
                // terminator and stop.
                base = None;
                had_terminator = true;
                self.consume_user_key_forward(user_key);
                break;
            }
            let value = self.inner.value().map(|s| s.to_vec()).unwrap_or_default();
            match vt {
                VALUE_TYPE_MERGE => {
                    operands_newest_first.push(value);
                    self.inner.advance()?;
                }
                VALUE_TYPE_VALUE => {
                    base = Some(value);
                    had_terminator = true;
                    self.consume_user_key_forward(user_key);
                    break;
                }
                VALUE_TYPE_DELETION => {
                    base = None;
                    had_terminator = true;
                    self.consume_user_key_forward(user_key);
                    break;
                }
                _ => {
                    self.inner.advance()?;
                }
            }
        }
        let _ = had_terminator;

        if operands_newest_first.is_empty() {
            // No merges - shouldn't happen because caller saw a
            // merge at the head, but handle it safely.
            return Ok(base);
        }

        let operand_refs: Vec<&[u8]> = operands_newest_first
            .iter()
            .rev()
            .map(|v| v.as_slice())
            .collect();
        match merge_op.full_merge(user_key, base.as_deref(), &operand_refs) {
            Some(v) => Ok(Some(v)),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("merge operator {} failed", merge_op.name()),
            )),
        }
    }

    /// Walk the merging iterator **backward** until we find a user key
    /// whose newest visible version (at `snapshot_seq`) is a live value.
    ///
    /// In reverse walk, a user-key group is visited in *ascending* seq
    /// order (because higher seq produces a smaller internal key, which
    /// comes later in reverse order). We scan the whole group, keeping
    /// track of the highest visible seq we see - that is the winning
    /// version. If it's a tombstone the group is skipped; otherwise it
    /// is emitted.
    fn materialize_prev_visible(&mut self) {
        loop {
            let Some(ik) = self.inner.key() else {
                self.reverse_curr = None;
                return;
            };
            let (uk, _, _) = decode_internal_key(ik);
            let group = uk.to_vec();

            let rt_seq = self.covering_rt_seq(&group);
            let mut collected: Vec<(u64, u8, Vec<u8>)> = Vec::new();

            while let Some(ik2) = self.inner.key() {
                let (uk2, seq, vt) = decode_internal_key(ik2);
                if uk2 != group.as_slice() {
                    break;
                }
                if seq <= self.snapshot_seq && (rt_seq == 0 || seq > rt_seq) {
                    let v = self.inner.value().map(|s| s.to_vec()).unwrap_or_default();
                    collected.push((seq, vt, v));
                }
                if let Err(e) = self.inner.advance_backward() {
                    self.error = Some(e);
                    self.reverse_curr = None;
                    return;
                }
            }

            if collected.is_empty() {
                continue;
            }

            let mut terminator_idx: Option<usize> = None;
            for (i, (_, vt, _)) in collected.iter().enumerate().rev() {
                if *vt != VALUE_TYPE_MERGE {
                    terminator_idx = Some(i);
                    break;
                }
            }

            let (base, operand_range_start) = match terminator_idx {
                Some(i) => match collected[i].1 {
                    VALUE_TYPE_VALUE => (Some(collected[i].2.clone()), i + 1),
                    VALUE_TYPE_DELETION => (None, i + 1),
                    _ => (None, i + 1),
                },
                None => (None, 0),
            };

            let operand_slice = &collected[operand_range_start..];
            if operand_slice.is_empty() {
                match terminator_idx {
                    Some(i) if collected[i].1 == VALUE_TYPE_VALUE => {
                        self.reverse_curr = Some((group, base.unwrap()));
                        return;
                    }
                    _ => continue,
                }
            }

            let Some(merge_op) = self.merge_operator.clone() else {
                continue;
            };
            let operand_refs: Vec<&[u8]> = operand_slice.iter().map(|e| e.2.as_slice()).collect();
            match merge_op.full_merge(&group, base.as_deref(), &operand_refs) {
                Some(v) => {
                    self.reverse_curr = Some((group, v));
                    return;
                }
                None => {
                    self.error = Some(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("merge operator {} failed", merge_op.name()),
                    ));
                    self.reverse_curr = None;
                    return;
                }
            }
        }
    }

    /// Advance the merging iterator forward past every entry whose user
    /// key matches `user_key`.
    fn consume_user_key_forward(&mut self, user_key: &[u8]) {
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

    /// Switch from reverse to forward iteration. After a reverse pass
    /// just emitted `curr_user`, the merging iterator's level cursors
    /// are positioned in the *previous* user-key group. Re-seek every
    /// level forward to just past the last version of `curr_user` so the
    /// next forward step lands on the immediately following user key.
    fn flip_to_forward(&mut self) {
        let Some((uk, _)) = &self.reverse_curr else {
            return;
        };
        let probe = above_all_versions(uk);
        if let Err(e) = self.inner.seek(&probe) {
            self.error = Some(e);
        }
        self.reverse_curr = None;
        self.direction = Direction::Forward;
        self.last_forward_user_key = None;
    }

    /// Switch from forward to reverse iteration. After a forward pass
    /// just emitted `curr_user`, re-seek every level backward to just
    /// before the smallest internal key of `curr_user` so the next
    /// reverse step lands on the immediately preceding user key.
    fn flip_to_reverse(&mut self) {
        // In forward mode, the current user key is in curr_user_key
        // (for normal entries) or merge_result (for merge results).
        let uk: Vec<u8> = if let Some((k, _)) = &self.merge_result {
            k.clone()
        } else {
            self.curr_user_key.clone()
        };
        if uk.is_empty() {
            return;
        }
        let probe = LookupKey::from_prefixed(&uk, u64::MAX);
        if let Err(e) = self.inner.seek_for_prev(probe.internal()) {
            self.error = Some(e);
        }
        self.valid_entry = false;
        self.merge_result = None;
        self.direction = Direction::Reverse;
    }
}
