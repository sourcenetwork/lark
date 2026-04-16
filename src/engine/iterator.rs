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
//! 4. L1..Ln SSTables (one level at a time; within a level files are
//!    non-overlapping so their order is irrelevant)
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
//! - Treats a tombstone as "this user key is deleted" — the whole group
//!   is skipped, and any older versions in lower levels are suppressed.
//!
//! # Forward vs reverse materialization
//!
//! In **forward** mode, entries within a user-key group are visited newest-
//! seq first, so the first visible entry we see is the answer — the rest
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
//! user key in the new direction — see [`LarkIterator::flip_to_forward`]
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

use super::block_cache::BlockCache;
use super::internal_key::{
    compare_internal_keys, decode_internal_key, lookup_key, INTERNAL_KEY_SUFFIX_LEN,
    VALUE_TYPE_DELETION, VALUE_TYPE_MERGE, VALUE_TYPE_VALUE,
};
use super::manifest::Version;
use super::memtable::MemTable;
use super::range_tombstone::{max_covering_seq, RangeTombstone};
use super::sstable::SsTableReader;
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

    fn seek_to_last(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.seek_to_last();
                Ok(())
            }
            Self::SsTable(it) => it.seek_to_last(),
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
                if !it.reader.may_have_prefix(prefix) {
                    it.curr = None;
                    return Ok(());
                }
                it.seek(target)
            }
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

    fn advance_backward(&mut self) -> io::Result<()> {
        match self {
            Self::Memtable(it) => {
                it.advance_backward();
                Ok(())
            }
            Self::SsTable(it) => it.advance_backward(),
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

    fn seek_to_last(&mut self) {
        self.curr = self.mt.last_entry_before(Bound::Unbounded);
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

    fn advance_backward(&mut self) {
        if let Some((k, _)) = self.curr.take() {
            self.curr = self.mt.last_entry_before(Bound::Excluded(k.as_slice()));
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
        // Pick the block that could contain the largest entry `<= target`.
        //
        // `seek_block` returns the first block whose last key is `>= target`
        // — that is the "containing" block, i.e. the earliest block that
        // might hold `target` or the smallest key greater than `target`.
        //
        // If `seek_block` returns `None` every block's last key is `<
        // target`, which means `target` is greater than every entry in
        // the table. The correct answer is then the last entry of the
        // **last** block (not block 0).
        let num_blocks = self.reader.num_blocks();
        if num_blocks == 0 {
            self.curr = None;
            return Ok(());
        }
        let block_idx = self.reader.seek_block(target).unwrap_or(num_blocks - 1);

        self.block_idx = block_idx;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;

        // Within the block, find the largest entry `<= target` via linear
        // walk. One block holds ~hundreds of entries, so this is cheap.
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
                // Every entry in this block is `> target`, which can
                // happen when the containing block's first key already
                // exceeds `target`. The answer, if one exists, is the
                // last entry of the previous block — its last key is
                // known to be `< target` (that's why `seek_block`
                // skipped it).
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

    fn seek_to_last(&mut self) -> io::Result<()> {
        let n = self.reader.num_blocks();
        if n == 0 {
            self.curr = None;
            return Ok(());
        }
        self.block_idx = n - 1;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        if self.block_entries.is_empty() {
            self.curr = None;
            return Ok(());
        }
        self.entry_pos = self.block_entries.len() - 1;
        self.curr = self.block_entries.last().cloned();
        Ok(())
    }

    fn advance_backward(&mut self) -> io::Result<()> {
        if self.curr.is_none() {
            return Ok(());
        }
        if self.entry_pos > 0 {
            self.entry_pos -= 1;
            self.curr = self.block_entries.get(self.entry_pos).cloned();
            return Ok(());
        }
        // Move to the previous block.
        if self.block_idx == 0 {
            self.curr = None;
            return Ok(());
        }
        self.block_idx -= 1;
        self.block_entries = self
            .reader
            .load_block_entries(self.block_idx, &self.cache)?;
        if self.block_entries.is_empty() {
            self.curr = None;
            return Ok(());
        }
        self.entry_pos = self.block_entries.len() - 1;
        self.curr = self.block_entries.last().cloned();
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
    /// Most recently produced `(user_key, value)` pair, if any.
    curr_user: Option<(Vec<u8>, Vec<u8>)>,
    /// Flat snapshot of every range tombstone the iterator must honor
    /// — the union of memtable RTs and SSTable RTs from every level
    /// reachable through the pinned `Version`. Captured once at
    /// construction so the merging walk can consult it without
    /// re-locking memtables on each step. Range tombstone counts are
    /// expected to be small relative to point entries; a linear scan
    /// per user-key group is acceptable.
    range_tombstones: Vec<RangeTombstone>,
    /// Pinning handle — keeping this alive guarantees compaction cannot
    /// drop SSTable metadata out from under us. The SsTableReaders owned
    /// by the level iterators similarly keep file handles live.
    _version: Arc<Version>,
    /// Sticky error from the most recent I/O attempt, if any. Cleared on
    /// the next successful seek.
    error: Option<io::Error>,
    /// Exclusive upper bound set by [`LarkIterator::seek_prefix`]. When
    /// `Some`, forward iteration stops as soon as the next visible key
    /// is `>= upper_bound`, confining the scan to the originally seeked
    /// prefix. Cleared by any other seek.
    upper_bound: Option<Vec<u8>>,
    /// Prefix extractor captured at construction. Used by
    /// [`LarkIterator::seek_prefix`] to decide whether the caller's
    /// query prefix is a valid extracted prefix (in which case the
    /// per-SSTable prefix bloom can be consulted) or an arbitrary
    /// byte string (in which case we fall back to a plain upper-bound
    /// scan without bloom skipping).
    prefix_extractor: Option<Arc<dyn PrefixExtractor>>,
    /// Optional merge operator. When `Some`, the iterator collapses
    /// merge chains encountered during materialization into a
    /// single final value via [`MergeOperator::full_merge`].
    merge_operator: Option<Arc<dyn MergeOperator>>,
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
    /// version that still references them — so the iterator is immune to
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
        // L1+: within a level file order doesn't matter (non-overlapping).
        for level in 1..version.levels.len() {
            for file in &version.levels[level] {
                range_tombstones.extend(file.reader.range_tombstones().iter().cloned());
                levels.push(LevelIter::SsTable(SsTableLevelIter::new(
                    Arc::clone(&file.reader),
                    Arc::clone(&cache),
                )));
            }
        }

        Self {
            inner: MergingIter::new(levels),
            snapshot_seq,
            direction: Direction::Forward,
            curr_user: None,
            range_tombstones,
            _version: version,
            error: None,
            upper_bound: None,
            prefix_extractor,
            merge_operator,
        }
    }

    /// Largest seq of any range tombstone covering `user_key` that is
    /// visible at this iterator's snapshot. Returns 0 when none.
    fn covering_rt_seq(&self, user_key: &[u8]) -> u64 {
        if self.range_tombstones.is_empty() {
            return 0;
        }
        max_covering_seq(&self.range_tombstones, user_key, self.snapshot_seq)
    }

    pub(crate) fn seek_to_first(&mut self) {
        self.error = None;
        self.curr_user = None;
        self.upper_bound = None;
        self.direction = Direction::Forward;
        if let Err(e) = self.inner.seek_to_first() {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn seek_to_last(&mut self) {
        self.error = None;
        self.curr_user = None;
        self.upper_bound = None;
        self.direction = Direction::Reverse;
        if let Err(e) = self.inner.seek_to_last() {
            self.error = Some(e);
            return;
        }
        self.materialize_prev_visible();
    }

    pub(crate) fn seek(&mut self, target: &[u8]) {
        self.error = None;
        self.curr_user = None;
        self.upper_bound = None;
        self.direction = Direction::Forward;
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
    pub(crate) fn seek_prefix(&mut self, prefix: &[u8]) {
        self.error = None;
        self.curr_user = None;
        self.direction = Direction::Forward;
        self.upper_bound = prefix_upper_bound(prefix);

        // Only consult per-SSTable prefix blooms when the caller's
        // query prefix is itself a valid extracted prefix — otherwise
        // the bloom was hashed with different inputs and a negative
        // answer is meaningless. Fall back to a plain seek + upper
        // bound walk.
        let can_skip = match &self.prefix_extractor {
            Some(ex) => matches!(ex.extract(prefix), Some(p) if p == prefix),
            None => false,
        };

        let search_key = lookup_key(prefix, u64::MAX);
        let res = if can_skip {
            self.inner.seek_with_prefix_skip(&search_key, prefix)
        } else {
            self.inner.seek(&search_key)
        };
        if let Err(e) = res {
            self.error = Some(e);
            return;
        }
        self.materialize_next_visible();
    }

    pub(crate) fn next(&mut self) {
        if self.curr_user.is_none() || self.error.is_some() {
            return;
        }
        if self.direction == Direction::Reverse {
            self.flip_to_forward();
        }
        self.curr_user = None;
        self.materialize_next_visible();
    }

    pub(crate) fn prev(&mut self) {
        if self.curr_user.is_none() || self.error.is_some() {
            return;
        }
        if self.direction == Direction::Forward {
            self.flip_to_reverse();
        }
        self.curr_user = None;
        self.materialize_prev_visible();
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

            // Prefix-bounded scan: once we reach a user key at or past
            // the upper bound we're done. Bound is exclusive — a key
            // equal to `upper_bound` is already outside the prefix.
            if let Some(ub) = self.upper_bound.as_deref() {
                if uk >= ub {
                    self.curr_user = None;
                    return;
                }
            }

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

            // Newest visible entry for this user key. If a range
            // tombstone with a strictly greater seq covers it, the
            // whole user key is considered deleted — same effect as a
            // point tombstone. Otherwise emit the value (or skip if
            // it's itself a point tombstone; or collapse a merge
            // chain via the configured merge operator).
            let uk_owned = uk.to_vec();
            let rt_seq = self.covering_rt_seq(&uk_owned);
            if rt_seq > seq {
                self.consume_user_key_forward(&uk_owned);
                continue;
            }
            match vt {
                VALUE_TYPE_DELETION => {
                    self.consume_user_key_forward(&uk_owned);
                    continue;
                }
                VALUE_TYPE_MERGE => match self.collapse_merge_chain_forward(&uk_owned, rt_seq) {
                    Ok(Some(v)) => {
                        self.curr_user = Some((uk_owned, v));
                        return;
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        self.error = Some(e);
                        self.curr_user = None;
                        return;
                    }
                },
                _ => {
                    let v = self.inner.value().map(|s| s.to_vec()).unwrap_or_default();
                    self.curr_user = Some((uk_owned.clone(), v));
                    self.consume_user_key_forward(&uk_owned);
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
                // No merge operator — can't collapse. Treat
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
                // for the same user key — synthesize a deletion
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
            // No merges — shouldn't happen because caller saw a
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
    /// track of the highest visible seq we see — that is the winning
    /// version. If it's a tombstone the group is skipped; otherwise it
    /// is emitted.
    fn materialize_prev_visible(&mut self) {
        loop {
            let Some(ik) = self.inner.key() else {
                self.curr_user = None;
                return;
            };
            let (uk, _, _) = decode_internal_key(ik);
            let group = uk.to_vec();

            // Collect every visible entry in this group in walk
            // order (ascending seq). After the walk we know the
            // full group and can classify its winning outcome.
            // Within a user-key group, entries with
            // `seq > snapshot_seq` and entries hidden by a range
            // tombstone are skipped. We retain merges so we can
            // collapse them at the end via the configured operator.
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
                    self.curr_user = None;
                    return;
                }
            }

            // `collected` is ascending by seq (oldest visible first,
            // newest visible last). Find the newest terminator and
            // fold any operands that sit on top of it.
            //
            // Walk backwards to locate the terminator seq; collect
            // operands that are strictly newer than the terminator.
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

            // Operands that apply are everything from `terminator_idx + 1`
            // through the end. If there's no terminator, every entry
            // is a merge operand and the base is None.
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
                // Plain value or deletion — no merge fold required.
                match terminator_idx {
                    Some(i) if collected[i].1 == VALUE_TYPE_VALUE => {
                        self.curr_user = Some((group, base.unwrap()));
                        return;
                    }
                    _ => continue, // deletion or nothing visible
                }
            }

            // Collapse the merge chain. We need a merge operator to
            // materialize it — without one, treat the group as
            // missing.
            let Some(merge_op) = self.merge_operator.clone() else {
                continue;
            };
            // `operand_slice` is already oldest-first.
            let operand_refs: Vec<&[u8]> = operand_slice.iter().map(|e| e.2.as_slice()).collect();
            match merge_op.full_merge(&group, base.as_deref(), &operand_refs) {
                Some(v) => {
                    self.curr_user = Some((group, v));
                    return;
                }
                None => {
                    self.error = Some(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("merge operator {} failed", merge_op.name()),
                    ));
                    self.curr_user = None;
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
        let Some((uk, _)) = &self.curr_user else {
            return;
        };
        let probe = above_all_versions(uk);
        if let Err(e) = self.inner.seek(&probe) {
            self.error = Some(e);
        }
        self.direction = Direction::Forward;
    }

    /// Switch from forward to reverse iteration. After a forward pass
    /// just emitted `curr_user`, re-seek every level backward to just
    /// before the smallest internal key of `curr_user` so the next
    /// reverse step lands on the immediately preceding user key.
    fn flip_to_reverse(&mut self) {
        let Some((uk, _)) = &self.curr_user else {
            return;
        };
        // `lookup_key(uk, u64::MAX)` is the smallest internal key for
        // `uk` under `compare_internal_keys`: same user key, trailer
        // `!MAX_SEQ || 0` = `[0,0,...,0]` which is the lowest possible
        // trailer. `seek_for_prev` is inclusive (lands on the largest
        // entry `<=` the probe). Since no real entry has seq=u64::MAX
        // (the engine starts at 0 and counts up), nothing in the skip
        // list will match the probe exactly, and `seek_for_prev` lands
        // on the last entry of the user key immediately preceding `uk`.
        let probe = lookup_key(uk, u64::MAX);
        if let Err(e) = self.inner.seek_for_prev(&probe) {
            self.error = Some(e);
        }
        self.direction = Direction::Reverse;
    }
}
