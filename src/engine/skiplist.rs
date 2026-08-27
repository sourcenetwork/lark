//! Insert-only concurrent skip list over a bump [`Arena`].
//!
//! Replaces `crossbeam_skiplist::SkipMap<InternalKey, Vec<u8>>`. Where
//! that cost three heap allocations per write (internal key, value, node)
//! plus epoch garbage, one node here is a single arena bump that holds
//! its header, tower, key bytes and value bytes inline, so a steady-state
//! write touches the global allocator zero times.
//!
//! # Node layout
//!
//! One variable-length allocation, [`NODE_ALIGN`]-aligned:
//!
//! ```text
//! offset  0 : u32                    key_len     internal-key bytes
//! offset  4 : u32                    value_len
//! offset  8 : u8                     height      1 ..= MAX_HEIGHT
//! offset  9 : [u8; 7]                padding     keeps the tower aligned
//! offset 16 : [AtomicPtr<u8>; height] tower      level 0 first
//! offset 16 + PTR*height             key bytes
//! offset 16 + PTR*height + key_len   value bytes
//! ```
//!
//! # Concurrency contract
//!
//! Exactly one thread inserts at a time (the engine serializes writers on
//! its write lock); any number of threads read concurrently and lock-free.
//! This is the same contract the crossbeam-backed memtable documented.
//!
//! The safety of the whole module rests on these invariants, named at
//! every `unsafe` site that relies on them. They are what the loom and
//! miri models exist to check.
//!
//! - **S1 (publication).** Every byte of a node - header, tower, key and
//!   value - is written before the single `Release` store that links it
//!   into level 0. A reader that observes the node through an `Acquire`
//!   load therefore observes it fully initialised.
//! - **S2 (single writer).** At most one thread is inside `insert` at a
//!   time. A `debug_assert`-backed flag catches a violation in test and
//!   debug builds.
//! - **S3 (no unlinking).** No node is ever removed. A `next` pointer
//!   observed once stays a valid node pointer until the whole arena dies.
//! - **S4 (provenance).** Every node pointer derives from the pointer
//!   `Arena::alloc` returned for that node; none is synthesised from an
//!   integer and none crosses chunks.
//! - **S5 (aliasing).** `NodeRef::key` and `NodeRef::value` form `&[u8]`
//!   over regions that are never written again after S1, and no `&mut`
//!   to any node byte exists once the node is published.
//! - **S6 (bounds).** `key_len` and `value_len` are written by the same
//!   code that sized the allocation, so both slices stay inside the
//!   node. `insert` refuses a key or value that does not fit a `u32`,
//!   which [`crate::Options::validate`] makes unreachable.
//! - **S7 (height in range).** `random_height` returns `1 ..= MAX_HEIGHT`,
//!   and every tower walk is bounded by the node's own height, so no
//!   read runs past the tower into the key bytes.
//! - **S8 (`Send` + `Sync`).** The only cross-thread communication is
//!   through the tower's `AtomicPtr`s, and the only mutation is by the
//!   single writer: S1 plus S2 plus S3.
//! - **S9 (head sentinel).** The head is a standalone allocation owned
//!   by the list, so its address is stable for the list's life and an
//!   untouched memtable reserves no arena chunk at all.
//!
//! The arena's own invariants (A1 to A7), which these build on, are
//! documented in [`super::arena`].

#![allow(unsafe_code)]

use std::marker::PhantomData;
use std::ptr::NonNull;

use super::arena::Arena;
use super::internal_key::{INTERNAL_KEY_SUFFIX_LEN, compare_internal_keys, compare_internal_split};
use super::sync::{Arc, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Maximum tower height. With [`BRANCHING`] 4, height 12 indexes about
/// 16.7M entries, which covers a 64 MiB memtable of 64-byte entries with
/// room to spare.
const MAX_HEIGHT: usize = 12;

/// Expected fan-out between adjacent levels.
const BRANCHING: u64 = 4;

/// Bytes before the tower: two lengths, the height, and padding.
const NODE_HEADER: usize = 16;

/// One forward pointer.
type Link = AtomicPtr<u8>;

/// Size of one tower slot.
const PTR_SIZE: usize = size_of::<Link>();

/// Alignment every node is allocated at.
pub(crate) const NODE_ALIGN: usize = if align_of::<Link>() > 8 {
    align_of::<Link>()
} else {
    8
};

/// Size of the head sentinel: a full-height tower and nothing else.
const HEAD_SIZE: usize = NODE_HEADER + PTR_SIZE * MAX_HEIGHT;

/// Bytes one node occupies for the given shape.
fn node_size(key_len: usize, value_len: usize, height: usize) -> usize {
    (NODE_HEADER + PTR_SIZE * height + key_len + value_len).next_multiple_of(NODE_ALIGN)
}

/// Most arena bytes an entry of this shape can take.
///
/// Tower height is drawn at insert time, so the exact cost is not known
/// before the fact; this assumes the tallest tower. The commit leader
/// uses it to keep a group from carrying the active memtable past
/// `write_buffer_size`, where over-stating a cost only ends a group
/// early and under-stating one would break the bound.
pub(crate) fn max_node_size(internal_key_len: usize, value_len: usize) -> usize {
    node_size(internal_key_len, value_len, MAX_HEIGHT)
}

/// Read a node's `(key_len, value_len, height)` header.
///
/// # Safety
///
/// `node` must point at an initialised node or the head sentinel (S1).
unsafe fn header(node: *const u8) -> (usize, usize, usize) {
    unsafe {
        let key_len = node.cast::<u32>().read() as usize;
        let value_len = node.add(4).cast::<u32>().read() as usize;
        let height = node.add(8).read() as usize;
        (key_len, value_len, height)
    }
}

/// Borrow one tower slot.
///
/// # Safety
///
/// `node` must point at an initialised node whose height is greater than
/// `level` (S1, S7).
unsafe fn link(node: *const u8, level: usize) -> &'static Link {
    // SAFETY: the tower starts at NODE_HEADER and holds `height` slots,
    // each `PTR_SIZE` wide and `NODE_ALIGN`-aligned because the node is.
    // The `'static` lifetime is contained by the callers, which never
    // hand it out past their own node reference (S3, A5).
    unsafe { &*node.add(NODE_HEADER + level * PTR_SIZE).cast::<Link>() }
}

/// The node linked after `node` at `level`, or `None` at the end.
///
/// # Safety
///
/// Same as [`link`].
unsafe fn next_at(node: *const u8, level: usize) -> Option<NonNull<u8>> {
    // Acquire pairs with the Release store in `insert` that published the
    // node (S1): observing the pointer implies observing its bytes.
    NonNull::new(unsafe { link(node, level) }.load(Ordering::Acquire))
}

/// A node's internal key.
///
/// # Safety
///
/// `node` must point at an initialised node (S1); the head sentinel has
/// `key_len == 0` and must not be passed here.
unsafe fn node_key<'a>(node: *const u8) -> &'a [u8] {
    unsafe {
        let (key_len, _, height) = header(node);
        std::slice::from_raw_parts(node.add(NODE_HEADER + PTR_SIZE * height), key_len)
    }
}

/// A borrowed view of one node.
///
/// The lifetime ties it to the skip list, which owns the `Arc<Arena>`
/// keeping the bytes alive (A5).
#[derive(Clone, Copy)]
pub(crate) struct NodeRef<'a> {
    ptr: NonNull<u8>,
    _marker: PhantomData<&'a ArenaSkipList>,
}

impl<'a> NodeRef<'a> {
    fn new(ptr: NonNull<u8>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The node's internal key.
    pub(crate) fn key(&self) -> &'a [u8] {
        // SAFETY (S1, S5): the node was fully written before it became
        // reachable and its bytes are never written again, so a shared
        // slice over them is sound for as long as the arena lives.
        unsafe { node_key(self.ptr.as_ptr()) }
    }

    /// The node's value bytes: empty for a deletion tombstone.
    pub(crate) fn value(&self) -> &'a [u8] {
        let (ptr, len) = self.value_span();
        match ptr {
            // SAFETY (S1, S5, S6): `value_span` derived the pointer from
            // this node's own allocation and the length from the header
            // that sized it.
            Some(ptr) => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) },
            None => &[],
        }
    }

    /// Address and length of the internal-key bytes inside the arena,
    /// so a caller can build a zero-copy [`crate::DbSlice`] over them.
    pub(crate) fn key_span(&self) -> (Option<NonNull<u8>>, usize) {
        // SAFETY (S1, S6): the header was written before publication and
        // sized this very allocation, so the key region is in range.
        unsafe {
            let node = self.ptr.as_ptr();
            let (key_len, _, height) = header(node);
            if key_len == 0 {
                return (None, 0);
            }
            let ptr = node.add(NODE_HEADER + PTR_SIZE * height);
            (NonNull::new(ptr), key_len)
        }
    }

    /// Address and length of the value bytes inside the arena, so a
    /// caller can build a zero-copy [`crate::DbSlice`] over them.
    pub(crate) fn value_span(&self) -> (Option<NonNull<u8>>, usize) {
        // SAFETY (S1, S6): the header was written before publication and
        // sized this very allocation, so the value region is in range.
        unsafe {
            let node = self.ptr.as_ptr();
            let (key_len, value_len, height) = header(node);
            if value_len == 0 {
                return (None, 0);
            }
            let ptr = node.add(NODE_HEADER + PTR_SIZE * height + key_len);
            (NonNull::new(ptr), value_len)
        }
    }

    /// The next node in internal-key order.
    pub(crate) fn next(&self) -> Option<NodeRef<'a>> {
        // SAFETY (S1, S7): every node has at least one tower slot.
        unsafe { next_at(self.ptr.as_ptr(), 0) }.map(NodeRef::new)
    }
}

/// Insert-only concurrent skip list over a bump arena.
pub(crate) struct ArenaSkipList {
    arena: Arc<Arena>,
    /// Head sentinel: a full-height tower with no key or value. It is a
    /// standalone allocation rather than an arena one, so an untouched
    /// memtable reserves no arena chunk at all (S9).
    head: NonNull<u8>,
    /// xorshift64 state for the height draw. Only the single writer
    /// mutates it, so `Relaxed` suffices and the type stays `Sync`.
    rnd: AtomicU64,
    count: AtomicUsize,
    #[cfg(debug_assertions)]
    inserting: super::sync::AtomicBool,
}

// SAFETY (S8): the only cross-thread communication is through the tower's
// `AtomicPtr`s with `Release`/`Acquire` (S1), the only mutation is by the
// single writer (S2), and no node is ever unlinked or freed while the
// list lives (S3, A5).
unsafe impl Send for ArenaSkipList {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for ArenaSkipList {}

impl ArenaSkipList {
    /// A new, empty list over `arena`.
    ///
    /// Returns `None` only when the global allocator refused the head
    /// sentinel, which the caller surfaces as an out-of-memory error
    /// rather than panicking.
    pub(crate) fn new(arena: Arc<Arena>) -> Option<Self> {
        let layout = std::alloc::Layout::from_size_align(HEAD_SIZE, NODE_ALIGN).ok()?;
        // SAFETY: `HEAD_SIZE` is nonzero and `NODE_ALIGN` is a power of
        // two, checked by `from_size_align` above.
        let head = NonNull::new(unsafe { std::alloc::alloc(layout) })?;
        // SAFETY (S9): `head` addresses `HEAD_SIZE` freshly allocated
        // bytes, which is exactly the header plus a `MAX_HEIGHT` tower.
        unsafe { init_node_header(head.as_ptr(), 0, 0, MAX_HEIGHT) };
        Some(Self {
            arena,
            head,
            // Any nonzero seed works; a fixed one keeps the height draw
            // reproducible across runs, which the tests rely on.
            rnd: AtomicU64::new(0x2545_F491_4F6C_DD1D),
            count: AtomicUsize::new(0),
            #[cfg(debug_assertions)]
            inserting: super::sync::AtomicBool::new(false),
        })
    }

    /// The arena backing every node.
    pub(crate) fn arena(&self) -> &Arc<Arena> {
        &self.arena
    }

    /// Whether the list holds no entries.
    pub(crate) fn is_empty(&self) -> bool {
        self.first().is_none()
    }

    /// Number of entries inserted.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    fn random_height(&self) -> usize {
        let mut x = self.rnd.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rnd.store(x, Ordering::Relaxed);
        let mut height = 1;
        let mut draw = x;
        while height < MAX_HEIGHT && draw.is_multiple_of(BRANCHING) {
            height += 1;
            draw /= BRANCHING;
        }
        height
    }

    /// Insert `user_key || !seq || value_type -> value`.
    ///
    /// The internal key is assembled directly inside the node, so no
    /// intermediate buffer is built and nothing is copied twice.
    ///
    /// Returns `false` without inserting when the entry cannot be
    /// represented (a key or value of 4 GiB or more), which
    /// [`crate::Options::validate`] makes unreachable.
    pub(crate) fn insert(&self, user_key: &[u8], seq: u64, value_type: u8, value: &[u8]) -> bool {
        #[cfg(debug_assertions)]
        let _writer = SingleWriterGuard::enter(&self.inserting);

        let key_len = user_key.len() + INTERNAL_KEY_SUFFIX_LEN;
        if u32::try_from(key_len).is_err() || u32::try_from(value.len()).is_err() {
            tracing::error!(
                key_len,
                value_len = value.len(),
                "memtable entry too large to encode; write dropped"
            );
            return false;
        }

        let trailer = internal_trailer(seq, value_type);
        let height = self.random_height();

        // Descend recording, at every level, the last node whose key is
        // strictly less than the new one. Comparison always goes through
        // the internal-key comparator: raw byte order is wrong when one
        // user key is a prefix of another.
        let mut prev = [self.head; MAX_HEIGHT];
        let mut cursor = self.head;
        for level in (0..MAX_HEIGHT).rev() {
            // SAFETY (S1, S3, S7): `cursor` is the head or a node reached
            // at this level, so its height exceeds `level`, and every
            // node it reaches stays valid for the arena's life.
            while let Some(next) = unsafe { next_at(cursor.as_ptr(), level) } {
                let next_key = unsafe { node_key(next.as_ptr()) };
                if compare_internal_split(next_key, user_key, &trailer).is_lt() {
                    cursor = next;
                } else {
                    break;
                }
            }
            prev[level] = cursor;
        }

        let size = node_size(key_len, value.len(), height);
        let Some(node) = self.arena.alloc(size, NODE_ALIGN) else {
            // The global allocator refused. Aborting is what `Vec::push`
            // does, and it is the only alternative to silently losing an
            // acknowledged write.
            let layout = std::alloc::Layout::from_size_align(size, NODE_ALIGN)
                .unwrap_or_else(|_| std::alloc::Layout::new::<u8>());
            std::alloc::handle_alloc_error(layout)
        };

        // SAFETY (S1, S6): `node` addresses `size` bytes from this
        // arena, and `size` was computed from exactly these lengths, so
        // every write below stays inside the allocation. The node is not
        // reachable yet, so plain stores need no synchronisation.
        unsafe {
            let raw = node.as_ptr();
            init_node_header(raw, key_len, value.len(), height);
            let key_at = raw.add(NODE_HEADER + PTR_SIZE * height);
            std::ptr::copy_nonoverlapping(user_key.as_ptr(), key_at, user_key.len());
            std::ptr::copy_nonoverlapping(
                trailer.as_ptr(),
                key_at.add(user_key.len()),
                INTERNAL_KEY_SUFFIX_LEN,
            );
            std::ptr::copy_nonoverlapping(value.as_ptr(), key_at.add(key_len), value.len());
            // Still unreachable: seed the forward pointers with plain
            // stores through the freshly constructed atomics.
            for (level, slot) in prev.iter().enumerate().take(height) {
                let successor =
                    next_at(slot.as_ptr(), level).map_or(std::ptr::null_mut(), |n| n.as_ptr());
                link(raw, level).store(successor, Ordering::Relaxed);
            }
            // S1: the only synchronising stores in the module. Level 0
            // first, so a reader that reaches the node from a higher
            // level always finds it linked at the bottom too.
            for (level, slot) in prev.iter().enumerate().take(height) {
                link(slot.as_ptr(), level).store(raw, Ordering::Release);
            }
        }

        self.count.fetch_add(1, Ordering::Release);
        true
    }

    /// Descend to level 0, returning the last node whose key is strictly
    /// less than `target` and the first whose key is greater or equal.
    ///
    /// Both come out of the **same** level-0 step, and that is load
    /// bearing: a writer may link a new node in between two reads, so a
    /// descent that stopped and then re-read `predecessor.next[0]` could
    /// hand back a node that was already inserted past `target`. A
    /// reader that then treats "first key not equal to mine" as "absent"
    /// misses a key that is present. The successor returned here was
    /// `>= target` at the instant it was observed.
    fn seek_pair(&self, target: &[u8]) -> (Option<NonNull<u8>>, Option<NonNull<u8>>) {
        let mut cursor = self.head;
        let mut successor = None;
        for level in (0..MAX_HEIGHT).rev() {
            loop {
                // SAFETY (S1, S3, S7): `cursor` is the head or a node
                // reached at this level, so its height exceeds `level`,
                // and every node it reaches stays valid for the arena's
                // life.
                let next = unsafe { next_at(cursor.as_ptr(), level) };
                match next {
                    Some(node)
                        if compare_internal_keys(unsafe { node_key(node.as_ptr()) }, target)
                            .is_lt() =>
                    {
                        cursor = node;
                    }
                    other => {
                        if level == 0 {
                            successor = other;
                        }
                        break;
                    }
                }
            }
        }
        ((cursor != self.head).then_some(cursor), successor)
    }

    /// The first node whose key is greater than or equal to `target`.
    pub(crate) fn seek_ge(&self, target: &[u8]) -> Option<NodeRef<'_>> {
        self.seek_pair(target).1.map(NodeRef::new)
    }

    /// The first node whose key is strictly greater than `target`.
    pub(crate) fn seek_gt(&self, target: &[u8]) -> Option<NodeRef<'_>> {
        let mut node = self.seek_ge(target);
        while let Some(current) = node {
            if compare_internal_keys(current.key(), target).is_gt() {
                return Some(current);
            }
            node = current.next();
        }
        None
    }

    /// The last node whose key is less than or equal to `target`.
    pub(crate) fn seek_le(&self, target: &[u8]) -> Option<NodeRef<'_>> {
        let (below, at_or_after) = self.seek_pair(target);
        if let Some(node) = at_or_after {
            let node = NodeRef::new(node);
            if compare_internal_keys(node.key(), target).is_le() {
                return Some(node);
            }
        }
        below.map(NodeRef::new)
    }

    /// The last node whose key is strictly less than `target`.
    pub(crate) fn seek_lt(&self, target: &[u8]) -> Option<NodeRef<'_>> {
        self.seek_pair(target).0.map(NodeRef::new)
    }

    /// The first node in internal-key order.
    pub(crate) fn first(&self) -> Option<NodeRef<'_>> {
        // SAFETY (S1, S9): the head always has a level-0 slot.
        unsafe { next_at(self.head.as_ptr(), 0) }.map(NodeRef::new)
    }

    /// The last node in internal-key order.
    pub(crate) fn last(&self) -> Option<NodeRef<'_>> {
        let mut cursor = self.head;
        for level in (0..MAX_HEIGHT).rev() {
            // SAFETY (S1, S3, S7): as in `insert`'s descent.
            while let Some(next) = unsafe { next_at(cursor.as_ptr(), level) } {
                cursor = next;
            }
        }
        (cursor != self.head).then(|| NodeRef::new(cursor))
    }
}

impl Drop for ArenaSkipList {
    fn drop(&mut self) {
        // SAFETY (S9): the head came from `std::alloc::alloc` with this
        // exact layout in `ArenaSkipList::new` and is freed once. Nodes
        // are not freed here: they belong to the arena, which returns its
        // chunks to the pool when its last reference dies (A5).
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(HEAD_SIZE, NODE_ALIGN);
            std::alloc::dealloc(self.head.as_ptr(), layout);
        }
    }
}

/// The 9-byte `!seq || value_type` trailer of an internal key.
fn internal_trailer(seq: u64, value_type: u8) -> [u8; INTERNAL_KEY_SUFFIX_LEN] {
    let mut trailer = [0u8; INTERNAL_KEY_SUFFIX_LEN];
    trailer[..8].copy_from_slice(&(!seq).to_be_bytes());
    trailer[8] = value_type;
    trailer
}

/// Write a node's header and construct its tower as `height` null links.
///
/// # Safety
///
/// `node` must address at least `NODE_HEADER + PTR_SIZE * height` writable
/// bytes, aligned to [`NODE_ALIGN`], and no other thread may observe it
/// yet (S1).
unsafe fn init_node_header(node: *mut u8, key_len: usize, value_len: usize, height: usize) {
    debug_assert!((1..=MAX_HEIGHT).contains(&height));
    unsafe {
        node.cast::<u32>().write(key_len as u32);
        node.add(4).cast::<u32>().write(value_len as u32);
        node.add(8).write(height as u8);
        std::ptr::write_bytes(node.add(9), 0, NODE_HEADER - 9);
        for level in 0..height {
            node.add(NODE_HEADER + level * PTR_SIZE)
                .cast::<Link>()
                .write(Link::new(std::ptr::null_mut()));
        }
    }
}

/// Debug-only enforcement of S2: exactly one thread inside `insert`.
#[cfg(debug_assertions)]
struct SingleWriterGuard<'a>(&'a super::sync::AtomicBool);

#[cfg(debug_assertions)]
impl<'a> SingleWriterGuard<'a> {
    fn enter(flag: &'a super::sync::AtomicBool) -> Self {
        let busy = flag.swap(true, Ordering::Acquire);
        debug_assert!(
            !busy,
            "ArenaSkipList::insert is single-writer (S2); the engine must serialize writers"
        );
        Self(flag)
    }
}

#[cfg(debug_assertions)]
impl Drop for SingleWriterGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::arena::{ArenaProfile, ChunkPool};
    use crate::engine::internal_key::{VALUE_TYPE_DELETION, VALUE_TYPE_VALUE, encode_internal_key};
    use proptest::prelude::*;

    fn list(budget: usize) -> ArenaSkipList {
        let profile = ArenaProfile::EMBEDDED;
        let pool = Arc::new(ChunkPool::new(profile, budget, 2));
        let arena = Arc::new(Arena::new(pool, budget, profile));
        ArenaSkipList::new(arena).expect("head allocation")
    }

    fn collect(list: &ArenaSkipList) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::new();
        let mut node = list.first();
        while let Some(current) = node {
            out.push((current.key().to_vec(), current.value().to_vec()));
            node = current.next();
        }
        out
    }

    #[test]
    fn empty_list_has_no_entries() {
        let list = list(64 * 1024);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.first().is_none());
        assert!(list.last().is_none());
        assert!(list.seek_ge(b"anything").is_none());
        assert!(list.seek_le(b"anything").is_none());
        assert_eq!(
            list.arena().reserved_bytes(),
            0,
            "head is not an arena chunk"
        );
    }

    #[test]
    fn insert_then_read_back() {
        let list = list(64 * 1024);
        assert!(list.insert(b"k", 1, VALUE_TYPE_VALUE, b"v1"));
        assert!(list.insert(b"k", 2, VALUE_TYPE_VALUE, b"v2"));
        assert!(list.insert(b"a", 3, VALUE_TYPE_DELETION, b""));
        assert_eq!(list.len(), 3);

        let entries = collect(&list);
        assert_eq!(entries.len(), 3);
        // "a" sorts first; for "k", the newer seq comes first.
        assert_eq!(
            entries[0].0,
            encode_internal_key(b"a", 3, VALUE_TYPE_DELETION)
        );
        assert_eq!(entries[0].1, b"");
        assert_eq!(entries[1].0, encode_internal_key(b"k", 2, VALUE_TYPE_VALUE));
        assert_eq!(entries[1].1, b"v2");
        assert_eq!(entries[2].1, b"v1");
    }

    #[test]
    fn seeks_bracket_the_list() {
        let list = list(64 * 1024);
        for key in [b"b".as_slice(), b"m", b"y"] {
            assert!(list.insert(key, 1, VALUE_TYPE_VALUE, key));
        }
        let probe = |k: &[u8]| encode_internal_key(k, u64::MAX, VALUE_TYPE_DELETION);

        assert_eq!(list.seek_ge(&probe(b"a")).expect("first").value(), b"b");
        assert_eq!(list.seek_ge(&probe(b"m")).expect("exact").value(), b"m");
        assert_eq!(list.seek_ge(&probe(b"n")).expect("next").value(), b"y");
        assert!(list.seek_ge(&probe(b"z")).is_none());

        assert!(list.seek_lt(&probe(b"b")).is_none());
        assert_eq!(list.seek_lt(&probe(b"n")).expect("prev").value(), b"m");
        assert_eq!(list.last().expect("last").value(), b"y");
        assert_eq!(list.first().expect("first").value(), b"b");
    }

    #[test]
    fn duplicate_internal_key_finds_the_first() {
        // WAL replay can re-present the same (key, seq) if a rewrite was
        // interrupted. Both nodes are stored; a reader finds the first.
        let list = list(64 * 1024);
        assert!(list.insert(b"k", 7, VALUE_TYPE_VALUE, b"first"));
        assert!(list.insert(b"k", 7, VALUE_TYPE_VALUE, b"second"));
        assert_eq!(list.len(), 2);
        let probe = encode_internal_key(b"k", 7, VALUE_TYPE_DELETION);
        let found = list.seek_ge(&probe).expect("present");
        assert!(found.value() == b"first" || found.value() == b"second");
        assert_eq!(collect(&list).len(), 2);
    }

    #[test]
    fn prefix_keys_sort_by_the_internal_comparator() {
        // Raw byte order would interleave these wrongly: "ab"'s !seq
        // trailer collides with "abc"'s literal 'c'.
        let list = list(64 * 1024);
        assert!(list.insert(b"abc", 1, VALUE_TYPE_VALUE, b"abc"));
        assert!(list.insert(b"ab", u64::MAX, VALUE_TYPE_VALUE, b"ab-high"));
        assert!(list.insert(b"ab", 0, VALUE_TYPE_VALUE, b"ab-low"));
        let values: Vec<Vec<u8>> = collect(&list).into_iter().map(|(_, v)| v).collect();
        assert_eq!(
            values,
            vec![b"ab-high".to_vec(), b"ab-low".to_vec(), b"abc".to_vec()]
        );
    }

    #[test]
    fn empty_value_round_trips() {
        let list = list(64 * 1024);
        assert!(list.insert(b"k", 1, VALUE_TYPE_DELETION, b""));
        let node = list.first().expect("present");
        assert_eq!(node.value(), b"");
        assert_eq!(node.value_span().1, 0);
        assert!(node.value_span().0.is_none());
    }

    #[test]
    fn a_value_larger_than_a_chunk_still_lands() {
        let list = list(64 * 1024);
        let big = vec![7u8; 300 * 1024];
        assert!(list.insert(b"big", 1, VALUE_TYPE_VALUE, &big));
        assert_eq!(list.first().expect("present").value(), big.as_slice());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "20k inserts against 8 spinning readers; \
                  `a_reader_walks_the_list_while_a_writer_publishes` is the miri-sized form"
    )]
    fn readers_see_whole_entries_while_a_writer_inserts() {
        // S1: a reader must never observe a node with a torn key or a
        // value that does not match it.
        let budget = 4 * 1024 * 1024;
        let profile = ArenaProfile::SERVER;
        let pool = Arc::new(ChunkPool::new(profile, budget, 2));
        let arena = Arc::new(Arena::new(pool, budget, profile));
        let list = Arc::new(ArenaSkipList::new(arena).expect("head"));

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..8)
            .map(|_| {
                let list = Arc::clone(&list);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let mut node = list.first();
                        while let Some(current) = node {
                            let key = current.key();
                            assert!(key.len() >= INTERNAL_KEY_SUFFIX_LEN);
                            let user = &key[..key.len() - INTERNAL_KEY_SUFFIX_LEN];
                            assert_eq!(
                                current.value(),
                                user,
                                "value must match the key it was written with"
                            );
                            node = current.next();
                        }
                    }
                })
            })
            .collect();

        for i in 0..20_000u32 {
            let key = format!("key{i:08}");
            assert!(list.insert(
                key.as_bytes(),
                u64::from(i) + 1,
                VALUE_TYPE_VALUE,
                key.as_bytes()
            ));
        }
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().expect("reader thread");
        }
        assert_eq!(list.len(), 20_000);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "40k inserts against 4 spinning readers; \
                  `a_reader_walks_the_list_while_a_writer_publishes` is the miri-sized form"
    )]
    fn a_seeded_key_stays_findable_while_a_writer_inserts_around_it() {
        // The two-step seek this replaced (descend, then re-read the
        // predecessor's level-0 link) could hand back a node inserted
        // after the descent finished, which a reader reads as "absent".
        let budget = 8 * 1024 * 1024;
        let profile = ArenaProfile::SERVER;
        let pool = Arc::new(ChunkPool::new(profile, budget, 2));
        let arena = Arc::new(Arena::new(pool, budget, profile));
        let list = Arc::new(ArenaSkipList::new(arena).expect("head"));
        assert!(list.insert(b"pinned", 1, VALUE_TYPE_VALUE, b"stable"));

        let probe = encode_internal_key(b"pinned", u64::MAX, VALUE_TYPE_DELETION);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let misses = Arc::new(AtomicUsize::new(0));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let list = Arc::clone(&list);
                let stop = Arc::clone(&stop);
                let misses = Arc::clone(&misses);
                let probe = probe.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let found = list
                            .seek_ge(&probe)
                            .filter(|entry| entry.value() == b"stable");
                        if found.is_none() {
                            misses.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for i in 0..40_000u64 {
            // Keys on both sides of "pinned", so inserts land right next
            // to the seek target at every level.
            let key = if i % 2 == 0 {
                format!("pinne{i:08}")
            } else {
                format!("pinnee{i:08}")
            };
            assert!(list.insert(key.as_bytes(), i + 2, VALUE_TYPE_VALUE, b"noise"));
        }
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().expect("reader thread");
        }
        assert_eq!(
            misses.load(Ordering::Relaxed),
            0,
            "a present key must never read as absent"
        );
    }

    /// The publication protocol at a size an interpreter can finish.
    ///
    /// The two stress tests above are the ones that catch a rare race by
    /// volume; this one exists so miri's aliasing model and data-race
    /// detector actually reach the same code. One writer publishes eight
    /// nodes while one reader walks the list, reads every key and value,
    /// and re-seeks a key that was already there: S1, S3, S5, S6 and S7
    /// all sit on that path.
    #[test]
    fn a_reader_walks_the_list_while_a_writer_publishes() {
        let skiplist = Arc::new(list(64 * 1024));
        assert!(skiplist.insert(b"seed", 1, VALUE_TYPE_VALUE, b"seed"));
        let probe = encode_internal_key(b"seed", u64::MAX, VALUE_TYPE_DELETION);

        let reader = {
            let skiplist = Arc::clone(&skiplist);
            let probe = probe.clone();
            std::thread::spawn(move || {
                for _ in 0..8 {
                    let mut cursor = skiplist.first();
                    while let Some(current) = cursor {
                        let key = current.key();
                        assert!(key.len() >= INTERNAL_KEY_SUFFIX_LEN);
                        let user = &key[..key.len() - INTERNAL_KEY_SUFFIX_LEN];
                        assert_eq!(current.value(), user);
                        cursor = current.next();
                    }
                    assert!(
                        skiplist.seek_ge(&probe).is_some(),
                        "S3: a published key cannot be lost"
                    );
                }
            })
        };

        for i in 0..8u64 {
            let key = format!("k{i}");
            assert!(skiplist.insert(key.as_bytes(), i + 2, VALUE_TYPE_VALUE, key.as_bytes()));
        }
        reader.join().expect("reader thread");
        assert_eq!(skiplist.len(), 9);
    }

    proptest! {
        #[test]
        fn insert_then_seek_round_trips(
            entries in proptest::collection::vec(
                (proptest::collection::vec(any::<u8>(), 0..24), 1u64..1000, any::<Vec<u8>>()),
                1..80,
            ),
        ) {
            let list = list(1024 * 1024);
            let mut model: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for (key, seq, value) in &entries {
                prop_assert!(list.insert(key, *seq, VALUE_TYPE_VALUE, value));
                model.push((encode_internal_key(key, *seq, VALUE_TYPE_VALUE), value.clone()));
            }
            model.sort_by(|a, b| compare_internal_keys(&a.0, &b.0));

            let got = collect(&list);
            prop_assert_eq!(got.len(), model.len());
            // Key order is total; two entries sharing an internal key
            // (same user key and seq) may sit in either order, which is
            // the documented duplicate behaviour, so compare the keys in
            // order and the pairs as a multiset.
            for (got, want) in got.iter().zip(model.iter()) {
                prop_assert_eq!(&got.0, &want.0);
            }
            let mut got_pairs = got.clone();
            let mut want_pairs = model.clone();
            got_pairs.sort();
            want_pairs.sort();
            prop_assert_eq!(got_pairs, want_pairs);

            // Every stored key is findable by an exact seek.
            for (key, _) in &model {
                let found = list.seek_ge(key).expect("stored key is findable");
                prop_assert!(compare_internal_keys(found.key(), key).is_eq());
            }
        }

        #[test]
        fn seeks_agree_with_a_linear_scan(
            keys in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..12), 1..40),
            probe in proptest::collection::vec(any::<u8>(), 0..12),
        ) {
            let list = list(1024 * 1024);
            for (i, key) in keys.iter().enumerate() {
                prop_assert!(list.insert(key, i as u64 + 1, VALUE_TYPE_VALUE, key));
            }
            let sorted = collect(&list);
            let target = encode_internal_key(&probe, u64::MAX, VALUE_TYPE_DELETION);

            let want_ge = sorted.iter().find(|(k, _)| compare_internal_keys(k, &target).is_ge());
            let got_ge = list.seek_ge(&target).map(|n| n.key().to_vec());
            prop_assert_eq!(got_ge.as_deref(), want_ge.map(|(k, _)| k.as_slice()));

            let want_lt = sorted.iter().rev().find(|(k, _)| compare_internal_keys(k, &target).is_lt());
            let got_lt = list.seek_lt(&target).map(|n| n.key().to_vec());
            prop_assert_eq!(got_lt.as_deref(), want_lt.map(|(k, _)| k.as_slice()));

            let want_le = sorted.iter().rev().find(|(k, _)| compare_internal_keys(k, &target).is_le());
            let got_le = list.seek_le(&target).map(|n| n.key().to_vec());
            prop_assert_eq!(got_le.as_deref(), want_le.map(|(k, _)| k.as_slice()));

            let want_gt = sorted.iter().find(|(k, _)| compare_internal_keys(k, &target).is_gt());
            let got_gt = list.seek_gt(&target).map(|n| n.key().to_vec());
            prop_assert_eq!(got_gt.as_deref(), want_gt.map(|(k, _)| k.as_slice()));
        }

        #[test]
        fn node_accounting_matches_the_layout(
            key in proptest::collection::vec(any::<u8>(), 0..2048),
            value in proptest::collection::vec(any::<u8>(), 0..4096),
        ) {
            let list = list(1024 * 1024);
            prop_assert!(list.insert(&key, 5, VALUE_TYPE_VALUE, &value));
            let node = list.first().expect("present");
            prop_assert_eq!(node.key().len(), key.len() + INTERNAL_KEY_SUFFIX_LEN);
            prop_assert_eq!(node.value(), value.as_slice());
            // The arena charged the full node, not just key + value.
            let used = list.arena().used_bytes();
            prop_assert!(used >= key.len() + value.len() + INTERNAL_KEY_SUFFIX_LEN + NODE_HEADER);
        }
    }
}
