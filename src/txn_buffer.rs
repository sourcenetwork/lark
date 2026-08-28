//! The buffer a transaction accumulates its own keys in.
//!
//! Buffering takes `&self` so one transaction can be shared across
//! threads, which rules out a plain `BTreeMap`. A general concurrent map
//! is the obvious substitute and the wrong one here: it allocates a table
//! the moment it exists, and a transaction holds three of these buffers
//! and lives for microseconds. Measured on the transaction benchmark,
//! paying for those tables cost 61% to 67% of uncontended commit
//! throughput.
//!
//! This is sized for what a transaction actually holds. It allocates
//! nothing until the first entry, then one node per entry, and answers a
//! lookup by walking the list. A transaction touching a handful of keys
//! beats a hash table on both counts. Past
//! [`crate::Options::transaction_keys_inline`] entries the walk would stop
//! being cheap, so the buffer indexes itself with a `HopscotchMap`.
//!
//! The index holds pointers into the list, not copies of what the list
//! holds. Copying would put a second key and a second value on the heap
//! for every entry, which on a large transaction is the memory the caller
//! was trying not to spend.
//!
//! # Why the reclamation problem does not arise
//!
//! Nodes are never unlinked. An overwrite prepends, and the walk returns
//! the first match, so the newest value for a key is the one found. The
//! list is freed only in `Drop`, which takes `&mut self` and therefore
//! cannot run beside a reader. That is what makes a plain `AtomicPtr`
//! list sound here without epochs or hazard pointers, and it is a
//! property of how a transaction is used rather than of this type: it
//! holds because a transaction is resolved exclusively.

#![allow(unsafe_code)]

use core::hash::Hash;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

use kovan_map::HopscotchMap as ConcurrentMap;

struct Node<K, V> {
    key: K,
    value: V,
    next: *mut Node<K, V>,
}

/// A pointer to a listed node, for the index to hold instead of a copy.
struct NodeRef<K: 'static, V: 'static>(*mut Node<K, V>);

impl<K, V> Clone for NodeRef<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, V> Copy for NodeRef<K, V> {}

// SAFETY: the pointer reaches a node this buffer allocated, which is
// immutable once published and freed only under `&mut self`. Sharing a
// `NodeRef` shares read-only bytes for as long as the buffer lives, and
// the index never outlives the buffer.
unsafe impl<K: Send + Sync, V: Send + Sync> Send for NodeRef<K, V> {}
// SAFETY: see the `Send` impl above.
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for NodeRef<K, V> {}

/// A transaction's buffered entries, keyed and lock-free.
pub(crate) struct TxnBuffer<K: 'static, V: 'static> {
    /// Newest first. Null while the buffer is empty, which is the state
    /// that has to cost nothing.
    head: AtomicPtr<Node<K, V>>,
    len: AtomicUsize,
    /// An index over the list, built once it grows past `spill_at`.
    /// Values are pointers into the list rather than copies of it.
    spill: OnceLock<ConcurrentMap<K, NodeRef<K, V>>>,
    /// Entries buffered before the index is built. `0` never indexes.
    spill_at: usize,
    /// Set once the spill map has absorbed the entries that predate it,
    /// after which a lookup can trust the map alone.
    spill_seeded: AtomicBool,
}

// SAFETY: the pointers reach nodes this buffer allocated and never hands
// out, and nothing is freed until `Drop` takes `&mut self`. Sharing the
// buffer therefore shares immutable nodes plus atomics, so it is `Send`
// and `Sync` exactly when the data it holds is.
unsafe impl<K: Send + Sync + 'static, V: Send + Sync + 'static> Send for TxnBuffer<K, V> {}
// SAFETY: see the `Send` impl above.
unsafe impl<K: Send + Sync + 'static, V: Send + Sync + 'static> Sync for TxnBuffer<K, V> {}

impl<K: 'static, V: 'static> Default for TxnBuffer<K, V> {
    fn default() -> Self {
        Self {
            head: AtomicPtr::new(core::ptr::null_mut()),
            len: AtomicUsize::new(0),
            spill: OnceLock::new(),
            spill_at: 0,
            spill_seeded: AtomicBool::new(false),
        }
    }
}

impl<K, V> TxnBuffer<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// A buffer that indexes itself past `spill_at` entries. `0` never
    /// indexes and always walks.
    pub(crate) fn new(spill_at: usize) -> Self {
        Self {
            head: AtomicPtr::new(core::ptr::null_mut()),
            len: AtomicUsize::new(0),
            spill: OnceLock::new(),
            spill_at,
            spill_seeded: AtomicBool::new(false),
        }
    }

    /// Entries buffered, counting an overwritten key once per write.
    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Buffer `key` at `value`, replacing whatever this buffer held for
    /// it. O(1): the node goes on the front and the walk finds it first.
    pub(crate) fn insert(&self, key: K, value: V) {
        let indexed = self.spill.get();
        // The index needs its own key; it never needs its own value.
        let index_key = indexed.map(|_| key.clone());
        let node = Box::into_raw(Box::new(Node {
            key,
            value,
            next: core::ptr::null_mut(),
        }));
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            // SAFETY: `node` is this thread's fresh allocation and is not
            // published until the CAS below succeeds.
            unsafe { (*node).next = head };
            match self
                .head
                .compare_exchange_weak(head, node, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
        // Only after the node is on the list, so a lookup that finds it
        // in the index can always follow the pointer.
        if let (Some(spill), Some(index_key)) = (indexed, index_key) {
            spill.insert(index_key, NodeRef(node));
        }
        let len = self.len.fetch_add(1, Ordering::AcqRel) + 1;
        if self.spill_at > 0 && len > self.spill_at && self.spill.get().is_none() {
            self.start_spilling();
        }
    }

    /// Value buffered for `key`, or `None`.
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: core::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(spill) = self.spill.get() {
            if let Some(NodeRef(node)) = spill.get(key) {
                // SAFETY: the index only ever holds pointers to listed
                // nodes, which live until `Drop` takes `&mut self`.
                return Some(unsafe { &*node }.value.clone());
            }
            // A miss is only conclusive once seeding has folded in the
            // entries that predate the map.
            if self.spill_seeded.load(Ordering::Acquire) {
                return None;
            }
        }
        self.walk(key)
    }

    fn walk<Q>(&self, key: &Q) -> Option<V>
    where
        K: core::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            // SAFETY: nodes live until `Drop`, which needs `&mut self`
            // and so cannot be running now.
            let node = unsafe { &*cursor };
            if node.key.borrow() == key {
                return Some(node.value.clone());
            }
            cursor = node.next;
        }
        None
    }

    /// Publish the spill map, then fold in everything already listed.
    ///
    /// Order matters: publishing first means every insert that follows
    /// mirrors itself, and seeding second means nothing already listed is
    /// missed. A key touched by both paths keeps the mirrored value,
    /// because seeding never overwrites.
    fn start_spilling(&self) {
        if self.spill.set(ConcurrentMap::new()).is_err() {
            // Another thread got there first and is seeding it.
            return;
        }
        let spill = self.spill.get().expect("just set");
        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            // SAFETY: as in `walk`.
            let node = unsafe { &*cursor };
            spill.insert_if_absent(node.key.clone(), NodeRef(cursor));
            cursor = node.next;
        }
        self.spill_seeded.store(true, Ordering::Release);
    }

    /// The value for `key`, inserting `value` and returning it if the
    /// buffer has none. Linearizable: concurrent callers for one key all
    /// receive the same value.
    pub(crate) fn get_or_insert(&self, key: K, value: V) -> V {
        if let Some(existing) = self.get(&key) {
            return existing;
        }
        // A racing insert for the same key can land between the lookup
        // and the CAS, and both callers have to come away with the same
        // value. Re-reading settles it: the list is newest-first and both
        // nodes are on it, so every caller walks to the same one.
        self.insert(key.clone(), value);
        self.get(&key)
            .expect("the entry just inserted is on the list")
    }

    /// Every buffered entry, newest write of each key only, without
    /// consuming the buffer.
    ///
    /// Bounded by what this transaction has written, never by what the
    /// database holds, which is why materializing it is affordable where
    /// materializing the database side would not be.
    pub(crate) fn snapshot(&self) -> Vec<(K, V)> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            // SAFETY: as in `walk`.
            let node = unsafe { &*cursor };
            // Newest first, so the first sighting of a key is its current
            // value and later ones are writes it replaced.
            if seen.insert(node.key.clone()) {
                out.push((node.key.clone(), node.value.clone()));
            }
            cursor = node.next;
        }
        out
    }

    /// Every buffered entry, newest write of each key only.
    ///
    /// Takes `&mut self` because it consumes the buffer: draining is what
    /// a resolved transaction does, and a resolved transaction is
    /// exclusive.
    /// Newest write first, duplicates included. Deduplicating here would
    /// mean cloning every key to track what was seen; the caller collects
    /// into a keyed structure anyway and gets it for free by keeping the
    /// first value it sees for a key.
    pub(crate) fn drain(&mut self) -> Vec<(K, V)> {
        let mut drained = Vec::with_capacity(self.len.load(Ordering::Acquire));
        let mut cursor = self.head.swap(core::ptr::null_mut(), Ordering::AcqRel);
        while !cursor.is_null() {
            // SAFETY: this thread took the list out of the buffer and has
            // `&mut self`, so it is the only owner of these nodes.
            let node = unsafe { Box::from_raw(cursor) };
            cursor = node.next;
            drained.push((node.key, node.value));
        }
        self.len.store(0, Ordering::Release);
        self.spill = OnceLock::new();
        self.spill_seeded.store(false, Ordering::Release);
        drained
    }
}

impl<K: 'static, V: 'static> Drop for TxnBuffer<K, V> {
    fn drop(&mut self) {
        let mut cursor = self.head.swap(core::ptr::null_mut(), Ordering::AcqRel);
        while !cursor.is_null() {
            // SAFETY: `&mut self` means no reader can be walking the
            // list, and each node was allocated by `Box::into_raw`.
            let node = unsafe { Box::from_raw(cursor) };
            cursor = node.next;
        }
    }
}
