//! Bump allocator for one memtable, with tiered chunk sizing and a
//! bounded recycling pool.
//!
//! One [`Arena`] backs one memtable. Every skip-list node, key and value
//! that memtable holds is a bump allocation inside one of its chunks, so
//! [`Arena::used_bytes`] is the memtable's true payload cost (node
//! header, tower, key and value, rounded to alignment) and
//! [`Arena::reserved_bytes`] is exactly what was taken from the global
//! allocator to hold it.
//!
//! # Tiering
//!
//! Chunks are powers of two. Chunk `n` is `initial_chunk_size << n`,
//! capped at `max_chunk_size` and stepped back down so the arena never
//! reserves a class larger than what is left of its budget. A single
//! entry larger than `max_chunk_size` gets a dedicated, exactly-sized
//! chunk instead of raising the cap for everyone. See [`ArenaProfile`].
//!
//! # Recycling
//!
//! A dropped arena hands its chunks to a shared [`ChunkPool`] rather
//! than to the global allocator, and the next memtable takes them back.
//! The pool is bounded in **bytes** by
//! `write_buffer_size * max_write_buffer_number`; chunks offered past
//! that bound are freed immediately, because an unbounded cache of
//! chunks is a memory leak with better manners.
//!
//! # Lifetime, and why a refcount rather than epoch reclamation
//!
//! The memtable is insert-only: no node is ever unlinked and no node is
//! ever freed on its own. The only lifetime question is "may this chunk
//! be recycled yet", and an `Arc<Arena>` refcount answers it exactly.
//! Every live [`crate::DbSlice`] taken from a memtable holds one, so a
//! chunk cannot return to the pool while a reader still points into it.
//! Per-node safe memory reclamation would have nothing to reclaim here.
//!
//! # Safety invariants
//!
//! Named at every `unsafe` site that relies on them, and the target of
//! the miri model:
//!
//! - **A1 (layout).** Every chunk is allocated with
//!   `Layout::from_size_align(size, CHUNK_ALIGN)` and freed with the
//!   identical layout. `ChunkHandle` carries its own `size`, so `Drop`
//!   and [`ChunkPool::give`] cannot disagree about it.
//! - **A2 (range).** `alloc` checks the fit before forming a pointer,
//!   because constructing an out-of-range pointer is undefined even if
//!   it is never read.
//! - **A3 (alignment).** Chunk bases are `CHUNK_ALIGN`-aligned by A1 and
//!   the bump cursor is only ever advanced to an aligned offset, so
//!   alignment is inductive.
//! - **A4 (initialisation).** Bytes handed out by `alloc` are
//!   uninitialised, and nothing reads them before the caller fills them.
//! - **A5 (lifetime).** A chunk's bytes stay valid until `Arena::drop`,
//!   which needs the last `Arc<Arena>` to die. Every live
//!   [`crate::DbSlice`] taken from the memtable holds one. This is the
//!   entire recycling safety argument.
//! - **A6 (ownership).** [`ChunkPool::give`] transfers a chunk exactly
//!   once: it is either moved into a ring slot or dropped. A full ring
//!   hands it back, and that path frees it, so nothing is leaked and
//!   nothing is freed twice.
//! - **A7 (accounting).** `parked` is incremented only by the size of a
//!   chunk that really entered a ring, and decremented only by the size
//!   of one that really left, so it can neither underflow nor
//!   over-report the bound.

#![allow(unsafe_code)]

use std::alloc::Layout;
use std::ptr::NonNull;

use kovan_queue::array_queue::ArrayQueue;

use super::sync::{Arc, AtomicUsize, Mutex, Ordering};

/// Alignment every chunk is allocated at. Covers every alignment the
/// skip-list asks for, so a fresh chunk never needs leading padding.
pub(crate) const CHUNK_ALIGN: usize = 16;

/// Smallest chunk the arena will ever request, so a pathologically small
/// `write_buffer_size` still produces a usable chunk.
const MIN_CHUNK: usize = 64;

/// Largest number of chunks the pool parks in any one size class. The
/// byte budget is the real bound; this only caps the ring metadata.
const MAX_CHUNKS_PER_CLASS: usize = 256;

/// How a memtable arena sizes its chunks.
///
/// One arena with one sizing policy serves every deployment: the profile
/// is two numbers, so the embedded path is the same code as the server
/// path and cannot rot untested.
///
/// Both fields must be nonzero powers of two with
/// `initial_chunk_size <= max_chunk_size`; [`crate::Options::validate`]
/// enforces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaProfile {
    /// Size of the arena's first chunk, in bytes.
    pub initial_chunk_size: usize,
    /// Largest chunk requested from the global allocator for an ordinary
    /// allocation. An entry larger than this gets its own exactly-sized
    /// chunk rather than raising the cap.
    pub max_chunk_size: usize,
}

impl ArenaProfile {
    /// 64 KiB initial, 1 MiB cap. A 64 MiB memtable is about 70 chunks,
    /// so the global allocator is touched roughly once per megabyte
    /// written, and only until the pool is warm. This is the default.
    pub const SERVER: Self = Self {
        initial_chunk_size: 64 * 1024,
        max_chunk_size: 1024 * 1024,
    };

    /// 4 KiB initial, 64 KiB cap. A 256 KiB memtable is a handful of
    /// chunks with no rounding waste, and 64 KiB is exactly one wasm32
    /// page, so no chunk ever straddles more pages than its contents
    /// need. Selected by [`crate::Options::embedded`].
    pub const EMBEDDED: Self = Self {
        initial_chunk_size: 4 * 1024,
        max_chunk_size: 64 * 1024,
    };

    /// Whether both sizes are nonzero powers of two in the right order.
    pub fn is_valid(&self) -> bool {
        self.initial_chunk_size.is_power_of_two()
            && self.max_chunk_size.is_power_of_two()
            && self.initial_chunk_size <= self.max_chunk_size
    }
}

impl Default for ArenaProfile {
    fn default() -> Self {
        Self::SERVER
    }
}

/// Largest power of two that is `<= n`, and `1` for `n == 0`.
fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

/// The smallest chunk class an arena with this profile and budget uses.
///
/// A budget smaller than `initial_chunk_size` would otherwise reserve a
/// whole initial chunk and rotate after a single entry, so the class
/// floor tracks the budget down. Arenas and the pool must agree on this
/// number or recycling silently stops working, so both call here.
fn base_class(profile: ArenaProfile, budget: usize) -> usize {
    let base = profile
        .initial_chunk_size
        .min(prev_power_of_two(budget))
        .max(MIN_CHUNK);
    base.min(profile.max_chunk_size.max(MIN_CHUNK))
}

/// One chunk of arena memory, owning its allocation.
struct ChunkHandle {
    ptr: NonNull<u8>,
    size: usize,
}

// SAFETY (A1): a `ChunkHandle` owns its allocation exclusively; the raw
// pointer is never aliased by another handle and the bytes carry no
// thread affinity, so moving one between threads only moves ownership.
unsafe impl Send for ChunkHandle {}
// SAFETY: a shared `&ChunkHandle` exposes no interior mutability; the
// bytes are reached through pointers copied out of it, whose safety is
// argued at their own use sites (A2..A5).
unsafe impl Sync for ChunkHandle {}

impl ChunkHandle {
    /// Allocate `size` bytes at [`CHUNK_ALIGN`], or `None` if the global
    /// allocator refused.
    fn alloc(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, CHUNK_ALIGN).ok()?;
        // SAFETY (A1): `Layout::from_size_align` succeeded, so `size` is
        // nonzero (callers never pass 0) and rounds up without overflow.
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).map(|ptr| Self { ptr, size })
    }
}

impl Drop for ChunkHandle {
    fn drop(&mut self) {
        // SAFETY (A1): this allocation came from `ChunkHandle::alloc`
        // with exactly `Layout::from_size_align(self.size, CHUNK_ALIGN)`,
        // which was validated there, and it is freed exactly once
        // because `ChunkHandle` is neither `Copy` nor `Clone`.
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, CHUNK_ALIGN);
            std::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Shared, bounded pool of retired arena chunks.
///
/// # Bound
///
/// The pool never parks more than `write_buffer_size *
/// max_write_buffer_number` bytes. Chunks offered past that bound are
/// freed rather than hoarded. The bound is stated in bytes, not chunk
/// count, because bytes are the resource that runs out.
///
/// One pool per engine, so a process with two open databases has two
/// independently bounded pools.
///
/// # Why it is mandatory on wasm32
///
/// wasm32 linear memory grows in 64 KiB pages and never shrinks, so
/// every high-water mark is permanent for the life of the instance.
/// Without recycling, each flush-and-refill cycle would hand chunks back
/// to a `dlmalloc` that may not reuse them, and the fragmentation would
/// be permanent growth. With it, the second memtable takes the first
/// one's chunk addresses back, so `memory.grow` runs once per size class,
/// ever.
pub(crate) struct ChunkPool {
    /// One MPMC ring per power-of-two class in `[base, max_chunk_size]`.
    classes: Box<[ArrayQueue<ChunkHandle>]>,
    base: usize,
    parked: AtomicUsize,
    budget: usize,
    /// Requests this pool could not serve, so the arena had to take a
    /// fresh chunk from the global allocator. Per-pool, so a test can
    /// attribute it while other tests run in parallel.
    #[cfg(test)]
    misses: AtomicUsize,
}

impl ChunkPool {
    /// Build a pool for arenas of `write_buffer_size` bytes using
    /// `profile`, bounded at `write_buffer_size * max_write_buffer_number`
    /// parked bytes.
    pub(crate) fn new(
        profile: ArenaProfile,
        write_buffer_size: usize,
        max_write_buffer_number: usize,
    ) -> Self {
        let profile = if profile.is_valid() {
            profile
        } else {
            ArenaProfile::SERVER
        };
        let base = base_class(profile, write_buffer_size);
        let max = profile.max_chunk_size.max(base);
        let budget = write_buffer_size.saturating_mul(max_write_buffer_number.max(1));

        let class_count = (max.trailing_zeros() - base.trailing_zeros()) as usize + 1;
        let classes: Vec<ArrayQueue<ChunkHandle>> = (0..class_count)
            .map(|i| {
                let size = base << i;
                let cap = (budget / size).clamp(1, MAX_CHUNKS_PER_CLASS);
                ArrayQueue::new(cap)
            })
            .collect();

        Self {
            classes: classes.into_boxed_slice(),
            base,
            parked: AtomicUsize::new(0),
            budget,
            #[cfg(test)]
            misses: AtomicUsize::new(0),
        }
    }

    /// Ring index for `size`, or `None` when `size` is not a pool class
    /// (a dedicated oversized chunk, or one below the class floor).
    fn class_index(&self, size: usize) -> Option<usize> {
        if !size.is_power_of_two() || size < self.base {
            return None;
        }
        let idx = (size.trailing_zeros() - self.base.trailing_zeros()) as usize;
        (idx < self.classes.len()).then_some(idx)
    }

    /// Take a chunk of exactly `size` bytes, or `None` when the class is
    /// empty. Never calls the global allocator.
    fn take(&self, size: usize) -> Option<ChunkHandle> {
        let chunk = self
            .class_index(size)
            .and_then(|idx| self.classes[idx].pop());
        match chunk {
            Some(chunk) => {
                // A7: decrement only for a chunk that really left the ring.
                self.parked.fetch_sub(chunk.size, Ordering::Relaxed);
                Some(chunk)
            }
            None => {
                #[cfg(test)]
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Requests this pool could not serve. Test-only: it is how the
    /// steady-state recycling claim is checked.
    #[cfg(test)]
    fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    /// Offer a retired chunk back. Frees it when the pool is at its byte
    /// bound, when the ring for its class is full, or when its size is
    /// not a pool class. Never calls the global allocator to grow.
    fn give(&self, chunk: ChunkHandle) {
        let Some(idx) = self.class_index(chunk.size) else {
            return;
        };
        let size = chunk.size;
        // A7: reserve the bytes before the push so two concurrent givers
        // cannot both see room for the last chunk, and give them back on
        // any path that does not park the chunk.
        let mut parked = self.parked.load(Ordering::Relaxed);
        loop {
            if parked.saturating_add(size) > self.budget {
                return;
            }
            match self.parked.compare_exchange_weak(
                parked,
                parked + size,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => parked = observed,
            }
        }
        // A6: `push` hands the chunk back on a full ring, so it is
        // dropped (and freed) exactly once here rather than leaked.
        if self.classes[idx].push(chunk).is_err() {
            self.parked.fetch_sub(size, Ordering::Relaxed);
        }
    }

    /// Bytes currently parked across every class.
    pub(crate) fn parked_bytes(&self) -> usize {
        self.parked.load(Ordering::Relaxed)
    }

    /// The pool's hard byte bound.
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }
}

/// Chunks owned by one arena, plus the bump cursor into the newest one.
///
/// Only the writer touches this; readers follow node pointers, which are
/// absolute addresses and never go through the chunk list.
struct ArenaState {
    chunks: Vec<ChunkHandle>,
    /// Bytes handed out of `chunks.last()`.
    offset: usize,
}

/// Bump allocator over a list of pooled chunks.
pub(crate) struct Arena {
    state: Mutex<ArenaState>,
    reserved: AtomicUsize,
    used: AtomicUsize,
    pool: Arc<ChunkPool>,
    budget: usize,
    base: usize,
    max_chunk_size: usize,
}

// SAFETY (A5): the chunk list is behind a `Mutex` and every pointer
// `alloc` returns addresses memory this arena owns until it is dropped,
// which cannot happen while any `Arc<Arena>` (including one held by a
// live `DbSlice`) survives. The atomics carry the byte counters.
unsafe impl Send for Arena {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for Arena {}

impl Arena {
    /// A new, empty arena. Allocates nothing: the first chunk is taken
    /// on the first [`Arena::alloc`], so an untouched memtable costs no
    /// linear memory at all.
    pub(crate) fn new(pool: Arc<ChunkPool>, budget: usize, profile: ArenaProfile) -> Self {
        let profile = if profile.is_valid() {
            profile
        } else {
            ArenaProfile::SERVER
        };
        let base = base_class(profile, budget);
        Self {
            state: Mutex::new(ArenaState {
                chunks: Vec::new(),
                offset: 0,
            }),
            reserved: AtomicUsize::new(0),
            used: AtomicUsize::new(0),
            pool,
            budget,
            base,
            max_chunk_size: profile.max_chunk_size.max(base),
        }
    }

    /// Bump-allocate `size` bytes aligned to `align`.
    ///
    /// Returns `None` only when the global allocator refused a new
    /// chunk. The arena does not refuse at its budget: it records the
    /// overshoot in [`Arena::used_bytes`], and the engine rotates the
    /// memtable at the next write. `align` must be a power of two no
    /// larger than [`CHUNK_ALIGN`].
    pub(crate) fn alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(align.is_power_of_two() && align <= CHUNK_ALIGN);
        if size == 0 {
            return None;
        }
        let mut state = self.state.lock();
        if let Some(ptr) = Self::bump(&mut state, size, align) {
            self.used.fetch_add(size, Ordering::Relaxed);
            return Some(ptr);
        }

        let chunk_size = self.next_chunk_size(state.chunks.len(), size);
        let chunk = match self.pool.take(chunk_size) {
            Some(chunk) => chunk,
            None => ChunkHandle::alloc(chunk_size)?,
        };
        self.reserved.fetch_add(chunk.size, Ordering::Relaxed);
        state.chunks.push(chunk);
        state.offset = 0;

        let ptr = Self::bump(&mut state, size, align)?;
        self.used.fetch_add(size, Ordering::Relaxed);
        Some(ptr)
    }

    /// Carve `size` bytes out of the newest chunk, or `None` if they do
    /// not fit.
    fn bump(state: &mut ArenaState, size: usize, align: usize) -> Option<NonNull<u8>> {
        let chunk = state.chunks.last()?;
        let base = chunk.ptr.as_ptr() as usize;
        // A2: the fit is checked before any pointer is formed, because
        // forming an out-of-range pointer is undefined even unread.
        let start = base.checked_add(state.offset)?.next_multiple_of(align);
        let end = start.checked_add(size)?;
        if end > base.checked_add(chunk.size)? {
            return None;
        }
        state.offset = end - base;
        // SAFETY (A2, A3): `start - base` and `end - base` are both
        // within `chunk.size`, so the result stays inside the chunk's
        // allocation, and `start` was rounded up to `align`.
        let ptr = unsafe { chunk.ptr.as_ptr().add(start - base) };
        NonNull::new(ptr)
    }

    /// Size of chunk number `chunk_index`, given that the allocation
    /// forcing it needs `need` bytes.
    fn next_chunk_size(&self, chunk_index: usize, need: usize) -> usize {
        if need > self.max_chunk_size {
            // A dedicated, exactly-sized chunk: one huge entry must not
            // raise the cap for every later chunk.
            return need.next_multiple_of(CHUNK_ALIGN);
        }
        // `base` is a power of two, so it can be shifted left exactly
        // `leading_zeros` times before it overflows. Clamping the shift
        // rather than letting it wrap matters: a 64 MiB memtable reaches
        // chunk 48, where an unclamped `base << 48` wraps to zero and
        // would collapse every later chunk to the size of one entry.
        let max_shift = self.base.leading_zeros() as usize;
        let class = (self.base << chunk_index.min(max_shift)).min(self.max_chunk_size);
        // Step back down so the arena never reserves a class larger than
        // what is left of its budget. This is what keeps a 1 MiB budget
        // from reserving 2 MiB on its way up the ladder.
        let remaining = self
            .budget
            .saturating_sub(self.reserved.load(Ordering::Relaxed));
        let mut size = class;
        while size > self.base && size > remaining {
            size >>= 1;
        }
        size.max(need.next_power_of_two()).min(self.max_chunk_size)
    }

    /// Bytes taken from the global allocator (or the pool) to back this
    /// arena: the sum of its chunk sizes.
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Relaxed)
    }

    /// Bytes handed out by [`Arena::alloc`]. For a memtable this is the
    /// full per-entry cost - node header, tower, key and value, rounded
    /// to alignment - and it excludes only the unused tail of the newest
    /// chunk. This is the number `write_buffer_size` bounds.
    pub(crate) fn used_bytes(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// How many chunks this arena holds.
    #[cfg(test)]
    pub(crate) fn chunk_count(&self) -> usize {
        self.state.lock().chunks.len()
    }

    /// The sizes of this arena's chunks, oldest first.
    #[cfg(test)]
    pub(crate) fn chunk_sizes(&self) -> Vec<usize> {
        self.state.lock().chunks.iter().map(|c| c.size).collect()
    }
}

impl Drop for Arena {
    /// Return every chunk to the pool. Runs only when the last
    /// `Arc<Arena>` dies, which includes every outstanding
    /// [`crate::DbSlice`] taken from this memtable (A5).
    fn drop(&mut self) {
        let state = self.state.get_mut();
        for chunk in state.chunks.drain(..) {
            self.pool.give(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn pool(profile: ArenaProfile, budget: usize, count: usize) -> Arc<ChunkPool> {
        Arc::new(ChunkPool::new(profile, budget, count))
    }

    /// Budget for the tests that fill an arena one small allocation at a
    /// time. Miri interprets every one of those, so a 64 MiB fill is
    /// hours of wall clock; 8 MiB still climbs the ladder past the cap
    /// and leaves every assertion below unchanged. Chunk indices beyond
    /// the ladder's top are covered arithmetically by
    /// `the_class_ladder_saturates_instead_of_wrapping`, which costs
    /// nothing.
    #[cfg(not(miri))]
    const LADDER_BUDGET: usize = 64 * 1024 * 1024;
    #[cfg(miri)]
    const LADDER_BUDGET: usize = 8 * 1024 * 1024;

    /// Recycling cycles. The bound only needs more cycles than the
    /// `2 * m` live arenas the test keeps, so a shorter run checks the
    /// same invariant.
    #[cfg(not(miri))]
    const RECYCLE_CYCLES: usize = 50;
    #[cfg(miri)]
    const RECYCLE_CYCLES: usize = 8;

    #[test]
    fn empty_arena_reserves_nothing() {
        let arena = Arena::new(
            pool(ArenaProfile::SERVER, 64 * 1024, 2),
            64 * 1024,
            ArenaProfile::SERVER,
        );
        assert_eq!(arena.reserved_bytes(), 0);
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.chunk_count(), 0);
    }

    #[test]
    fn alloc_is_aligned_and_in_range() {
        let profile = ArenaProfile::EMBEDDED;
        let arena = Arena::new(pool(profile, 64 * 1024, 2), 64 * 1024, profile);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for i in 1..200usize {
            let size = i * 7;
            let ptr = arena.alloc(size, 8).expect("arena has budget");
            assert_eq!(ptr.as_ptr() as usize % 8, 0, "8-aligned");
            let start = ptr.as_ptr() as usize;
            for (other, other_len) in &seen {
                assert!(
                    start + size <= *other || *other + *other_len <= start,
                    "allocations must not overlap"
                );
            }
            seen.push((start, size));
        }
        assert!(arena.used_bytes() >= seen.iter().map(|(_, l)| l).sum::<usize>());
    }

    #[test]
    fn chunk_sizes_follow_the_growth_rule() {
        let profile = ArenaProfile::EMBEDDED;
        let budget = 256 * 1024;
        let arena = Arena::new(pool(profile, budget, 2), budget, profile);
        // Fill the arena past its budget one 1 KiB block at a time.
        while arena.used_bytes() < budget {
            arena.alloc(1024, 8).expect("allocator");
        }
        let sizes = arena.chunk_sizes();
        assert_eq!(
            sizes,
            vec![4096, 8192, 16384, 32768, 65536, 65536, 65536, 4096],
            "4 KiB doubling up to the 64 KiB cap, stepped down at the budget"
        );
        assert_eq!(sizes.iter().sum::<usize>(), budget);
    }

    #[test]
    fn the_class_ladder_saturates_instead_of_wrapping() {
        // Chunk 48 of a 64 KiB-based ladder is where `base << index`
        // overflows a 64-bit shift. Every chunk from there on must stay
        // at the cap, not collapse to the size of a single entry.
        let profile = ArenaProfile::SERVER;
        let budget = 64 * 1024 * 1024;
        let arena = Arena::new(pool(profile, budget, 2), budget, profile);
        for index in [0usize, 4, 47, 48, 64, 200, usize::MAX] {
            let size = arena.next_chunk_size(index, 512);
            assert!(
                size >= arena.base && size <= profile.max_chunk_size,
                "chunk {index} sized {size}, outside [{}, {}]",
                arena.base,
                profile.max_chunk_size
            );
        }
        assert_eq!(arena.next_chunk_size(48, 512), profile.max_chunk_size);
    }

    #[test]
    fn a_large_budget_keeps_full_size_chunks_all_the_way_up() {
        let profile = ArenaProfile::SERVER;
        let budget = LADDER_BUDGET;
        let arena = Arena::new(pool(profile, budget, 2), budget, profile);
        while arena.used_bytes() < budget {
            arena.alloc(4096, 8).expect("allocator");
        }
        let sizes = arena.chunk_sizes();
        assert!(
            sizes.len() < 80,
            "a 64 MiB memtable must be a few dozen chunks, got {}",
            sizes.len()
        );
        let at_cap = sizes
            .iter()
            .filter(|size| **size == profile.max_chunk_size)
            .count();
        assert!(
            at_cap >= sizes.len() - 6,
            "all but the ladder's first few chunks must be at the cap: {sizes:?}"
        );
    }

    #[test]
    fn growth_never_overshoots_the_budget_by_more_than_one_class() {
        for budget in [1usize, 100, 4096, 6000, 1024 * 1024, LADDER_BUDGET] {
            let profile = ArenaProfile::SERVER;
            let arena = Arena::new(pool(profile, budget, 2), budget, profile);
            while arena.used_bytes() < budget {
                arena.alloc(512, 8).expect("allocator");
            }
            let bound = budget + profile.max_chunk_size;
            assert!(
                arena.reserved_bytes() <= bound,
                "budget {budget}: reserved {} > {bound}",
                arena.reserved_bytes()
            );
        }
    }

    #[test]
    fn a_small_budget_does_not_reserve_a_full_initial_chunk() {
        // The 4 KiB `write_buffer_size` the test suite uses everywhere
        // must reserve 4 KiB, not the 64 KiB server initial chunk.
        let profile = ArenaProfile::SERVER;
        let arena = Arena::new(pool(profile, 4096, 2), 4096, profile);
        arena.alloc(64, 8).expect("allocator");
        assert_eq!(arena.reserved_bytes(), 4096);
    }

    #[test]
    fn oversized_entry_gets_a_dedicated_chunk() {
        let profile = ArenaProfile::EMBEDDED;
        let budget = 256 * 1024;
        let arena = Arena::new(pool(profile, budget, 2), budget, profile);
        let big = profile.max_chunk_size * 3 + 5;
        arena.alloc(big, 8).expect("allocator");
        let sizes = arena.chunk_sizes();
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0], big.next_multiple_of(CHUNK_ALIGN));
        // A dedicated chunk is not a pool class, so it is freed rather
        // than parked.
        let pool = Arc::clone(&arena.pool);
        drop(arena);
        assert_eq!(pool.parked_bytes(), 0);
    }

    #[test]
    fn pool_bound_is_respected() {
        let profile = ArenaProfile::EMBEDDED;
        // Budget of one 4 KiB chunk: the pool parks at most 4 KiB.
        let pool = pool(profile, 4096, 1);
        assert_eq!(pool.budget(), 4096);
        for _ in 0..64 {
            pool.give(ChunkHandle::alloc(4096).expect("allocator"));
            assert!(pool.parked_bytes() <= pool.budget());
        }
        assert_eq!(pool.parked_bytes(), 4096);
        assert!(pool.take(4096).is_some());
        assert_eq!(pool.parked_bytes(), 0);
        assert!(pool.take(4096).is_none());
    }

    #[test]
    fn pool_refuses_sizes_that_are_not_classes() {
        let profile = ArenaProfile::EMBEDDED;
        let pool = pool(profile, 256 * 1024, 2);
        pool.give(ChunkHandle::alloc(3000).expect("allocator"));
        pool.give(ChunkHandle::alloc(1024 * 1024).expect("allocator"));
        assert_eq!(pool.parked_bytes(), 0, "neither size is a class");
        pool.give(ChunkHandle::alloc(8192).expect("allocator"));
        assert_eq!(pool.parked_bytes(), 8192);
    }

    #[test]
    fn steady_state_recycling_touches_the_allocator_zero_times() {
        let profile = ArenaProfile::EMBEDDED;
        let budget = 64 * 1024;
        let pool = pool(profile, budget, 2);

        let fill = |arena: &Arena| {
            while arena.used_bytes() < budget {
                arena.alloc(256, 8).expect("allocator");
            }
        };

        // Cycle 1 and 2 warm the pool.
        for _ in 0..2 {
            let arena = Arena::new(Arc::clone(&pool), budget, profile);
            fill(&arena);
        }
        assert!(pool.parked_bytes() > 0, "pool must hold the retired chunks");

        // Cycle 3 onwards must take every chunk back from the pool.
        for cycle in 3..8 {
            let before = pool.misses();
            let arena = Arena::new(Arc::clone(&pool), budget, profile);
            fill(&arena);
            let chunks = arena.chunk_count();
            drop(arena);
            let missed = pool.misses() - before;
            assert_eq!(
                missed, 0,
                "cycle {cycle}: {missed} of {chunks} chunks came from the global allocator"
            );
        }
    }

    #[test]
    fn high_water_mark_is_bounded_across_many_cycles() {
        // The wasm32 claim: peak live + parked bytes never exceeds
        // `2 * M * (W + c) + M * W`, and is flat after the pool warms.
        let profile = ArenaProfile::EMBEDDED;
        let w = 64 * 1024usize;
        let m = 2usize;
        let c = profile.max_chunk_size;
        let pool = pool(profile, w, m);
        let bound = 2 * m * (w + c) + m * w;

        let mut peak = 0usize;
        let mut live: Vec<Arena> = Vec::new();
        for _ in 0..RECYCLE_CYCLES {
            let arena = Arena::new(Arc::clone(&pool), w, profile);
            while arena.used_bytes() < w {
                arena.alloc(128, 8).expect("allocator");
            }
            live.push(arena);
            if live.len() > 2 * m {
                live.remove(0);
            }
            let total: usize =
                live.iter().map(|a| a.reserved_bytes()).sum::<usize>() + pool.parked_bytes();
            peak = peak.max(total);
        }
        assert!(peak <= bound, "peak {peak} exceeds bound {bound}");
    }

    proptest! {
        #[test]
        fn allocations_never_overlap_or_leave_their_chunk(
            sizes in proptest::collection::vec(1usize..3000, 1..200),
            budget_kib in 1usize..64,
        ) {
            let profile = ArenaProfile::EMBEDDED;
            let budget = budget_kib * 1024;
            let arena = Arena::new(pool(profile, budget, 2), budget, profile);
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for size in sizes {
                let ptr = arena.alloc(size, 8).expect("allocator");
                let start = ptr.as_ptr() as usize;
                prop_assert_eq!(start % 8, 0);
                for (other, len) in &spans {
                    prop_assert!(start + size <= *other || other + len <= start);
                }
                spans.push((start, size));
            }
            prop_assert!(arena.used_bytes() >= spans.iter().map(|(_, l)| l).sum::<usize>());
            prop_assert!(arena.reserved_bytes() >= arena.used_bytes());
        }

        #[test]
        fn base_class_is_always_a_valid_pool_class(budget in 0usize..(1 << 28)) {
            for profile in [ArenaProfile::SERVER, ArenaProfile::EMBEDDED] {
                let base = base_class(profile, budget);
                prop_assert!(base.is_power_of_two());
                prop_assert!(base <= profile.max_chunk_size);
                let pool = ChunkPool::new(profile, budget, 2);
                prop_assert_eq!(pool.class_index(base), Some(0));
            }
        }
    }
}
