//! Column families — multiple logically isolated keyspaces inside
//! one [`crate::Db`].
//!
//! # Design
//!
//! Column families in lark are implemented as **key-prefix
//! namespaces** on top of the single underlying LSM engine. Every
//! logical operation on a CF wraps the caller's user key in a
//! 4-byte big-endian `cf_id` prefix before handing it to the
//! engine. The engine sees one global keyspace; the logical
//! isolation is airtight because distinct `cf_id`s produce disjoint
//! byte ranges.
//!
//! ## Trade-offs vs per-CF memtables
//!
//! This design shares:
//!
//! - One memtable, one WAL, one manifest, one compaction thread.
//! - One block cache, one bloom / prefix-bloom budget.
//!
//! That means **cross-CF writes are atomic for free** (a
//! [`crate::WriteBatch`] touching multiple CFs lands in a single
//! underlying `apply_batch` call and crash-recovers as a unit), but
//! per-CF `write_buffer_size` / per-CF compaction strategies are
//! not available. The issue that introduced this feature explicitly
//! scopes those out of v1.
//!
//! ## Atomic flush across column families
//!
//! Because every CF shares one memtable, one WAL, and one
//! manifest, multi-CF writes are atomic in lark by construction.
//! A flush produces one SSTable that either contains every key in
//! a batch or none of them; the WAL is the source of truth until
//! the manifest edit lands, so a crash mid-flush replays the
//! whole batch on reopen.
//!
//! [`crate::Options::atomic_flush`] is accepted for parity with
//! storage engines that require an explicit opt-in to get this
//! guarantee — under lark's design its value is irrelevant, the
//! guarantee is always on.
//!
//! ## Metadata storage
//!
//! A reserved [`META_CF_ID`] = `0` holds CF registry entries:
//!
//! - `[0,0,0,0] || "next_id"` → `u32` big-endian counter (next id
//!   to hand out to a freshly created CF).
//! - `[0,0,0,0] || "name:" || <name>` → `u32` big-endian id of the
//!   CF with that name.
//!
//! Users cannot create a CF with id `0`; the default user-facing
//! CF is [`DEFAULT_CF_ID`] = `1` and is created lazily on first
//! [`crate::Db::open`] of a database that doesn't already contain
//! it. User keys in any CF cannot collide with metadata keys
//! because metadata lives under the reserved prefix `[0,0,0,0]`,
//! which no user-facing CF ever produces.
//!
//! ## Dropping a CF
//!
//! [`crate::Db::drop_column_family`] issues a `delete_range` over
//! the CF's prefix range and then removes the `name:` and the
//! handle's lookups. The range delete is O(1) write work regardless
//! of how many keys the CF contained; space is reclaimed on the
//! next compaction over the range. Dropped CFs leave no trace on
//! reopen.

use std::sync::Arc;

use parking_lot::Mutex;

/// Reserved column-family id used to store the CF registry. Users
/// cannot create a CF with this id; user-facing CFs start at
/// [`DEFAULT_CF_ID`].
pub(crate) const META_CF_ID: u32 = 0;

/// Id of the default user-facing column family. Auto-created on
/// the first [`crate::Db::open`] of any database and always present
/// thereafter.
pub(crate) const DEFAULT_CF_ID: u32 = 1;

/// Name of the default column family.
pub const DEFAULT_CF_NAME: &str = "default";

/// A handle to a column family. Cheap to clone; carries only the
/// CF's name and numeric id. Handles to dropped CFs are inert —
/// calls that use them will see no data and writes land in a
/// tombstoned range, but the Db won't panic.
#[derive(Debug, Clone)]
pub struct ColumnFamilyHandle {
    pub(crate) name: Arc<String>,
    pub(crate) id: u32,
}

impl ColumnFamilyHandle {
    /// The CF's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Internal: the numeric id used to derive the key prefix.
    pub(crate) fn id(&self) -> u32 {
        self.id
    }
}

impl PartialEq for ColumnFamilyHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ColumnFamilyHandle {}

/// Encode a user key for a given CF. Returns
/// `cf_id_be(4) || user_key`.
pub(crate) fn prefix_key(cf_id: u32, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + key.len());
    out.extend_from_slice(&cf_id.to_be_bytes());
    out.extend_from_slice(key);
    out
}

/// Shortest byte string strictly greater than every key prefixed
/// with `cf_id`. Used by CF drop to issue a
/// `delete_range(prefix, upper_bound)` and by `iter_cf` to bound
/// the scan.
pub(crate) fn cf_upper_bound(cf_id: u32) -> Vec<u8> {
    (cf_id + 1).to_be_bytes().to_vec()
}

/// Lower bound (inclusive) of the CF's key range.
pub(crate) fn cf_lower_bound(cf_id: u32) -> Vec<u8> {
    cf_id.to_be_bytes().to_vec()
}

/// Well-known metadata keys in the reserved [`META_CF_ID`] CF.
/// Callers use these directly via `Db::put` / `Db::get` on the
/// meta CF prefix.
pub(crate) mod meta {
    use super::META_CF_ID;

    pub(crate) fn next_id_key() -> Vec<u8> {
        let mut k = META_CF_ID.to_be_bytes().to_vec();
        k.extend_from_slice(b"next_id");
        k
    }

    pub(crate) fn name_key(name: &str) -> Vec<u8> {
        let mut k = META_CF_ID.to_be_bytes().to_vec();
        k.extend_from_slice(b"name:");
        k.extend_from_slice(name.as_bytes());
        k
    }

    /// Prefix used to walk every `name:*` entry in the meta CF
    /// via a range scan at open time.
    pub(crate) fn name_scan_prefix() -> Vec<u8> {
        let mut k = META_CF_ID.to_be_bytes().to_vec();
        k.extend_from_slice(b"name:");
        k
    }

    /// Exclusive upper bound of the `name:*` range, built by
    /// incrementing the trailing `:` of the prefix to `;`.
    pub(crate) fn name_scan_upper() -> Vec<u8> {
        let mut k = META_CF_ID.to_be_bytes().to_vec();
        k.extend_from_slice(b"name;");
        k
    }

    /// Parse the name component out of a `name:<name>` meta key.
    pub(crate) fn name_from_key(key: &[u8]) -> Option<&str> {
        let prefix = name_scan_prefix();
        key.strip_prefix(prefix.as_slice())
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }
}

/// In-memory cache of the on-disk CF registry. Guarded by a
/// single mutex — CF creation and drop are rare relative to point
/// writes, and the common path (`column_family(name)`) only reads.
pub(crate) struct CfRegistry {
    inner: Mutex<CfRegistryInner>,
}

struct CfRegistryInner {
    /// Next CF id to allocate. Persisted in meta under
    /// [`meta::next_id_key`].
    next_id: u32,
    /// name → id map. Kept in sync with the on-disk `name:*`
    /// entries.
    by_name: std::collections::HashMap<String, u32>,
    /// id → name reverse lookup. Populated lazily.
    by_id: std::collections::HashMap<u32, String>,
}

impl CfRegistry {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(CfRegistryInner {
                next_id: DEFAULT_CF_ID + 1,
                by_name: std::collections::HashMap::new(),
                by_id: std::collections::HashMap::new(),
            }),
        }
    }

    pub(crate) fn load(&self, entries: impl IntoIterator<Item = (String, u32)>, next_id: u32) {
        let mut inner = self.inner.lock();
        inner.next_id = next_id;
        inner.by_name.clear();
        inner.by_id.clear();
        for (name, id) in entries {
            inner.by_name.insert(name.clone(), id);
            inner.by_id.insert(id, name);
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<ColumnFamilyHandle> {
        let inner = self.inner.lock();
        inner.by_name.get(name).map(|&id| ColumnFamilyHandle {
            name: Arc::new(name.to_string()),
            id,
        })
    }

    /// Allocate a fresh CF id for `name`. The caller is responsible
    /// for persisting `(next_id_after, name → id)` to the meta CF
    /// atomically before returning the handle to user code.
    pub(crate) fn allocate(&self, name: &str) -> (ColumnFamilyHandle, u32) {
        let mut inner = self.inner.lock();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.by_name.insert(name.to_string(), id);
        inner.by_id.insert(id, name.to_string());
        (
            ColumnFamilyHandle {
                name: Arc::new(name.to_string()),
                id,
            },
            inner.next_id,
        )
    }

    /// Remove a CF from the in-memory registry. The caller is
    /// responsible for deleting the on-disk `name:*` entry and
    /// issuing the data-range delete.
    pub(crate) fn remove(&self, name: &str) {
        let mut inner = self.inner.lock();
        if let Some(id) = inner.by_name.remove(name) {
            inner.by_id.remove(&id);
        }
    }

    /// Snapshot of every registered CF name in arbitrary order.
    pub(crate) fn names(&self) -> Vec<String> {
        self.inner.lock().by_name.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_key_contains_cf_id() {
        let k = prefix_key(5, b"hello");
        assert_eq!(&k[0..4], &[0, 0, 0, 5]);
        assert_eq!(&k[4..], b"hello");
    }

    #[test]
    fn cf_bounds_are_adjacent() {
        let lo = cf_lower_bound(7);
        let hi = cf_upper_bound(7);
        assert_eq!(lo, vec![0, 0, 0, 7]);
        assert_eq!(hi, vec![0, 0, 0, 8]);
        assert!(lo < hi);
    }

    #[test]
    fn meta_keys_live_under_reserved_cf() {
        let next = meta::next_id_key();
        assert_eq!(&next[0..4], &[0, 0, 0, 0]);
        let name = meta::name_key("foo");
        assert_eq!(&name[0..4], &[0, 0, 0, 0]);
        assert!(name.ends_with(b"foo"));
    }

    #[test]
    fn meta_name_from_key_parses() {
        let key = meta::name_key("widgets");
        assert_eq!(meta::name_from_key(&key), Some("widgets"));
    }

    #[test]
    fn meta_scan_range_is_tight() {
        let lo = meta::name_scan_prefix();
        let hi = meta::name_scan_upper();
        assert!(lo < hi);
        let foo = meta::name_key("foo");
        assert!(lo <= foo);
        assert!(foo < hi);
    }

    #[test]
    fn registry_allocate_is_monotonic() {
        let r = CfRegistry::new();
        let (a, _) = r.allocate("alpha");
        let (b, _) = r.allocate("beta");
        assert!(a.id < b.id);
        assert_eq!(r.get("alpha").unwrap(), a);
        assert_eq!(r.get("beta").unwrap(), b);
    }
}
