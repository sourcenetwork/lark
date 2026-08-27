//! Fast on-disk checksums for accidental corruption detection.
//!
//! These helpers use xxh3 without a secret. They are intended to catch torn
//! writes, truncated records, and random bit rot; they are not cryptographic
//! hashes and do not authenticate data against an attacker who can rewrite
//! files. Add a keyed MAC or cryptographic hash at the storage boundary if
//! adversarial tamper detection becomes a requirement.

use std::io::{self, Read};

use xxhash_rust::xxh3::Xxh3Default;

// Hash domain separators: never displayed, never compared against a
// name, and present only to keep one checksum family from colliding with
// another. The bytes are part of the on-disk format, so they are frozen
// from regolith's first release onward and a change to any of them is a
// format break that needs a new version byte alongside it.
const WAL_RECORD_DOMAIN: &[u8] = b"regolith/wal-record/v2";
const WAL_STAMP_DOMAIN: &[u8] = b"regolith/wal-stamp/v1";
const MANIFEST_RECORD_DOMAIN: &[u8] = b"regolith/manifest-record/v2";
const SST_BLOCK_DOMAIN: &[u8] = b"regolith/sst-block/v2";
const SST_META_DOMAIN: &[u8] = b"regolith/sst-meta/v1";
const BACKUP_MANIFEST_DOMAIN: &[u8] = b"regolith/backup-manifest/v2";
const BACKUP_SHARED_FILE_DOMAIN: &[u8] = b"regolith/backup-shared-file/v2";
const OPFS_SLOT_HEADER_DOMAIN: &[u8] = b"regolith/opfs-slot-header/v1";

pub(crate) fn wal_record(len: u32, record_type: u8, data: &[u8]) -> u32 {
    let len = len.to_le_bytes();
    let record_type = [record_type];
    u32_parts(WAL_RECORD_DOMAIN, &[&len, &record_type, data])
}

/// Checksum over a WAL file's stamp: the magic, the format and the
/// reserved field. Covers the stamp only, so a stamp that survives says
/// nothing about the records after it.
pub(crate) fn wal_stamp(magic: &[u8; 4], format: u16, reserved: u16) -> u32 {
    let format = format.to_le_bytes();
    let reserved = reserved.to_le_bytes();
    u32_parts(WAL_STAMP_DOMAIN, &[magic, &format, &reserved])
}

pub(crate) fn manifest_record(len: u32, data: &[u8]) -> u32 {
    let len = len.to_le_bytes();
    u32_parts(MANIFEST_RECORD_DOMAIN, &[&len, data])
}

pub(crate) fn sst_block(compression_type: u8, payload: &[u8]) -> u32 {
    let compression_type = [compression_type];
    u32_parts(SST_BLOCK_DOMAIN, &[&compression_type, payload])
}

/// Kind tags for the SSTable metadata regions. They are mixed into the
/// checksum so a region cannot validate as a different kind of region
/// after a damaged offset sends the reader to the wrong bytes.
pub(crate) const META_KIND_INDEX: u8 = 1;
pub(crate) const META_KIND_INDEX_LEAF: u8 = 2;
pub(crate) const META_KIND_BLOOM: u8 = 3;
pub(crate) const META_KIND_RANGE_TOMBSTONE: u8 = 4;
pub(crate) const META_KIND_FOOTER: u8 = 5;

/// Checksum one SSTable metadata region: the index block, a partitioned
/// index leaf, the bloom region or the range-tombstone block. `kind` is
/// one of the `META_KIND_*` tags.
pub(crate) fn sst_meta(kind: u8, payload: &[u8]) -> u32 {
    u32_parts(SST_META_DOMAIN, &[&[kind], payload])
}

/// Checksum an SSTable footer: its seven fixed fields plus the magic
/// that says how to parse them.
pub(crate) fn sst_footer(fields: &[u8], magic: u64) -> u64 {
    u64_parts(
        SST_META_DOMAIN,
        &[&[META_KIND_FOOTER], fields, &magic.to_le_bytes()],
    )
}

/// Checksum of an OPFS slot header: the fixed-width fields followed by
/// the logical path they describe.
#[cfg_attr(
    not(all(target_arch = "wasm32", target_os = "unknown")),
    allow(dead_code)
)]
pub(crate) fn opfs_slot_header(fixed: &[u8], path: &[u8]) -> u32 {
    u32_parts(OPFS_SLOT_HEADER_DOMAIN, &[fixed, path])
}

pub(crate) fn backup_manifest(body: &[u8]) -> u64 {
    u64_parts(BACKUP_MANIFEST_DOMAIN, &[body])
}

pub(crate) fn backup_shared_file(reader: &mut impl Read) -> io::Result<u128> {
    let mut hasher = new_hasher(BACKUP_SHARED_FILE_DOMAIN);
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.digest128())
}

fn u32_parts(domain: &[u8], parts: &[&[u8]]) -> u32 {
    u64_parts(domain, parts) as u32
}

fn u64_parts(domain: &[u8], parts: &[&[u8]]) -> u64 {
    let mut hasher = new_hasher(domain);
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.digest()
}

fn new_hasher(domain: &[u8]) -> Xxh3Default {
    let mut hasher = Xxh3Default::new();
    hasher.update(domain);
    hasher
}
