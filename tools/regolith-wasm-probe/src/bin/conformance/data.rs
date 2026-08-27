//! The dataset, the expected-state model, and the digest.
//!
//! Every byte this harness writes is a pure function of a record
//! index, so a verifier needs no state carried over from the process
//! that wrote it. That is what lets step 3 (reopen in a fresh process)
//! and step 4 (reopen after a kill) check byte identity without
//! trusting anything the previous process left behind, and what lets
//! the native and wasm transcripts be compared line for line.

/// Bulk records written by the lifecycle phase.
pub const RECORDS: u64 = 5_000;

/// Records written after the snapshot is taken.
pub const LATE_RECORDS: u64 = 500;

/// Records the crash phase writes with `WriteOptions::sync`, which
/// fsyncs the WAL before returning.
pub const CRASH_SYNC: u64 = 200;

/// Records the crash phase writes with the database default
/// (`DurabilityMode::Eventual`).
pub const CRASH_ASYNC: u64 = 800;

/// Bulk records whose value the lifecycle phase rewrites after taking
/// its snapshot: `[OVERWRITE_LO, OVERWRITE_HI)`.
pub const OVERWRITE_LO: u64 = 1_000;
/// Exclusive upper bound of the overwrite band.
pub const OVERWRITE_HI: u64 = 1_100;

/// Bulk records the lifecycle phase deletes after taking its snapshot:
/// `[LATE_DELETE_LO, LATE_DELETE_HI)`.
pub const LATE_DELETE_LO: u64 = 2_000;
/// Exclusive upper bound of the late-delete band.
pub const LATE_DELETE_HI: u64 = 2_100;

/// Value generation written by the bulk load.
pub const GEN_BULK: u8 = 0;
/// Value generation written by the post-snapshot overwrite.
pub const GEN_OVERWRITE: u8 = 1;
/// Value generation used for the `late/` keys.
pub const GEN_LATE: u8 = 2;
/// Value generation used for the crash phase.
pub const GEN_CRASH: u8 = 3;

/// Bulk key for record `i`. Fixed width, so lexicographic order and
/// numeric order agree and an ordering check means something.
pub fn bulk_key(i: u64) -> Vec<u8> {
    format!("key/{i:08}").into_bytes()
}

/// Key for the `j`th record written after the snapshot was taken.
pub fn late_key(j: u64) -> Vec<u8> {
    format!("late/{j:08}").into_bytes()
}

/// Key for the `i`th fsynced record written by the crash phase.
pub fn crash_sync_key(i: u64) -> Vec<u8> {
    format!("crash/sync/{i:06}").into_bytes()
}

/// Key for the `i`th non-fsynced record written by the crash phase.
pub fn crash_async_key(i: u64) -> Vec<u8> {
    format!("crash/async/{i:06}").into_bytes()
}

/// Length of the value for `(index, generation)`.
///
/// Varied on purpose: most values straddle the block size, one in 250
/// is far larger than the 8 KiB write buffer in front of the WAL so
/// the large-write bypass path is exercised too.
pub fn value_len(i: u64, gen: u8) -> usize {
    if i % 250 == 0 {
        return 20_000 + (i % 17) as usize;
    }
    (16 + (i.wrapping_mul(7919).wrapping_add(gen as u64 * 131)) % 1_985) as usize
}

/// The value for `(index, generation)`, byte for byte.
///
/// Each record gets its own stream. A generator that walked one
/// shared sequence would make record `i + 1` a one-word shift of
/// record `i`, and the block compressor would then fold the whole
/// dataset down to a fraction of its size: the SSTables would stop
/// reflecting how much data is really being stored, and the paths
/// that matter for a large write would go untested. Seeding per
/// record and indexing by a counter keeps the corpus incompressible.
pub fn value(i: u64, gen: u8) -> Vec<u8> {
    let len = value_len(i, gen);
    let seed =
        mix(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (gen as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut out = Vec::with_capacity(len + 8);
    let mut counter = 0u64;
    while out.len() < len {
        let word = mix(seed.wrapping_add(counter.wrapping_mul(0x8EBC_6AF0_9C88_C6E3)));
        out.extend_from_slice(&word.to_le_bytes());
        counter += 1;
    }
    out.truncate(len);
    out
}

/// The splitmix64 finalizer: a bijection on `u64` with good avalanche.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What a key is expected to hold at a given point in the run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Expect {
    /// The key must not be readable.
    Absent,
    /// The key must read back as `value(i, gen)`.
    Present(u64, u8),
}

/// State of bulk record `i` immediately after the bulk load.
pub fn expect_after_load(i: u64) -> Expect {
    Expect::Present(i, GEN_BULK)
}

/// State of bulk record `i` after every third key has been deleted.
pub fn expect_after_delete(i: u64) -> Expect {
    if i % 3 == 0 {
        Expect::Absent
    } else {
        Expect::Present(i, GEN_BULK)
    }
}

/// State of bulk record `i` at the end of the lifecycle phase, after
/// the post-snapshot overwrites and deletes have landed. Every later
/// phase expects exactly this, which is what makes a fresh process
/// able to verify without inheriting anything.
pub fn expect_final(i: u64) -> Expect {
    if i % 3 == 0 {
        return Expect::Absent;
    }
    if (OVERWRITE_LO..OVERWRITE_HI).contains(&i) {
        return Expect::Present(i, GEN_OVERWRITE);
    }
    if (LATE_DELETE_LO..LATE_DELETE_HI).contains(&i) {
        return Expect::Absent;
    }
    Expect::Present(i, GEN_BULK)
}

/// FNV-1a over a canonical length-prefixed encoding of an ordered
/// entry sequence. Order-sensitive by construction, so it detects a
/// wrong scan order as well as wrong bytes, and it collapses a whole
/// scan into one line the native and wasm transcripts can be diffed
/// on.
pub struct Digest(u64);

impl Digest {
    /// A digest over an empty sequence.
    pub fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }

    /// Fold one key-value pair into the digest.
    pub fn entry(&mut self, key: &[u8], value: &[u8]) {
        self.bytes(&(key.len() as u32).to_le_bytes());
        self.bytes(key);
        self.bytes(&(value.len() as u32).to_le_bytes());
        self.bytes(value);
    }

    fn bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// The digest value.
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// The digest of the ascending `key/` sequence implied by `expect`.
pub fn expected_bulk_digest(expect: fn(u64) -> Expect) -> (u64, u64) {
    let mut digest = Digest::new();
    let mut count = 0;
    for i in 0..RECORDS {
        if let Expect::Present(idx, gen) = expect(i) {
            digest.entry(&bulk_key(i), &value(idx, gen));
            count += 1;
        }
    }
    (digest.finish(), count)
}

/// The digest of the descending `key/` sequence implied by `expect`.
pub fn expected_bulk_digest_reverse(expect: fn(u64) -> Expect) -> (u64, u64) {
    let mut digest = Digest::new();
    let mut count = 0;
    for i in (0..RECORDS).rev() {
        if let Expect::Present(idx, gen) = expect(i) {
            digest.entry(&bulk_key(i), &value(idx, gen));
            count += 1;
        }
    }
    (digest.finish(), count)
}

/// The digest of the ascending `late/` sequence.
pub fn expected_late_digest() -> (u64, u64) {
    let mut digest = Digest::new();
    for j in 0..LATE_RECORDS {
        digest.entry(&late_key(j), &value(j, GEN_LATE));
    }
    (digest.finish(), LATE_RECORDS)
}
