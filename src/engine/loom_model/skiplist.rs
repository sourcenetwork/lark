//! Models of `engine::skiplist`: one writer publishing nodes
//! against concurrent readers, and the two-step seek that must not be.

use loom::sync::Arc;

use super::super::internal_key::{VALUE_TYPE_VALUE, decode_internal_key, user_key_of};
use super::{explore, memtable, probe};

/// Invariants S1, S3 and S7: an insert publishes a whole node and never
/// hides one that is already published.
///
/// The writer inserts `b`, which sorts strictly between the two keys the
/// reader was seeded with, so every level of the reader's descent can be
/// diverted by the new node. The reader asserts that `a` is always
/// found (S3: nothing is unlinked, and re-linking a predecessor cannot
/// lose it) and that `b`, if seen at all, carries its own sequence and
/// its own value (S1: the node was complete before it was reachable).
pub fn insert_publishes_a_whole_node_to_a_concurrent_reader() {
    explore(
        "insert_publishes_a_whole_node_to_a_concurrent_reader",
        16,
        4,
        |witness| {
            let mt = Arc::new(memtable());
            mt.put(probe(b"a").prefixed_user_key(), b"va", 1);
            mt.put(probe(b"c").prefixed_user_key(), b"vc", 1);

            let writer = {
                let mt = Arc::clone(&mt);
                loom::thread::spawn(move || {
                    mt.put(probe(b"b").prefixed_user_key(), b"vb", 2);
                })
            };
            let reader = {
                let mt = Arc::clone(&mt);
                let witness = witness.clone();
                loom::thread::spawn(move || {
                    let (seq, value) = mt.get(&probe(b"a")).expect("S3: `a` cannot be lost");
                    assert_eq!(seq, 1);
                    assert_eq!(value.expect("`a` is a live value").as_slice(), b"va");

                    if let Some((seq, value)) = mt.get(&probe(b"b")) {
                        witness.record();
                        assert_eq!(seq, 2, "S1: a visible node carries its own sequence");
                        assert_eq!(
                            value.expect("`b` is a live value").as_slice(),
                            b"vb",
                            "S1: a visible node carries its own value"
                        );
                    }
                })
            };

            writer.join().expect("writer");
            reader.join().expect("reader");

            for (key, want) in [(&b"a"[..], &b"va"[..]), (b"b", b"vb"), (b"c", b"vc")] {
                let (_, value) = mt
                    .get(&probe(key))
                    .expect("every key is present after the join");
                assert_eq!(value.expect("live value").as_slice(), want);
            }
        },
    );
}

/// The regression that a two-load seek reintroduces: a key already in
/// the memtable stays findable while a writer inserts immediately before
/// it.
///
/// `seek_ge` folds the level-0 step into the descent for exactly this
/// reason. Loom drives the interleaving that a stress test can only
/// stumble into.
pub fn a_seeded_key_stays_findable_while_a_writer_inserts_before_it() {
    explore(
        "a_seeded_key_stays_findable_while_a_writer_inserts_before_it",
        16,
        4,
        |witness| {
            let mt = Arc::new(memtable());
            mt.put(probe(b"a").prefixed_user_key(), b"va", 1);
            mt.put(probe(b"c").prefixed_user_key(), b"vc", 1);

            let writer = {
                let mt = Arc::clone(&mt);
                loom::thread::spawn(move || {
                    mt.put(probe(b"b").prefixed_user_key(), b"vb", 2);
                })
            };
            let reader = {
                let mt = Arc::clone(&mt);
                let witness = witness.clone();
                loom::thread::spawn(move || {
                    let (_, value) = mt
                        .get(&probe(b"c"))
                        .expect("`c` is memtable-resident and cannot go missing");
                    assert_eq!(value.expect("live value").as_slice(), b"vc");
                    // The seek raced the insert at least once: `b` was
                    // already linked when this reader looked.
                    if mt.get(&probe(b"b")).is_some() {
                        witness.record();
                    }
                })
            };

            writer.join().expect("writer");
            reader.join().expect("reader");
        },
    );
}

/// Calibration for [`a_seeded_key_stays_findable_while_a_writer_inserts_before_it`].
///
/// Descend to the predecessor, then load its level-0 link a second time:
/// the shape `seek_ge` had before the fix. If loom explores the schedule
/// where the writer links its node between the two loads, this loses a
/// key that is present, and the model must fail. A model that cannot
/// fail here is not exploring the interleaving the fix exists for, and
/// the model above would be passing vacuously.
pub fn the_two_step_seek_loses_a_key_that_is_present() {
    explore(
        "the_two_step_seek_loses_a_key_that_is_present",
        4,
        1,
        |witness| {
            let mt = Arc::new(memtable());
            mt.put(probe(b"a").prefixed_user_key(), b"va", 1);
            mt.put(probe(b"c").prefixed_user_key(), b"vc", 1);

            let writer = {
                let mt = Arc::clone(&mt);
                loom::thread::spawn(move || {
                    mt.put(probe(b"b").prefixed_user_key(), b"vb", 2);
                })
            };
            let reader = {
                let mt = Arc::clone(&mt);
                let witness = witness.clone();
                loom::thread::spawn(move || {
                    let target = probe(b"c");
                    let list = mt.list();
                    // Descend to the predecessor, then load its level-0 link
                    // a second time. `MemTable::get` treats "the first key
                    // at or after mine is not mine" as "absent", so the same
                    // comparison decides the answer here.
                    let successor = match list.seek_lt(target.internal()) {
                        Some(predecessor) => predecessor.next(),
                        None => list.first(),
                    };
                    witness.record();
                    let found = successor
                        .is_some_and(|node| user_key_of(node.key()) == target.prefixed_user_key());
                    assert!(found, "the two-step seek lost `c`");
                })
            };

            writer.join().expect("writer");
            reader.join().expect("reader");
        },
    );
}

/// Invariants S2 and S5 under the engine's own serialization: two
/// writers taking a lock in turn, as the commit pipeline makes them, put
/// two versions of one key while a reader walks it.
///
/// The reader may legally see nothing, the older version or the newer
/// one; what it must never see is a value belonging to the other
/// sequence, or a key that decodes to something else. After the join
/// both versions are present, newest sequence first.
pub fn two_serialized_writers_and_a_reader_share_one_key() {
    explore(
        "two_serialized_writers_and_a_reader_share_one_key",
        1_000,
        100,
        |witness| {
            let mt = Arc::new(memtable());
            let pipeline = Arc::new(loom::sync::Mutex::new(()));

            let writers: Vec<_> = [(1u64, &b"v1"[..]), (2, b"v2")]
                .into_iter()
                .map(|(seq, value)| {
                    let mt = Arc::clone(&mt);
                    let pipeline = Arc::clone(&pipeline);
                    loom::thread::spawn(move || {
                        let _lock = pipeline.lock().expect("pipeline");
                        mt.put(probe(b"k").prefixed_user_key(), value, seq);
                    })
                })
                .collect();

            let reader = {
                let mt = Arc::clone(&mt);
                let witness = witness.clone();
                loom::thread::spawn(move || {
                    if let Some((seq, value)) = mt.get(&probe(b"k")) {
                        let value = value.expect("both writers write live values");
                        let want: &[u8] = if seq == 1 { b"v1" } else { b"v2" };
                        assert!(seq == 1 || seq == 2, "S5: sequence {seq} was never written");
                        assert_eq!(value.as_slice(), want, "S5: value belongs to another entry");
                        // The intermediate state: only the older writer
                        // has landed, so the reader is inside the pair.
                        if seq == 1 {
                            witness.record();
                        }
                    }
                })
            };

            for writer in writers {
                writer.join().expect("writer");
            }
            reader.join().expect("reader");

            let entries = mt.iter_internal();
            assert_eq!(entries.len(), 2, "S2: one entry per serialized insert");
            let decoded: Vec<_> = entries
                .iter()
                .map(|(key, value)| {
                    let (user_key, seq, value_type) = decode_internal_key(key);
                    (user_key.to_vec(), seq, value_type, value.clone())
                })
                .collect();
            assert_eq!(decoded[0].1, 2, "the newer sequence sorts first");
            assert_eq!(decoded[0].3, b"v2");
            assert_eq!(decoded[1].1, 1);
            assert_eq!(decoded[1].3, b"v1");
            for entry in &decoded {
                assert_eq!(entry.0, probe(b"k").prefixed_user_key());
                assert_eq!(entry.2, VALUE_TYPE_VALUE);
            }
        },
    );
}

/// Calibration for invariant S2: the guard that enforces "exactly one
/// thread inside `insert`" really fires when the contract is broken.
///
/// S2 is a contract the engine has to keep, not something the skip list
/// can enforce for free, so what is checkable is the detector. Two
/// writers insert with nothing serializing them; loom finds the schedule
/// where the second enters `insert` before the first has left, and the
/// guard's `debug_assert` must trip. Nothing races past it: the guard is
/// the first thing `insert` does, so the losing thread never touches a
/// node.
///
/// Only a debug build has the guard, so only a debug build has this
/// model. `tests/loom_memtable.rs` gates the test the same way.
#[cfg(debug_assertions)]
pub fn an_unserialized_insert_trips_the_single_writer_guard() {
    explore(
        "an_unserialized_insert_trips_the_single_writer_guard",
        2,
        1,
        |witness| {
            let mt = Arc::new(memtable());
            witness.record();

            let writers: Vec<_> = [(1u64, &b"v1"[..]), (2, b"v2")]
                .into_iter()
                .map(|(seq, value)| {
                    let mt = Arc::clone(&mt);
                    loom::thread::spawn(move || {
                        mt.put(probe(b"k").prefixed_user_key(), value, seq);
                    })
                })
                .collect();

            for writer in writers {
                if let Err(payload) = writer.join() {
                    std::panic::resume_unwind(payload);
                }
            }
        },
    );
}
