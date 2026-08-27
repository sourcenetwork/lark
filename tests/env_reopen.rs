//! Reopening a database has to read the filesystem it was written to.
//!
//! Every path `Db::open` takes must go through the configured [`Env`].
//! A single `std::fs` call that slips past it looks fine on the default
//! backend, where `Env` *is* the real filesystem, and makes reopening
//! impossible on every other one: the call looks for the database on
//! disk, does not find it, and the open fails with a bare `NotFound`
//! that names nothing.
//!
//! That is exactly what happened to WAL replay, which opened its log
//! with `std::fs::File::open` while everything around it used `Env`.
//! The default backend never noticed. `MemEnv` is the cheapest backend
//! that would, so it is what guards the rule here.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use lark_kv::{Db, MemEnv, Options, WriteBatch};

fn opts(env: &Arc<MemEnv>) -> Options {
    Options {
        env: env.clone(),
        max_background_compactions: 0,
        ..Options::default()
    }
}

/// The regression: a database on a non-default `Env` must reopen and
/// still hold everything it acknowledged.
#[test]
fn a_database_on_a_memory_env_reopens_with_every_acknowledged_write() {
    let env = Arc::new(MemEnv::new());

    {
        let db = Db::open("/reopen", opts(&env)).expect("first open");
        for i in 0..200u32 {
            db.put(format!("k{i:04}").as_bytes(), format!("v{i}").as_bytes())
                .expect("put");
        }
        let mut batch = WriteBatch::new();
        batch.put(b"batched", b"yes");
        batch.delete(b"k0000");
        db.write(batch).expect("batch");
        db.close().expect("close");
    }

    let db = Db::open("/reopen", opts(&env)).expect(
        "reopening a database on a non-default Env must work; a NotFound here means some \
         path took std::fs instead of the Env and looked on the real filesystem",
    );
    assert_eq!(db.get(b"k0001").expect("get").as_deref(), Some(&b"v1"[..]));
    assert_eq!(
        db.get(b"batched").expect("get").as_deref(),
        Some(&b"yes"[..])
    );
    assert_eq!(
        db.get(b"k0000").expect("get"),
        None,
        "the batched delete must survive"
    );
    assert_eq!(db.scan(None, None).expect("scan").len(), 200);
    db.close().expect("close");
}

/// Reopening must survive an unclean close, which is the path that
/// actually replays the WAL rather than reading a flushed table.
#[test]
fn a_memory_env_database_replays_its_wal_after_an_unclean_close() {
    let env = Arc::new(MemEnv::new());

    {
        let db = Db::open("/unclean", opts(&env)).expect("first open");
        for i in 0..50u32 {
            db.put(format!("u{i:04}").as_bytes(), b"v").expect("put");
        }
        // No `close`: the data lives only in the WAL, so the reopen has
        // to replay it rather than read a table.
        drop(db);
    }

    let db = Db::open("/unclean", opts(&env)).expect("reopen after an unclean close");
    for i in 0..50u32 {
        assert_eq!(
            db.get(format!("u{i:04}").as_bytes())
                .expect("get")
                .as_deref(),
            Some(&b"v"[..]),
            "u{i:04} did not survive WAL replay through the Env",
        );
    }
    db.close().expect("close");
}
