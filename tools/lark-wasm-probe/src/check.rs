//! The read helpers every phase shares.
//!
//! Each one names the offending key in its error, so a failure under
//! wasmtime says which record went wrong rather than only that
//! something did.

use lark_kv::Db;

use crate::dataset;

/// A key rendered for a human, lossily. Every key this probe writes is
/// ASCII, so the lossy conversion never actually loses anything.
pub fn show(key: &[u8]) -> String {
    String::from_utf8_lossy(key).into_owned()
}

/// Read `key`, turning an engine error into a message naming it.
pub fn get(db: &Db, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
    db.get(key)
        .map_err(|e| format!("get {} failed: {e}", show(key)))
}

/// Require that `key` is absent. `why` explains what removed it.
pub fn expect_absent(db: &Db, key: &[u8], why: &str) -> Result<(), String> {
    match get(db, key)? {
        None => Ok(()),
        Some(v) => Err(format!(
            "{}: expected absent ({why}) but read {} bytes",
            show(key),
            v.len()
        )),
    }
}

/// Require that `key` holds exactly `want`.
pub fn expect_value(db: &Db, key: &[u8], want: &[u8]) -> Result<(), String> {
    let label = show(key);
    match get(db, key)? {
        Some(got) => dataset::check(&label, want, &got),
        None => Err(format!("{label}: expected {} bytes, got none", want.len())),
    }
}
