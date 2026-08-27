//! Fuzz target that interprets arbitrary bytes as a sequence of
//! database operations (put, get, delete, scan, snapshot) and
//! asserts that no operation panics.
//!
//! Run with:
//!
//! ```sh
//! cd fuzz
//! cargo +nightly fuzz run fuzz_api_ops -- -max_total_time=60
//! ```

#![no_main]

use regolith::{Db, Options};
use libfuzzer_sys::fuzz_target;
use tempfile::TempDir;

fuzz_target!(|data: &[u8]| {
    let dir = TempDir::new().unwrap();
    let opts = Options {
        write_buffer_size: 4 * 1024,
        ..Options::default()
    };
    let Ok(db) = Db::open(dir.path(), opts) else {
        return;
    };

    let mut pos = 0;
    while pos < data.len() {
        let op = data[pos];
        pos += 1;

        match op % 5 {
            // put(key_len, key, val_len, val)
            0 => {
                let Some(&key_len_raw) = data.get(pos) else {
                    break;
                };
                pos += 1;
                let key_len = (key_len_raw as usize % 64)
                    .max(1)
                    .min(data.len().saturating_sub(pos));
                if key_len == 0 {
                    break;
                }
                let key = &data[pos..pos + key_len];
                pos += key_len;
                let val_len_raw = data.get(pos).copied().unwrap_or(0);
                pos = (pos + 1).min(data.len());
                let val_len = (val_len_raw as usize % 128).min(data.len().saturating_sub(pos));
                let val = &data[pos..pos + val_len];
                pos += val_len;
                let _ = db.put(key, val);
            }
            // get(key_len, key)
            1 => {
                let Some(key_len) = data.get(pos).copied() else {
                    break;
                };
                pos += 1;
                let key_len = (key_len as usize % 64).max(1);
                if pos + key_len > data.len() {
                    break;
                }
                let key = &data[pos..pos + key_len];
                pos += key_len;
                let _ = db.get(key);
            }
            // delete(key_len, key)
            2 => {
                let Some(key_len) = data.get(pos).copied() else {
                    break;
                };
                pos += 1;
                let key_len = (key_len as usize % 64).max(1);
                if pos + key_len > data.len() {
                    break;
                }
                let key = &data[pos..pos + key_len];
                pos += key_len;
                let _ = db.delete(key);
            }
            // scan(start_len, start, end_len, end)
            3 => {
                let Some(&start_len_raw) = data.get(pos) else {
                    break;
                };
                pos += 1;
                let start_len = (start_len_raw as usize % 32).min(data.len().saturating_sub(pos));
                let start = &data[pos..pos + start_len];
                pos += start_len;
                let end_len_raw = data.get(pos).copied().unwrap_or(0);
                pos = (pos + 1).min(data.len());
                let end_len = (end_len_raw as usize % 32).min(data.len().saturating_sub(pos));
                let end = &data[pos..pos + end_len];
                pos += end_len;
                let _ = db.scan(Some(start), Some(end));
            }
            // snapshot + get through it
            _ => {
                let snap = db.snapshot();
                let Some(key_len) = data.get(pos).copied() else {
                    break;
                };
                pos += 1;
                let key_len = (key_len as usize % 64).max(1);
                if pos + key_len > data.len() {
                    break;
                }
                let key = &data[pos..pos + key_len];
                pos += key_len;
                let _ = snap.get(key);
            }
        }
    }
});
