//! Fuzz target for `WriteBatch`: construct a batch from arbitrary
//! bytes, commit it, and read every written key back — asserting
//! no panics at any stage.
//!
//! Run with:
//!
//! ```sh
//! cd fuzz
//! cargo +nightly fuzz run fuzz_writebatch -- -max_total_time=60
//! ```

#![no_main]

use regolith::{Db, Options, WriteBatch};
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

    let mut batch = WriteBatch::new();
    let mut pos = 0;

    while pos < data.len() {
        let op = data[pos];
        pos += 1;

        match op % 3 {
            // put(key_len, key, val_len, val)
            0 => {
                let Some(&kl) = data.get(pos) else { break };
                pos += 1;
                let kl = (kl as usize % 64)
                    .max(1)
                    .min(data.len().saturating_sub(pos));
                if kl == 0 {
                    break;
                }
                let key = &data[pos..pos + kl];
                pos += kl;
                let vl = data.get(pos).copied().unwrap_or(0) as usize % 128;
                pos = (pos + 1).min(data.len());
                let vl = vl.min(data.len().saturating_sub(pos));
                let val = &data[pos..pos + vl];
                pos += vl;
                batch.put(key, val);
            }
            // delete(key_len, key)
            1 => {
                let Some(&kl) = data.get(pos) else { break };
                pos += 1;
                let kl = (kl as usize % 64)
                    .max(1)
                    .min(data.len().saturating_sub(pos));
                if kl == 0 {
                    break;
                }
                let key = &data[pos..pos + kl];
                pos += kl;
                batch.delete(key);
            }
            // delete_range(start_len, start, end_len, end)
            _ => {
                let sl = data.get(pos).copied().unwrap_or(0) as usize % 32;
                pos = (pos + 1).min(data.len());
                let sl = sl.min(data.len().saturating_sub(pos));
                let start = &data[pos..pos + sl];
                pos += sl;
                let el = data.get(pos).copied().unwrap_or(0) as usize % 32;
                pos = (pos + 1).min(data.len());
                let el = el.min(data.len().saturating_sub(pos));
                let end = &data[pos..pos + el];
                pos += el;
                batch.delete_range(start, end);
            }
        }
    }

    let _ = db.write(batch);
    let _ = db.scan(None, None);
});
