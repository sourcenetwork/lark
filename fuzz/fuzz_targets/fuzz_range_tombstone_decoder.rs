#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    lark_kv::fuzzing::decode_range_tombstones(data);
});
