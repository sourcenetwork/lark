//! Resident-memory sampling.
//!
//! On wasm the number that matters is the size of the module's linear
//! memory. It grows in 64 KiB pages and never shrinks, so the current
//! size *is* the high-water mark: every page the run ever touched is
//! still committed. On Linux the comparable figure is `VmHWM`, the
//! peak resident set, which the kernel reports separately from the
//! current RSS.
//!
//! The two are not interchangeable, so every reading carries the label
//! of what produced it and a report prints that label rather than
//! implying one number can be compared with the other.

/// Current memory reading in kibibytes and what produced it, or `None`
/// on a platform this probe cannot measure.
///
/// `None` is reported as "not measured"; it is never rendered as a
/// zero.
pub fn sample() -> Option<(u64, &'static str)> {
    platform_sample()
}

#[cfg(target_arch = "wasm32")]
fn platform_sample() -> Option<(u64, &'static str)> {
    let pages = core::arch::wasm32::memory_size(0) as u64;
    Some((
        pages * 64,
        "wasm linear memory (high-water by construction)",
    ))
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
fn platform_sample() -> Option<(u64, &'static str)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some((kib, "process VmHWM"));
        }
    }
    None
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "linux")))]
fn platform_sample() -> Option<(u64, &'static str)> {
    None
}
