//! Token-bucket rate limiter for background I/O.
//!
//! A [`RateLimiter`] throttles byte-denominated work (flush and
//! compaction writes) so bursts of background I/O don't saturate the
//! disk and push foreground latency off a cliff. The engine calls
//! [`RateLimiter::request`] with a byte count and a [`Priority`]; the
//! call blocks until enough tokens have accumulated, then returns.
//!
//! [`TokenBucketRateLimiter`] is the stock implementation: a single
//! token bucket with a configurable refill rate and burst capacity,
//! served by a FIFO queue that always drains [`Priority::High`] waiters
//! before [`Priority::Low`]. It is the only implementation lark ships;
//! the trait is public so callers can drop in their own (e.g. for test
//! harnesses or a shared limiter across multiple databases).

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

/// Priority of a rate-limited I/O request. High-priority waiters are
/// always served before low-priority waiters; within a priority class
/// waiters are served in FIFO order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Foreground work. Served first.
    High,
    /// Background work (flush, compaction). Yields to [`Priority::High`].
    Low,
}

impl Priority {
    fn class(self) -> u8 {
        match self {
            Priority::High => 0,
            Priority::Low => 1,
        }
    }
}

/// A byte-denominated rate limiter.
///
/// Implementations throttle callers of [`RateLimiter::request`] to at
/// most `get_bytes_per_second()` bytes over time. Requests block until
/// quota is available; shutdown (on drop or via an implementation-
/// specific `stop()` call) must wake every blocked waiter.
pub trait RateLimiter: Send + Sync + 'static {
    /// Request `bytes` worth of I/O quota. Blocks until the request is
    /// served or the limiter is shut down.
    fn request(&self, bytes: u64, pri: Priority);

    /// Update the refill rate. Takes effect on the next refill tick.
    fn set_bytes_per_second(&self, bytes_per_second: u64);

    /// Return the currently configured refill rate in bytes/sec.
    fn get_bytes_per_second(&self) -> u64;

    /// Total bytes successfully served at `pri` since construction.
    /// Excludes in-flight requests.
    fn get_total_bytes_through(&self, pri: Priority) -> u64;
}

/// Internal waiter identity: priority class first so the high class
/// sorts before the low class, seq second for FIFO within a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WaiterKey {
    class: u8,
    seq: u64,
}

struct State {
    bytes_per_second: u64,
    available: i128,
    last_refill: Instant,
    next_seq: u64,
    waiters: BTreeSet<WaiterKey>,
    shutdown: bool,
}

/// Default rate-limiter implementation: a single token bucket refilled
/// at `bytes_per_second` bytes/sec with a burst capacity of `burst_bytes`.
///
/// All waiters share one [`Mutex`] and one [`Condvar`]; on wakeup, each
/// waiter checks whether it is at the front of the FIFO queue (highest
/// priority, lowest seq) and, if so, whether enough tokens have
/// accumulated. Waiters that aren't at the front simply go back to
/// sleep.
///
/// `refill_period` controls the granularity at which tokens are
/// credited to the bucket: a smaller period smooths bursts at the cost
/// of more wakeups; a larger period is cheaper but chunkier. 100 ms is
/// a reasonable starting point for most workloads.
pub struct TokenBucketRateLimiter {
    state: Mutex<State>,
    cv: Condvar,
    burst_bytes: u64,
    refill_period: Duration,
    total_high: AtomicU64,
    total_low: AtomicU64,
}

impl TokenBucketRateLimiter {
    /// Construct a new limiter.
    ///
    /// * `bytes_per_second` — sustained refill rate. `0` disables the
    ///   limiter (every request is served instantly).
    /// * `refill_period` — how often tokens are credited. Must be > 0.
    /// * `burst_bytes` — maximum number of tokens the bucket can hold.
    ///   A fresh bucket starts full so the first `burst_bytes` worth of
    ///   work is served without blocking.
    pub fn new(bytes_per_second: u64, refill_period: Duration, burst_bytes: u64) -> Self {
        assert!(
            !refill_period.is_zero(),
            "refill_period must be greater than zero"
        );
        let burst_bytes = burst_bytes.max(1);
        Self {
            state: Mutex::new(State {
                bytes_per_second,
                available: burst_bytes as i128,
                last_refill: Instant::now(),
                next_seq: 0,
                waiters: BTreeSet::new(),
                shutdown: false,
            }),
            cv: Condvar::new(),
            burst_bytes,
            refill_period,
            total_high: AtomicU64::new(0),
            total_low: AtomicU64::new(0),
        }
    }

    /// Wake every blocked waiter and return. Subsequent calls to
    /// [`RateLimiter::request`] also return immediately without
    /// consuming tokens. Called automatically when the limiter is
    /// dropped.
    pub fn stop(&self) {
        let mut state = self.state.lock();
        state.shutdown = true;
        self.cv.notify_all();
    }

    /// Refill the bucket based on elapsed time since `last_refill`.
    /// Caller holds the lock.
    fn refill_locked(&self, state: &mut State, now: Instant) {
        let elapsed = now.saturating_duration_since(state.last_refill);
        if elapsed < self.refill_period {
            return;
        }
        let period_nanos: u128 = self.refill_period.as_nanos().max(1);
        let periods = (elapsed.as_nanos() / period_nanos) as u64;
        if periods == 0 {
            return;
        }
        // tokens = rate * (periods * refill_period)
        let rate = state.bytes_per_second as u128;
        let tokens = rate
            .saturating_mul(period_nanos)
            .saturating_mul(periods as u128)
            / 1_000_000_000u128;
        state.available = (state.available + tokens as i128).min(self.burst_bytes as i128);
        state.last_refill += self.refill_period * periods as u32;
    }

    /// Internal: serve one chunk (at most `burst_bytes`).
    fn request_chunk(&self, bytes: u64, pri: Priority) -> bool {
        if bytes == 0 {
            return true;
        }
        let class = pri.class();
        let mut state = self.state.lock();
        if state.shutdown {
            return false;
        }
        let my_seq = state.next_seq;
        state.next_seq += 1;
        let key = WaiterKey { class, seq: my_seq };
        state.waiters.insert(key);

        let served = loop {
            if state.shutdown {
                break false;
            }

            let now = Instant::now();
            self.refill_locked(&mut state, now);

            let front = state
                .waiters
                .iter()
                .next()
                .copied()
                .expect("self is in waiters");
            if front == key && state.available >= bytes as i128 {
                state.available -= bytes as i128;
                break true;
            }

            // Either we aren't at the front or tokens aren't ready.
            // Compute how long until the next refill and sleep that
            // long — a spurious wakeup just re-enters the loop.
            let wait = self
                .refill_period
                .saturating_sub(Instant::now().saturating_duration_since(state.last_refill));
            let wait = if wait.is_zero() {
                self.refill_period
            } else {
                wait
            };
            self.cv.wait_for(&mut state, wait);
        };

        state.waiters.remove(&key);
        // Front of queue may have changed; wake the new front.
        self.cv.notify_all();

        if served {
            match pri {
                Priority::High => {
                    self.total_high.fetch_add(bytes, Ordering::Relaxed);
                }
                Priority::Low => {
                    self.total_low.fetch_add(bytes, Ordering::Relaxed);
                }
            }
        }
        served
    }
}

impl Drop for TokenBucketRateLimiter {
    fn drop(&mut self) {
        // Dropping doesn't actually wake external waiters (they hold
        // &self), but `stop()` is idempotent and flags state.shutdown
        // for any future request calls that race the drop.
        self.stop();
    }
}

impl RateLimiter for TokenBucketRateLimiter {
    fn request(&self, bytes: u64, pri: Priority) {
        if self.get_bytes_per_second() == 0 {
            match pri {
                Priority::High => {
                    self.total_high.fetch_add(bytes, Ordering::Relaxed);
                }
                Priority::Low => {
                    self.total_low.fetch_add(bytes, Ordering::Relaxed);
                }
            }
            return;
        }
        let mut remaining = bytes;
        while remaining > 0 {
            let chunk = remaining.min(self.burst_bytes);
            if !self.request_chunk(chunk, pri) {
                // Shutdown: stop trying.
                return;
            }
            remaining -= chunk;
        }
    }

    fn set_bytes_per_second(&self, bytes_per_second: u64) {
        let mut state = self.state.lock();
        // Catch up on any pending refill at the old rate before
        // switching, so the change takes effect cleanly from "now".
        let now = Instant::now();
        self.refill_locked(&mut state, now);
        state.bytes_per_second = bytes_per_second;
        self.cv.notify_all();
    }

    fn get_bytes_per_second(&self) -> u64 {
        self.state.lock().bytes_per_second
    }

    fn get_total_bytes_through(&self, pri: Priority) -> u64 {
        match pri {
            Priority::High => self.total_high.load(Ordering::Relaxed),
            Priority::Low => self.total_low.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn single_request_within_burst_is_instant() {
        let lim = TokenBucketRateLimiter::new(1_000_000, Duration::from_millis(100), 1_000_000);
        let start = Instant::now();
        lim.request(500_000, Priority::Low);
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(lim.get_total_bytes_through(Priority::Low), 500_000);
    }

    #[test]
    fn ten_mb_through_one_mbps_takes_at_least_nine_seconds() {
        // The bucket starts full (1 MB burst) so a 10 MB request
        // sees 1 MB of free credit up front — expected wait is
        // ~9 seconds, not 10. Assert >= 9 to match.
        let lim = TokenBucketRateLimiter::new(1_000_000, Duration::from_millis(100), 1_000_000);
        let start = Instant::now();
        lim.request(10_000_000, Priority::Low);
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_secs(9),
            "10 MB through 1 MB/s took {:?}, expected >= 9s",
            elapsed
        );
        assert_eq!(lim.get_total_bytes_through(Priority::Low), 10_000_000);
    }

    #[test]
    fn high_priority_preempts_low() {
        // Tune the bucket so every waiter needs several refill
        // periods' worth of tokens: that gives us a wide window to
        // inject a high-priority request into the queue while low-
        // priority waiters are still blocked, and lets the priority
        // order determine who drains each refill.
        //
        // rate:          100_000 bytes/sec
        // refill_period: 100 ms  → +10_000 bytes per period
        // burst:         10_000 bytes
        //
        // Each waiter asks for 30_000 bytes, i.e. three refills'
        // worth of credit.
        let lim = Arc::new(TokenBucketRateLimiter::new(
            100_000,
            Duration::from_millis(100),
            10_000,
        ));

        // Drain the initial burst so every following waiter has to
        // queue for refills.
        lim.request(10_000, Priority::Low);

        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        let low_handles: Vec<_> = (0..2)
            .map(|i| {
                let lim = lim.clone();
                let order = order.clone();
                thread::spawn(move || {
                    lim.request(30_000, Priority::Low);
                    let label = if i == 0 { "lo1" } else { "lo2" };
                    order.lock().push(label);
                })
            })
            .collect();

        // Brief sleep to ensure both low waiters have registered
        // in the BTreeSet before we enqueue the high-priority one.
        // Well below one refill_period so no low waiter can have
        // been served yet.
        thread::sleep(Duration::from_millis(30));

        let high = {
            let lim = lim.clone();
            let order = order.clone();
            thread::spawn(move || {
                lim.request(30_000, Priority::High);
                order.lock().push("high");
            })
        };

        high.join().unwrap();
        for h in low_handles {
            h.join().unwrap();
        }

        let order = order.lock();
        let high_idx = order.iter().position(|&s| s == "high").unwrap();
        // High must land strictly before at least one low waiter
        // despite arriving last in wall-clock order.
        assert!(
            high_idx < 2,
            "high did not preempt any low: order = {:?}",
            *order
        );
    }

    #[test]
    fn shutdown_wakes_blocked_waiters() {
        let lim = Arc::new(TokenBucketRateLimiter::new(
            1_000,
            Duration::from_secs(60),
            1_000,
        ));
        // Drain the burst so the next request has to wait ~60s.
        lim.request(1_000, Priority::Low);

        let blocked = {
            let lim = lim.clone();
            thread::spawn(move || {
                let start = Instant::now();
                lim.request(1_000, Priority::Low);
                start.elapsed()
            })
        };

        thread::sleep(Duration::from_millis(100));
        lim.stop();
        let waited = blocked.join().unwrap();
        assert!(
            waited < Duration::from_secs(5),
            "blocked waiter did not wake promptly after stop: {:?}",
            waited
        );
    }

    #[test]
    fn set_bytes_per_second_live_update_is_respected() {
        let lim = Arc::new(TokenBucketRateLimiter::new(
            100_000,
            Duration::from_millis(50),
            100_000,
        ));
        lim.request(100_000, Priority::Low); // drain burst
        assert_eq!(lim.get_bytes_per_second(), 100_000);

        // Bump the rate and confirm a subsequent large request
        // completes faster than it would have at the old rate.
        lim.set_bytes_per_second(10_000_000);
        assert_eq!(lim.get_bytes_per_second(), 10_000_000);
        let start = Instant::now();
        lim.request(1_000_000, Priority::Low);
        // At the old rate 1 MB would need ~10 seconds; at 10 MB/s
        // it should need ~100 ms. Give it generous slack for CI.
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "request took {:?} after rate bump",
            start.elapsed()
        );
    }

    #[test]
    fn zero_rate_disables_limiter() {
        let lim = TokenBucketRateLimiter::new(0, Duration::from_millis(100), 1);
        let start = Instant::now();
        lim.request(100_000_000, Priority::Low);
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(lim.get_total_bytes_through(Priority::Low), 100_000_000);
    }

    #[test]
    fn get_total_bytes_through_tracks_both_classes() {
        let lim = TokenBucketRateLimiter::new(10_000_000, Duration::from_millis(50), 10_000_000);
        lim.request(1_000, Priority::High);
        lim.request(2_000, Priority::Low);
        lim.request(3_000, Priority::High);
        assert_eq!(lim.get_total_bytes_through(Priority::High), 4_000);
        assert_eq!(lim.get_total_bytes_through(Priority::Low), 2_000);
    }
}
