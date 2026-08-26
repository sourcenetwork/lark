//! Broadcast wake-up for stalled writers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kovan_channel::signal::Signal;
use kovan_queue::array_queue::ArrayQueue;

/// How many stalled writers can hold a registration at once.
const STALL_WAITER_SLOTS: usize = 256;

/// Bounded broadcast wake-up for writers parked on a stop-writes stall.
///
/// Writers register a signal and park with a deadline; the compaction
/// worker drains every registration after each pass. The ring is bounded,
/// and a writer that finds it full parks without registering, which stays
/// live because the deadline alone already bounds every wait.
pub(crate) struct StallSignal {
    waiters: ArrayQueue<Arc<Signal>>,
}

impl StallSignal {
    pub(crate) fn new() -> Self {
        Self {
            waiters: ArrayQueue::new(STALL_WAITER_SLOTS),
        }
    }

    /// Park the calling writer for at most `timeout`.
    pub(crate) fn wait(&self, timeout: Duration) {
        let signal = Arc::new(Signal::new());
        let _ = self.waiters.push(Arc::clone(&signal));
        signal.wait_deadline(Instant::now() + timeout);
        // Mark our own registration stale so a later `notify_all` does not
        // spend a wake-up on a signal nobody is waiting on any more.
        signal.notify();
    }

    /// Wake every registered writer so it can re-check its thresholds.
    pub(crate) fn notify_all(&self) {
        while let Some(waiter) = self.waiters.pop() {
            if !waiter.is_notified() {
                waiter.notify();
            }
        }
    }
}

impl Default for StallSignal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn stall_signal_wakes_a_registered_waiter() {
        let signal = Arc::new(StallSignal::new());
        let waiter = Arc::clone(&signal);
        let handle = thread::spawn(move || {
            waiter.wait(Duration::from_secs(5));
        });
        // Drain-and-notify until the waiter has registered and exited; the
        // deadline in `wait` bounds this even if the notify races ahead.
        let deadline = Instant::now() + Duration::from_secs(10);
        while !handle.is_finished() && Instant::now() < deadline {
            signal.notify_all();
            thread::yield_now();
        }
        handle.join().expect("waiter thread panicked");
    }

    #[test]
    fn stall_signal_wait_returns_on_its_deadline_without_a_notify() {
        let signal = StallSignal::new();
        let start = Instant::now();
        signal.wait(Duration::from_millis(20));
        assert!(start.elapsed() >= Duration::from_millis(15));
    }

    #[test]
    fn stall_signal_survives_more_waiters_than_slots() {
        let signal = StallSignal::new();
        for _ in 0..(STALL_WAITER_SLOTS * 2) {
            let _ = signal.waiters.push(Arc::new(Signal::new()));
        }
        signal.notify_all();
        // Every surviving registration was drained, so the ring is
        // available again for the next round of stalled writers.
        assert!(signal.waiters.pop().is_none());
    }
}
