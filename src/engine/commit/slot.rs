//! The per-writer handoff slot.
//!
//! One slot per writer thread, allocated once and reused, so the steady
//! write path never allocates a handoff. The `state` atomic is the whole
//! synchronisation: each transition hands exclusive access to `payload` to
//! the other party, which is why the payload mutex is never contended.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread::{self, Thread};

use parking_lot::Mutex;

use super::request::WriteRequest;

/// Handoff states, cycled `Idle -> Pending -> Done -> Idle`.
const SLOT_IDLE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_DONE: u8 = 2;

struct SlotPayload {
    request: WriteRequest,
    outcome: Option<io::Result<()>>,
}

/// One writer thread's handoff slot, created once and reused for every
/// write that thread ever makes.
///
/// `payload` is a mutex rather than a raw cell because the state machine
/// already guarantees only one party touches it at a time: the mutex is
/// uncontended by construction, costs two uncontended atomics, and removes
/// an entire unsafe surface from the write path.
pub(crate) struct WriteSlot {
    state: AtomicU8,
    payload: Mutex<SlotPayload>,
    /// The owning thread. A slot is thread-local, so this is captured once
    /// and is always the thread that will park on it.
    thread: Thread,
}

impl WriteSlot {
    pub(super) fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_IDLE),
            payload: Mutex::new(SlotPayload {
                request: WriteRequest::Idle,
                outcome: None,
            }),
            thread: thread::current(),
        }
    }

    /// Load `request` and mark the slot pending.
    ///
    /// Returns the request back when the slot is not idle, which means a
    /// previous ticket is still outstanding on this thread. The caller
    /// commits inline instead of racing an in-flight handoff.
    pub(super) fn arm(&self, request: WriteRequest) -> Result<(), WriteRequest> {
        if self.state.load(Ordering::Acquire) != SLOT_IDLE {
            return Err(request);
        }
        {
            let mut payload = self.payload.lock();
            payload.request = request;
            payload.outcome = None;
        }
        self.state.store(SLOT_PENDING, Ordering::Release);
        Ok(())
    }

    /// Leader side: take the armed request, leaving the slot empty.
    pub(super) fn take_request(&self) -> WriteRequest {
        std::mem::replace(&mut self.payload.lock().request, WriteRequest::Idle)
    }

    /// Leader side: record the group's outcome and release the writer.
    pub(super) fn complete(&self, outcome: io::Result<()>) {
        self.payload.lock().outcome = Some(outcome);
        self.state.store(SLOT_DONE, Ordering::Release);
        self.thread.unpark();
    }

    pub(super) fn is_done(&self) -> bool {
        self.state.load(Ordering::Acquire) == SLOT_DONE
    }

    /// Writer side: collect the outcome and return the slot to service.
    pub(super) fn finish(&self) -> io::Result<()> {
        let outcome = self.payload.lock().outcome.take();
        self.state.store(SLOT_IDLE, Ordering::Release);
        outcome.unwrap_or_else(|| Err(io::Error::other("commit slot completed with no outcome")))
    }
}

thread_local! {
    /// One slot per writer thread. A write is a blocking call, so a thread
    /// has at most one ticket in flight and one slot suffices.
    static WRITE_SLOT: Arc<WriteSlot> = Arc::new(WriteSlot::new());
}

/// This thread's handoff slot, created on its first write and reused for
/// every write it ever makes.
pub(super) fn thread_slot() -> Arc<WriteSlot> {
    WRITE_SLOT.with(Arc::clone)
}

#[cfg(test)]
mod tests {
    use super::super::super::DurabilityMode;
    use super::*;

    fn put(key: &[u8], value: &[u8]) -> WriteRequest {
        WriteRequest::Put {
            key: key.to_vec(),
            value: value.to_vec(),
            durability: DurabilityMode::Eventual,
            disable_wal: false,
        }
    }

    #[test]
    fn slot_round_trips_a_request_and_an_outcome() {
        let slot = WriteSlot::new();
        assert!(slot.arm(put(b"k", b"v")).is_ok());
        assert!(!slot.is_done());

        let taken = slot.take_request();
        assert_eq!(taken.op_count(), 1);

        slot.complete(Ok(()));
        assert!(slot.is_done());
        assert!(slot.finish().is_ok());
        // Back in service for the next write on this thread.
        assert!(slot.arm(put(b"k2", b"v2")).is_ok());
    }

    #[test]
    fn arm_refuses_a_slot_with_an_outstanding_ticket() {
        let slot = WriteSlot::new();
        assert!(slot.arm(put(b"k", b"v")).is_ok());
        match slot.arm(put(b"k2", b"v2")) {
            Err(returned) => assert_eq!(returned.op_count(), 1),
            Ok(()) => panic!("a pending slot must refuse a second request"),
        }
    }

    #[test]
    fn a_completed_slot_with_no_outcome_fails_loud() {
        let slot = WriteSlot::new();
        slot.state.store(SLOT_DONE, Ordering::Release);
        let err = slot.finish().expect_err("missing outcome must be an error");
        assert!(err.to_string().contains("no outcome"));
    }
}
