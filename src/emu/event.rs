//! This module handles event-driven simulation

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::BinaryHeap;
use std::sync::Mutex;

struct Entry {
    time: u64,
    handler: Box<Fn()>,
}

// #region Ordering relation for Entry
//

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Smaller time needs to come larger as BinaryHeap is a max-heap.
        other.time.cmp(&self.time)
    }
}

//
// #endregion

#[repr(C)]
pub struct EventLoop {
    cycle: AtomicU64,
    next_event: AtomicU64,
    // This has to be a Box to allow repr(C)
    events: Box<Mutex<BinaryHeap<Entry>>>,
}

extern {
    fn fiber_event_loop();
}

impl EventLoop {
    /// Create a new event loop.
    pub fn new() -> EventLoop {
        EventLoop {
            cycle: AtomicU64::new(0),
            next_event: AtomicU64::new(u64::max_value()),
            events: Box::new(Mutex::new(BinaryHeap::new())),
        }
    }

    /// Creata a fiber for the event loop
    pub fn create_fiber(self) -> crate::fiber::Fiber {
        let event_fiber = crate::fiber::Fiber::new();
        let ptr = event_fiber.data_pointer();
        unsafe { std::ptr::write(ptr, self) }
        event_fiber.set_fn(fiber_event_loop);
        event_fiber
    }

    /// Query the current cycle count.
    #[inline(always)]
    pub fn cycle(&self) -> u64 {
        self.cycle.load(Ordering::Relaxed)
    }

    /// Add a new event to the event loop for triggering. If it happens in the past it will be
    /// dequeued and triggered as soon as `cycle` increments for the next time.
    pub fn queue(&self, time: u64, handler: Box<Fn()->()>) {
        let mut guard = self.events.lock().unwrap();
        guard.push(Entry {
            time,
            handler,
        });
        // It's okay to be relaxed because guard's release op will order it.
        self.next_event.store(match guard.peek() {
            Some(it) => it.time,
            None => u64::max_value(),
        }, Ordering::Relaxed);
    }
}

#[no_mangle]
extern "C" fn event_loop_handle(this: &EventLoop) {
    let cycle = this.cycle();
    let mut guard = this.events.lock().unwrap();
    loop {
        let entry = guard.pop().unwrap();
        (entry.handler)();
        let next_event = match guard.peek() {
            Some(it) => it.time,
            None => u64::max_value(),
        };
        if cycle < next_event {
            // It's okay to be relaxed because guard's release op will order it.
            this.next_event.store(next_event, Ordering::Relaxed);
            break
        }
    }
}
