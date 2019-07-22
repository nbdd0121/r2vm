//! This module handles event-driven simulation

use std::collections::BinaryHeap;
use std::sync::Mutex;

#[cfg(not(feature = "thread"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "thread")]
use std::sync::Condvar;
#[cfg(feature = "thread")]
use std::time::{Duration, Instant};

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
    #[cfg(not(feature = "thread"))]
    cycle: AtomicU64,
    #[cfg(not(feature = "thread"))]
    next_event: AtomicU64,
    #[cfg(feature = "thread")]
    epoch: Instant,
    #[cfg(feature = "thread")]
    condvar: Condvar,
    // This has to be a Box to allow repr(C)
    events: Box<Mutex<BinaryHeap<Entry>>>,
}

extern {
    // See also `event.s` for this function
    fn fiber_event_loop();
}

#[inline(always)]
fn scaling_const() -> u64 {
    if cfg!(feature = "thread") {
        1
    } else if cfg!(feature = "fast") {
        20
    } else {
        100
    }
}

impl EventLoop {
    /// Create a new event loop.
    #[cfg(not(feature = "thread"))]
    pub fn new() -> EventLoop {
        EventLoop {
            cycle: AtomicU64::new(0),
            next_event: AtomicU64::new(u64::max_value()),
            events: Box::new(Mutex::new(BinaryHeap::new())),
        }
    }

    /// Create a new event loop.
    #[cfg(feature = "thread")]
    pub fn new() -> EventLoop {
        EventLoop {
            epoch: Instant::now(),
            condvar: Condvar::new(),
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
    #[cfg(not(feature = "thread"))]
    #[inline(always)]
    pub fn cycle(&self) -> u64 {
        self.cycle.load(Ordering::Relaxed)
    }

    /// Query the current cycle count.
    #[cfg(feature = "thread")]
    pub fn cycle(&self) -> u64 {
        let duration = Instant::now().duration_since(self.epoch);
        (duration.as_micros() as u64) * scaling_const()
    }

    /// Add a new event to the event loop for triggering. If it happens in the past it will be
    /// dequeued and triggered as soon as `cycle` increments for the next time.
    #[cfg(not(feature = "thread"))]
    pub fn queue(&self, cycle: u64, handler: Box<Fn()->()>) {
        let mut guard = self.events.lock().unwrap();
        guard.push(Entry {
            time: cycle,
            handler,
        });
        // It's okay to be relaxed because guard's release op will order it.
        self.next_event.store(match guard.peek() {
            Some(it) => it.time,
            None => u64::max_value(),
        }, Ordering::Relaxed);
    }

    /// Add a new event to the event loop for triggering. If it happens in the past it will be
    /// dequeued and triggered as soon as `cycle` increments for the next time.
    #[cfg(feature = "thread")]
    pub fn queue(&self, cycle: u64, handler: Box<Fn()->()>) {
        let mut guard = self.events.lock().unwrap();
        guard.push(Entry {
            time: cycle,
            handler,
        });

        // If the event just queued is the next event, we need to wake the event loop up.
        if guard.peek().unwrap().time == cycle {
            self.condvar.notify_one()
        }
    }

    /// Query the current time (we pretend to be operating at 100MHz at the moment)
    #[inline(always)]
    pub fn time(&self) -> u64 {
        self.cycle() / scaling_const()
    }

    pub fn queue_time(&self, time: u64, handler: Box<Fn()->()>) {
        self.queue(time * scaling_const(), handler);
    }

    /// Handle all events at or before `cycle`, and return the cycle of next event if any.
    fn handle_events(&self, guard: &mut std::sync::MutexGuard<BinaryHeap<Entry>>, cycle: u64) -> Option<u64> {
        loop {
            let time = match guard.peek() {
                None => return None,
                Some(v) => v.time,
            };
            if time > cycle {
                return Some(time)
            }
            let entry = guard.pop().unwrap();
            (entry.handler)();
        }
    }

}

#[cfg(not(feature = "thread"))]
#[no_mangle]
extern "C" fn event_loop_handle(this: &EventLoop) {
    let cycle = this.cycle();
    let next_event = this.handle_events(&mut this.events.lock().unwrap(), cycle).unwrap_or(u64::max_value());
    this.next_event.store(next_event, Ordering::Relaxed);
}

#[cfg(feature = "thread")]
#[no_mangle]
extern "C" fn event_loop_handle(this: &EventLoop) {
    let mut guard = this.events.lock().unwrap();
    loop {
        let cycle = this.cycle();
        let result = this.handle_events(&mut guard, cycle);
        guard = match result {
            None => this.condvar.wait(guard).unwrap(),
            Some(v) => this.condvar.wait_timeout(guard, Duration::from_micros(v - cycle)).unwrap().0,
        }
    }
}
