//! This module handles event-driven simulation

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::{Mutex, MutexGuard, Condvar};

struct Entry {
    time: u64,
    handler: Box<dyn FnOnce() + Send>,
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
    // Only used in non-threaded mode
    cycle: AtomicU64,
    next_event: AtomicU64,
    // Only used in threaded mode
    epoch: crate::util::RoCell<Instant>,
    condvar: Condvar,
    // This has to be a Box to allow repr(C)
    events: Mutex<BinaryHeap<Entry>>,
    shutdown: AtomicBool,
}

extern {
    // See also `event.s` for this function
    fn event_loop_wait();
}

impl EventLoop {
    /// Create a new event loop.
    pub fn new() -> EventLoop {
        EventLoop {
            cycle: AtomicU64::new(0),
            next_event: AtomicU64::new(u64::max_value()),
            epoch: crate::util::RoCell::new(Instant::now()),
            condvar: Condvar::new(),
            events: Mutex::new(BinaryHeap::new()),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Stop this event loop.
    pub fn shutdown(&self) {
        // Acquire the lock so nobody is modifying the data structures when we are touching some
        // time-sensitive data.
        let guard = self.events.lock();
        // As event loops can be restarted after shutdown, we need to synchonize non-threaded and
        // threaded counters.
        if crate::threaded() {
            // This won't conflict with `event_loop_wait` as they are in different mode.
            self.cycle.store(self.cycle(), Ordering::Relaxed);
        } else {
            // Calculate number of micros since now. We can only round-up as cycle shouldn't go back.
            let micro = (self.cycle() + 99) / 100;
            // No need to worry about data race here due to mode difference.
            unsafe { crate::util::RoCell::replace(&self.epoch, Instant::now() - Duration::from_micros(micro)) };
        }
        std::mem::drop(guard);

        self.shutdown.store(true, Ordering::Relaxed);
        // Queue a no-op event to wake the loop up.
        self.queue(0, Box::new(|| {}));
    }

    /// Query the current cycle count.
    pub fn cycle(&self) -> u64 {
        if crate::threaded() {
            let duration = Instant::now().duration_since(*self.epoch);
            duration.as_micros() as u64 * 100
        } else {
            self.cycle.load(Ordering::Relaxed)
        }
    }

    /// Add a new event to the event loop for triggering. If it happens in the past it will be
    /// dequeued and triggered as soon as `cycle` increments for the next time.
    pub fn queue(&self, cycle: u64, handler: Box<dyn FnOnce() + Send>) {
        let mut guard = self.events.lock();
        guard.push(Entry {
            time: cycle,
            handler,
        });

        if crate::threaded() {
            // If the event just queued is the next event, we need to wake the event loop up.
            if guard.peek().unwrap().time == cycle {
                self.condvar.notify_one();
            }
        } else {
            // It's okay to be relaxed because guard's release op will order it.
            self.next_event.store(match guard.peek() {
                Some(it) => it.time,
                None => u64::max_value(),
            }, Ordering::Relaxed);
        }
    }

    /// Query the current time (we pretend to be operating at 100MHz at the moment)
    pub fn time(&self) -> u64 {
        self.cycle() / 100
    }

    pub fn queue_time(&self, time: u64, handler: Box<dyn FnOnce() + Send>) {
        self.queue(time * 100, handler);
    }

    /// Handle all events at or before `cycle`, and return the cycle of next event if any.
    fn handle_events(&self, guard: &mut MutexGuard<BinaryHeap<Entry>>, cycle: u64) -> Option<u64> {
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

    pub fn event_loop(&self) {
        let mut guard = self.events.lock();
        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                self.shutdown.store(false, Ordering::Relaxed);
                return;
            }
            let cycle = self.cycle();
            let result = self.handle_events(&mut guard, cycle);
            if crate::threaded() {
                match result {
                    None => {
                        self.condvar.wait(&mut guard);
                    }
                    Some(v) => {
                        self.condvar.wait_for(&mut guard, Duration::from_micros(v - cycle));
                    }
                }
            } else {
                self.next_event.store(result.unwrap_or(u64::max_value()), Ordering::Relaxed);
                MutexGuard::unlocked(&mut guard, || {
                    unsafe { event_loop_wait() }
                });
            }
        }
    }
}
