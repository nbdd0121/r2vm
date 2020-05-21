//! This module handles event-driven simulation

use futures::future::BoxFuture;
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

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
    /// The base when we need to know how many cycles are spent in lock-step. When switching from
    /// lockstep mode to non-lockstep mode, `cycle` will be updated, so it does not reflect number
    /// of cycles in lockstep mode, we therefore need a base to keep track.
    lockstep_cycle_base: AtomicU64,
}

extern "C" {
    // See also `event.s` for this function
    fn event_loop_wait();
}

impl EventLoop {
    /// Create a new event loop.
    pub fn new() -> EventLoop {
        EventLoop {
            cycle: AtomicU64::new(0),
            lockstep_cycle_base: AtomicU64::new(0),
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
            let cycle = self.cycle();
            let lockstep_cycle = self.get_lockstep_cycles();
            self.cycle.store(cycle, Ordering::Relaxed);
            self.lockstep_cycle_base.store(cycle - lockstep_cycle, Ordering::Relaxed);
        } else {
            // Calculate number of micros since now. We can only round-up as cycle shouldn't go back.
            let micro = (self.cycle() + 99) / 100;
            // No need to worry about data race here due to mode difference.
            unsafe {
                crate::util::RoCell::replace(
                    &self.epoch,
                    Instant::now() - Duration::from_micros(micro),
                )
            };
        }
        std::mem::drop(guard);

        self.shutdown.store(true, Ordering::Relaxed);
        // Queue a no-op event to wake the loop up.
        self.queue(0, Box::new(|| {}));
    }

    /// Query the number of cycles spent in lockstep execution mode.
    pub fn get_lockstep_cycles(&self) -> u64 {
        self.cycle.load(Ordering::Relaxed) - self.lockstep_cycle_base.load(Ordering::Relaxed)
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
        guard.push(Entry { time: cycle, handler });

        if crate::threaded() {
            // If the event just queued is the next event, we need to wake the event loop up.
            if guard.peek().unwrap().time == cycle {
                self.condvar.notify_one();
            }
        } else {
            // It's okay to be relaxed because guard's release op will order it.
            self.next_event.store(
                match guard.peek() {
                    Some(it) => it.time,
                    None => u64::max_value(),
                },
                Ordering::Relaxed,
            );
        }
    }

    /// Create a future that resolves after the specified cycle.
    pub fn on_cycle(&self, cycle: u64) -> impl Future<Output = ()> + Send + 'static {
        struct Timer(u64, bool);
        impl Future for Timer {
            type Output = ();

            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                // Needed to have a 'static lifetime, which is more useful fore user.
                let event_loop = crate::event_loop();
                if event_loop.cycle() >= self.0 {
                    return Poll::Ready(());
                }
                if !self.1 {
                    self.1 = true;
                    let waker = cx.waker().clone();
                    event_loop.queue(self.0, Box::new(move || waker.wake()));
                }
                Poll::Pending
            }
        }
        Timer(cycle, false)
    }

    /// Query the current time (we pretend to be operating at 100MHz at the moment)
    pub fn time(&self) -> u64 {
        self.cycle() / 100
    }

    pub fn queue_time(&self, time: u64, handler: Box<dyn FnOnce() + Send>) {
        self.queue(time * 100, handler);
    }

    pub fn on_time(&self, time: u64) -> impl Future<Output = ()> + Send + 'static {
        self.on_cycle(time * 100)
    }

    /// Handle all events at or before `cycle`, and return the cycle of next event if any.
    fn handle_events(
        &self,
        mut guard: &mut MutexGuard<BinaryHeap<Entry>>,
        cycle: u64,
    ) -> Option<u64> {
        loop {
            let time = match guard.peek() {
                None => return None,
                Some(v) => v.time,
            };
            if time > cycle {
                return Some(time);
            }
            let entry = guard.pop().unwrap();
            MutexGuard::unlocked(&mut guard, || {
                (entry.handler)();
            });
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
                MutexGuard::unlocked(&mut guard, || unsafe { event_loop_wait() });
            }
        }
    }

    /// Spawn a [`Future`] on this event loop.
    pub fn spawn(&self, future: BoxFuture<'static, ()>) {
        use futures::task::ArcWake;

        struct Task {
            future: Mutex<Option<BoxFuture<'static, ()>>>,
        }

        impl ArcWake for Task {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.clone().wake();
            }

            fn wake(self: Arc<Self>) {
                // We're a bit lazy here without capturing `self`. But as we will only have 1
                // single event loop, so this is probably okay.
                let event_loop = crate::event_loop();
                // Schedule an event to poll the task again
                event_loop.queue(
                    0,
                    Box::new(move || {
                        let waker_ref = futures::task::waker_ref(&self);
                        let mut context = Context::from_waker(&waker_ref);

                        let mut lock = self.future.lock();
                        if let Some(ref mut future) = *lock {
                            // This is safe because we are pinned by Arc.
                            let poll = unsafe { Pin::new_unchecked(future) }.poll(&mut context);
                            if poll.is_ready() {
                                // When we polled a context to ready, drop the Future.
                                *lock = None;
                            }
                        }
                    }),
                );
            }
        }

        let task = Arc::new(Task { future: Mutex::new(Some(future)) });
        task.wake();
    }
}
