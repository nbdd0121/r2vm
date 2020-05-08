#[macro_use]
extern crate offset_of;

use parking_lot::{Condvar as PCondvar, Mutex as PMutex};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

mod map;
mod mutex;
mod park;

pub use mutex::{Condvar, Mutex, MutexGuard, RawMutex};

extern "C" {
    fn fiber_start(cell: FiberStack) -> FiberStack;
    fn fiber_current() -> FiberStack;
    fn fiber_sleep(num: usize);
}

thread_local! {
    static IN_FIBER: Cell<bool> = Cell::new(false);
}

#[inline]
fn in_fiber() -> bool {
    IN_FIBER.with(|x| x.get())
}

#[inline]
fn assert_in_fiber() {
    assert!(in_fiber(), "not in fiber");
}

/// A `FiberStack` is the basic data structure that keeps fiber.
///
/// A `FiberStack` is always aligned at 2MB boundary, and is exactly 2MB in size. This design
/// guarantees that we can use stack pointer to retrieve the base pointer easily, so we can have
/// reasonable performance when yielding in the middle of Rust code.
///
/// The fiber stack is logically parititoned to three regions. The stack grows downwards from the
/// very top; the fiber's book-keeping area is 64-bytes, and the area beyond 64-bytes are for data
/// storage. The base pointer usually points to the location just go past these 64-bytes.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FiberStack(std::num::NonZeroUsize);

/// Management data structure of a fiber.
#[repr(C)]
struct FiberData {
    group: *const FiberGroupData,
    prev: FiberStack,
    next: FiberStack,
    paused: AtomicBool,
    _pad: usize,
    cycles_to_sleep: usize,
    next_avail: std::sync::atomic::AtomicUsize,
    stack_pointer: usize,
}

impl FiberStack {
    fn allocate() -> Self {
        // This is what is assumed in fiber.s
        assert_eq!(std::mem::size_of::<FiberData>(), 64);
        assert_eq!(offset_of!(FiberData, next), 16);
        assert_eq!(offset_of!(FiberData, cycles_to_sleep), 40);
        assert_eq!(offset_of!(FiberData, next_avail), 48);
        assert_eq!(offset_of!(FiberData, stack_pointer), 56);

        // Allocate 2M memory for stack. This must be aligned properly to allow efficient context
        // retrieval in yield function.
        let map = unsafe { libc::memalign(0x200000, 0x200000) };
        if map.is_null() {
            panic!("cannot create fiber stack");
        }
        let stack = FiberStack(
            std::num::NonZeroUsize::new(map as usize + std::mem::size_of::<FiberData>()).unwrap(),
        );
        stack.init();
        stack
    }

    #[allow(dead_code)]
    unsafe fn deallocate(self) {
        libc::free((self.0.get() - std::mem::size_of::<FiberData>()) as _)
    }

    unsafe fn data(self) -> &'static mut FiberData {
        &mut *((self.0.get() - std::mem::size_of::<FiberData>()) as *mut FiberData)
    }

    fn init(self) {
        unsafe {
            let data = self.data();
            data.stack_pointer = self.0.get() - std::mem::size_of::<FiberData>() + 0x200000 - 32;
            data.next = self;
            data.prev = self;
            data.next_avail.store(self.0.get(), Ordering::Relaxed);
            data.paused = AtomicBool::new(false);
            data.cycles_to_sleep = 0;
        }
    }

    fn data_pointer(self) -> usize {
        self.0.get()
    }
}

pub struct Fiber(FiberStack);

impl Fiber {
    pub fn new() -> Fiber {
        Fiber(FiberStack::allocate())
    }

    pub fn data_pointer<T>(&self) -> *mut T {
        self.0.data_pointer() as _
    }

    pub fn set_fn(&self, f: fn()) {
        unsafe { *((self.0.data_pointer() - 64 + 0x200000 - 32) as *mut usize) = f as usize };
    }

    /// Yield the current fiber for `num` many times.
    #[inline]
    pub fn sleep(num: usize) {
        assert_in_fiber();
        if num > 0 {
            unsafe { fiber_sleep(num - 1) }
        }
    }

    pub fn scratchpad<T>() -> *mut T {
        unsafe { fiber_current().data_pointer() as _ }
    }
}

impl Drop for Fiber {
    fn drop(&mut self) {
        panic!("dropping fiber");
    }
}

struct FiberGroupData {
    fibers: Vec<Fiber>,
    first: Option<FiberStack>,
    last: Option<FiberStack>,
    mutex: PMutex<()>,
    condvar: PCondvar,
}

// Box to have a stable address
pub struct FiberGroup(Box<FiberGroupData>);

impl FiberGroup {
    pub fn new() -> FiberGroup {
        FiberGroup(Box::new(FiberGroupData {
            fibers: Vec::new(),
            first: None,
            last: None,
            mutex: PMutex::new(()),
            condvar: PCondvar::new(),
        }))
    }

    /// Add a fiber to this fiber group.
    pub fn add(&mut self, fiber: Fiber) {
        let inner = &mut *self.0;

        let fiber_data = unsafe { fiber.0.data() };
        fiber_data.group = inner;

        if inner.first.is_none() {
            inner.first = Some(fiber.0);
            inner.last = Some(fiber.0);
        } else {
            let last = inner.last.unwrap();

            // This is essentially insertion to linked list
            unsafe {
                let last_data = last.data();
                let next = last_data.next;
                let next_data = next.data();
                let next_avail = last_data.next_avail.load(Ordering::Relaxed);
                next_data.prev = fiber.0;
                fiber_data.next = next;
                fiber_data.next_avail.store(next_avail, Ordering::Relaxed);
                last_data.next = fiber.0;
                // The newly inserted fiber is available.
                last_data.next_avail.store((fiber.0).0.get(), Ordering::Relaxed);
                fiber_data.prev = last;
            }

            inner.last = Some(fiber.0);
        }
        inner.fibers.push(fiber);
    }

    unsafe fn prepare_pause(stack: FiberStack) {
        stack.data().paused.store(true, Ordering::Relaxed);
    }

    /// Put the fiber into sleep. Note that the current thread must be executing this FiberGroup.
    ///
    /// Returns true if awaken already.
    unsafe fn pause(stack: FiberStack) -> bool {
        let group = &*stack.data().group;

        // All modifications to fiber structures must hold the lock.
        let mut guard = group.mutex.lock();

        // Unpause has been called on thie fiber between `prepare_pause` and `pause`.
        if !stack.data().paused.load(Ordering::Relaxed) {
            return true;
        }

        // Wait until there is a fiber to schedule
        while stack.data().next_avail.load(Ordering::Relaxed) == stack.0.get() {
            group.condvar.wait(&mut guard);
            // This fiber has been waken up
            if !stack.data().paused.load(Ordering::Relaxed) {
                return true;
            }
        }

        let mut prev = stack.data().prev;
        loop {
            let prev_data = prev.data();
            // Safe to do this store because we have exclusive acceess to this field.
            prev_data
                .next_avail
                .store(stack.data().next_avail.load(Ordering::Relaxed), Ordering::Relaxed);
            if !prev_data.paused.load(Ordering::Relaxed) {
                break;
            }
            prev = prev_data.prev;
        }

        false
    }

    unsafe fn unpause(stack: FiberStack) {
        let fiber = stack.data();
        let group = &*fiber.group;

        // All modifications to fiber structures must hold the lock.
        let mut _guard = group.mutex.lock();

        fiber.paused.store(false, Ordering::Relaxed);
        let mut prev = fiber.prev;
        loop {
            let prev_data = prev.data();
            // Safe to do this: if the fiber group is currently running and is accessing this field
            // during an yield operation it will either yield to the old fiber or the new one,
            // both of which are available to run.
            prev_data.next_avail.store(stack.0.get(), Ordering::Relaxed);
            if !prev_data.paused.load(Ordering::Relaxed) {
                break;
            }
            prev = prev_data.prev;
        }

        // Wake up the fiber thread if currently no fiber is available.
        group.condvar.notify_all();
    }

    /// Start executing this fiber group. Exits when there are no running fibers.
    /// The ownership of all fibers are returned (THIS IS A HACK).
    pub fn run(&mut self) -> Vec<Fiber> {
        IN_FIBER.with(|x| {
            assert!(!x.get(), "FiberGroup re-entry");
            x.set(true)
        });
        let inner = &mut self.0;
        loop {
            // Run fiber group. Function will return when any fiber exits.
            let stack = unsafe { fiber_start(inner.first.unwrap()) };
            let next = unsafe { stack.data().next };
            if stack == next {
                stack.init();
                // The removing fiber is the only fiber, returning
                inner.first = None;
                inner.last = None;
                IN_FIBER.with(|x| x.set(false));
                return std::mem::replace(&mut inner.fibers, Vec::new());
            }
            unsafe {
                Self::prepare_pause(stack);
                Self::pause(stack);
                let prev = stack.data().prev;
                if stack == inner.first.unwrap() {
                    inner.first = Some(next);
                }
                if stack == inner.last.unwrap() {
                    inner.last = Some(prev);
                }
                prev.data().next = next;
                next.data().prev = prev;
            }
            // Prepare for next use
            stack.init();
        }
    }
}

impl Drop for FiberGroup {
    fn drop(&mut self) {
        if self.0.first.is_some() {
            panic!("dropping fiber");
        }
    }
}
