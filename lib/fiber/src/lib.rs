#[macro_use]
extern crate offset_of;

use std::cell::Cell;

extern "C" {
    fn fiber_start(cell: FiberStack) -> FiberStack;
    fn fiber_current() -> FiberStack;
    fn fiber_sleep(num: usize);
}

thread_local! {
    static IN_FIBER: Cell<bool> = Cell::new(false);
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
    _pad0: usize,
    _pad1: usize,
    cycles_to_sleep: usize,
    _pad2: usize,
    stack_pointer: usize,
}

impl FiberStack {
    fn allocate() -> Self {
        // This is what is assumed in fiber.s
        assert_eq!(std::mem::size_of::<FiberData>(), 64);
        assert_eq!(offset_of!(FiberData, next), 16);
        assert_eq!(offset_of!(FiberData, cycles_to_sleep), 40);
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
            data.stack_pointer = self.0.get() - 64 + 0x200000 - 32;
            data.next = self;
            data.prev = self;
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
        IN_FIBER.with(|x| assert!(x.get(), "not in fiber"));
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
}

// Box to have a stable address
pub struct FiberGroup(Box<FiberGroupData>);

impl FiberGroup {
    pub fn new() -> FiberGroup {
        FiberGroup(Box::new(FiberGroupData {
            fibers: Vec::new(),
            first: None,
            last: None,
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
                next_data.prev = fiber.0;
                fiber_data.next = next;
                last_data.next = fiber.0;
                fiber_data.prev = last;
            }

            inner.last = Some(fiber.0);
        }
        inner.fibers.push(fiber);
    }

    /// Start executing this fiber group. Exits when there are no running fibers.
    /// The ownership of all fibers are returned (THIS IS A HACK).
    pub fn run(&mut self) -> Vec<Fiber> {
        IN_FIBER.with(|x| x.set(true));
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
