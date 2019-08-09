extern {
    fn fiber_start(cell: FiberStack) -> FiberStack;
    fn fiber_sleep(num: usize);
}

/// A `FiberStack` is the basic data structure that keeps fiber.
///
/// A `FiberStack` is always aligned at 2MB boundary, and is exactly 2MB in size. This design
/// guarantees that we can use stack pointer to retrieve the base pointer easily, so we can have
/// reasonable performance when yielding in the middle of Rust code.
///
/// The fiber stack is logically parititoned to three regions. The stack grows downwards from the
/// very top; the fiber's book-keeping area is 32-bytes, and the area beyond 32-bytes are for data
/// storage. The base pointer usually points to the location just go past these 32-bytes.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FiberStack(std::num::NonZeroUsize);

impl FiberStack {
    fn allocate() -> Self {
        // Allocate 2M memory for stack. This must be aligned properly to allow efficient context
        // retrieval in yield function.
        let map = unsafe { libc::memalign(0x200000, 0x200000) };
        if map == libc::MAP_FAILED {
            panic!("cannot create fiber stack");
        }
        let stack = FiberStack(std::num::NonZeroUsize::new(map as usize + 32).unwrap());

        unsafe {
            *stack.sp() = map as usize + 0x200000 - 32;
            *stack.next() = stack;
            *stack.prev() = stack;
        }
        stack
    }

    #[allow(dead_code)]
    unsafe fn deallocate(self) {
        libc::free((self.0.get() - 32) as _)
    }

    fn data_pointer(self) -> usize {
        self.0.get()
    }

    fn prev(self) -> *mut FiberStack {
        (self.0.get() - 8) as *mut FiberStack
    }

    fn next(self) -> *mut FiberStack {
        (self.0.get() - 16) as *mut FiberStack
    }

    fn sp(self) -> *mut usize {
        (self.0.get() - 32) as *mut usize
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

    pub fn set_fn(&self, f: unsafe extern "C" fn()->()) {
        unsafe { *((self.0.data_pointer() - 32 + 0x200000 - 32) as *mut usize) = f as usize };
    }

    /// Yield the current fiber for `num` many times.
    ///
    /// Note: this signature is marked safe just for convience.
    /// If the current code is running under fiber then it is completely safe to call this, but
    /// it will just crash if the current code is not running inside fiber.
    pub fn sleep(num: usize) {
        unsafe { fiber_sleep(num) }
    }
}

impl Drop for Fiber {
    fn drop(&mut self) {
        panic!("dropping fiber");
    }
}

struct FiberGroupData {
    first: Option<FiberStack>,
    last: Option<FiberStack>,
}

pub struct FiberGroup(FiberGroupData);

impl FiberGroup {
    pub fn new() -> FiberGroup {
        FiberGroup(FiberGroupData {
            first: None,
            last: None,
        })
    }

    /// Add a fiber to this fiber group.
    pub fn add(&mut self, fiber: Fiber) {
        let inner = &mut self.0;
        if inner.first.is_none() {
            inner.first = Some(fiber.0);
            inner.last = Some(fiber.0);
        } else {
            let last = inner.last.unwrap();

            // This is essentially insertion to linked list
            unsafe {
                let next = *last.next();
                *next.prev() = fiber.0;
                *fiber.0.next() = next;
                *last.next() = fiber.0;
                *fiber.0.prev() = last;
            }

            inner.last = Some(fiber.0);
        }
        std::mem::forget(fiber);
    }

    /// Start executing this fiber group. Exits when there are no running fibers.
    pub fn run(&mut self) {
        let inner = &mut self.0;
        loop {
            // Run fiber group. Function will return when any fiber exits.
            let stack = unsafe { fiber_start(inner.last.unwrap()) };
            let next = unsafe { *stack.next() };
            if stack == next {
                // The removing fiber is the only fiber, returning
                inner.first = None;
                inner.last = None;
                return;
            }
            unsafe {
                let prev = *stack.prev();
                if stack == inner.first.unwrap() {
                    inner.first = Some(next);
                }
                if stack == inner.last.unwrap() {
                    inner.last = Some(prev);
                }
                *prev.next() = next;
                *next.prev() = prev;
            }
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
