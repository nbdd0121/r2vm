extern {
    fn fiber_start(cell: usize);
    fn fiber_sleep(num: usize);
}

/// A `FiberStack` is the basic data structure that keeps fiber. Usually when a fiber is running,
/// the base pointer points to just past the FiberStack, which can be used by the fiber to store
/// useful information.
///
/// A `FiberStack` is always aligned at 2MB boundary, and is exactly 2MB in size. This design
/// guarantees that we can use stack pointer to retrieve the base pointer easily, so we can have
/// reasonable performance when yielding in the middle of Rust code.
#[repr(C)]
struct FiberStack {
    sp: usize,
    _pad: usize,
    next: usize,
    prev: usize,
}

pub struct Fiber(*mut FiberStack);

impl Fiber {
    pub fn new() -> Fiber {
        // Allocate 2M memory for stack. This must be aligned properly to allow efficient context
        // retrieval in yield function.
        let map = unsafe { libc::memalign(0x200000, 0x200000) };
        if map == libc::MAP_FAILED {
            panic!("cannot create fiber stack");
        }
        let map = map as *mut FiberStack;

        unsafe {
            (*map).sp = map as usize + 0x200000;
            (*map).next = map as usize + 32;
            (*map).prev = map as usize + 32;
        }
        Fiber(map)
    }

    fn get_stack_pointer(&self) -> usize {
        unsafe { (*self.0).sp }
    }

    fn set_stack_pointer(&self, sp: usize) {
        unsafe { (*self.0).sp = sp }
    }

    pub fn data_pointer<T>(&self) -> *mut T {
        (self.0 as usize + 32) as _
    }

    pub fn set_fn(&self, f: unsafe extern "C" fn()->()) {
        let sp = self.get_stack_pointer();
        unsafe { *((sp - 8) as *mut usize) = f as usize };
        self.set_stack_pointer(sp - 8);
    }

    pub fn chain(&self, next: &Fiber) {
        unsafe {
            // This is essentially insertion to linked list
            let original_next = ((*self.0).next as usize - 32) as *mut FiberStack;
            (*original_next).prev = next.0 as usize + 32;
            (*next.0).next = original_next as usize + 32;
            (*self.0).next = next.0 as usize + 32;
            (*next.0).prev = self.0 as usize + 32;
        }
    }

    /// Start the fiber. This should never return.
    pub unsafe fn enter(&self) {
        fiber_start(self.0 as usize);
        unreachable!();
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
