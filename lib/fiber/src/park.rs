//! Implementation of parking lot for fibers
//!
//! For detailed implementation, check out the following URLs
//! * https://webkit.org/blog/6161/locking-in-webkit/
//! * https://docs.rs/parking_lot

use super::raw::fiber_sleep;
use super::{fiber_current, FiberGroup, FiberStack};
use once_cell::sync::Lazy;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
struct WaitEntry {
    fiber: FiberStack,
    token: UnparkToken,
    next: Option<NonNull<WaitEntry>>,
}

struct WaitList {
    head: NonNull<WaitEntry>,
    tail: NonNull<WaitEntry>,
}

unsafe impl Send for WaitList {}

static WAIT_LIST_MAP: Lazy<super::map::ConcurrentMap<usize, WaitList>> =
    Lazy::new(|| super::map::ConcurrentMap::new());

#[derive(Clone, Copy, Debug)]
pub struct UnparkToken(pub usize);

pub fn park(
    key: usize,
    validate: impl FnOnce() -> bool,
    before_sleep: impl FnOnce(),
) -> Option<UnparkToken> {
    // Required before calling fiber_current.
    super::assert_in_fiber();

    let cur = unsafe { fiber_current() };
    let mut entry = WaitEntry { fiber: cur, token: UnparkToken(0), next: None };

    let valid = WAIT_LIST_MAP.with(key, |list| {
        // Deadlock prevention: must acquire group lock after list lock.

        // Give the caller a chance, under strong synchronisation guarantee, to do last check and possibly abort.
        if !validate() {
            return false;
        }

        match list {
            None => {
                *list = Some(WaitList { head: (&mut entry).into(), tail: (&mut entry).into() });
            }
            Some(ref mut list) => {
                unsafe { list.tail.as_mut().next = Some((&mut entry).into()) };
                list.tail = (&mut entry).into();
            }
        }

        unsafe { FiberGroup::prepare_pause(cur) };
        true
    });

    if !valid {
        return None;
    }

    before_sleep();

    unsafe {
        let awaken = FiberGroup::pause(cur);
        if !awaken {
            fiber_sleep(0);
        }
    };

    std::sync::atomic::fence(Ordering::Acquire);
    Some(entry.token)
}

pub fn unpark_all(key: usize, token: UnparkToken) {
    let list = WAIT_LIST_MAP.with(key, |list| list.take());
    if let Some(list) = list {
        let mut ptr = Some(list.head);
        while let Some(mut entry) = ptr {
            let entry = unsafe { entry.as_mut() };
            entry.token = token;
            ptr = entry.next;
            unsafe { FiberGroup::unpause(entry.fiber) };
        }
    }
}

pub struct UnparkResult {
    pub unparked: bool,
    pub have_more: bool,
}

pub fn unpark_one(key: usize, callback: impl FnOnce(UnparkResult) -> UnparkToken) {
    let fiber = WAIT_LIST_MAP.with(key, |list| {
        if let Some(ref mut inner) = list {
            let entry = unsafe { &mut *inner.head.as_ptr() };
            match entry.next {
                None => *list = None,
                Some(next) => inner.head = next,
            }
            entry.token = callback(UnparkResult { unparked: true, have_more: list.is_some() });
            Some(entry.fiber)
        } else {
            callback(UnparkResult { unparked: false, have_more: list.is_some() });
            None
        }
    });
    if let Some(fiber) = fiber {
        unsafe { FiberGroup::unpause(fiber) };
    }
}
