#![feature(test)]

//! This benchmark tests that fiber::Mutex and parking_lot::Mutex have no significant performance
//! difference in multithreaded contexts.

extern crate test;

use fiber::{FiberContext, FiberGroup, RawRwLock, RwLock};
use lock_api::RwLock as LRwLock;
use parking_lot::RawRwLock as PRawRwLock;
use std::sync::Arc;
use test::Bencher;

#[bench]
fn rwlock_create(b: &mut Bencher) {
    b.iter(|| RwLock::new(()));
}

#[bench]
fn rwlock_contention(b: &mut Bencher) {
    b.iter(|| run::<RawRwLock>(4, 0, 1000));
}

#[bench]
fn rwlock_contention_p(b: &mut Bencher) {
    b.iter(|| run::<PRawRwLock>(4, 0, 1000));
}

#[bench]
fn rwlock_no_contention(b: &mut Bencher) {
    b.iter(|| run::<RawRwLock>(1, 0, 4000));
}

#[bench]
fn rwlock_no_contention_p(b: &mut Bencher) {
    b.iter(|| run::<PRawRwLock>(1, 0, 4000));
}

#[bench]
fn rwlock_read(b: &mut Bencher) {
    b.iter(|| run::<RawRwLock>(0, 4, 1000));
}

#[bench]
fn rwlock_read_p(b: &mut Bencher) {
    b.iter(|| run::<PRawRwLock>(0, 4, 1000));
}

#[bench]
fn rwlock_rw(b: &mut Bencher) {
    b.iter(|| run::<RawRwLock>(2, 2, 1000));
}

#[bench]
fn rwlock_rw_p(b: &mut Bencher) {
    b.iter(|| run::<PRawRwLock>(2, 2, 1000));
}

fn run<R: lock_api::RawRwLock + Send + Sync + 'static>(
    wthread: usize,
    rthread: usize,
    iter: usize,
) {
    let m = Arc::new(LRwLock::<R, _>::new(()));
    let mut threads = Vec::new();

    let barrier = Arc::new(std::sync::Barrier::new(wthread + rthread + 1));

    for _ in 0..wthread {
        let m = m.clone();
        let b = barrier.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = FiberContext::new(());
            FiberGroup::with(|group| {
                group.spawn(&mut ctx, || {
                    for _ in 0..iter {
                        let _ = m.write();
                    }
                });
                b.wait();
            })
        }));
    }

    for _ in 0..rthread {
        let m = m.clone();
        let b = barrier.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = FiberContext::new(());
            FiberGroup::with(|group| {
                group.spawn(&mut ctx, || {
                    for _ in 0..iter {
                        let _ = m.read();
                    }
                });
                b.wait();
            })
        }));
    }

    barrier.wait();
    for t in threads {
        t.join().unwrap();
    }
}
