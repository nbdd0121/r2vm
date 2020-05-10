#![feature(test)]

//! This benchmark tests that fiber::Mutex and parking_lot::Mutex have no significant performance
//! difference in multithreaded contexts.

extern crate test;

use fiber::{FiberContext, FiberGroup, Mutex, RawMutex};
use lock_api::Mutex as LMutex;
use parking_lot::RawMutex as PRawMutex;
use std::sync::Arc;
use test::Bencher;

#[bench]
fn mutex_create(b: &mut Bencher) {
    b.iter(|| Mutex::new(()));
}

#[bench]
fn mutex_contention(b: &mut Bencher) {
    b.iter(|| run::<RawMutex>(4, 1000));
}

#[bench]
fn mutex_contention_p(b: &mut Bencher) {
    b.iter(|| run::<PRawMutex>(4, 1000));
}

#[bench]
fn mutex_no_contention(b: &mut Bencher) {
    b.iter(|| run::<RawMutex>(1, 4000));
}

#[bench]
fn mutex_no_contention_p(b: &mut Bencher) {
    b.iter(|| run::<PRawMutex>(1, 4000));
}

fn run<R: lock_api::RawMutex + Send + Sync + 'static>(thread: usize, iter: usize) {
    let m = Arc::new(LMutex::<R, _>::new(()));
    let mut threads = Vec::new();

    for _ in 0..thread {
        let m = m.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = FiberContext::new(());
            FiberGroup::with(|group| {
                group.spawn(&mut ctx, || {
                    for _ in 0..iter {
                        let _ = m.lock();
                    }
                });
            })
        }));
    }

    for t in threads {
        t.join().unwrap();
    }
}
