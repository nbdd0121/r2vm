#![feature(test, llvm_asm, naked_functions)]

extern crate test;

use fiber::{self, FiberContext, FiberGroup};
use test::Bencher;

#[bench]
fn bench_sleep(b: &mut Bencher) {
    b.iter(|| run(10, 10000));
}

#[bench]
fn bench_sleep_asm(b: &mut Bencher) {
    b.iter(|| run_asm(10, 10000));
}

fn run(fiber: usize, iter: usize) {
    let mut ctx: Vec<_> = (0..fiber).into_iter().map(|_| FiberContext::new(())).collect();
    FiberGroup::with(|group| {
        for ctx in ctx.iter_mut() {
            group.spawn(ctx, || {
                for _ in 0..iter {
                    fiber::sleep(1);
                }
            });
        }
    });
}

#[inline(never)]
#[naked]
unsafe extern "C" fn test(_x: usize) {
    llvm_asm!("call fiber_save_raw
    push rdi
    l:
    call fiber_yield_raw
    sub qword ptr [rsp], 1
    jnz l
    pop rdi
    jmp fiber_restore_ret_raw"::::"intel");
}

fn run_asm(fiber: usize, iter: usize) {
    let mut ctx: Vec<_> = (0..fiber).into_iter().map(|_| FiberContext::new(())).collect();
    FiberGroup::with(|group| {
        for ctx in ctx.iter_mut() {
            group.spawn(ctx, || {
                unsafe { test(iter) };
            });
        }
    });
}
