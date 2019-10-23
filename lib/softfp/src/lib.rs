//! Software floating point operation library.
//!
//! Before using the library for any floating point operations, you must register handlers using
//! [`register_get_rounding_mode`] and [`register_set_exception_flag`]. A simplest way would be
//! track them using
//! [`thread_local!`](https://doc.rust-lang.org/nightly/std/macro.thread_local.html)
//! variables. They are not included in this crate as this crate supports `#![no_std]`
//! environments.

#![no_std]
mod fp;
mod int;

pub use fp::*;
