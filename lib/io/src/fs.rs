//! File system shares.
//!
//! This module provides a [`FileSystem`] trait which file system shares and I/O devices that
///! provide them to the guest OS.

#[doc(no_inline)]
pub use p9::FileSystem;
#[doc(no_inline)]
pub use p9::Passthrough;
