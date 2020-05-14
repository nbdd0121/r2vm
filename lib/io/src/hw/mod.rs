//! Hardware devices emulated.
//!
//! This module contains all emulated hardware devices currently supported by R2VM. If the hardware
//! rely on some host resources that can be abstracted (e.g. block device or network device), they
//! should live within their own module under the root of crate.
//!
//! # Lifetime of devices
//! As usually emulation of hardware devices requires running event loop, which requires data to be
//! shared across multiple threads as well as the user of struct that implements [`IoMemory`]. This
//! may needs a lot of [`Arc`]s and [`Mutex`]s and may cause the lifetime to be difficult to manage
//! or even cause memory leaks.
//!
//! As a rule of thumb, we recommend all implementations to hide [`Arc`]s and [`Weak`]s inside
//! their implementation and not expose in the signature. The device constructor therefore should
//! look like `fn new(...) -> Self` rather than `fn new(...) -> Arc<Self>`. This allows users to
//! own the device. When the owned device instance is dropped, necessary steps should be taken to
//! shutdown the event loops and free up resources. To avoid memory leak, any references handed out
//! by the device implementation (e.g. through [`IrqPin`]) should contain a [`Weak`] instead of an
//! `[`Arc`].
//!
//! [`IoMemory`]: crate::IoMemory
//! [`IrqPin`]: crate::IrqPin
//! [`Arc`]: std::sync::Arc
//! [`Weak`]: std::sync::Weak
//! [`Mutex`]: std::sync::Mutex

pub mod intc;
pub mod rtc;
