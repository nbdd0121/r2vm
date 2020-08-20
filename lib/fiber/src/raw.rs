//! Raw function exposed for DBT usage.

extern "C" {
    /// Save all volatile registers.
    ///
    /// Upon returning, the stack pointer is changed and RBP is set to the context data pointer of
    /// the active fiber. This should only be used at the beginning of
    // an assembly function and should be paired with `fiber_restore_ret_raw`. Only call this function
    /// from inside fiber environment.
    pub fn fiber_save_raw();

    /// Restore all saved registers by `fiber_save_raw` and return.
    ///
    /// Must be called at the end of an assembly function that calls `fiber_save_raw` at the beginning.
    /// `jmp fiber_restore_ret_raw` should replace `ret`.
    pub fn fiber_restore_ret_raw();

    /// Yield for `num + 1` cycles.
    ///
    /// Because this cannot be called from oridinary Rust/C functions and only assemblies, we omit
    /// the `num` argument from the signature.
    ///
    /// All registers are tempered excluding the stack pointer. Upon returning RBP is set to the
    /// context data pointer of the active fiber. Must be called within fiber environment.
    pub fn fiber_sleep_raw();

    /// Yield for 1 cycle.
    ///
    /// All registers are tempered excluding the stack pointer. Upon returning RBP is set to the
    /// context data pointer of the active fiber. Must be called within fiber environment.
    pub fn fiber_yield_raw();

    /// Yield for `num + 1` cycles.
    ///
    /// Must be called within fiber environment.
    pub fn fiber_sleep(num: usize);

    /// Return the context data pointer of the active fiber.
    ///
    /// Must be called within fiber environment. Consider using [`with_context`] if possible.
    ///
    /// [`with_context`]: super::with_context
    pub fn fiber_current() -> std::num::NonZeroUsize;
}
