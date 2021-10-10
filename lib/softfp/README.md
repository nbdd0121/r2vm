# softfp

Software floating point operation library.

Many floating operations require a rounding mode and can set a few exception flags.
This library therefore expects two callbacks, `softfp_get_rounding_mode` and
`softfp_set_exception_flags`. You can define them with `#[no_mangle]`. The calling
convention is Rust, so there is no need for `extern "C"`.

If you prefer not to define these callbacks with `#[no_mangle]`, you can use the
enabled-by-default "register" feature which allow you to register the function
via `register_get_rounding_mode` and `register_set_exception_flags` instead.
You must register them before using the library for any floating point operations.
The simplest way would be track them using
[`thread_local!`](https://doc.rust-lang.org/nightly/std/macro.thread_local.html)
variables.
