extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/emu/safe_memory.s")
        .compile("foo");
}
