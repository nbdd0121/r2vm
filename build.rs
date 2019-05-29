extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/emu/safe_memory.s")
        .file("src/fiber/fiber.s")
        .compile("foo");
}
