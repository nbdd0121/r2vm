extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/emu/safe_memory.s")
        .file("src/emu/event.s")
        .file("src/fiber/fiber.s")
        .compile("foo");
}
