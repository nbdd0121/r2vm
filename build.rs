extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/emu/event.s")
        .file("src/dbt/helper.s")
        .file("src/fiber/fiber.s")
        .compile("foo");
}
