extern crate cc;

fn main() {
    cc::Build::new().file("src/fiber.s").compile("fiber");
}
