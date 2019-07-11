mod split_int;
pub use split_int::SplitInt;

pub mod int;
pub mod softfp;

mod code;
pub use code::Code;

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        unsafe { &(*(0 as *const $ty)).$field as *const _ as usize }
    }
}
