/// Calculate the offset of the specified (possibly nested) field from the start of the struct.
#[macro_export]
macro_rules! offset_of {
    ($ty:ty, $($field:ident).*) => {
        unsafe { &(*(std::ptr::null() as *const $ty)) $(.$field)* as *const _ as usize }
    }
}
