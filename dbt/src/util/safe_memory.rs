extern {
    fn memory_probe_read(ptr: usize) -> bool;
    fn memory_probe_write(ptr: usize) -> bool;
}

pub fn probe_read<T>(ptr: *const T) -> Result<(), ()> {
    if unsafe { memory_probe_read(ptr as usize) } { Err(()) } else { Ok(()) }
}

pub fn probe_write<T>(ptr: *mut T) -> Result<(), ()> {
    if unsafe { memory_probe_write(ptr as usize) } { Err(()) } else { Ok(()) }
}
