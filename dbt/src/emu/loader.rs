use std::path::Path;
use std::fs::File;
use std::os::unix::io::{IntoRawFd, AsRawFd, FromRawFd};
use std::ffi::CStr;
use rand::RngCore;
use super::abi;
use super::ureg;

const PF_R    : u32 = 0x4;
const PF_W    : u32 = 0x2;
const PF_X    : u32 = 0x1;
const ET_EXEC : libc::Elf64_Half = 2;
const ET_DYN  : libc::Elf64_Half = 3;
const EM_RISCV: libc::Elf64_Half = 243;

#[repr(C)]
pub struct Loader {
    fd: libc::c_int,
    file_size: libc::c_ulong,
    memory: *mut libc::c_void,
}

impl Drop for Loader {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.memory, self.file_size as usize) };
        unsafe { File::from_raw_fd(self.fd) };
    }
}

struct PhdrIter<'a> {
    i: usize,
    ehdr: &'a libc::Elf64_Ehdr,
}

impl<'a> Iterator for PhdrIter<'a> {
    type Item = &'a libc::Elf64_Phdr;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.ehdr.e_phnum as usize {
            None
        } else {
            let ptr = self.ehdr as *const _ as usize + self.ehdr.e_phoff as usize + self.ehdr.e_phentsize as usize * self.i;
            self.i += 1;
            Some(unsafe { &*(ptr as *const libc::Elf64_Phdr) })
        }
    }
}

impl Loader {
    fn ehdr(&self) -> &libc::Elf64_Ehdr {
        unsafe { &*(self.memory as *const libc::Elf64_Ehdr) }
    }

    fn phdr(&self) -> PhdrIter {
        PhdrIter { i: 0, ehdr: self.ehdr() }
    }

    pub fn new(file: &Path) -> std::io::Result<Loader> {
        let file = File::open(super::syscall::translate_path(file))?;
        // Get the size of the file.
        let file_size = file.metadata()?.len();
        // Map the file to memory.
        let memory = unsafe { libc::mmap(std::ptr::null_mut(), file_size as usize, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0) };
        if memory == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error())
        }
        Ok(Loader {
            fd: file.into_raw_fd(),
            file_size: file_size,
            memory
        })
    }

    pub fn validate_elf(&self) -> Result<(), &'static str> {
        let header = unsafe { &*(self.memory as *const libc::Elf64_Ehdr) };

        // Check the ELF magic numbers
        if &header.e_ident[0..4] != "\x7FELF".as_bytes() {
            return Err("the program to be loaded is not elf.");
        }

        // We can only proceed with executable or dynamic binary.
        if header.e_type != ET_EXEC && header.e_type != ET_DYN {
            return Err("the binary is not executable.");
        }

        // Check that the ELF is for RISC-V
        if header.e_machine != EM_RISCV {
            return Err("the binary is not for RISC-V.");
        }

        Ok(())
    }

    fn find_interpreter(&self) -> Option<&str> {
        let header = unsafe { &*(self.memory as *const libc::Elf64_Ehdr) };

        for i in 0..(header.e_phnum as usize) {
            let h = unsafe { &*((self.memory as usize + header.e_phoff as usize + header.e_phentsize as usize * i) as *const libc::Elf64_Phdr) };

            if h.p_type == libc::PT_INTERP {
                let content = unsafe {std::slice::from_raw_parts((self.memory as usize + h.p_offset as usize) as *const u8, h.p_filesz as usize)};
                return Some(CStr::from_bytes_with_nul(content).unwrap().to_str().unwrap());
            }
        }
        None
    }

    unsafe fn load_image(&self, load_addr: &mut ureg, brk: &mut ureg) -> ureg {
        let ehdr = self.ehdr();

        // Scan the bounds of the image.
        let mut loaddr = ureg::max_value();
        let mut hiaddr = 0;
        for h in self.phdr() {
            if h.p_type == libc::PT_LOAD {
                loaddr = std::cmp::min(loaddr, h.p_vaddr);
                hiaddr = std::cmp::max(hiaddr, h.p_vaddr + h.p_memsz);
            }
        }

        loaddr &=! 4095;
        hiaddr = (hiaddr + 4095) &! 4095;

        // For dynamic binaries, we need to allocate a location for it.
        let bias = if ehdr.e_type == ET_DYN {
            let map = libc::mmap(
                std::ptr::null_mut(),
                (hiaddr - loaddr) as _,
                libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0
            );
            if map == libc::MAP_FAILED {
                panic!("mmap failed while loading");
            }
            map as usize as ureg - loaddr
        } else {
            let map = libc::mmap(
                loaddr as usize as _,
                (hiaddr - loaddr) as _,
                libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED, -1 ,0
            );
            if map == libc::MAP_FAILED {
                panic!("mmap failed while loading");
            }
            0
        };

        for h in self.phdr() {
            if h.p_type == libc::PT_LOAD {
                // size in memory cannot be smaller than size in file
                if h.p_filesz > h.p_memsz {
                    panic!("invalid elf file: constraint p_filesz <= p_memsz is not satisified");
                }

                let vaddr_map_end = h.p_vaddr + h.p_filesz;
                let vaddr_end = h.p_vaddr + h.p_memsz;
                let mut file_offset = h.p_offset;
                let page_start = h.p_vaddr &! 4095;
                let map_end = vaddr_map_end &! 4095;
                let page_end = (vaddr_end + 4095) &! 4095;

                let mut prot = 0;
                if (h.p_flags & PF_R) != 0 { prot |= libc::PROT_READ };
                if (h.p_flags & PF_W) != 0 { prot |= libc::PROT_WRITE };
                if (h.p_flags & PF_X) != 0 { prot |= libc::PROT_READ };

                // First page is not aligned, we need t adjust it so that it is aligned.
                if h.p_vaddr != page_start {
                    file_offset -= h.p_vaddr - page_start;
                }

                // Map until map_end.
                if map_end != page_start {
                    let map = libc::mmap(
                        (bias + page_start) as usize as _,
                        (map_end - page_start) as _,
                        prot, libc::MAP_PRIVATE | libc::MAP_FIXED, self.fd, file_offset as _
                    );
                    if map == libc::MAP_FAILED {
                        panic!("mmap failed while loading");
                    }
                }

                // For those beyond map_end, we need those beyond filesz to be zero. As we know the memory location will
                // initially be zero, we just make them writable and copy the remaining part across.
                if map_end != page_end {
                    let rem_ptr = (bias + map_end) as usize as _;

                    // Make it writable.
                    libc::mprotect(rem_ptr, (page_end - map_end) as usize, libc::PROT_READ | libc::PROT_WRITE);

                    // Copy across.
                    libc::memcpy(
                        rem_ptr,
                        (self.memory as usize + file_offset as usize + (map_end - page_start) as usize) as _,
                        (vaddr_map_end - map_end) as usize
                    );

                    // Change protection if it shouldn't be writable.
                    if (h.p_flags & PF_W) == 0 {
                        libc::mprotect(rem_ptr, (page_end - map_end) as usize, prot);
                    }
                }

                // Set brk to the address past the last program segment.
                if vaddr_end > *brk {
                    *brk = vaddr_end;
                }
            }
        }

        // Return information needed by the caller.
        *load_addr = bias + loaddr;
        *brk = bias + ((*brk + 4095) &! 4095);
        return bias + ehdr.e_entry;
    }

    unsafe fn load_elf(&self, sp: &mut ureg) -> ureg {
        self.validate_elf().unwrap();

        let mut load_addr = 0;
        let mut brk = 0;
        let entry = self.load_image(&mut load_addr, &mut brk);
        let mut actual_entry = entry;
        let mut interp_addr = 0;

        // If an interpreter exists, load it as well.
        if let Some(interp) = self.find_interpreter() {
            let interp_file = Loader::new(interp.as_ref()).unwrap();
            interp_file.validate_elf().unwrap();

            if interp_file.find_interpreter().is_some() {
                panic!("interpreter cannot require other interpreters");
            }

            let mut interp_brk = 0;
            actual_entry = interp_file.load_image(&mut interp_addr, &mut interp_brk);
        }

        // Setup brk.
        brk = (brk + 4095) &! 4095;
        super::syscall::init_brk(brk);

        let mut push = |value: ureg| {
            *sp -= std::mem::size_of::<ureg>() as ureg;
            *(*sp as usize as *mut ureg) = value;
        };

        // Setup auxillary vectors.
        let header = &*(self.memory as *const libc::Elf64_Ehdr);
        push(load_addr + header.e_phoff);
        push(abi::AT_PHDR);
        push(header.e_phentsize as _);
        push(abi::AT_PHENT);
        push(header.e_phnum as _);
        push(abi::AT_PHNUM);
        push(4096);
        push(abi::AT_PAGESZ);
        push(interp_addr);
        push(abi::AT_BASE);
        push(0);
        push(abi::AT_FLAGS);
        push(entry);
        push(abi::AT_ENTRY);

        actual_entry
    }

    unsafe fn load_bin(&self, location: ureg) {
        libc::memcpy(location as usize as _, self.memory, self.file_size as _);
    }
}

pub unsafe fn load(file: &Loader, ctx: &mut crate::riscv::interp::Context, args: &mut dyn Iterator<Item=String>) {
    if crate::get_flags().user_only {
        // Set sp to be the highest possible address.
        let mut sp: ureg = 0x7fff0000;
        let map = libc::mmap(
            (sp - 0x800000) as usize as _,
            0x800000,
            libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED, -1, 0
        );
        if map == libc::MAP_FAILED {
            panic!("mmap failed while loading");
        }

        let sp_alloc = |sp: &mut ureg, size: usize| {
            *sp -= size as ureg;
            std::slice::from_raw_parts_mut(*sp as usize as _, size)
        };

        // This contains (guest) pointers to all argument strings annd environment variables.
        let mut env_pointers = Vec::new();
        let mut arg_pointers = Vec::new();

        // Copy all environment variables into guest user space.
        for (var_k, var_v) in std::env::vars() {
            sp_alloc(&mut sp, 1)[0] = 0;
            sp_alloc(&mut sp, var_v.len()).copy_from_slice(var_v.as_bytes());
            sp_alloc(&mut sp, 1)[0] = '=' as u8;
            sp_alloc(&mut sp, var_k.len()).copy_from_slice(var_k.as_bytes());
            env_pointers.push(sp);
        }

        // Copy all arguments into guest user space.
        for arg in args {
            sp_alloc(&mut sp, 1)[0] = 0;
            sp_alloc(&mut sp, arg.len()).copy_from_slice(arg.as_bytes());
            arg_pointers.push(sp);
        }

        // Align the stack to 8-byte boundary.
        sp &= !7;

        let push = |sp: &mut ureg, value: ureg| {
            *sp -= std::mem::size_of::<ureg>() as ureg;
            *(*sp as usize as *mut ureg) = value;
        };

        // Random data
        let mut rng = rand::rngs::OsRng::new().unwrap();
        push(&mut sp, rng.next_u64());
        push(&mut sp, rng.next_u64());
        push(&mut sp, rng.next_u64());
        push(&mut sp, rng.next_u64());
        let random_data = sp;

        // Setup auxillary vectors.
        push(&mut sp, 0);
        push(&mut sp, abi::AT_NULL);

        // Initialize context, and set up ELF-specific auxillary vectors.
        ctx.pc = file.load_elf(&mut sp);

        push(&mut sp, libc::getuid() as _);
        push(&mut sp, abi::AT_UID);
        push(&mut sp, libc::geteuid() as _);
        push(&mut sp, abi::AT_EUID);
        push(&mut sp, libc::getgid() as _);
        push(&mut sp, abi::AT_GID);
        push(&mut sp, libc::getegid() as _);
        push(&mut sp, abi::AT_EGID);
        push(&mut sp, 0);
        push(&mut sp, abi::AT_HWCAP);
        push(&mut sp, 100);
        push(&mut sp, abi::AT_CLKTCK);
        push(&mut sp, random_data);
        push(&mut sp, abi::AT_RANDOM);

        // fill in environ, last is nullptr
        push(&mut sp, 0);
        for v in env_pointers.into_iter().rev() { push(&mut sp, v) };

        // fill in argv, last is nullptr
        push(&mut sp, 0);
        for &v in arg_pointers.iter().rev() { push(&mut sp, v) };

        // set argc
        push(&mut sp, arg_pointers.len() as _);

        // sp
        ctx.registers[2] = sp;
        // libc adds this value into exit hook, so we need to make sure it is zero.
        ctx.registers[10] = 0;
        ctx.prv = 0;
    } else {
        // Allocate a 1G memory for physical address, starting at 0x200000.
        let map = libc::mmap(
            0x200000 as _,
            0x40000000 - 0x200000,
            libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED, -1, 0
        );
        if map == libc::MAP_FAILED {
            panic!("mmap failed while loading");
        }

        let size = file.file_size;
        file.load_bin(0x200000);
        Loader::new("dt".as_ref()).unwrap().load_bin(0x200000 + size);

        // a0 is the current hartid
        ctx.registers[10] = 0;
        // a1 should be the device tree
        ctx.registers[11] = 0x200000 + size;
        ctx.pc = 0x200000;
        ctx.prv = 1;
    }
}
