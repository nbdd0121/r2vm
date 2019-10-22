//! MMU-related helper functions and constants.

pub const PTE_V: u64 = 0x01;
pub const PTE_R: u64 = 0x02;
pub const PTE_W: u64 = 0x04;
pub const PTE_X: u64 = 0x08;
pub const PTE_U: u64 = 0x10;
pub const PTE_G: u64 = 0x20;
pub const PTE_A: u64 = 0x40;
pub const PTE_D: u64 = 0x80;

/// Type of access. This excludes STATUS, PRV and other states that may influence permission check.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum AccessType {
    Read,
    Write,
    Execute,
}

/// Walk the page table under SV39. We don't support SV48 at the moment.
pub fn walk_page(satp: u64, vpn: u64, mut read_mem: impl FnMut(u64) -> u64) -> u64 {
    // Check if the address is canonical.
    if (((vpn << (64 - 27)) as i64) >> (64 - 27 - 12)) as u64 >> 12 != vpn {
        return 0;
    }

    let mut ppn = satp & ((1u64 << 44) - 1);
    let mut global = false;

    for i in 0..3 {
        let bits_left = 18 - i * 9;
        let index = (vpn >> bits_left) & 511;
        let pte_addr = (ppn << 12) + index * 8;
        let pte = read_mem(pte_addr);
        ppn = pte >> 10;

        // Check for invalid PTE
        if pte & PTE_V == 0 {
            return 0;
        }

        // Check for malformed PTEs
        if pte & (PTE_R | PTE_W | PTE_X) == PTE_W {
            return 0;
        }
        if pte & (PTE_R | PTE_W | PTE_X) == PTE_W | PTE_X {
            return 0;
        }

        // A global bit will cause the page to be global regardless if this is leaf.
        if pte & PTE_G != 0 {
            global = true
        }

        // Not leaf yet
        if pte & (PTE_R | PTE_W | PTE_X) == 0 {
            continue;
        }

        // Check for misaligned huge page
        if ppn & ((1 << bits_left) - 1) != 0 {
            return 0;
        }

        // Synthesis a 4K PTE
        let ppn = ppn | (vpn & ((1 << bits_left) - 1));
        return ppn << 10 | pte & ((1 << 10) - 1) | (if global { PTE_G } else { 0 });
    }

    // Invalid if reached here
    0
}

pub fn check_permission(pte: u64, access: AccessType, prv: u8, status: u64) -> Result<(), ()> {
    if pte & PTE_V == 0 {
        return Err(());
    }

    if prv == 0 {
        if pte & PTE_U == 0 {
            return Err(());
        }
    } else {
        if pte & PTE_U != 0 && status & (1 << 18) == 0 {
            return Err(());
        }
    }

    if pte & PTE_A == 0 {
        return Err(());
    }

    match access {
        AccessType::Read => {
            if pte & PTE_R == 0 && (pte & PTE_X == 0 || status & (1 << 19) == 0) {
                return Err(());
            }
        }
        AccessType::Write => {
            if pte & PTE_W == 0 || pte & PTE_D == 0 {
                return Err(());
            }
        }
        AccessType::Execute => {
            if pte & PTE_X == 0 {
                return Err(());
            }
        }
    }

    Ok(())
}
