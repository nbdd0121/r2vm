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

#[derive(Clone, Copy)]
pub struct PageWalkResult {
    /// The leaf PTE entry.
    pub pte: u64,
    /// Size of this page. 4K page is 0, 2M page is 1, 1G page is 2.
    pub granularity: u8,
}

impl PageWalkResult {
    #[inline]
    pub fn invalid() -> Self {
        PageWalkResult { pte: 0, granularity: 0 }
    }

    #[inline]
    pub fn from_4k_pte(pte: u64) -> Self {
        PageWalkResult { pte, granularity: 0 }
    }

    /// Make up a fake 4K PTE for those that does not support multi-graunularity.
    ///
    /// The global bit is also synthesised accordingly.
    #[inline]
    pub fn synthesise_4k(&self, vaddr: u64) -> Self {
        let page_index_within_superpage = (vaddr >> 12) & ((1 << (self.granularity * 9)) - 1);
        let pte = self.pte | page_index_within_superpage << 10;
        Self { pte, granularity: 0 }
    }
}

/// Walk the page table under SV39. We don't support SV48 at the moment.
pub fn walk_page(satp: u64, vpn: u64, mut read_mem: impl FnMut(u64) -> u64) -> PageWalkResult {
    // Check if the address is canonical.
    if (((vpn << (64 - 27)) as i64) >> (64 - 27 - 12)) as u64 >> 12 != vpn {
        return PageWalkResult::invalid();
    }

    let mut ppn = satp & ((1u64 << 44) - 1);
    let mut global = false;

    for i in 0..3 {
        let bits_left = 18 - i * 9;
        let index = (vpn >> bits_left) & 511;
        let pte_addr = (ppn << 12) + index * 8;
        let mut pte = read_mem(pte_addr);
        ppn = pte >> 10;

        // Check for invalid PTE
        if pte & PTE_V == 0 {
            return PageWalkResult::invalid();
        }

        // Check for malformed PTEs
        if pte & (PTE_R | PTE_W | PTE_X) == PTE_W {
            return PageWalkResult::invalid();
        }
        if pte & (PTE_R | PTE_W | PTE_X) == PTE_W | PTE_X {
            return PageWalkResult::invalid();
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
            return PageWalkResult::invalid();
        }

        if global {
            pte |= PTE_G;
        }
        return PageWalkResult { pte, granularity: 2 - i };
    }

    // Invalid if reached here
    PageWalkResult::invalid()
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
