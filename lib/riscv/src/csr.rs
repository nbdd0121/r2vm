use core::fmt;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Csr(pub u16);

// We want these to look like as enum values, so keep cases like this.
#[allow(non_upper_case_globals)]
impl Csr {
    pub const Fflags: Csr = Csr(0x001);
    pub const Frm: Csr = Csr(0x002);
    pub const Fcsr: Csr = Csr(0x003);

    pub const Cycle: Csr = Csr(0xC00);
    pub const Time: Csr = Csr(0xC01);
    pub const Instret: Csr = Csr(0xC02);

    // These CSRs are Rv32I only, and they are considered invalid in RV64I
    pub const Cycleh: Csr = Csr(0xC80);
    pub const Timeh: Csr = Csr(0xC81);
    pub const Instreth: Csr = Csr(0xC82);

    pub const Sstatus: Csr = Csr(0x100);
    pub const Sie: Csr = Csr(0x104);
    pub const Stvec: Csr = Csr(0x105);
    pub const Scounteren: Csr = Csr(0x106);
    pub const Sscratch: Csr = Csr(0x140);
    pub const Sepc: Csr = Csr(0x141);
    pub const Scause: Csr = Csr(0x142);
    pub const Stval: Csr = Csr(0x143);
    pub const Sip: Csr = Csr(0x144);
    pub const Satp: Csr = Csr(0x180);

    pub const Mvendorid: Csr = Csr(0xF11);
    pub const Marchid: Csr = Csr(0xF12);
    pub const Mimpid: Csr = Csr(0xF13);
    pub const Mhartid: Csr = Csr(0xF14);
    pub const Mstatus: Csr = Csr(0x300);
    pub const Misa: Csr = Csr(0x301);
    pub const Medeleg: Csr = Csr(0x302);
    pub const Mideleg: Csr = Csr(0x303);
    pub const Mie: Csr = Csr(0x304);
    pub const Mtvec: Csr = Csr(0x305);
    pub const Mcounteren: Csr = Csr(0x306);
    pub const Mscratch: Csr = Csr(0x340);
    pub const Mepc: Csr = Csr(0x341);
    pub const Mcause: Csr = Csr(0x342);
    pub const Mtval: Csr = Csr(0x343);
    pub const Mip: Csr = Csr(0x344);

    pub const Mcycle: Csr = Csr(0xB00);
    pub const Mtime: Csr = Csr(0xB01);
    pub const Minstret: Csr = Csr(0xB02);
}

impl Csr {
    /// Get the minimal privilege level required to access the CSR
    pub fn min_prv_level(self) -> u8 {
        ((self.0 >> 8) & 0b11) as u8
    }

    pub fn readonly(self) -> bool {
        (self.0 >> 10) & 0b11 == 0b11
    }
}

impl fmt::Display for Csr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(match *self {
            Csr::Fflags => "fflags",
            Csr::Frm => "frm",
            Csr::Fcsr => "fcsr",
            Csr::Cycle => "cycle",
            Csr::Time => "time",
            Csr::Instret => "instret",
            Csr::Cycleh => "cycleh",
            Csr::Timeh => "timeh",
            Csr::Instreth => "instreth",
            Csr::Sstatus => "sstatus",
            Csr::Sie => "sie",
            Csr::Stvec => "stvec",
            Csr::Scounteren => "scounteren",
            Csr::Sscratch => "sscratch",
            Csr::Sepc => "sepc",
            Csr::Scause => "scause",
            Csr::Stval => "stval",
            Csr::Sip => "sip",
            Csr::Satp => "satp",
            v => return write!(f, "#0x{:x}", v.0),
        })
    }
}
