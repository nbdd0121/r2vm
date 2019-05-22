use std::fmt;

#[repr(u16)]
#[derive(Clone, Copy)]
pub enum Csr {
    Fflags = 0x001,
    Frm = 0x002,
    Fcsr = 0x003,

    Cycle = 0xC00,
    Time = 0xC01,
    Instret = 0xC02,
    
    /* Rv32I only */
    Cycleh = 0xC80,
    Timeh = 0xC81,
    Instreth = 0xC82,

    Sstatus = 0x100,
    Sie = 0x104,
    Stvec = 0x105,
    Scounteren = 0x106,

    Sscratch = 0x140,
    Sepc = 0x141,
    Scause = 0x142,
    Stval = 0x143,
    Sip = 0x144,

    Satp = 0x180,

    #[doc(hidden)]
    __Nonexhaustive,
}

impl From<u16> for Csr {
    fn from(value: u16) -> Csr {
        unsafe { std::mem::transmute(value) }
    }
}

impl fmt::Display for Csr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(match self {
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
            v => return write!(f, "{}", *v as u16),
        })
    }
}
