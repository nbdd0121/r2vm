use core::fmt;
use core::convert::TryFrom;
use num_traits::FromPrimitive;

#[repr(u16)]
#[derive(Clone, Copy, FromPrimitive)]
pub enum Csr {
    Fflags = 0x001,
    Frm = 0x002,
    Fcsr = 0x003,

    Cycle = 0xC00,
    Time = 0xC01,
    Instret = 0xC02,
    
    // These CSRs are Rv32I only, and they are considered invalid in RV64I
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
}

impl Csr {
    /// Get the minimal privilege level required to access the CSR
    pub fn min_prv_level(self) -> u8 {
        (((self as u16) >> 8) & 0b11) as u8
    }

    pub fn readonly(self) -> bool {
        ((self as u16) >> 10) & 0b11 == 0b11
    }
}

impl TryFrom<u16> for Csr {
    type Error = ();
    fn try_from(value: u16) -> Result<Csr, ()> {
        match Csr::from_u64(value as u64) {
            Some(v) => Ok(v),
            None => Err(())
        }
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
        })
    }
}
