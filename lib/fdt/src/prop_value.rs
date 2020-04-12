use std::convert::{TryFrom, TryInto};
use std::error::Error;

/// Error indicating failure to convert property to the desired type.
#[derive(Debug)]
pub struct PropConversionError;

impl<T: Error> From<T> for PropConversionError {
    fn from(_: T) -> Self {
        PropConversionError
    }
}

/// A wrapper around underlying bytes of a device tree property.
/// Use provided `{TryFrom}` trait implementations to convert types from and to this type.
#[derive(Clone)]
pub struct PropValue(pub Box<[u8]>);

// Empty
impl TryFrom<()> for PropValue {
    type Error = PropConversionError;
    fn try_from(_: ()) -> Result<Self, Self::Error> {
        Ok(PropValue(Box::new([])))
    }
}

impl TryFrom<&PropValue> for () {
    type Error = PropConversionError;
    fn try_from(raw: &PropValue) -> Result<Self, Self::Error> {
        if raw.0.len() != 0 {
            return Err(PropConversionError);
        }
        Ok(())
    }
}

// String
impl TryFrom<&str> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let cstr = std::ffi::CString::new(value)?;
        Ok(PropValue(cstr.into_bytes_with_nul().into_boxed_slice()))
    }
}

impl<'a> TryFrom<&'a PropValue> for &'a str {
    type Error = PropConversionError;
    fn try_from(raw: &'a PropValue) -> Result<Self, Self::Error> {
        let cstr = std::ffi::CStr::from_bytes_with_nul(&raw.0)?;
        Ok(cstr.to_str()?)
    }
}

impl TryFrom<&[&str]> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: &[&str]) -> Result<Self, Self::Error> {
        let mut vec = Vec::new();
        for v in value {
            vec.extend_from_slice(v.as_bytes());
            vec.push(0);
        }
        Ok(PropValue(vec.into_boxed_slice()))
    }
}

// Cell
impl TryFrom<u32> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(PropValue(Box::new(value.to_be_bytes())))
    }
}

impl TryFrom<&PropValue> for u32 {
    type Error = PropConversionError;
    fn try_from(raw: &PropValue) -> Result<Self, Self::Error> {
        if raw.0.len() != 4 {
            return Err(PropConversionError);
        }
        Ok(u32::from_be_bytes(raw.0[..].try_into().unwrap()))
    }
}

impl TryFrom<u64> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(PropValue(Box::new(value.to_be_bytes())))
    }
}

impl TryFrom<&PropValue> for u64 {
    type Error = PropConversionError;
    fn try_from(raw: &PropValue) -> Result<Self, Self::Error> {
        if raw.0.len() != 8 {
            return Err(PropConversionError);
        }
        Ok(u64::from_be_bytes(raw.0[..].try_into().unwrap()))
    }
}

// Cells
impl TryFrom<&[u32]> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: &[u32]) -> Result<Self, Self::Error> {
        let mut vec = Vec::with_capacity(value.len() * 4);
        for v in value {
            vec.extend_from_slice(&v.to_be_bytes());
        }
        Ok(PropValue(vec.into_boxed_slice()))
    }
}

impl TryFrom<&PropValue> for Box<[u32]> {
    type Error = PropConversionError;
    fn try_from(raw: &PropValue) -> Result<Self, Self::Error> {
        if raw.0.len() % 4 != 0 {
            return Err(PropConversionError);
        }
        let len = raw.0.len() / 4;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(u32::from_be_bytes(raw.0[i * 4..i * 4 + 4].try_into().unwrap()));
        }
        Ok(vec.into_boxed_slice())
    }
}

impl TryFrom<&[u64]> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: &[u64]) -> Result<Self, Self::Error> {
        let mut vec = Vec::with_capacity(value.len() * 8);
        for v in value {
            vec.extend_from_slice(&v.to_be_bytes());
        }
        Ok(PropValue(vec.into_boxed_slice()))
    }
}

impl TryFrom<&PropValue> for Box<[u64]> {
    type Error = PropConversionError;
    fn try_from(raw: &PropValue) -> Result<Self, Self::Error> {
        if raw.0.len() % 8 != 0 {
            return Err(PropConversionError);
        }
        let len = raw.0.len() / 8;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(u64::from_be_bytes(raw.0[i * 8..i * 8 + 8].try_into().unwrap()));
        }
        Ok(vec.into_boxed_slice())
    }
}

// Raw bytes
impl TryFrom<&[u8]> for PropValue {
    type Error = PropConversionError;
    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(PropValue(value.to_vec().into_boxed_slice()))
    }
}
