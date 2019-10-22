extern crate fnv;

use fnv::FnvHashMap;
use std::convert::TryInto;

mod prop_value;
pub use prop_value::{PropConversionError, PropValue};

#[derive(Clone)]
pub struct Node {
    pub name: String,
    pub properties: Vec<Prop>,
    pub child: Vec<Node>,
}

impl Node {
    pub fn new(name: impl Into<String>) -> Self {
        Node { name: name.into(), properties: Vec::new(), child: Vec::new() }
    }

    /// Convience method for append a property with given name and value to the current node.
    pub fn add_prop(
        &mut self,
        name: impl Into<String>,
        value: impl TryInto<PropValue, Error = PropConversionError>,
    ) {
        self.properties.push(Prop::new(name, value));
    }

    /// Convience method for adding a child node to the current node.
    pub fn add_node(&mut self, name: impl Into<String>) -> &mut Node {
        self.child.push(Self::new(name));
        self.child.last_mut().unwrap()
    }
}

#[derive(Clone)]
pub struct Prop {
    pub name: String,
    pub value: PropValue,
}

impl Prop {
    pub fn new(
        name: impl Into<String>,
        value: impl TryInto<PropValue, Error = PropConversionError>,
    ) -> Prop {
        let value = value.try_into().unwrap();
        Prop { name: name.into(), value }
    }
}

const FDT_MAGIC: u32 = 0xd00dfeed;
const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;

struct Encoder<'a> {
    dt_struct: Vec<u8>,
    dt_strings: Vec<u8>,
    string_map: FnvHashMap<&'a str, usize>,
}

impl<'a> Encoder<'a> {
    fn push(&mut self, value: u32) {
        self.dt_struct.extend_from_slice(&value.to_be_bytes());
    }

    fn align_to_word(&mut self) {
        let pad_bytes = ((self.dt_struct.len() + 3) & !3) - self.dt_struct.len();
        for _ in 0..pad_bytes {
            self.dt_struct.push(0);
        }
    }

    fn add_string(&mut self, str: &'a str) -> usize {
        use std::collections::hash_map::Entry;
        match self.string_map.entry(str) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(v) => {
                let len = self.dt_strings.len();
                self.dt_strings.extend_from_slice(&str.as_bytes());
                self.dt_strings.push(0);
                v.insert(len);
                len
            }
        }
    }

    fn encode_node(&mut self, node: &'a Node) {
        self.push(FDT_BEGIN_NODE);
        self.dt_struct.extend_from_slice(node.name.as_bytes());
        self.dt_struct.push(0);
        self.align_to_word();
        for prop in node.properties.iter() {
            self.push(FDT_PROP);
            self.push(prop.value.0.len() as u32);
            let nameoff = self.add_string(&prop.name);
            self.push(nameoff as u32);
            self.dt_struct.extend_from_slice(&prop.value.0);
            self.align_to_word();
        }
        for child in node.child.iter() {
            self.encode_node(child);
        }
        self.push(FDT_END_NODE);
    }
}

pub fn encode(node: &Node) -> Vec<u8> {
    let mut enc = Encoder {
        dt_struct: Vec::new(),
        dt_strings: Vec::new(),
        string_map: FnvHashMap::default(),
    };
    enc.encode_node(node);
    enc.push(9);
    let mut vec = Vec::new();
    vec.extend_from_slice(&FDT_MAGIC.to_be_bytes());
    let total_size = enc.dt_struct.len() + enc.dt_strings.len() + 16 + 40;
    vec.extend_from_slice(&(total_size as u32).to_be_bytes());
    vec.extend_from_slice(&56u32.to_be_bytes());
    vec.extend_from_slice(&(enc.dt_struct.len() as u32 + 56).to_be_bytes());
    vec.extend_from_slice(&40u32.to_be_bytes());
    vec.extend_from_slice(&17u32.to_be_bytes());
    vec.extend_from_slice(&16u32.to_be_bytes());
    vec.extend_from_slice(&0u32.to_be_bytes());
    vec.extend_from_slice(&(enc.dt_strings.len() as u32).to_be_bytes());
    vec.extend_from_slice(&(enc.dt_struct.len() as u32).to_be_bytes());
    assert_eq!(vec.len(), 40);
    vec.extend_from_slice(&0u64.to_be_bytes());
    vec.extend_from_slice(&0u64.to_be_bytes());
    vec.append(&mut enc.dt_struct);
    vec.append(&mut enc.dt_strings);
    vec
}
