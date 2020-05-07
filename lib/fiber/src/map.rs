use fnv::FnvHashMap;
use parking_lot::Mutex;
use std::hash::Hash;

pub struct ConcurrentMap<K, V> {
    map: Mutex<FnvHashMap<K, Option<V>>>,
}

impl<K, V> ConcurrentMap<K, V> {
    pub fn new() -> Self {
        ConcurrentMap { map: Mutex::new(FnvHashMap::default()) }
    }
}

impl<K: Eq + Hash, V> ConcurrentMap<K, V> {
    pub fn with<R>(&self, key: K, f: impl FnOnce(&mut Option<V>) -> R) -> R {
        use std::collections::hash_map::Entry;

        let mut inner = self.map.lock();
        match inner.entry(key) {
            Entry::Occupied(mut entry) => {
                let r = f(entry.get_mut());
                if entry.get().is_none() {
                    entry.remove();
                }
                r
            }
            Entry::Vacant(entry) => {
                let mut buf = None;
                let r = f(&mut buf);
                if buf.is_some() {
                    entry.insert(buf);
                }
                r
            }
        }
    }
}
