use super::replacement_policy::{Fifo, ReplacementPolicy};

pub struct CacheSet<T, R: ?Sized = Fifo> {
    arr: Box<[Option<T>]>,
    rp: R,
}

impl<T, R> std::ops::Index<usize> for CacheSet<T, R> {
    type Output = Option<T>;

    fn index(&self, index: usize) -> &Option<T> {
        &self.arr[index]
    }
}

impl<T, R> std::ops::IndexMut<usize> for CacheSet<T, R> {
    fn index_mut(&mut self, index: usize) -> &mut Option<T> {
        &mut self.arr[index]
    }
}

impl<T, R: ReplacementPolicy> CacheSet<T, R> {
    pub fn new(size: usize) -> Self {
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(None);
        }
        CacheSet { arr: vec.into_boxed_slice(), rp: R::new(size) }
    }

    /// Search in the cache. Returns found entry and an insertion/overwrite pointer.
    pub fn find(&mut self, mut matcher: impl FnMut(&mut T) -> bool) -> (Option<&mut T>, usize) {
        for i in 0..self.arr.len() {
            if let Some(entry) = &mut self.arr[i] {
                if !matcher(entry) {
                    continue;
                }

                // Due to lifetime rules, entry does not live long enough, so we have to
                // re-borrow here.
                return (self.arr[i].as_mut(), i);
            }
        }
        (None, self.rp.select())
    }

    pub fn remove(&mut self, ptr: usize) -> Option<T> {
        self.rp.invalidate(ptr);
        std::mem::replace(&mut self.arr[ptr], None)
    }

    pub fn insert(&mut self, insert_ptr: usize, insert: T) -> Option<T> {
        self.rp.insert(insert_ptr);
        std::mem::replace(&mut self.arr[insert_ptr], Some(insert))
    }

    pub fn retain(&mut self, mut filter: impl FnMut(&mut T) -> bool) {
        let associativity = self.arr.len();
        for i in 0..associativity {
            match self.arr[i] {
                None => continue,
                Some(ref mut entry) => {
                    if filter(entry) {
                        continue;
                    }
                }
            }
            self.arr[i] = None;
            self.rp.invalidate(i);
        }
    }
}
