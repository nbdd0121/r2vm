pub use fifo::Fifo;
pub use random::Random;

pub trait ReplacementPolicy {
    fn new(assoc: usize) -> Self
    where
        Self: Sized;

    /// Insert an new entry.
    fn insert(&mut self, index: usize);

    /// Mark an entry as invalidate.
    fn invalidate(&mut self, index: usize);

    /// Mark an entry as recently accessed.
    ///
    /// Due to the natural of R2VM's design, not all memory requests would reach the memory model,
    /// thus the replacement policy. If the policy would need `touch` to work, it must invalidate
    /// relevant entries from L0.
    ///
    /// To support these the ReplacementPolicy would need to report if they would want to capture
    /// accesses to certain indexes via `indexes_to_capture`. We currently haven't implemented
    /// the functionality just yet.
    fn touch(&mut self, index: usize);

    /// Select an entry for eviction
    fn select(&mut self) -> usize;

    /// Retrieve the indexes that `touch`ing them upon access is required to maintain simulation
    /// accuracy.
    ///
    /// This method only needs to report a difference. It only needs to report those indexes
    /// that are newly required after their last `insert` or `touch`.
    fn indexes_to_capture(&mut self) -> Vec<usize> {
        Vec::new()
    }
}

mod random {

    use super::ReplacementPolicy;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng as Rng;

    pub struct Random {
        valid: Box<[bool]>,
        rng: Rng,
    }

    impl ReplacementPolicy for Random {
        fn new(assoc: usize) -> Self {
            Self {
                valid: vec![false; assoc].into_boxed_slice(),
                rng: SeedableRng::seed_from_u64(0),
            }
        }

        fn insert(&mut self, index: usize) {
            self.valid[index] = true;
        }

        fn invalidate(&mut self, index: usize) {
            self.valid[index] = false;
        }

        fn touch(&mut self, _index: usize) {}

        fn select(&mut self) -> usize {
            for i in 0..self.valid.len() {
                if !self.valid[i] {
                    return i;
                }
            }
            self.rng.next_u32() as usize % self.valid.len()
        }
    }
}

mod fifo {

    use super::ReplacementPolicy;

    pub struct Fifo {
        // TODO: Optimise memory usage with bitset
        valid: Box<[bool]>,
        ptr: usize,
    }

    impl ReplacementPolicy for Fifo {
        fn new(assoc: usize) -> Self {
            Self { valid: vec![false; assoc].into_boxed_slice(), ptr: 0 }
        }

        fn insert(&mut self, index: usize) {
            if self.ptr == index {
                self.ptr = if self.ptr == self.valid.len() - 1 { 0 } else { self.ptr + 1 }
            }
            self.valid[index] = true;
        }

        fn invalidate(&mut self, index: usize) {
            self.valid[index] = false;
        }

        fn touch(&mut self, _index: usize) {}

        fn select(&mut self) -> usize {
            for i in 0..self.valid.len() {
                if !self.valid[i] {
                    return i;
                }
            }
            self.ptr
        }
    }
}
