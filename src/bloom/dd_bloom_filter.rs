use bit_vec::BitVec;
use rand::{Rng, XorShiftRng};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::f64::consts;
use std::hash::Hash;
use std::marker::PhantomData;
use util;

const PRIME: u64 = 0xFFFF_FFFF_FFFF_FFC5;

/// A space-efficient probabilistic data structure for data deduplication in streams.
///
/// This particular implementation uses Biased Sampling to determine whether data is distinct.
/// Past work for data deduplication include Stable Bloom Filters, but it suffers from a high
/// false negative rate and slow convergence rate.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::BSBloomFilter;
///
/// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 70);
/// assert_eq!(filter.bit_count(), 10);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
pub struct BSBloomFilter<T> {
    bit_vec: BitVec,
    hashers: [SipHasher; 2],
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BSBloomFilter<T> {
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `BSBloomFilter` with `bit_count` bits per filter and a false
    /// positive probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    /// ```
    pub fn new(bit_count: usize, fpp: f64) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        let mut rng = XorShiftRng::new_unseeded();
        BSBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hashers: [
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            ],
            rng,
            bit_count,
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Inserts an element into the bloom filter and returns if it is distinct
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let hashes = util::get_hashes::<T, U>(&self.hashers, item);
        if !self.contains_hashes(hashes) {
            let bit_count = self.bit_count();

            (0..self.hasher_count).for_each(|index| {
                let index = index * bit_count + self.rng.gen_range(0, bit_count);
                self.bit_vec.set(index, false);
            });

            for index in 0..self.hasher_count {
                let mut offset = (index as u64).wrapping_mul(hashes[1]) % PRIME;
                offset = hashes[0].wrapping_add(offset);
                offset %= self.bit_count as u64;
                offset += (index * self.bit_count) as u64;

                self.bit_vec.set(offset as usize, true);
            }
        }
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let hashes = util::get_hashes::<T, U>(&self.hashers, item);
        self.contains_hashes(hashes)
    }

    fn contains_hashes(&self, hashes: [u64; 2]) -> bool {
        (0..self.hasher_count).all(|index| {
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % PRIME;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_count as u64;
            offset += (index * self.bit_count) as u64;
            self.bit_vec[offset as usize]
        })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.len(), 70);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bits in each partition in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.bit_count(), 10);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.hasher_count(), 7);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Clears the bloom filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.bit_vec.set_all(false)
    }

    /// Returns the number of set bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bit_vec.count_ones()
    }

    /// Returns the number of unset bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let mut filter = BSBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 63);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }
}

/// A space-efficient probabilistic data structure for data deduplication in streams.
///
/// This particular implementation uses Biased Sampling with Single Deletion to determine whether
/// data is distinct. Past work for data deduplication include Stable Bloom Filters, but it suffers
/// from a high false negative rate and slow convergence rate.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::BSSDBloomFilter;
///
/// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 70);
/// assert_eq!(filter.bit_count(), 10);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
pub struct BSSDBloomFilter<T> {
    bit_vec: BitVec,
    hashers: [SipHasher; 2],
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BSSDBloomFilter<T> {
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `BSSDBloomFilter` with `bit_count` bits per filter and a false
    /// positive probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    /// ```
    pub fn new(bit_count: usize, fpp: f64) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        let mut rng = XorShiftRng::new_unseeded();
        BSSDBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hashers: [
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            ],
            rng,
            bit_count,
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Inserts an element into the bloom filter and returns if it is distinct
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        if !self.contains(item) {
            let hashes = util::get_hashes::<T, U>(&self.hashers, item);
            let bit_count = self.bit_count();

            let filter_index = self.rng.gen_range(0, self.hasher_count);
            let index = self.rng.gen_range(0, self.bit_count);

            self.bit_vec.set(filter_index * bit_count + index, false);

            for index in 0..self.hasher_count {
                let mut offset = (index as u64).wrapping_mul(hashes[1]) % PRIME;
                offset = hashes[0].wrapping_add(offset);
                offset %= self.bit_count as u64;
                offset += (index * self.bit_count) as u64;

                self.bit_vec.set(offset as usize, true);
            }
        }
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let hashes = util::get_hashes::<T, U>(&self.hashers, item);
        (0..self.hasher_count).all(|index| {
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % PRIME;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_count as u64;
            offset += (index * self.bit_count) as u64;
            self.bit_vec[offset as usize]
        })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.len(), 70);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bits in each partition in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.bit_count(), 10);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.hasher_count(), 7);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Clears the bloom filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.bit_vec.set_all(false)
    }

    /// Returns the number of set bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bit_vec.count_ones()
    }

    /// Returns the number of unset bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 63);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }
}

/// A space-efficient probabilistic data structure for data deduplication in streams.
///
/// This particular implementation uses Randomized Load Balanced Biased Sampling to determine
/// whether data is distinct. Past work for data deduplication include Stable Bloom Filters, but it
/// suffers from a high false negative rate and slow convergence rate.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::RLBSBloomFilter;
///
/// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 70);
/// assert_eq!(filter.bit_count(), 10);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
pub struct RLBSBloomFilter<T> {
    bit_vecs: Vec<BitVec>,
    hashers: [SipHasher; 2],
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> RLBSBloomFilter<T> {
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `RLBSBloomFilter` with `bit_count` bits per filter and a false
    /// positive probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    /// ```
    pub fn new(bit_count: usize, fpp: f64) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        let mut rng = XorShiftRng::new_unseeded();
        RLBSBloomFilter {
            bit_vecs: vec![BitVec::new(bit_count); hasher_count],
            hashers: [
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
                SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            ],
            rng,
            bit_count,
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Inserts an element into the bloom filter and returns if it is distinct
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        if !self.contains(item) {
            let hashes = util::get_hashes::<T, U>(&self.hashers, item);
            let bit_count = self.bit_count();

            (0..self.hasher_count).for_each(|filter_index| {
                let prob = self.bit_vecs[filter_index].count_ones() as f64 / bit_count as f64;
                let index = self.rng.gen_range(0, bit_count);
                if self.rng.gen::<f64>() < prob {
                    self.bit_vecs[filter_index].set(index, false);
                }
            });

            for filter_index in 0..self.hasher_count {
                let mut offset = (filter_index as u64).wrapping_mul(hashes[1]) % PRIME;
                offset = hashes[0].wrapping_add(offset);
                offset %= self.bit_count as u64;

                self.bit_vecs[filter_index].set(offset as usize, true);
            }
        }
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let hashes = util::get_hashes::<T, U>(&self.hashers, item);
        (0..self.hasher_count).all(|filter_index| {
            let mut offset = (filter_index as u64).wrapping_mul(hashes[1]) % PRIME;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_count as u64;
            self.bit_vecs[filter_index][offset as usize]
        })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.len(), 70);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_count * self.hasher_count
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bits in each partition in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.bit_count(), 10);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// assert_eq!(filter.hasher_count(), 7);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Clears the bloom filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.bit_vecs
            .iter_mut()
            .for_each(|bit_vec| bit_vec.set_all(false));
    }

    /// Returns the number of set bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bit_vecs
            .iter()
            .map(|bit_vec| bit_vec.count_ones())
            .sum()
    }

    /// Returns the number of unset bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 63);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vecs
            .iter()
            .map(|bit_vec| bit_vec.count_zeros())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::BSBloomFilter;
    use super::BSSDBloomFilter;
    use super::RLBSBloomFilter;

    #[test]
    fn test_bs() {
        let mut filter = BSBloomFilter::<String>::new(10, 0.01);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 63);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 70);
        assert_eq!(filter.bit_count(), 10);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_bssd() {
        let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 63);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 70);
        assert_eq!(filter.bit_count(), 10);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_rlbs() {
        let mut filter = RLBSBloomFilter::<String>::new(10, 0.01);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 63);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 70);
        assert_eq!(filter.bit_count(), 10);
        assert_eq!(filter.hasher_count(), 7);
    }
}
