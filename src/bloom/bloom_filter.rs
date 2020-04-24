use crate::bit_vec::BitVec;
use crate::util;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::hash::Hash;
use std::marker::PhantomData;

/// A space-efficient probabilistic data structure to test for membership in a set.
///
/// At its core, a bloom filter is a bit array, initially all set to zero. `K` hash functions
/// map each element to `K` bits in the bit array. An element definitely does not exist in the
/// bloom filter if any of the `K` bits are unset. An element is possibly in the set if all of the
/// `K` bits are set. This particular implementation of a bloom filter uses two hash functions to
/// simulate `K` hash functions. Additionally, it operates on only one "slice" in order to have
/// predictable memory usage.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::BloomFilter;
///
/// let mut filter = BloomFilter::<String>::new(10, 0.01);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 96);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct BloomFilter<T> {
    bit_vec: BitVec,
    hashers: [SipHasher; 2],
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BloomFilter<T> {
    fn get_hasher_count(bit_count: usize, item_count: usize) -> usize {
        ((bit_count as f64) / (item_count as f64) * 2f64.ln()).ceil() as usize
    }

    /// Constructs a new, empty `BloomFilter` with an estimated max capacity of `item_count` items
    /// and a maximum false positive probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::new(10, 0.01);
    /// ```
    pub fn new(item_count: usize, fpp: f64) -> Self {
        let bit_count = (-fpp.log2() * (item_count as f64) / 2f64.ln()).ceil() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: util::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits and an estimated max capacity
    /// of `item_count` items.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::from_item_count(100, 10);
    /// ```
    pub fn from_item_count(bit_count: usize, item_count: usize) -> Self {
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: util::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits and a maximum false positive
    /// probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::from_fpp(100, 0.01);
    /// ```
    pub fn from_fpp(bit_count: usize, fpp: f64) -> Self {
        let item_count = -(2f64.ln() * (bit_count as f64) / fpp.log2()).floor() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: util::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
            _marker: PhantomData,
        }
    }

    /// Inserts an element into the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::new(10, 0.01);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let hashes = util::get_hashes::<T, U>(&self.hashers, item);
        for index in 0..self.hasher_count {
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_vec.len() as u64;
            self.bit_vec.set(offset as usize, true);
        }
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::new(10, 0.01);
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
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_vec.len() as u64;
            self.bit_vec[offset as usize]
        })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::from_fpp(100, 0.01);
    ///
    /// assert_eq!(filter.len(), 100);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::from_fpp(100, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.bit_vec.is_empty()
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::new(10, 0.01);
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
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::new(10, 0.01);
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
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::from_fpp(100, 0.01);
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
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::from_fpp(100, 0.01);
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 93);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }

    /// Returns the estimated false positive probability of the bloom filter. This value will
    /// increase as more items are added.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::<String>::new(100, 0.01);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.01);
    /// ```
    pub fn estimated_fpp(&self) -> f64 {
        let single_fpp = self.bit_vec.count_ones() as f64 / self.bit_vec.len() as f64;
        single_fpp.powi(self.hasher_count as i32)
    }
}

impl<T> PartialEq for BloomFilter<T> {
    fn eq(&self, other: &BloomFilter<T>) -> bool {
        self.hasher_count == other.hasher_count
            && self.hashers[0].keys() == other.hashers[0].keys()
            && self.hashers[1].keys() == other.hashers[1].keys()
            && self.bit_vec == other.bit_vec
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;

    #[test]
    fn test_new() {
        let mut filter = BloomFilter::<String>::new(10, 0.01);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 89);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 96);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_from_fpp() {
        let mut filter = BloomFilter::<String>::from_fpp(100, 0.01);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 93);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 100);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_from_item_count() {
        let mut filter = BloomFilter::<String>::from_item_count(100, 10);

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 93);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 100);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_estimated_fpp() {
        let mut filter = BloomFilter::<String>::new(100, 0.01);
        assert!(filter.estimated_fpp() < std::f64::EPSILON);

        filter.insert("foo");

        let expected_fpp = (7f64 / 959f64).powi(7);
        assert!((filter.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_ser_de() {
        let mut filter = BloomFilter::<String>::new(100, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: BloomFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.hasher_count, de_filter.hasher_count);
        // SipHasher doesn't implement PartialEq, but it does implement Debug,
        // and its Debug impl does print all internal state.
        for i in 0..2 {
            assert_eq!(
                format!("{:?}", filter.hashers[i]),
                format!("{:?}", de_filter.hashers[i])
            );
        }
    }
}
