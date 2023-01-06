use crate::bit_vec::BitVec;
use crate::{DoubleHasher, SipHasherBuilder};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct BloomFilter<T, B = SipHasherBuilder> {
    bit_vec: BitVec,
    hasher: DoubleHasher<T, B>,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BloomFilter<T> {
    /// Constructs a new, empty `BloomFilter` with an estimated max capacity of `item_count` items,
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
        Self::with_hashers(
            item_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits, and an estimated max capacity
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
        Self::from_item_count_with_hashers(
            bit_count,
            item_count,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits, and a maximum false positive
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
        Self::from_fpp_with_hashers(
            bit_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> BloomFilter<T, B>
where
    B: BuildHasher,
{
    fn get_hasher_count(bit_count: usize, item_count: usize) -> usize {
        ((bit_count as f64) / (item_count as f64) * 2f64.ln()).ceil() as usize
    }

    /// Constructs a new, empty `BloomFilter` with an estimated max capacity of `item_count` items,
    /// a maximum false positive probability of `fpp`, and two hasher builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = BloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(item_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let bit_count = (-fpp.log2() * (item_count as f64) / 2f64.ln()).ceil() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits, an estimated max capacity
    /// of `item_count` items, and two hasher builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = BloomFilter::<String>::from_item_count_with_hashers(
    ///     100,
    ///     10,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_item_count_with_hashers(
        bit_count: usize,
        item_count: usize,
        hash_builders: [B; 2],
    ) -> Self {
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits, a maximum false positive
    /// probability of `fpp`, and two hasher builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = BloomFilter::<String>::from_fpp_with_hashers(
    ///     100,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_fpp_with_hashers(bit_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let item_count = -(2f64.ln() * (bit_count as f64) / fpp.log2()).floor() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
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
        self.hasher
            .hash(item)
            .take(self.hasher_count)
            .for_each(|hash| {
                let offset = hash % self.bit_vec.len() as u64;
                self.bit_vec.set(offset as usize, true);
            })
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
        self.hasher.hash(item).take(self.hasher_count).all(|hash| {
            let offset = hash % self.bit_vec.len() as u64;
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BloomFilter::<String>::from_fpp_with_hashers(
    ///     100,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BloomFilter::<String>::from_fpp_with_hashers(
    ///     100,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
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

    /// Returns a reference to the bloom filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::<String>::new(10, 0.01);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;
    use crate::util::tests::{hash_builder_1, hash_builder_2};

    #[test]
    fn test_new() {
        let mut filter =
            BloomFilter::<String>::with_hashers(10, 0.01, [hash_builder_1(), hash_builder_2()]);

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
        let mut filter = BloomFilter::<String>::from_fpp_with_hashers(
            100,
            0.01,
            [hash_builder_1(), hash_builder_2()],
        );

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
        let mut filter = BloomFilter::<String>::from_item_count_with_hashers(
            100,
            10,
            [hash_builder_1(), hash_builder_2()],
        );

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
        let mut filter =
            BloomFilter::<String>::with_hashers(100, 0.01, [hash_builder_1(), hash_builder_2()]);
        assert!(filter.estimated_fpp() < std::f64::EPSILON);

        filter.insert("foo");

        let expected_fpp = (7f64 / 959f64).powi(7);
        assert!((filter.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut filter = BloomFilter::<String>::new(100, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: BloomFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.hasher_count(), de_filter.hasher_count());
        assert_eq!(filter.hashers(), de_filter.hashers());
    }
}
