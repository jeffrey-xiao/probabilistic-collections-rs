use crate::bit_vec::BitVec;
use crate::{DoubleHasher, HashIter, SipHasherBuilder};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::f64::consts;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

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
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct BSBloomFilter<T, B = SipHasherBuilder> {
    bit_vec: BitVec,
    hasher: DoubleHasher<T, B>,
    #[cfg_attr(feature = "serde", serde(skip, default = "XorShiftRng::from_entropy"))]
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BSBloomFilter<T> {
    /// Constructs a new, empty `BSBloomFilter` with `bit_count` bits per filter, and a false
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
        Self::with_hashers(
            bit_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> BSBloomFilter<T, B>
where
    B: BuildHasher,
{
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `BSBloomFilter` with `bit_count` bits per filter, a false
    /// positive probability of `fpp`, and two hash builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = BSBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(bit_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        BSBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            rng: XorShiftRng::from_entropy(),
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
        let hashes = self.hasher.hash(item);
        if !self.contains_hashes(hashes) {
            (0..self.hasher_count).for_each(|index| {
                let index = index * self.bit_count + self.rng.gen_range(0, self.bit_count);
                self.bit_vec.set(index, false);
            });

            hashes
                .take(self.hasher_count)
                .enumerate()
                .for_each(|(index, hash)| {
                    let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
                    self.bit_vec.set(offset as usize, true);
                })
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
        self.contains_hashes(self.hasher.hash(item))
    }

    fn contains_hashes(&self, hashes: HashIter) -> bool {
        hashes
            .take(self.hasher_count)
            .enumerate()
            .all(|(index, hash)| {
                let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BSBloomFilter::<String>::with_hashers(
    ///     10,
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
    /// use probabilistic_collections::bloom::BSBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BSBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 63);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }

    /// Returns a reference to the bloom filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSBloomFilter;
    ///
    /// let filter = BSBloomFilter::<String>::new(10, 0.01);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
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
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct BSSDBloomFilter<T, B = SipHasherBuilder> {
    bit_vec: BitVec,
    hasher: DoubleHasher<T, B>,
    #[cfg_attr(feature = "serde", serde(skip, default = "XorShiftRng::from_entropy"))]
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> BSSDBloomFilter<T> {
    /// Constructs a new, empty `BSSDBloomFilter` with `bit_count` bits per filter, and a false
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
        Self::with_hashers(
            bit_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> BSSDBloomFilter<T, B>
where
    B: BuildHasher,
{
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `BSSDBloomFilter` with `bit_count` bits per filter, a false
    /// positive probability of `fpp`, and two hash builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = BSSDBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(bit_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        BSSDBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            rng: XorShiftRng::from_entropy(),
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
        let hashes = self.hasher.hash(item);
        if !self.contains_hashes(hashes) {
            let filter_index = self.rng.gen_range(0, self.hasher_count);
            let index = self.rng.gen_range(0, self.bit_count);

            self.bit_vec
                .set(filter_index * self.bit_count + index, false);

            hashes
                .take(self.hasher_count)
                .enumerate()
                .for_each(|(index, hash)| {
                    let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
                    self.bit_vec.set(offset as usize, true);
                })
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
        self.contains_hashes(self.hasher.hash(item))
    }

    fn contains_hashes(&self, hashes: HashIter) -> bool {
        hashes
            .take(self.hasher_count)
            .enumerate()
            .all(|(index, hash)| {
                let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::with_hashers(
    ///     10,
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
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = BSSDBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 63);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }

    /// Returns a reference to the bloom filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::BSSDBloomFilter;
    ///
    /// let filter = BSSDBloomFilter::<String>::new(10, 0.01);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
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
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct RLBSBloomFilter<T, B = SipHasherBuilder> {
    bit_vecs: Vec<BitVec>,
    hasher: DoubleHasher<T, B>,
    #[cfg_attr(feature = "serde", serde(skip, default = "XorShiftRng::from_entropy"))]
    rng: XorShiftRng,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> RLBSBloomFilter<T> {
    /// Constructs a new, empty `RLBSBloomFilter` with `bit_count` bits per filter, and a false
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
        Self::with_hashers(
            bit_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> RLBSBloomFilter<T, B>
where
    B: BuildHasher,
{
    fn get_hasher_count(fpp: f64) -> usize {
        ((1.0 + fpp.ln() / (1.0 - 1.0 / consts::E).ln() + 1.0) / 2.0).ceil() as usize
    }

    /// Constructs a new, empty `RLBSBloomFilter` with `bit_count` bits per filter, a false
    /// positive probability of `fpp`, and two hash builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = RLBSBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(bit_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        RLBSBloomFilter {
            bit_vecs: vec![BitVec::new(bit_count); hasher_count],
            hasher: DoubleHasher::with_hashers(hash_builders),
            rng: XorShiftRng::from_entropy(),
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
        let hashes = self.hasher.hash(item);
        if !self.contains_hashes(hashes) {
            (0..self.hasher_count).for_each(|filter_index| {
                let prob = self.bit_vecs[filter_index].count_ones() as f64 / self.bit_count as f64;
                let index = self.rng.gen_range(0, self.bit_count);
                if self.rng.gen::<f64>() < prob {
                    self.bit_vecs[filter_index].set(index, false);
                }
            });

            hashes
                .take(self.hasher_count)
                .enumerate()
                .for_each(|(filter_index, hash)| {
                    let offset = hash % self.bit_count as u64;
                    self.bit_vecs[filter_index].set(offset as usize, true);
                })
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
        self.contains_hashes(self.hasher.hash(item))
    }

    fn contains_hashes(&self, hashes: HashIter) -> bool {
        hashes
            .take(self.hasher_count)
            .enumerate()
            .all(|(filter_index, hash)| {
                let offset = hash % self.bit_count as u64;
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = RLBSBloomFilter::<String>::with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
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

    /// Returns a reference to the bloom filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::RLBSBloomFilter;
    ///
    /// let filter = RLBSBloomFilter::<String>::new(10, 0.01);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
    }
}

#[cfg(test)]
mod tests {
    use super::{BSBloomFilter, BSSDBloomFilter, RLBSBloomFilter};
    use crate::util::tests::{hash_builder_1, hash_builder_2};

    #[test]
    fn test_bs() {
        let mut filter =
            BSBloomFilter::<String>::with_hashers(10, 0.01, [hash_builder_1(), hash_builder_2()]);

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

    #[cfg(feature = "serde")]
    #[test]
    fn test_bs_ser_de() {
        let mut filter = BSBloomFilter::<String>::new(10, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: BSBloomFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.bit_count, de_filter.bit_count);
        assert_eq!(filter.hasher_count(), de_filter.hasher_count());
        assert_eq!(filter.hashers(), de_filter.hashers());
    }

    #[test]
    fn test_bssd() {
        let mut filter =
            BSSDBloomFilter::<String>::with_hashers(10, 0.01, [hash_builder_1(), hash_builder_2()]);

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

    #[cfg(feature = "serde")]
    #[test]
    fn test_bssd_ser_de() {
        let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: BSBloomFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.bit_count, de_filter.bit_count);
        assert_eq!(filter.hasher_count(), de_filter.hasher_count());
        assert_eq!(filter.hashers(), de_filter.hashers());
    }

    #[test]
    fn test_rlbs() {
        let mut filter =
            RLBSBloomFilter::<String>::with_hashers(10, 0.01, [hash_builder_1(), hash_builder_2()]);

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

    #[cfg(feature = "serde")]
    #[test]
    fn test_rlbs_ser_de() {
        let mut filter = BSSDBloomFilter::<String>::new(10, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: BSBloomFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.bit_count, de_filter.bit_count);
        assert_eq!(filter.hasher_count(), de_filter.hasher_count());
        assert_eq!(filter.hashers(), de_filter.hashers());
    }
}
