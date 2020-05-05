use crate::bit_array_vec::BitArrayVec;
use crate::cuckoo::{DEFAULT_ENTRIES_PER_INDEX, DEFAULT_FINGERPRINT_BIT_COUNT, DEFAULT_MAX_KICKS};
use crate::SipHasherBuilder;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

/// A space-efficient probabilistic data structure to test for membership in a set. Cuckoo filters
/// also provide the flexibility to remove items.
///
/// A cuckoo filter is based on cuckoo hashing and is essentially a cuckoo hash table storing
/// each keys' fingerprint. Cuckoo filters can be highly compact and serve as an improvement over
/// variations of tradition Bloom filters that support deletion (E.G. counting Bloom filters).
///
/// # Examples
///
/// ```
/// use probabilistic_collections::cuckoo::CuckooFilter;
///
/// let mut filter = CuckooFilter::<String>::new(100);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.remove("foo");
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 0);
/// assert_eq!(filter.capacity(), 128);
/// assert_eq!(filter.bucket_len(), 32);
/// assert_eq!(filter.fingerprint_bit_count(), 8);
/// ```
#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct CuckooFilter<T, B = SipHasherBuilder> {
    max_kicks: usize,
    entries_per_index: usize,
    fingerprint_vec: BitArrayVec,
    pub(super) extra_items: Vec<(u64, usize)>,
    hash_builders: [B; 2],
    #[cfg_attr(feature = "serde", serde(skip, default = "XorShiftRng::from_entropy"))]
    rng: XorShiftRng,
    _marker: PhantomData<T>,
}

impl<T> CuckooFilter<T> {
    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`. By
    /// default, the cuckoo filter will have 8 bits per item fingerprint, 4 entries per index, and
    /// a maximum of 512 item displacements before terminating the insertion process. The cuckoo
    /// filter will have an estimated maximum false positive probability of 3%.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    /// ```
    pub fn new(item_count: usize) -> Self {
        Self::with_hashers(
            item_count,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, a
    /// fingerprint bit count of `fingerprint_bit_count`, `entries_per_index` entries per index,
    /// and a maximum of 512 item displacements before terminating the insertion process. This
    /// method provides no guarantees on the false positive probability of the cuckoo filter.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0, if `fingerprint_bit_count` less than 1 or greater than 64, or
    /// if `entries_per_index` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::from_parameters(100, 16, 8);
    /// ```
    pub fn from_parameters(
        item_count: usize,
        fingerprint_bit_count: usize,
        entries_per_index: usize,
    ) -> Self {
        Self::from_parameters_with_hashers(
            item_count,
            fingerprint_bit_count,
            entries_per_index,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, an
    /// estimated maximum false positive probability of `fpp`, `entries_per_index` entries per
    /// index, and a maximum of 512 item displacements before terminating the insertion process.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0 or if `entries_per_index` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::from_entries_per_index(100, 0.01, 4);
    /// ```
    pub fn from_entries_per_index(item_count: usize, fpp: f64, entries_per_index: usize) -> Self {
        Self::from_entries_per_index_with_hashers(
            item_count,
            fpp,
            entries_per_index,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, an
    /// estimated maximum false positive probability of `fpp`, a fingerprint bit count of
    /// `fingerprint_bit_count`, and a maximum of 512 item displacements before terminating the
    /// insertion process.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0, if `fingerprint_bit_count` is less than 1 or greater than 64,
    /// or if it is impossible to achieve the given maximum false positive probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::from_fingerprint_bit_count(100, 0.01, 10);
    /// ```
    pub fn from_fingerprint_bit_count(
        item_count: usize,
        fpp: f64,
        fingerprint_bit_count: usize,
    ) -> Self {
        Self::from_fingerprint_bit_count_with_hashers(
            item_count,
            fpp,
            fingerprint_bit_count,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> CuckooFilter<T, B>
where
    B: BuildHasher,
{
    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, and
    /// two hasher builders for double hashing. By default, the cuckoo filter will have 8 bits per
    /// item fingerprint, 4 entries per index, and a maximum of 512 item displacements before
    /// terminating the insertion process. The cuckoo filter will have an estimated maximum false
    /// positive probability of 3%.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = CuckooFilter::<String>::with_hashers(
    ///     100,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(item_count: usize, hash_builders: [B; 2]) -> Self {
        assert!(item_count > 0);
        let bucket_len = ((item_count + DEFAULT_ENTRIES_PER_INDEX - 1) / DEFAULT_ENTRIES_PER_INDEX)
            .next_power_of_two();
        CuckooFilter {
            max_kicks: DEFAULT_MAX_KICKS,
            entries_per_index: DEFAULT_ENTRIES_PER_INDEX,
            fingerprint_vec: BitArrayVec::new(
                DEFAULT_FINGERPRINT_BIT_COUNT,
                bucket_len * DEFAULT_ENTRIES_PER_INDEX,
            ),
            extra_items: Vec::new(),
            hash_builders,
            rng: XorShiftRng::from_entropy(),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, a
    /// fingerprint bit count of `fingerprint_bit_count`, `entries_per_index` entries per index, a
    /// maximum of 512 item displacements before terminating the insertion process, and two hasher
    /// builders for double hashing. This method provides no guarantees on the false positive
    /// probability of the cuckoo filter.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0, if `fingerprint_bit_count` less than 1 or greater than 64, or
    /// if `entries_per_index` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = CuckooFilter::<String>::from_parameters_with_hashers(
    ///     100,
    ///     16,
    ///     8,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_parameters_with_hashers(
        item_count: usize,
        fingerprint_bit_count: usize,
        entries_per_index: usize,
        hash_builders: [B; 2],
    ) -> Self {
        assert!(
            item_count > 0
                && fingerprint_bit_count > 1
                && fingerprint_bit_count <= 64
                && entries_per_index > 0
        );
        let exact_bucket_len = (item_count + entries_per_index - 1) / entries_per_index;
        let bucket_len = exact_bucket_len.next_power_of_two();
        CuckooFilter {
            max_kicks: DEFAULT_MAX_KICKS,
            entries_per_index,
            fingerprint_vec: BitArrayVec::new(
                fingerprint_bit_count,
                bucket_len * entries_per_index,
            ),
            extra_items: Vec::new(),
            hash_builders,
            rng: XorShiftRng::from_entropy(),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, an
    /// estimated maximum false positive probability of `fpp`, `entries_per_index` entries per
    /// index, a maximum of 512 item displacements before terminating the insertion process, and
    /// two hasher builders for doubling hashing.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0 or if `entries_per_index` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = CuckooFilter::<String>::from_entries_per_index_with_hashers(
    ///     100,
    ///     0.01,
    ///     4,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_entries_per_index_with_hashers(
        item_count: usize,
        fpp: f64,
        entries_per_index: usize,
        hash_builders: [B; 2],
    ) -> Self {
        assert!(item_count > 0);
        assert!(entries_per_index > 0);
        let power = 2.0 / (1.0 - (1.0 - fpp).powf(1.0 / (2.0 * entries_per_index as f64)));
        let fingerprint_bit_count = power.log2().ceil() as usize;
        let exact_bucket_len = (item_count + entries_per_index - 1) / entries_per_index;
        let bucket_len = exact_bucket_len.next_power_of_two();
        CuckooFilter {
            max_kicks: DEFAULT_MAX_KICKS,
            entries_per_index,
            fingerprint_vec: BitArrayVec::new(
                fingerprint_bit_count,
                bucket_len * entries_per_index,
            ),
            extra_items: Vec::new(),
            hash_builders,
            rng: XorShiftRng::from_entropy(),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `CuckooFilter` with an estimated max capacity of `item_count`, an
    /// estimated maximum false positive probability of `fpp`, a fingerprint bit count of
    /// `fingerprint_bit_count`, a maximum of 512 item displacements before terminating the
    /// insertion process, and two hasher builders for double hashing.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `item_count` is 0, if `fingerprint_bit_count` is less than 1 or greater than 64,
    /// or if it is impossible to achieve the given maximum false positive probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = CuckooFilter::<String>::from_fingerprint_bit_count_with_hashers(
    ///     100,
    ///     0.01,
    ///     10,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_fingerprint_bit_count_with_hashers(
        item_count: usize,
        fpp: f64,
        fingerprint_bit_count: usize,
        hash_builders: [B; 2],
    ) -> Self {
        assert!(item_count > 0);
        assert!(fingerprint_bit_count > 1 && fingerprint_bit_count <= 64);
        let fingerprints_count = 2.0f64.powi(fingerprint_bit_count as i32);
        let single_fpp = (fingerprints_count - 2.0) / (fingerprints_count - 1.0);
        let entries_per_index = ((1.0 - fpp).log(single_fpp) / 2.0).floor() as usize;
        assert!(entries_per_index > 0);
        let exact_bucket_len = (item_count + entries_per_index - 1) / entries_per_index;
        let bucket_len = exact_bucket_len.next_power_of_two();
        CuckooFilter {
            max_kicks: DEFAULT_MAX_KICKS,
            entries_per_index,
            fingerprint_vec: BitArrayVec::new(
                fingerprint_bit_count,
                bucket_len * entries_per_index,
            ),
            extra_items: Vec::new(),
            hash_builders,
            rng: XorShiftRng::from_entropy(),
            _marker: PhantomData,
        }
    }

    pub(super) fn get_fingerprint(raw_fingerprint: u64) -> Vec<u8> {
        (0..8)
            .map(|index| ((raw_fingerprint >> (index * 8)) & (0xFF)) as u8)
            .collect()
    }

    fn get_raw_fingerprint(fingerprint: &[u8]) -> u64 {
        let mut ret = 0u64;
        for (index, byte) in fingerprint.iter().enumerate() {
            ret |= (u64::from(*byte)) << (index * 8)
        }
        ret
    }

    #[inline]
    fn get_vec_index(&self, index: usize, bucket_index: usize) -> usize {
        index * self.entries_per_index + bucket_index
    }

    fn get_fingerprint_and_indexes<U>(&self, item: &U) -> (Vec<u8>, usize, usize)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let trailing_zeros = 64 - self.fingerprint_bit_count();
        let mut hasher = self.hash_builders[0].build_hasher();
        item.hash(&mut hasher);
        let mut h0 = hasher.finish();
        let mut raw_fingerprint = h0 << trailing_zeros >> trailing_zeros;

        // rehash when fingerprint is all 0s
        while raw_fingerprint == 0 {
            let mut hasher = self.hash_builders[0].build_hasher();
            h0.hash(&mut hasher);
            h0 = hasher.finish();
            raw_fingerprint = h0 << trailing_zeros >> trailing_zeros;
        }
        let fingerprint = Self::get_fingerprint(raw_fingerprint);

        let mut hasher = self.hash_builders[1].build_hasher();
        item.hash(&mut hasher);
        let h1 = hasher.finish();

        let index_1 = h1 as usize % self.bucket_len();
        let index_2 = (index_1 ^ raw_fingerprint as usize) % self.bucket_len();
        (fingerprint, index_1, index_2)
    }

    /// Inserts an element into the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::new(100);
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (mut fingerprint, index_1, index_2) = self.get_fingerprint_and_indexes(item);
        if !self.contains_fingerprint(&fingerprint, index_1, index_2) {
            if self.insert_fingerprint(fingerprint.as_slice(), index_1) {
                return;
            }

            if self.insert_fingerprint(fingerprint.as_slice(), index_2) {
                return;
            }

            // have to kick out an entry
            let mut index = if self.rng.gen::<bool>() {
                index_1
            } else {
                index_2
            };
            let mut prev_index = index;

            for _ in 0..self.max_kicks {
                let bucket_index = self.rng.gen_range(0, self.entries_per_index);
                let vec_index = self.get_vec_index(index, bucket_index);
                let new_fingerprint = self.fingerprint_vec.get(vec_index);
                self.fingerprint_vec.set(vec_index, fingerprint.as_slice());
                fingerprint = new_fingerprint;
                prev_index = index;
                index = prev_index ^ Self::get_raw_fingerprint(&fingerprint) as usize;
                index %= self.bucket_len();
                if self.insert_fingerprint(fingerprint.as_slice(), index) {
                    return;
                }
            }

            self.extra_items.push((
                Self::get_raw_fingerprint(&fingerprint),
                cmp::min(prev_index, index),
            ));
        }
    }

    pub(super) fn insert_fingerprint(&mut self, fingerprint: &[u8], index: usize) -> bool {
        let entries_per_index = self.entries_per_index;
        for bucket_index in 0..entries_per_index {
            let vec_index = self.get_vec_index(index, bucket_index);
            let is_empty = self
                .fingerprint_vec
                .get(vec_index)
                .iter()
                .all(|byte| *byte == 0);

            if is_empty {
                self.fingerprint_vec.set(vec_index, fingerprint);
                return true;
            }
        }
        false
    }

    /// Removes an element from the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::new(100);
    ///
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    ///
    /// filter.remove("foo");
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn remove<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (fingerprint, index_1, index_2) = self.get_fingerprint_and_indexes(item);
        self.remove_fingerprint(&fingerprint, index_1, index_2)
    }

    fn remove_fingerprint(&mut self, fingerprint: &[u8], index_1: usize, index_2: usize) {
        let raw_fingerprint = Self::get_raw_fingerprint(fingerprint);
        let min_index = cmp::min(index_1, index_2);
        let entries_per_index = self.entries_per_index;
        if let Some(index) = self
            .extra_items
            .iter()
            .position(|item| *item == (raw_fingerprint, min_index))
        {
            self.extra_items.swap_remove(index);
        }
        for bucket_index in 0..entries_per_index {
            let vec_index_1 = self.get_vec_index(index_1, bucket_index);
            let vec_index_2 = self.get_vec_index(index_2, bucket_index);
            let fingerprint_1 = &self.fingerprint_vec.get(vec_index_1);
            let fingerprint_2 = &self.fingerprint_vec.get(vec_index_2);
            if Self::get_raw_fingerprint(fingerprint_1) == raw_fingerprint {
                let empty_fingerprint = Self::get_fingerprint(0);
                self.fingerprint_vec
                    .set(vec_index_1, empty_fingerprint.as_slice());
            }
            if Self::get_raw_fingerprint(fingerprint_2) == raw_fingerprint {
                let empty_fingerprint = Self::get_fingerprint(0);
                self.fingerprint_vec
                    .set(vec_index_2, empty_fingerprint.as_slice());
            }
        }
    }

    /// Checks if an element is possibly in the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::new(100);
    ///
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (fingerprint, index_1, index_2) = self.get_fingerprint_and_indexes(item);
        self.contains_fingerprint(&fingerprint, index_1, index_2)
    }

    fn contains_fingerprint(&self, fingerprint: &[u8], index_1: usize, index_2: usize) -> bool {
        let raw_fingerprint = Self::get_raw_fingerprint(fingerprint);
        let min_index = cmp::min(index_1, index_2);
        let entries_per_index = self.entries_per_index;
        if self.extra_items.contains(&(raw_fingerprint, min_index)) {
            return true;
        }
        (0..entries_per_index).any(|bucket_index| {
            let vec_index_1 = self.get_vec_index(index_1, bucket_index);
            let vec_index_2 = self.get_vec_index(index_2, bucket_index);
            self.fingerprint_vec
                .get(vec_index_1)
                .iter()
                .zip(fingerprint)
                .all(|pair| pair.0 == pair.1)
                || self
                    .fingerprint_vec
                    .get(vec_index_2)
                    .iter()
                    .zip(fingerprint)
                    .all(|pair| pair.0 == pair.1)
        })
    }

    /// Clears the cuckoo filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::new(100);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.fingerprint_vec.clear();
        self.extra_items.clear();
    }

    /// Returns the number of occupied entries in the cuckoo filter. It does account for items in
    /// the extra items vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert_eq!(filter.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.fingerprint_vec.occupied_len()
    }

    /// Returns `true` if there are no occupied entries in the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert!(filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the cuckoo filter. Items stay spill into the extra items
    /// vector even through the length of the cuckoo filter is less than the capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert_eq!(filter.capacity(), 128);
    /// ```
    pub fn capacity(&self) -> usize {
        self.fingerprint_vec.capacity()
    }

    /// Returns the length of each bucket in the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert_eq!(filter.bucket_len(), 32);
    /// ```
    pub fn bucket_len(&self) -> usize {
        self.fingerprint_vec.capacity() / self.entries_per_index
    }

    /// Returns the number of entries per index in the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert_eq!(filter.entries_per_index(), 4);
    /// ```
    pub fn entries_per_index(&self) -> usize {
        self.entries_per_index
    }

    /// Returns the number of items that could not be inserted into the CuckooFilter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = CuckooFilter::<String>::from_parameters_with_hashers(
    ///     1,
    ///     8,
    ///     1,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    ///
    /// filter.insert("foo");
    /// filter.insert("foobar");
    /// assert_eq!(filter.extra_items_len(), 1);
    /// ```
    pub fn extra_items_len(&self) -> usize {
        self.extra_items.len()
    }

    /// Returns `true` if there are any items that could not be inserted into the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::from_parameters(1, 8, 1);
    ///
    /// filter.insert("foo");
    /// filter.insert("foobar");
    /// assert!(filter.is_nearly_full());
    /// ```
    pub fn is_nearly_full(&self) -> bool {
        !self.extra_items.is_empty()
    }

    /// Returns the number of bits in each item fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    ///
    /// assert_eq!(filter.fingerprint_bit_count(), 8);
    /// ```
    pub fn fingerprint_bit_count(&self) -> usize {
        self.fingerprint_vec.bit_count()
    }

    /// Returns the estimated false positive probability of the cuckoo filter. This value will
    /// increase as more items are added.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let mut filter = CuckooFilter::<String>::new(100);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.01);
    /// ```
    pub fn estimated_fpp(&self) -> f64 {
        let fingerprints_count = 2.0f64.powi(self.fingerprint_bit_count() as i32);
        let single_fpp = (fingerprints_count - 2.0) / (fingerprints_count - 1.0);
        let occupied_len = self.fingerprint_vec.occupied_len();
        let occupied_ratio = occupied_len as f64 / self.capacity() as f64;
        1.0 - single_fpp.powf(2.0 * self.entries_per_index() as f64 * occupied_ratio)
    }

    /// Returns a reference to the cuckoo filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::CuckooFilter;
    ///
    /// let filter = CuckooFilter::<String>::new(100);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        &self.hash_builders
    }
}

impl<T> PartialEq for CuckooFilter<T> {
    fn eq(&self, other: &CuckooFilter<T>) -> bool {
        self.max_kicks == other.max_kicks
            && self.entries_per_index == other.entries_per_index
            && self.fingerprint_vec == other.fingerprint_vec
            && self.extra_items == other.extra_items
            && self.hash_builders == other.hash_builders
    }
}

#[cfg(test)]
mod tests {
    use super::CuckooFilter;
    use crate::util::tests::{HASH_BUILDER_1, HASH_BUILDER_2};

    #[test]
    fn test_get_fingerprint() {
        let fingerprint = CuckooFilter::<String>::get_fingerprint(0x7FBF_DFEF_F7FB_FDFE);
        assert_eq!(
            CuckooFilter::<String>::get_raw_fingerprint(&fingerprint),
            0x7FBF_DFEF_F7FB_FDFE,
        );
    }

    #[test]
    fn test_get_raw_fingerprint() {
        let fingerprint = vec![0xFF, 0xFF];
        assert_eq!(
            CuckooFilter::<String>::get_raw_fingerprint(&fingerprint),
            0xFFFF,
        );
    }

    #[test]
    fn test_new() {
        let filter = CuckooFilter::<String>::with_hashers(100, [HASH_BUILDER_1, HASH_BUILDER_2]);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert_eq!(filter.capacity(), 128);
        assert_eq!(filter.bucket_len(), 32);
        assert_eq!(filter.fingerprint_bit_count(), 8);
        assert_eq!(filter.entries_per_index(), 4);
    }

    #[test]
    fn test_from_parameters() {
        let filter = CuckooFilter::<String>::from_parameters_with_hashers(
            100,
            16,
            8,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert_eq!(filter.capacity(), 128);
        assert_eq!(filter.bucket_len(), 16);
        assert_eq!(filter.fingerprint_bit_count(), 16);
        assert_eq!(filter.entries_per_index(), 8);
    }

    #[test]
    fn test_from_entries_per_index() {
        let filter = CuckooFilter::<String>::from_entries_per_index_with_hashers(
            100,
            0.01,
            4,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert_eq!(filter.capacity(), 128);
        assert_eq!(filter.bucket_len(), 32);
        assert_eq!(filter.fingerprint_bit_count(), 11);
        assert_eq!(filter.entries_per_index(), 4);
    }

    #[test]
    fn test_from_fingerprint_bit_count() {
        let filter = CuckooFilter::<String>::from_fingerprint_bit_count_with_hashers(
            100,
            0.01,
            10,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert_eq!(filter.capacity(), 160);
        assert_eq!(filter.bucket_len(), 32);
        assert_eq!(filter.fingerprint_bit_count(), 10);
        assert_eq!(filter.entries_per_index(), 5);
    }

    #[test]
    fn test_insert() {
        let mut filter =
            CuckooFilter::<String>::with_hashers(100, [HASH_BUILDER_1, HASH_BUILDER_2]);
        filter.insert("foo");
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_insert_existing_item() {
        let mut filter =
            CuckooFilter::<String>::with_hashers(100, [HASH_BUILDER_1, HASH_BUILDER_2]);
        filter.insert("foo");
        filter.insert("foo");

        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_insert_extra_items() {
        let mut filter = CuckooFilter::<String>::from_parameters_with_hashers(
            1,
            8,
            1,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        filter.insert("foo");
        filter.insert("foobar");

        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        assert_eq!(filter.extra_items.len(), 1);
        assert!(filter.is_nearly_full());

        assert!(filter.contains("foo"));
        assert!(filter.contains("foobar"));
    }

    #[test]
    fn test_remove() {
        let mut filter =
            CuckooFilter::<String>::with_hashers(100, [HASH_BUILDER_1, HASH_BUILDER_2]);
        filter.insert("foo");
        filter.remove("foo");

        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert!(!filter.contains("foo"));
    }

    #[test]
    fn test_remove_extra_items() {
        let mut filter = CuckooFilter::<String>::from_parameters_with_hashers(
            1,
            8,
            1,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        filter.insert("foo");
        filter.insert("foobar");

        filter.remove("foo");
        filter.remove("foobar");

        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert_eq!(filter.extra_items.len(), 0);
        assert!(!filter.is_nearly_full());
        assert!(!filter.contains("foo"));
        assert!(!filter.contains("foobar"));
    }
    #[test]
    fn test_remove_both_indexes() {
        let mut filter = CuckooFilter::<String>::from_parameters_with_hashers(
            2,
            8,
            1,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        filter.insert("foobar");
        filter.insert("barfoo");
        filter.insert("baz");
        filter.insert("qux");

        filter.remove("baz");
        filter.remove("qux");
        filter.remove("foobar");
        filter.remove("barfoo");

        assert!(!filter.contains("baz"));
        assert!(!filter.contains("qux"));
        assert!(!filter.contains("foobar"));
        assert!(!filter.contains("barfoo"));
    }

    #[test]
    fn test_clear() {
        let mut filter = CuckooFilter::<String>::from_parameters_with_hashers(
            2,
            8,
            1,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        filter.insert("foobar");
        filter.insert("barfoo");
        filter.insert("baz");
        filter.insert("qux");

        filter.clear();

        assert!(!filter.contains("baz"));
        assert!(!filter.contains("qux"));
        assert!(!filter.contains("foobar"));
        assert!(!filter.contains("barfoo"));
    }

    #[test]
    fn test_estimated_fpp() {
        let mut filter = CuckooFilter::<String>::from_entries_per_index_with_hashers(
            100,
            0.01,
            4,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );
        assert!(filter.estimated_fpp() < std::f64::EPSILON);

        filter.insert("foo");

        let expected_fpp = 1.0 - ((2f64.powi(11) - 2.0) / (2f64.powi(11) - 1.0)).powf(8.0 / 128.0);
        assert!((filter.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut filter = CuckooFilter::<String>::from_entries_per_index(100, 0.01, 4);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: CuckooFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.max_kicks, de_filter.max_kicks);
        assert_eq!(filter.entries_per_index, de_filter.entries_per_index);
        assert_eq!(filter.fingerprint_vec, de_filter.fingerprint_vec);
        assert_eq!(filter.extra_items, de_filter.extra_items);
        assert_eq!(filter.hash_builders, de_filter.hash_builders);
    }
}
