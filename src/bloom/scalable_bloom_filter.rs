use crate::bloom::BloomFilter;
use crate::SipHasherBuilder;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

/// A growable, space-efficient probabilistic data structure to test for membership in a set.
///
/// A scalable bloom filter uses multiple bloom filters to progressively grow as more items are
/// added to the scalable bloom filter. The optimal fill ratio of a bloom filter is 50%, so as
/// soon as the number of ones exceeds 50% of the total bits, then another bloom filter is added to
/// the scalable bloom filter. The new filter will have its size based on the growth ratio, and the
/// number of hash functions based on the tightening ratio. The overall false positive probability
/// of the scalable bloom filter will be `initial_fpp * 1 / (1 - tightening_ratio)`.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::ScalableBloomFilter;
///
/// let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 100);
/// assert_eq!(filter.filter_count(), 1);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct ScalableBloomFilter<T, B = SipHasherBuilder> {
    filters: Vec<BloomFilter<T, B>>,
    approximate_bits_used: usize,
    initial_fpp: f64,
    growth_ratio: f64,
    tightening_ratio: f64,
}

impl<T> ScalableBloomFilter<T> {
    /// Constructs a new, empty `ScalableBloomFilter` with initially `initial_bit_count` bits, an
    /// initial maximum false positive probability of `fpp`. Every time a new bloom filter is
    /// added, the size will be `growth_ratio` multiplied by the previous size, and the false
    /// positive probability will be `tightening_ratio` multipled by the previous false positive
    /// probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// ```
    pub fn new(
        initial_bit_count: usize,
        fpp: f64,
        growth_ratio: f64,
        tightening_ratio: f64,
    ) -> Self {
        Self::with_hashers(
            initial_bit_count,
            fpp,
            growth_ratio,
            tightening_ratio,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> ScalableBloomFilter<T, B>
where
    B: BuildHasher + Clone + Copy,
{
    /// Constructs a new, empty `ScalableBloomFilter` with initially `initial_bit_count` bits, an
    /// initial maximum false positive probability of `fpp`, and two hasher builders for double
    /// hashing. Every time a new bloom filter is added, the size will be `growth_ratio` multiplied
    /// by the previous size, and the false positive probability will be `tightening_ratio`
    /// multipled by the previous false positive probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = ScalableBloomFilter::<String>::with_hashers(
    ///     100,
    ///     0.01,
    ///     2.0,
    ///     0.5,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(
        initial_bit_count: usize,
        fpp: f64,
        growth_ratio: f64,
        tightening_ratio: f64,
        hash_builders: [B; 2],
    ) -> Self {
        ScalableBloomFilter {
            filters: vec![BloomFilter::from_fpp_with_hashers(
                initial_bit_count,
                fpp,
                hash_builders,
            )],
            approximate_bits_used: 0,
            initial_fpp: fpp,
            growth_ratio,
            tightening_ratio,
        }
    }

    fn try_grow(&mut self) {
        let mut new_filter = None;
        {
            let filter = self.filters.last().expect("Expected non-empty filters.");

            if self.approximate_bits_used * 2 >= filter.len() {
                self.approximate_bits_used = filter.count_ones();
                if self.approximate_bits_used * 2 >= filter.len() {
                    let exponent = self.filters.len() as i32;
                    new_filter = Some(BloomFilter::from_fpp_with_hashers(
                        (filter.len() as f64 * self.growth_ratio).ceil() as usize,
                        self.initial_fpp * self.tightening_ratio.powi(exponent),
                        *filter.hashers(),
                    ));
                    self.approximate_bits_used = 0;
                }
            }
        }

        if let Some(new_filter) = new_filter {
            self.filters.push(new_filter);
        }
    }

    /// Inserts an element into the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        if !self.filters.iter().any(|filter| filter.contains(item)) {
            let filter = self
                .filters
                .last_mut()
                .expect("Expected non-empty filters.");
            filter.insert(item);
            self.approximate_bits_used += filter.hasher_count();
        }
        self.try_grow();
    }

    /// Checks if an element is possibly in the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
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
        self.filters.iter().any(|filter| filter.contains(item))
    }

    /// Returns the number of bits in the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.len(), 100);
    /// ```
    pub fn len(&self) -> usize {
        self.filters.iter().map(|filter| filter.len()).sum()
    }

    /// Returns `true` if the scalable bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bloom filters used by the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.filter_count(), 1);
    /// ```
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Clears the scalable bloom filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        let initial_filter = self.filters.first().expect("Expected non-empty filters.");
        self.filters = vec![BloomFilter::from_fpp_with_hashers(
            initial_filter.len(),
            self.initial_fpp,
            *initial_filter.hashers(),
        )];
        self.approximate_bits_used = 0;
    }

    /// Returns the number of set bits in the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::with_hashers(
    ///     100,
    ///     0.01,
    ///     2.0,
    ///     0.5,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.filters.iter().map(|filter| filter.count_ones()).sum()
    }

    /// Returns the number of unset bits in the scalable bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::with_hashers(
    ///     100,
    ///     0.01,
    ///     2.0,
    ///     0.5,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 93);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.filters.iter().map(|filter| filter.count_zeros()).sum()
    }

    /// Returns the estimated false positive probability of the scalable bloom filter. This value
    /// will increase as more items are added.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.01);
    /// ```
    pub fn estimated_fpp(&self) -> f64 {
        1.0 - self
            .filters
            .iter()
            .map(|filter| 1.0 - filter.estimated_fpp())
            .product::<f64>()
    }

    /// Returns a reference to the scalable cuckoo filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::ScalableBloomFilter;
    ///
    /// let filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.filters.first().expect("Expected non-empty filters.").hashers()
    }
}

#[cfg(test)]
mod tests {
    use super::ScalableBloomFilter;
    use crate::util::tests::{HASH_BUILDER_1, HASH_BUILDER_2};

    #[test]
    fn test_scalable_bloom_filter() {
        let mut filter = ScalableBloomFilter::<String>::with_hashers(
            100,
            0.01,
            2.0,
            0.5,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.approximate_bits_used, 7);
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 93);

        filter.clear();
        assert!(!filter.contains("foo"));
        assert_eq!(filter.approximate_bits_used, 0);

        assert_eq!(filter.len(), 100);
        assert_eq!(filter.filter_count(), 1);
    }

    #[test]
    fn test_grow() {
        let mut filter = ScalableBloomFilter::<u32>::with_hashers(
            100,
            0.01,
            2.0,
            0.5,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );

        for i in 0..15 {
            filter.insert(&i);
        }

        assert_eq!(filter.len(), 300);
        assert_eq!(filter.filter_count(), 2);
        assert_eq!(filter.filters[0].hasher_count(), 7);
        assert_eq!(filter.filters[1].hasher_count(), 8);
    }

    #[test]
    fn test_estimated_fpp() {
        let mut filter = ScalableBloomFilter::<u32>::with_hashers(
            800,
            0.01,
            2.0,
            0.5,
            [HASH_BUILDER_1, HASH_BUILDER_2],
        );
        assert!(filter.estimated_fpp() < std::f64::EPSILON);

        for item in 0..200 {
            filter.insert(&item);
        }

        assert_eq!(filter.filter_count(), 2);
        let fpp_0 = 1.0 - filter.filters[0].estimated_fpp();
        let fpp_1 = 1.0 - filter.filters[1].estimated_fpp();
        let expected_fpp = 1.0 - (fpp_0 * fpp_1);
        assert!((filter.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut filter = ScalableBloomFilter::<String>::new(100, 0.01, 2.0, 0.5);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: ScalableBloomFilter<String> =
            bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.filters, de_filter.filters);
        assert_eq!(
            filter.approximate_bits_used,
            de_filter.approximate_bits_used
        );
        assert!((filter.initial_fpp - de_filter.initial_fpp).abs() < std::f64::EPSILON);
        assert!((filter.growth_ratio - de_filter.growth_ratio).abs() < std::f64::EPSILON);
        assert!((filter.tightening_ratio - de_filter.tightening_ratio).abs() < std::f64::EPSILON);
    }
}
