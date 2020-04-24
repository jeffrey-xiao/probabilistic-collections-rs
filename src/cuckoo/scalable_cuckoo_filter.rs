use crate::cuckoo::{CuckooFilter, DEFAULT_ENTRIES_PER_INDEX};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::Hash;

/// A growable, space-efficient probabilistic data structure to test for membership in a set.
/// Scalable cuckoo filters also provide the flexibility to remove items.
///
/// A cuckoo filter is based on cuckoo hashing and is essentially a cuckoo hash table storing
/// each keys' fingerprint. Cuckoo filters can be highly compact and serve as an improvement over
/// variations of tradition Bloom filters that support deletion (E.G. counting Bloom filters).
///
/// This implementation is a scalable version of a cuckoo filter inspired by
/// `ScalableBloomFilter`. Currently, the scalable cuckoo filter will naively insert into the last
/// inserted cuckoo filter despite the fact that deletions could free space in previously inserted
/// cuckoo filters. Checking if there is space in previously inserted cuckoo filters is fairly
/// expensive and would significantly slow down the scalable cuckoo filter.
///
/// The overall false positive probability of the scalable cuckoo filter will be `initial_fpp * 1 /
/// (1 - tightening_ratio)`.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
///
/// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
/// assert!(!filter.contains("foo"));
///
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.remove("foo");
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 0);
/// assert_eq!(filter.capacity(), 128);
/// assert_eq!(filter.filter_count(), 1);
/// ```
#[derive(Deserialize, Serialize)]
pub struct ScalableCuckooFilter<T> {
    filters: Vec<CuckooFilter<T>>,
    initial_item_count: usize,
    initial_fpp: f64,
    growth_ratio: f64,
    tightening_ratio: f64,
}

impl<T> ScalableCuckooFilter<T> {
    /// Constructs a new, empty `ScalableCuckooFilter` with an estimated initial item capacity of
    /// `item_count` and an initial maximum false positive probability of `fpp`. Every time a new
    /// cuckoo filter is added, the size will be approximately `growth_ratio` multiplied by the
    /// previous size, and the false positive probability will be `tightening_ratio` multipled by
    /// the previous false positive probability.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// ```
    pub fn new(item_count: usize, fpp: f64, growth_ratio: f64, tightening_ratio: f64) -> Self {
        ScalableCuckooFilter {
            filters: vec![CuckooFilter::from_entries_per_index(
                item_count,
                fpp,
                DEFAULT_ENTRIES_PER_INDEX,
            )],
            initial_item_count: item_count,
            initial_fpp: fpp,
            growth_ratio,
            tightening_ratio,
        }
    }

    /// Constructs a new, empty `ScalableCuckooFilter` with an estimated initial item capacity of
    /// `item_count`, an initial maximum false positive probability of `fpp`, and
    /// `entries_per_index` entries per index. Every time a new cuckoo filter is added, the size
    /// will be approximately `growth_ratio` multiplied by the previous size, and the false positive
    /// probability will be `tightening_ratio` multipled by the previous false positive probability.
    ///
    /// The length of each bucket will be rounded off to the next power of two.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::from_entries_per_index(100, 0.01, 4, 2.0, 0.5);
    /// ```
    pub fn from_entries_per_index(
        item_count: usize,
        fpp: f64,
        entries_per_index: usize,
        growth_ratio: f64,
        tightening_ratio: f64,
    ) -> Self {
        ScalableCuckooFilter {
            filters: vec![CuckooFilter::from_entries_per_index(
                item_count,
                fpp,
                entries_per_index,
            )],
            initial_item_count: item_count,
            initial_fpp: fpp,
            growth_ratio,
            tightening_ratio,
        }
    }

    fn try_grow(&mut self) {
        let mut new_filter_opt = None;
        {
            let exponent = self.filters.len() as i32;
            let filter = self
                .filters
                .last_mut()
                .expect("Expected non-empty filters.");

            if filter.is_nearly_full() {
                let mut new_filter = CuckooFilter::from_entries_per_index(
                    (filter.capacity() as f64 * self.growth_ratio).ceil() as usize,
                    self.initial_fpp * self.tightening_ratio.powi(exponent),
                    filter.entries_per_index(),
                );

                for fingerprint_entry in filter.extra_items.drain(..) {
                    let index_1 = fingerprint_entry.1;
                    let mut index_2 = fingerprint_entry.1 ^ fingerprint_entry.0 as usize;
                    index_2 %= new_filter.bucket_len();
                    let fingerprint = CuckooFilter::<T>::get_fingerprint(fingerprint_entry.0);
                    // Should always have room in a new filter.
                    if !new_filter.insert_fingerprint(fingerprint.as_slice(), index_1) {
                        new_filter.insert_fingerprint(fingerprint.as_slice(), index_2);
                    }
                }

                new_filter_opt = Some(new_filter);
            }
        }

        if let Some(new_filter) = new_filter_opt {
            self.filters.push(new_filter);
        }
    }

    /// Inserts an element into the scalable cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
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
        }
        self.try_grow();
    }

    /// Checks if an element is possibly in the scalable cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&mut self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        self.filters.iter().any(|filter| filter.contains(item))
    }

    /// Removes an element from the scalable cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// assert!(!filter.contains("foo"));
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
        for filter in &mut self.filters {
            filter.remove(item);
        }
    }

    /// Returns the number of occupied entries in the cuckoo filter
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.filters.iter().map(|filter| filter.len()).sum()
    }

    /// Returns `true` if there are no occupied entries in the cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert!(filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the cuckoo filter. The scalable filter may grow even
    /// through the length of the cuckoo filter is less than the capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.capacity(), 128);
    /// ```
    pub fn capacity(&self) -> usize {
        self.filters.iter().map(|filter| filter.capacity()).sum()
    }

    /// Returns the number of entries per index in each filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.entries_per_index(), 4);
    /// ```
    pub fn entries_per_index(&self) -> usize {
        let filter = self.filters.first().expect("Expected non-empty filters.");
        filter.entries_per_index()
    }

    /// Returns the number of cuckoo filters used by the scalable cuckoo filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// assert_eq!(filter.filter_count(), 1);
    /// ```
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Clears the scalable cuckoo filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        let default_entries_per_index = self
            .filters
            .first()
            .expect("Expected non-empty filters.")
            .entries_per_index();

        self.filters = vec![CuckooFilter::from_entries_per_index(
            self.initial_item_count,
            self.initial_fpp,
            default_entries_per_index,
        )];
    }

    /// Returns the estimated false positive probability of the scalable cuckoo filter. This value
    /// will increase as more items are added.
    ///
    /// # Examples
    ///
    /// use probabilistic_collections::cuckoo::ScalableCuckooFilter;
    ///
    /// let mut filter = ScalableCuckooFilter::<String>::new(100, 0.01, 2.0, 0.5);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.01);
    pub fn estimated_fpp(&self) -> f64 {
        1.0 - self
            .filters
            .iter()
            .map(|filter| 1.0 - filter.estimated_fpp())
            .product::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::ScalableCuckooFilter;

    #[test]
    pub fn test_new() {
        let mut scf = ScalableCuckooFilter::<usize>::new(100, 0.01, 2.0, 0.5);
        assert_eq!(scf.len(), 0);
        assert!(scf.is_empty());
        assert_eq!(scf.capacity(), 128);
        assert_eq!(scf.entries_per_index(), 4);

        for item in 0..130 {
            scf.insert(&item);
        }

        scf.clear();

        for item in 0..130 {
            assert!(!scf.contains(&item));
        }

        for item in 0..130 {
            scf.insert(&item);
            assert!(scf.contains(&item));
            assert_eq!(scf.len(), item + 1);
        }

        assert_eq!(scf.capacity(), 384);
        assert_eq!(scf.filter_count(), 2);
        assert_eq!(scf.filters[0].extra_items_len(), 0);
        assert_eq!(scf.filters[1].extra_items_len(), 0);
        assert_eq!(scf.filters[0].fingerprint_bit_count(), 11);
        assert_eq!(scf.filters[1].fingerprint_bit_count(), 12);

        for item in 0..130 {
            scf.remove(&item);
            assert!(!scf.contains(&item));
            assert_eq!(scf.len(), 130 - item - 1);
        }
    }

    #[test]
    pub fn test_from_entries_per_index() {
        let mut scf = ScalableCuckooFilter::<usize>::from_entries_per_index(100, 0.01, 8, 2.0, 0.5);
        assert_eq!(scf.len(), 0);
        assert!(scf.is_empty());
        assert_eq!(scf.capacity(), 128);
        assert_eq!(scf.entries_per_index(), 8);

        for item in 0..130 {
            scf.insert(&item);
        }

        scf.clear();

        for item in 0..130 {
            assert!(!scf.contains(&item));
        }

        for item in 0..130 {
            scf.insert(&item);
            assert!(scf.contains(&item));
            assert_eq!(scf.len(), item + 1);
        }

        assert_eq!(scf.capacity(), 384);
        assert_eq!(scf.filter_count(), 2);
        assert_eq!(scf.filters[0].extra_items_len(), 0);
        assert_eq!(scf.filters[1].extra_items_len(), 0);
        assert_eq!(scf.filters[0].fingerprint_bit_count(), 12);
        assert_eq!(scf.filters[1].fingerprint_bit_count(), 13);

        for item in 0..130 {
            scf.remove(&item);
            assert!(!scf.contains(&item));
            assert_eq!(scf.len(), 130 - item - 1);
        }
    }

    #[test]
    fn test_estimated_fpp() {
        let mut scf = ScalableCuckooFilter::<u32>::new(100, 0.01, 2.0, 0.5);
        assert!(scf.estimated_fpp() < std::f64::EPSILON);

        for item in 0..200 {
            scf.insert(&item);
        }

        let filter_fpp_0 = scf.filters[0].estimated_fpp();
        let filter_fpp_1 = scf.filters[1].estimated_fpp();
        let expected_fpp = 1.0 - (1.0 - filter_fpp_0) * (1.0 - filter_fpp_1);
        assert!((scf.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_ser_de() {
        let mut scf = ScalableCuckooFilter::<usize>::new(100, 0.01, 2.0, 0.5);
        scf.insert(&0);

        let serialized_scf = bincode::serialize(&scf).unwrap();
        let de_scf: ScalableCuckooFilter<usize> = bincode::deserialize(&serialized_scf).unwrap();

        assert!(scf.contains(&0));
        assert_eq!(scf.filters, de_scf.filters);
        assert_eq!(scf.initial_item_count, de_scf.initial_item_count);
        assert!((scf.initial_fpp - de_scf.initial_fpp).abs() < std::f64::EPSILON);
        assert!((scf.growth_ratio - de_scf.growth_ratio).abs() < std::f64::EPSILON);
        assert!((scf.tightening_ratio - de_scf.tightening_ratio).abs() < std::f64::EPSILON);
    }
}
