use crate::{DoubleHasher, SipHasherBuilder};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::hash::BuildHasher;
use std::hash::Hash;
use std::marker::PhantomData;

/// `MinHash` is a locality sensitive hashing scheme that can estimate the Jaccard Similarity
/// measure between two sets `s1` and `s2`. It uses multiple hash functions and for each hash
/// function `h`, finds the minimum hash value obtained from the hashing an item in `s1` using `h`
/// and hashing an item in `s2` using `h`. Our estimate for the Jaccard Similarity is the number of
/// minimum hash values that are equal divided by the number of total hash functions used.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
/// use probabilistic_collections::SipHasherBuilder;
///
/// let min_hash = MinHash::with_hashers(
///     100,
///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
/// );
///
/// assert_eq!(
///     min_hash.get_similarity(
///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
///     ),
///     0.49,
/// );
/// ```
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct MinHash<T, U, B = SipHasherBuilder> {
    hasher: DoubleHasher<U, B>,
    hasher_count: usize,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> MinHash<T, U>
where
    T: Iterator<Item = U>,
{
    /// Constructs a new `MinHash` with a specified number of hash functions to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    ///
    /// let min_hash = MinHash::<ShingleIterator<str>, _>::new(100);
    /// ```
    pub fn new(hasher_count: usize) -> Self {
        Self::with_hashers(
            hasher_count,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, U, B> MinHash<T, U, B>
where
    T: Iterator<Item = U>,
    B: BuildHasher,
{
    /// Constructs a new `MinHash` with a specified number of hash functions to use, and a hasher
    /// builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let min_hash = MinHash::<ShingleIterator<str>, _>::with_hashers(
    ///     100,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn with_hashers(hasher_count: usize, hash_builders: [B; 2]) -> Self {
        MinHash {
            hasher: DoubleHasher::with_hashers(hash_builders),
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Returns the minimum hash values obtained from a specified iterator `iter`. This function is
    /// used in conjunction with `get_similarity_from_hashes` when doing multiple comparisons.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let min_hash = MinHash::with_hashers(
    ///     100,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    ///
    /// let shingles1 = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
    /// let shingles2 = ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect());
    /// let min_hashes1 = min_hash.get_min_hashes(shingles1);
    /// let min_hashes2 = min_hash.get_min_hashes(shingles2);
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity_from_hashes(&min_hashes1, &min_hashes2),
    ///     0.49,
    /// );
    /// ```
    pub fn get_min_hashes(&self, mut iter: T) -> Vec<u64>
    where
        U: Hash,
    {
        if self.hasher_count == 0 {
            return Vec::new();
        }

        // If iterator is empty, return sentinel maxima
        let first = match iter.next() {
            Some(x) => x,
            None => return vec![u64::MAX; self.hasher_count],
        };

        let mut first_hash_iter = self.hasher.hash(&first);
        let mut mins: Vec<u64> = (0..self.hasher_count)
            .map(|_| first_hash_iter.next().expect("Expected hash"))
            .collect();

        for shingle in iter {
            let mut hash_iter = self.hasher.hash(&shingle);
            for i in 0..self.hasher_count {
                let h = hash_iter.next().expect("Expected hash");
                if h < mins[i] {
                    mins[i] = h;
                }
            }
        }

        mins
    }

    /// Returns the minimum hash values and their counts (number of shingles achieving
    /// the minimum) for each hash index. If there are no shingles and `hasher_count > 0`,
    /// the minima are set to `u64::MAX` and counts to 0. If `hasher_count == 0`, both
    /// returned vectors are empty.
    pub fn get_count_min_sketch(&self, mut iter: T) -> (Vec<u64>, Vec<usize>)
    where
        U: Hash,
    {
        if self.hasher_count == 0 {
            return (Vec::new(), Vec::new());
        }

        // Empty iterator: return sentinel maxima and zero counts
        let first = match iter.next() {
            Some(x) => x,
            None => {
                return (
                    vec![u64::MAX; self.hasher_count],
                    vec![0usize; self.hasher_count],
                )
            }
        };

        // Seed minima and counts from the first shingle
        let mut first_hash_iter = self.hasher.hash(&first);
        let mut mins: Vec<u64> = (0..self.hasher_count)
            .map(|_| first_hash_iter.next().expect("Expected hash"))
            .collect();
        let mut counts: Vec<usize> = vec![1; self.hasher_count];

        // Update with subsequent shingles
        for shingle in iter {
            let mut hash_iter = self.hasher.hash(&shingle);
            for i in 0..self.hasher_count {
                let h = hash_iter.next().expect("Expected hash");
                if h < mins[i] {
                    mins[i] = h;
                    counts[i] = 1;
                } else if h == mins[i] {
                    counts[i] = counts[i].saturating_add(1);
                }
            }
        }

        (mins, counts)
    }

    /// Returns the estimated Jaccard Similarity measure from the minimum hashes of two iterators.
    /// This function is used in conjunction with `get_min_hashes` when doing multiple comparisons.
    ///
    /// # Panics
    ///
    /// Panics if the length of the two hashes are not equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let min_hash = MinHash::with_hashers(
    ///     100,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    ///
    /// let shingles1 = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
    /// let shingles2 = ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect());
    /// let min_hashes1 = min_hash.get_min_hashes(shingles1);
    /// let min_hashes2 = min_hash.get_min_hashes(shingles2);
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity_from_hashes(&min_hashes1, &min_hashes2),
    ///     0.49,
    /// );
    /// ```
    pub fn get_similarity_from_hashes(&self, min_hashes_1: &[u64], min_hashes_2: &[u64]) -> f64 {
        assert_eq!(min_hashes_1.len(), min_hashes_2.len());
        let matches: u64 = min_hashes_1
            .iter()
            .zip(min_hashes_2.iter())
            .map(|(min_hash_1, min_hash_2)| u64::from(min_hash_1 == min_hash_2))
            .sum();

        (matches as f64) / (self.hasher_count as f64)
    }

    /// Returns the estimated Jaccard Similarity measure from two iterators `iter_1` and
    /// `iter_2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let min_hash = MinHash::with_hashers(
    ///     100,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity(
    ///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
    ///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
    ///     ),
    ///     0.49,
    /// );
    /// ```
    pub fn get_similarity(&self, iter_1: T, iter_2: T) -> f64
    where
        U: Hash,
    {
        self.get_similarity_from_hashes(&self.get_min_hashes(iter_1), &self.get_min_hashes(iter_2))
    }

    /// Returns the number of hash functions being used in `MinHash`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    ///
    /// let min_hash = MinHash::<ShingleIterator<str>, _>::new(100);
    /// assert_eq!(min_hash.hasher_count(), 100);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Returns a reference to the `MinHash`'s hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let min_hash = MinHash::<ShingleIterator<str>, _>::new(100);
    /// let hashers = min_hash.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
    }
}

#[cfg(test)]
mod tests {
    use super::MinHash;
    use crate::similarity::tests::{S1, S2, S3};
    use crate::similarity::{get_jaccard_similarity, ByteGrams, ShingleIterator};
    use crate::util::tests::{hash_builder_1, hash_builder_2};
    use std::f64;

    #[test]
    fn test_min_hash() {
        let min_hash = MinHash::with_hashers(100, [hash_builder_1(), hash_builder_2()]);

        let similarity = min_hash.get_similarity(
            ShingleIterator::new(2, S1.split(' ').collect()),
            ShingleIterator::new(2, S2.split(' ').collect()),
        );
        assert!(f64::abs(similarity - 0.49) < f64::EPSILON);

        let similarity = min_hash.get_similarity(
            ShingleIterator::new(2, S1.split(' ').collect()),
            ShingleIterator::new(2, S3.split(' ').collect()),
        );
        assert!(f64::abs(similarity - 0.00) < f64::EPSILON);

        let hash1 = min_hash.get_min_hashes(ShingleIterator::new(2, S1.split(' ').collect()));
        let hash2 = min_hash.get_min_hashes(ShingleIterator::new(2, S2.split(' ').collect()));
        let hash3 = min_hash.get_min_hashes(ShingleIterator::new(2, S3.split(' ').collect()));
        assert!(
            f64::abs(min_hash.get_similarity_from_hashes(&hash1, &hash2) - 0.49) < f64::EPSILON
        );
        assert!(
            f64::abs(min_hash.get_similarity_from_hashes(&hash1, &hash3) - 0.00) < f64::EPSILON
        );

        assert_eq!(min_hash.hasher_count(), 100);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let min_hash = MinHash::new(100);
        let serialized_min_hash = bincode::serialize(&min_hash).unwrap();
        let de_min_hash: MinHash<ShingleIterator<str>, _> =
            bincode::deserialize(&serialized_min_hash).unwrap();

        let sim = min_hash.get_similarity(
            ShingleIterator::new(2, S1.split(' ').collect()),
            ShingleIterator::new(2, S2.split(' ').collect()),
        );
        let de_sim = de_min_hash.get_similarity(
            ShingleIterator::new(2, S1.split(' ').collect()),
            ShingleIterator::new(2, S2.split(' ').collect()),
        );
        assert!((sim - de_sim).abs() < f64::EPSILON);

        assert_eq!(min_hash.hasher_count(), de_min_hash.hasher_count());
        assert_eq!(min_hash.hashers(), de_min_hash.hashers());
    }

    #[test]
    fn test_empty_iterator_returns_max_and_zero_counts() {
        let min_hash = MinHash::with_hashers(5, [hash_builder_1(), hash_builder_2()]);

        let empty: Vec<&str> = Vec::new();
        let mins = min_hash.get_min_hashes(empty.clone().into_iter());
        assert_eq!(mins, vec![u64::MAX; 5]);

        let (mins2, counts) = min_hash.get_count_min_sketch(empty.into_iter());
        assert_eq!(mins2, vec![u64::MAX; 5]);
        assert_eq!(counts, vec![0usize; 5]);
    }

    #[test]
    fn test_zero_hashers_returns_empty_vectors() {
        let min_hash = MinHash::with_hashers(0, [hash_builder_1(), hash_builder_2()]);
        let items = vec!["a", "b", "c"]; // non-empty iterator
        let mins = min_hash.get_min_hashes(items.clone().into_iter());
        assert!(mins.is_empty());
        let (mins2, counts2) = min_hash.get_count_min_sketch(items.into_iter());
        assert!(mins2.is_empty());
        assert!(counts2.is_empty());
    }

    #[test]
    fn test_counts_with_repeated_identical_shingles() {
        let k = 8;
        let min_hash = MinHash::with_hashers(k, [hash_builder_1(), hash_builder_2()]);

        // Single item establishes baseline minima and counts of 1
        let (mins_single, counts_single) = min_hash.get_count_min_sketch(vec!["x"].into_iter());
        assert_eq!(counts_single, vec![1usize; k]);

        // Repeating the same item N times should keep the same minima and increase counts to N
        let n: usize = 3;
        let (mins_repeated, counts_repeated) =
            min_hash.get_count_min_sketch(vec!["x", "x", "x"].into_iter());
        assert_eq!(mins_repeated, mins_single);
        assert_eq!(counts_repeated, vec![n; k]);

        // Mixing in different items should not reduce counts below 1 and should
        // preserve minima if the repeated item already yields minima.
        let (mins_mixed, counts_mixed) =
            min_hash.get_count_min_sketch(vec!["x", "y", "x", "z", "x"].into_iter());
        assert!(mins_mixed
            .iter()
            .zip(mins_single.iter())
            .all(|(a, b)| *a <= *b));
        assert!(counts_mixed.iter().all(|&c| c >= 1));
    }

    #[test]
    fn test_min_hash_with_byte_grams_approximates_jaccard() {
        let data1 = S1.as_bytes();
        let data2 = S2.as_bytes();
        let width = 3;

        let exact = get_jaccard_similarity(
            ByteGrams::new(data1, width),
            ByteGrams::new(data2, width),
        );

        let min_hash = MinHash::with_hashers(256, [hash_builder_1(), hash_builder_2()]);
        let approx = min_hash.get_similarity(
            ByteGrams::new(data1, width),
            ByteGrams::new(data2, width),
        );

        // Allow some tolerance due to probabilistic estimation; 0.1 should be safe for 256 hashes.
        assert!((approx - exact).abs() <= 0.1, "approx={} exact={}", approx, exact);
    }
}
