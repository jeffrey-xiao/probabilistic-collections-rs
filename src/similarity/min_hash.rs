use siphasher::sip::SipHasher;
use std::hash::Hash;
use std::marker::PhantomData;
use util;

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
///
/// let min_hash = MinHash::new(100);
///
/// assert_eq!(
///     min_hash.get_similarity(
///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
///     ),
///     0.42,
/// );
/// ```
pub struct MinHash<T, U> {
    hashers: [SipHasher; 2],
    hasher_count: usize,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> MinHash<T, U> {
    /// Constructs a new `MinHash` with a specified number of hash functions to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{MinHash, ShingleIterator};
    ///
    /// let min_hash = MinHash::<ShingleIterator<&str>, &str>::new(100);
    /// ```
    pub fn new(hasher_count: usize) -> Self {
        MinHash {
            hashers: util::get_hashers(),
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
    ///
    /// let min_hash = MinHash::new(100);
    ///
    /// let shingles1 = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
    /// let shingles2 = ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect());
    /// let min_hashes1 = min_hash.get_min_hashes(shingles1);
    /// let min_hashes2 = min_hash.get_min_hashes(shingles2);
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity_from_hashes(&min_hashes1, &min_hashes2),
    ///     0.42
    /// );
    /// ```
    pub fn get_min_hashes(&self, iter: T) -> Vec<u64>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let hash_pairs = iter
            .map(|shingle| util::get_hashes::<&U, U>(&self.hashers, &shingle))
            .collect::<Vec<_>>();
        (0..self.hasher_count)
            .map(|index| {
                hash_pairs
                    .iter()
                    .map(|hashes| util::get_hash(index, hashes))
                    .min()
                    .expect("Expected non-zero `hasher_count` and shingles.")
            })
            .collect()
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
    ///
    /// let min_hash = MinHash::new(100);
    ///
    /// let shingles1 = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
    /// let shingles2 = ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect());
    /// let min_hashes1 = min_hash.get_min_hashes(shingles1);
    /// let min_hashes2 = min_hash.get_min_hashes(shingles2);
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity_from_hashes(&min_hashes1, &min_hashes2),
    ///     0.42
    /// );
    /// ```
    pub fn get_similarity_from_hashes(&self, min_hashes_1: &[u64], min_hashes_2: &[u64]) -> f64 {
        assert_eq!(min_hashes_1.len(), min_hashes_2.len());
        let matches: u64 = min_hashes_1
            .iter()
            .zip(min_hashes_2.iter())
            .map(
                |(min_hash_1, min_hash_2)| {
                    if min_hash_1 == min_hash_2 {
                        1
                    } else {
                        0
                    }
                },
            )
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
    ///
    /// let min_hash = MinHash::new(100);
    ///
    /// assert_eq!(
    ///     min_hash.get_similarity(
    ///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
    ///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
    ///     ),
    ///     0.42,
    /// );
    /// ```
    pub fn get_similarity(&self, iter_1: T, iter_2: T) -> f64
    where
        T: Iterator<Item = U>,
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
    /// let min_hash = MinHash::<ShingleIterator<&str>, &str>::new(100);
    /// assert_eq!(min_hash.hasher_count(), 100);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }
}

#[cfg(test)]
mod tests {
    use super::MinHash;
    use similarity::ShingleIterator;

    static S1: &'static str = "the cat sat on a mat";
    static S2: &'static str = "the cat sat on the mat";
    static S3: &'static str = "we all scream for ice cream";

    #[test]
    fn test_min_hash() {
        let min_hash = MinHash::new(100);

        assert_eq!(
            min_hash.get_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S2.split(' ').collect()),
            ),
            0.42,
        );

        assert_eq!(
            min_hash.get_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S3.split(' ').collect()),
            ),
            0.00,
        );

        assert_eq!(min_hash.hasher_count(), 100);
    }
}
