use rand::{Rng, XorShiftRng};
use siphasher::sip::SipHasher;
use std::cmp;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::marker::PhantomData;

/// `SimHash` is a locality sensitive hashing scheme. If two sets `s1` and `s2` are similar,
/// `SimHash` will generate hashes for `s1` and `s2` that has a small Hamming Distance between
/// them.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
///
/// let sim_hash = SimHash::new();
///
/// assert_eq!(
///     sim_hash.get_sim_hash(ShingleIterator::new(
///         2,
///         "the cat sat on a mat".split(' ').collect()
///     )),
///     0b1111011011001001011100000010010110011011101011110110000010001001,
/// );
/// assert_eq!(
///     sim_hash.get_sim_hash(ShingleIterator::new(
///         2,
///         "the cat sat on the mat".split(' ').collect()
///     )),
///     0b0111011001000001011110000011011010011011101001100101101000000001,
/// );
/// ```
pub struct SimHash<T, U> {
    hasher: SipHasher,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> SimHash<T, U> {
    /// Constructs a new `SimHash`,
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    ///
    /// let sim_hash = SimHash::<ShingleIterator<&str>, &str>::new();
    /// ```
    pub fn new() -> Self {
        let mut rng = XorShiftRng::new_unseeded();
        SimHash {
            hasher: SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            _marker: PhantomData,
        }
    }

    fn get_hash(&self, item: &U) -> u64
    where
        U: Hash,
    {
        let mut sip = self.hasher;
        item.hash(&mut sip);
        sip.finish()
    }

    /// Returns the hash associated with iterator `iter`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    ///
    /// let sim_hash = SimHash::new();
    ///
    /// assert_eq!(
    ///     sim_hash.get_sim_hash(ShingleIterator::new(
    ///         2,
    ///         "the cat sat on a mat".split(' ').collect()
    ///     )),
    ///     0b1111011011001001011100000010010110011011101011110110000010001001,
    /// );
    /// ```
    pub fn get_sim_hash(&self, iter: T) -> u64
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        let mut counts = [0i64; 64];
        for hash in iter.map(|item| self.get_hash(&item)) {
            for i in 0..64 {
                if (hash >> i) & 1 == 0 {
                    counts[i] += 1;
                } else {
                    counts[i] -= 1;
                }
            }
        }

        counts.iter().fold(0, |acc, count| {
            if *count >= 0 {
                (acc << 1) | 1
            } else {
                acc << 1
            }
        })
    }

    /// Returns all pairs of indexes corresponding to iterators in `iter_vec` that could be similar
    /// based on `window_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    ///
    /// let sim_hash = SimHash::new();
    ///
    /// sim_hash.report_similarities(
    ///     2,
    ///     vec![
    ///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
    ///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
    ///         ShingleIterator::new(2, "we all scream for ice cream".split(' ').collect()),
    ///     ],
    /// );
    /// ```
    pub fn report_similarities(&self, window_size: usize, iter_vec: Vec<T>) -> Vec<(usize, usize)>
    where
        T: Iterator<Item = U>,
        U: Hash,
    {
        assert!(window_size > 1);
        let mut sim_hashes: Vec<_> = iter_vec
            .into_iter()
            .enumerate()
            .map(|(index, shingles)| (self.get_sim_hash(shingles), index))
            .collect();

        let mut similarities = HashSet::new();

        for _ in 0..64 {
            sim_hashes.sort();
            for i in 0..sim_hashes.len() - window_size + 1 {
                for j in i..i + window_size {
                    for k in j + 1..i + window_size {
                        similarities.insert((
                            cmp::min(sim_hashes[j].1, sim_hashes[k].1),
                            cmp::max(sim_hashes[j].1, sim_hashes[k].1),
                        ));
                    }
                }
            }

            for sim_hash in &mut sim_hashes {
                sim_hash.0.rotate_left(1);
            }
        }

        Vec::from_iter(similarities.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::SimHash;
    use similarity::ShingleIterator;

    static S1: &'static str = "the cat sat on a mat";
    static S2: &'static str = "the cat sat on the mat";
    static S3: &'static str = "we all scream for ice cream";

    #[test]
    fn test_sim_hash() {
        let sim_hash = SimHash::new();

        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S1.split(' ').collect())),
            0b1111011011001001011100000010010110011011101011110110000010001001,
        );
        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S2.split(' ').collect())),
            0b0111011001000001011110000011011010011011101001100101101000000001,
        );
        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S3.split(' ').collect())),
            0b11100101100111101010111000110101100011100100100001101000000000,
        );

        let similarities = sim_hash.report_similarities(
            2,
            vec![
                ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
                ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
                ShingleIterator::new(2, "we all scream for ice cream".split(' ').collect()),
            ],
        );

        assert!(similarities.contains(&(0, 1)));
        assert!(similarities.contains(&(1, 2)));
    }
}
