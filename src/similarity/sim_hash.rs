use crate::util;
use crate::SipHasherBuilder;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::cmp;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};
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
/// use probabilistic_collections::SipHasherBuilder;
///
/// let sim_hash = SimHash::with_hasher(SipHasherBuilder::from_seed(0, 0));
///
/// assert_eq!(
///     sim_hash.get_sim_hash(ShingleIterator::new(
///         2,
///         "the cat sat on a mat".split(' ').collect()
///     )),
///     0b1000_0101_1011_1000_0010_1111_1011_0000_1110_1000_0011_1011_0110_0100_0000_0100,
/// );
/// assert_eq!(
///     sim_hash.get_sim_hash(ShingleIterator::new(
///         2,
///         "the cat sat on the mat".split(' ').collect()
///     )),
///     0b0000_0101_0011_1000_0001_1111_1111_1000_0111_1011_0011_0001_1010_0001_0000_0110,
/// );
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct SimHash<T, U, B = SipHasherBuilder> {
    hash_builder: B,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> SimHash<T, U>
where
    T: Iterator<Item = U>,
{
    /// Constructs a new `SimHash`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    ///
    /// let sim_hash = SimHash::<ShingleIterator<str>, _>::new();
    /// ```
    pub fn new() -> Self {
        Self::with_hasher(SipHasherBuilder::from_entropy())
    }
}

impl<T, U, B> SimHash<T, U, B>
where
    T: Iterator<Item = U>,
    B: BuildHasher,
{
    /// Constructs a new `SimHash` with a specified hasher builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let sim_hash = SimHash::<ShingleIterator<str>, _>::with_hasher(
    ///     SipHasherBuilder::from_entropy(),
    /// );
    /// ```
    pub fn with_hasher(hash_builder: B) -> Self {
        SimHash {
            hash_builder,
            _marker: PhantomData,
        }
    }

    /// Returns the hash associated with iterator `iter`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let sim_hash = SimHash::with_hasher(SipHasherBuilder::from_seed(0, 0));
    ///
    /// assert_eq!(
    ///     sim_hash.get_sim_hash(ShingleIterator::new(
    ///         2,
    ///         "the cat sat on a mat".split(' ').collect()
    ///     )),
    ///     0b1000_0101_1011_1000_0010_1111_1011_0000_1110_1000_0011_1011_0110_0100_0000_0100,
    /// );
    /// ```
    pub fn get_sim_hash(&self, iter: T) -> u64
    where
        U: Hash,
    {
        let mut counts = [0i64; 64];
        for hash in iter.map(|item| util::hash(&self.hash_builder, &item)) {
            for (i, count) in counts.iter_mut().enumerate() {
                if (hash >> i) & 1 == 0 {
                    *count += 1;
                } else {
                    *count -= 1;
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
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let sim_hash = SimHash::with_hasher(SipHasherBuilder::from_seed(0, 0));
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
            for i in 0..=sim_hashes.len() - window_size {
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
                sim_hash.0 = sim_hash.0.rotate_left(1);
            }
        }

        Vec::from_iter(similarities.into_iter())
    }

    /// Returns a reference to the `SimHash`'s hasher builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::{ShingleIterator, SimHash};
    ///
    /// let sim_hash = SimHash::<ShingleIterator<str>, _>::new();
    /// let hasher = sim_hash.hasher();
    /// ```
    pub fn hasher(&self) -> &B {
        &self.hash_builder
    }
}

impl<T, U> Default for SimHash<T, U>
where
    T: Iterator<Item = U>,
{
    fn default() -> SimHash<T, U> {
        SimHash::new()
    }
}

#[cfg(test)]
mod tests {
    use super::SimHash;
    use crate::similarity::tests::{S1, S2, S3};
    use crate::similarity::ShingleIterator;
    use crate::util::tests::hash_builder_1;

    #[test]
    fn test_sim_hash() {
        let sim_hash = SimHash::with_hasher(hash_builder_1());

        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S1.split(' ').collect())),
            0b1000_0101_1011_1000_0010_1111_1011_0000_1110_1000_0011_1011_0110_0100_0000_0100,
        );
        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S2.split(' ').collect())),
            0b0000_0101_0011_1000_0001_1111_1111_1000_0111_1011_0011_0001_1010_0001_0000_0110,
        );
        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S3.split(' ').collect())),
            0b1001_0011_1010_1010_1100_1011_0000_0000_0010_1100_0010_0001_0101_1000_0111_1101,
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

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let sim_hash = SimHash::default();
        let serialized_sim_hash = bincode::serialize(&sim_hash).unwrap();
        let de_sim_hash: SimHash<ShingleIterator<str>, _> =
            bincode::deserialize(&serialized_sim_hash).unwrap();

        assert_eq!(
            sim_hash.get_sim_hash(ShingleIterator::new(2, S1.split(' ').collect())),
            de_sim_hash.get_sim_hash(ShingleIterator::new(2, S1.split(' ').collect())),
        );
        assert_eq!(sim_hash.hasher(), de_sim_hash.hasher());
    }
}
