//! Module for measuring similarities between sets.

mod min_hash;
mod sim_hash;

pub use self::min_hash::MinHash;
pub use self::sim_hash::SimHash;

use std::collections::HashSet;
use std::hash::Hash;
use std::iter::FromIterator;

/// A w-shingle iterator for an list of items.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::similarity::ShingleIterator;
///
/// let mut shingle_iter = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
///
/// assert_eq!(shingle_iter.next(), Some(vec!["the", "cat"]));
/// assert_eq!(shingle_iter.next(), Some(vec!["cat", "sat"]));
/// assert_eq!(shingle_iter.next(), Some(vec!["sat", "on"]));
/// assert_eq!(shingle_iter.next(), Some(vec!["on", "a"]));
/// assert_eq!(shingle_iter.next(), Some(vec!["a", "mat"]));
/// assert_eq!(shingle_iter.next(), None);
/// ```
pub struct ShingleIterator<'a, T>
where
    T: 'a + ?Sized,
{
    token_count: usize,
    index: usize,
    tokens: Vec<&'a T>,
}

impl<'a, T> ShingleIterator<'a, T>
where
    T: ?Sized,
{
    /// Constructs a new `ShingleIterator` that contains shingles of `token_count` tokens from
    /// `tokens`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::similarity::ShingleIterator;
    ///
    /// let mut shingle_iter = ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect());
    /// ```
    pub fn new(token_count: usize, tokens: Vec<&'a T>) -> Self {
        ShingleIterator {
            token_count,
            index: 0,
            tokens,
        }
    }
}

impl<'a, T> Iterator for ShingleIterator<'a, T>
where
    T: ?Sized,
{
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.tokens.len() - self.token_count {
            return None;
        }
        self.index += 1;
        Some(self.tokens[self.index - 1..self.index + self.token_count - 1].to_vec())
    }
}

/// Computes the Jaccard Similarity between two iterators. The Jaccard Similarity is the quotient
/// between the intersection and the union.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::similarity::{get_jaccard_similarity, ShingleIterator};
///
/// assert_eq!(
///     get_jaccard_similarity(
///         ShingleIterator::new(2, "the cat sat on a mat".split(' ').collect()),
///         ShingleIterator::new(2, "the cat sat on the mat".split(' ').collect()),
///     ),
///     3.0 / 7.0,
/// );
/// ```
pub fn get_jaccard_similarity<T, U>(iter_1: T, iter_2: T) -> f64
where
    T: Iterator<Item = U>,
    U: Hash + Eq,
{
    let h1 = HashSet::<U>::from_iter(iter_1);
    let h2 = HashSet::<U>::from_iter(iter_2);

    (h1.intersection(&h2).count() as f64) / (h1.union(&h2).count() as f64)
}

#[cfg(test)]
mod tests {
    use super::{get_jaccard_similarity, ShingleIterator};
    static S1: &'static str = "the cat sat on a mat";
    static S2: &'static str = "the cat sat on the mat";
    static S3: &'static str = "we all scream for ice cream";

    #[test]
    fn test_jaccard_similarity() {
        assert_eq!(
            get_jaccard_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S2.split(' ').collect()),
            ),
            3.0 / 7.0,
        );

        assert_eq!(
            get_jaccard_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S3.split(' ').collect()),
            ),
            0.0 / 7.0,
        );
    }
}
