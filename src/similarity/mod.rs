mod min_hash;
mod sim_hash;

pub use self::min_hash::MinHash;
pub use self::sim_hash::SimHash;

use std::collections::HashSet;
use std::hash::Hash;
use std::iter::FromIterator;

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
    T: ?Sized
{
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
    T: ?Sized
{
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.tokens.len() - self.token_count {
            return None;
        }
        self.index += 1;
        return Some(self.tokens[self.index - 1..self.index + self.token_count - 1].to_vec())
    }
}

pub fn get_jacard_similarity<T, U>(shingles_1: T, shingles_2: T) -> f64
where
    T: Iterator<Item=U>,
    U: Hash + Eq,
{
    let h1 = HashSet::<U>::from_iter(shingles_1);
    let h2 = HashSet::<U>::from_iter(shingles_2);

    return (h1.intersection(&h2).count() as f64) / (h1.union(&h2).count() as f64);
}

#[cfg(test)]
mod tests {
    use super::{get_jacard_similarity, ShingleIterator};
    static S1: &'static str = "the cat sat on a mat";
    static S2: &'static str = "the cat sat on the mat";
    static S3: &'static str = "we all scream for ice cream";


    #[test]
    fn test_jacard_similarity() {
        assert_eq!(
            get_jacard_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S2.split(' ').collect()),
            ),
            3.0 / 7.0,
        );

        assert_eq!(
            get_jacard_similarity(
                ShingleIterator::new(2, S1.split(' ').collect()),
                ShingleIterator::new(2, S3.split(' ').collect()),
            ),
            0.0 / 7.0,
        );
    }
}
