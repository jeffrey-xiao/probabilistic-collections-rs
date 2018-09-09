use siphasher::sip::SipHasher;
use std::hash::Hash;
use std::marker::PhantomData;
use util;

pub struct MinHash<T, U> {
    hashers: [SipHasher; 2],
    hasher_count: usize,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> MinHash<T, U> {
    pub fn new(hasher_count: usize) -> Self {
        MinHash {
            hashers: util::get_hashers(),
            hasher_count,
            _marker: PhantomData,
        }
    }

    pub fn get_min_hashes(&self, shingles: T) -> Vec<u64>
    where
        T: Iterator<Item=U>,
        U: Hash,
    {
        let hash_pairs = shingles
            .map(|shingle| util::get_hashes::<&U, U>(&self.hashers, &shingle))
            .collect::<Vec<_>>();
        (0..self.hasher_count)
            .map(|index| {
                hash_pairs
                    .iter()
                    .map(|hashes| util::get_hash(index, hashes))
                    .min()
                    .expect("Expected non-zero `hasher_count` and shingles")
            })
            .collect()
    }

    pub fn get_similarity_from_hashes(&self, min_hashes_1: Vec<u64>, min_hashes_2: Vec<u64>) -> f64 {
        assert_eq!(min_hashes_1.len(), min_hashes_2.len());
        let matches: u64 = min_hashes_1.iter()
            .zip(min_hashes_2.iter())
            .map(|(min_hash_1, min_hash_2)| {
                if min_hash_1 == min_hash_2 {
                    return 1
                } else {
                    return 0
                }
            })
            .sum();

        (matches as f64) / (self.hasher_count as f64)
    }

    pub fn get_similarity(&self, shingles_1: T, shingles_2: T) -> f64
    where
        T: Iterator<Item=U>,
        U: Hash,
    {
        self.get_similarity_from_hashes(
            self.get_min_hashes(shingles_1),
            self.get_min_hashes(shingles_2),
        )
    }

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
