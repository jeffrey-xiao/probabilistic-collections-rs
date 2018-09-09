use rand::{Rng, XorShiftRng};
use siphasher::sip::SipHasher;
use std::cmp;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::marker::PhantomData;

pub struct SimHash<T, U> {
    hasher: SipHasher,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> SimHash<T, U> {
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

    pub fn get_sim_hash(&self, s: T) -> u64
    where
        T: Iterator<Item=U>,
        U: Hash,
    {
        let mut counts = [0i64; 64];
        for hash in s.map(|item| self.get_hash(&item)) {
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

    pub fn report_similarities(
        &self,
        window_size: usize,
        shingles_vec: Vec<T>,
    ) -> Vec<(usize, usize)>
    where
        T: Iterator<Item=U>,
        U: Hash,
    {
        assert!(window_size > 1);
        let mut sim_hashes: Vec<_> = shingles_vec
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
