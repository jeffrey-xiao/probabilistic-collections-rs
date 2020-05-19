use rand::Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::hash::BuildHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::{cmp, fmt};

/// The default hash builder for all collections.
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy)]
pub struct SipHasherBuilder {
    k0: u64,
    k1: u64,
    hasher: SipHasher,
}

impl SipHasherBuilder {
    /// Constructs a new `SipHasherBuilder` that uses the thread-local RNG to seed itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let hash_builder = SipHasherBuilder::from_entropy();
    /// ```
    pub fn from_entropy() -> Self {
        let mut rng = rand::thread_rng();
        Self::from_seed(rng.gen(), rng.gen())
    }

    /// Constructs a new `SipHasherBuilder` that is seeded with the given keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let hash_builder = SipHasherBuilder::from_seed(0, 0);
    /// ```
    pub fn from_seed(k0: u64, k1: u64) -> Self {
        SipHasherBuilder {
            k0,
            k1,
            hasher: SipHasher::new_with_keys(k0, k1),
        }
    }
}

impl fmt::Debug for SipHasherBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SipHasherBuilder")
            .field("k0", &self.k0)
            .field("k1", &self.k1)
            .finish()
    }
}

impl cmp::PartialEq for SipHasherBuilder {
    fn eq(&self, other: &SipHasherBuilder) -> bool {
        self.k0 == other.k0 && self.k1 == other.k1
    }
}

impl BuildHasher for SipHasherBuilder {
    type Hasher = SipHasher;

    #[inline]
    fn build_hasher(&self) -> SipHasher {
        self.hasher
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, PartialEq)]
pub struct DoubleHasher<T, B = SipHasherBuilder> {
    hash_builders: [B; 2],
    _marker: PhantomData<T>,
}

impl<T> DoubleHasher<T> {
    pub fn new() -> Self {
        Self::with_hashers([
            SipHasherBuilder::from_entropy(),
            SipHasherBuilder::from_entropy(),
        ])
    }
}

impl<T, B> DoubleHasher<T, B>
where
    B: BuildHasher,
{
    pub fn with_hashers(hash_builders: [B; 2]) -> Self {
        DoubleHasher {
            hash_builders,
            _marker: PhantomData,
        }
    }

    pub fn hash<U>(&self, item: &U) -> HashIter
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        HashIter {
            a: hash(&self.hash_builders[0], &item),
            b: hash(&self.hash_builders[1], &item),
            c: 0,
        }
    }

    pub fn hashers(&self) -> &[B; 2] {
        &self.hash_builders
    }
}

pub fn hash(hash_builder: &impl BuildHasher, item: &impl Hash) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    item.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Copy)]
pub struct HashIter {
    a: u64,
    b: u64,
    c: u64,
}

impl Iterator for HashIter {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.a;
        self.a = self.a.wrapping_add(self.b);
        self.b = self.b.wrapping_add(self.c);
        self.c += 1;
        Some(ret)
    }
}

#[cfg(test)]
pub mod tests {
    use super::SipHasherBuilder;
    use siphasher::sip::SipHasher;

    pub fn hash_builder_1() -> SipHasherBuilder {
        SipHasherBuilder {
            k0: 0,
            k1: 0,
            hasher: SipHasher::new_with_keys(0, 0),
        }
    }

    pub fn hash_builder_2() -> SipHasherBuilder {
        SipHasherBuilder {
            k0: 1,
            k1: 1,
            hasher: SipHasher::new_with_keys(1, 1),
        }
    }
}
