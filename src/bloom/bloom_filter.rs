use bit_vec::BitVec;
use rand::{Rng, XorShiftRng};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use siphasher::sip::SipHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A space-efficient probabilistic data structure to test for membership in a set.
///
/// At its core, a bloom filter is a bit array, initially all set to zero. `K` hash functions
/// map each element to `K` bits in the bit array. An element definitely does not exist in the
/// bloom filter if any of the `K` bits are unset. An element is possibly in the set if all of the
/// `K` bits are set. This particular implementation of a bloom filter uses two hash functions to
/// simulate `K` hash functions. Additionally, it operates on only one "slice" in order to have
/// predictable memory usage.
///
/// # Examples
/// ```
/// use probabilistic_collections::bloom::BloomFilter;
///
/// let mut filter = BloomFilter::new(10, 0.01);
///
/// assert!(!filter.contains(&"foo"));
/// filter.insert(&"foo");
/// assert!(filter.contains(&"foo"));
///
/// filter.clear();
/// assert!(!filter.contains(&"foo"));
///
/// assert_eq!(filter.len(), 96);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
#[derive(Clone)]
pub struct BloomFilter {
    bit_vec: BitVec,
    hashers: [SipHasher; 2],
    hasher_count: usize,
}

impl BloomFilter {
    fn get_hashers() -> [SipHasher; 2] {
        let mut rng = XorShiftRng::new_unseeded();
        [
            SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
        ]
    }

    fn get_hasher_count(bit_count: usize, item_count: usize) -> usize {
        ((bit_count as f64) / (item_count as f64) * 2f64.ln()).ceil() as usize
    }

    /// Constructs a new, empty `BloomFilter` with an estimated max capacity of `item_count` items
    /// and a maximum false positive probability of `fpp`.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::new(10, 0.01);
    /// ```
    pub fn new(item_count: usize, fpp: f64) -> Self {
        let bit_count = (-fpp.log2() * (item_count as f64) / 2f64.ln()).ceil() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: Self::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits and an estimated max capacity
    /// of `item_count` items.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::from_item_count(100, 10);
    /// ```
    pub fn from_item_count(bit_count: usize, item_count: usize) -> Self {
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: Self::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
        }
    }

    /// Constructs a new, empty `BloomFilter` with `bit_count` bits and a maximum false positive
    /// probability of `fpp`.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::from_fpp(100, 0.01);
    /// ```
    pub fn from_fpp(bit_count: usize, fpp: f64) -> Self {
        let item_count = -(2f64.ln() * (bit_count as f64) / fpp.log2()).floor() as usize;
        BloomFilter {
            bit_vec: BitVec::new(bit_count),
            hashers: Self::get_hashers(),
            hasher_count: Self::get_hasher_count(bit_count, item_count),
        }
    }

    fn get_hashes<T>(&self, item: &T) -> [u64; 2]
    where
        T: Hash,
    {
        let mut ret = [0; 2];
        for (index, hash) in ret.iter_mut().enumerate() {
            let mut sip = self.hashers[index];
            item.hash(&mut sip);
            *hash = sip.finish();
        }
        ret
    }

    /// Inserts an element into the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(10, 0.01);
    ///
    /// filter.insert(&"foo");
    /// ```
    pub fn insert<T>(&mut self, item: &T)
    where
        T: Hash,
    {
        let hashes = self.get_hashes(item);
        for index in 0..self.hasher_count {
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_vec.len() as u64;
            self.bit_vec.set(offset as usize, true);
        }
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(10, 0.01);
    ///
    /// assert!(!filter.contains(&"foo"));
    /// filter.insert(&"foo");
    /// assert!(filter.contains(&"foo"));
    /// ```
    pub fn contains<T>(&self, item: &T) -> bool
    where
        T: Hash,
    {
        let hashes = self.get_hashes(item);
        (0..self.hasher_count).all(|index| {
            let mut offset = (index as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.bit_vec.len() as u64;
            self.bit_vec[offset as usize]
        })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::from_fpp(100, 0.01);
    ///
    /// assert_eq!(filter.len(), 100);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::from_fpp(100, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.bit_vec.is_empty()
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let filter = BloomFilter::new(10, 0.01);
    ///
    /// assert_eq!(filter.hasher_count(), 7);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Clears the bloom filter, removing all elements.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(10, 0.01);
    ///
    /// filter.insert(&"foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains(&"foo"));
    /// ```
    pub fn clear(&mut self) {
        self.bit_vec.set_all(false)
    }

    /// Returns the number of set bits in the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::from_fpp(100, 0.01);
    /// filter.insert(&"foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bit_vec.count_ones()
    }

    /// Returns the number of unset bits in the bloom filter.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::from_fpp(100, 0.01);
    /// filter.insert(&"foo");
    ///
    /// assert_eq!(filter.count_zeros(), 93);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }

    /// Returns the estimated false positive probability of the bloom filter. This value will
    /// increase as more items are added.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::bloom::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01);
    /// assert!(filter.estimate_fpp() < 1e-6);
    ///
    /// filter.insert(&0);
    /// assert!(filter.estimate_fpp() < 0.01);
    /// ```
    pub fn estimate_fpp(&self) -> f64 {
        let single_fpp = self.bit_vec.count_ones() as f64 / self.bit_vec.len() as f64;
        single_fpp.powi(self.hasher_count as i32)
    }
}

impl PartialEq for BloomFilter {
    fn eq(&self, other: &BloomFilter) -> bool {
        self.hasher_count == other.hasher_count
            && self.hashers[0].keys() == other.hashers[0].keys()
            && self.hashers[1].keys() == other.hashers[1].keys()
            && self.bit_vec == other.bit_vec
    }
}

impl Serialize for BloomFilter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BloomFilter", 4)?;
        state.serialize_field("hasher_count", &self.hasher_count)?;
        state.serialize_field("keys_0", &self.hashers[0].keys())?;
        state.serialize_field("keys_1", &self.hashers[1].keys())?;
        state.serialize_field("bit_vec", &self.bit_vec)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for BloomFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field { HasherCount, Keys0, Keys1, BitVec };

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`hasher_count`, `keys_0`, `keys_1`, or `bit_vec`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "hasher_count" => Ok(Field::HasherCount),
                            "keys_0" => Ok(Field::Keys0),
                            "keys_1" => Ok(Field::Keys1),
                            "bit_vec" => Ok(Field::BitVec),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct BloomFilterVisitor;

        impl<'de> Visitor<'de> for BloomFilterVisitor {
            type Value = BloomFilter;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct BloomFilter")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let hasher_count = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let keys_0: (u64, u64) = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let keys_1: (u64, u64) = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let bit_vec = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                Ok(BloomFilter {
                    hasher_count,
                    hashers: [
                        SipHasher::new_with_keys(keys_0.0, keys_0.1),
                        SipHasher::new_with_keys(keys_1.0, keys_1.1),
                    ],
                    bit_vec,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut hasher_count = None;
                let mut keys_0 = None;
                let mut keys_1 = None;
                let mut bit_vec = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::HasherCount => {
                            if hasher_count.is_some() {
                                return Err(de::Error::duplicate_field("hasher_count"));
                            }
                            hasher_count = Some(map.next_value()?);
                        },
                        Field::Keys0 => {
                            if keys_0.is_some() {
                                return Err(de::Error::duplicate_field("keys_0"));
                            }
                            keys_0 = Some(map.next_value()?);
                        },
                        Field::Keys1 => {
                            if keys_1.is_some() {
                                return Err(de::Error::duplicate_field("keys_1"));
                            }
                            keys_1 = Some(map.next_value()?);
                        },
                        Field::BitVec => {
                            if bit_vec.is_some() {
                                return Err(de::Error::duplicate_field("bit_vec"));
                            }
                            bit_vec = Some(map.next_value()?);
                        },
                    }
                }
                let hasher_count = hasher_count.ok_or_else(|| de::Error::missing_field("hasher_count"))?;
                let keys_0: (u64, u64) = keys_0.ok_or_else(|| de::Error::missing_field("keys_0"))?;
                let keys_1: (u64, u64) = keys_1.ok_or_else(|| de::Error::missing_field("keys_1"))?;
                let bit_vec = bit_vec.ok_or_else(|| de::Error::missing_field("bit_vec"))?;
                Ok(BloomFilter {
                    hasher_count,
                    hashers: [
                        SipHasher::new_with_keys(keys_0.0, keys_0.1),
                        SipHasher::new_with_keys(keys_1.0, keys_1.1),
                    ],
                    bit_vec,
                })
            }
        }
        const FIELDS: &[&str] = &["hasher_count", "keys_0", "keys_1", "bit_vec"];
        deserializer.deserialize_struct("BloomFilter", FIELDS, BloomFilterVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;
    use bincode::{deserialize, serialize};

    #[test]
    fn test_new() {
        let mut filter = BloomFilter::new(10, 0.01);

        assert!(!filter.contains(&"foo"));
        filter.insert(&"foo");
        assert!(filter.contains(&"foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 89);

        filter.clear();
        assert!(!filter.contains(&"foo"));

        assert_eq!(filter.len(), 96);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_from_fpp() {
        let mut filter = BloomFilter::from_fpp(100, 0.01);

        assert!(!filter.contains(&"foo"));
        filter.insert(&"foo");
        assert!(filter.contains(&"foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 93);

        filter.clear();
        assert!(!filter.contains(&"foo"));

        assert_eq!(filter.len(), 100);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_from_item_count() {
        let mut filter = BloomFilter::from_item_count(100, 10);

        assert!(!filter.contains(&"foo"));
        filter.insert(&"foo");
        assert!(filter.contains(&"foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 93);

        filter.clear();
        assert!(!filter.contains(&"foo"));

        assert_eq!(filter.len(), 100);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_estimate_fpp() {
        let mut filter = BloomFilter::new(100, 0.01);
        assert!(filter.estimate_fpp() < 1e-6);

        filter.insert(&0);

        let expected_fpp = (7f64 / 959f64).powi(7);
        assert!((filter.estimate_fpp() - expected_fpp).abs() < 1e-15);
    }

    #[test]
    fn test_ser_de() {
        let mut filter = BloomFilter::new(100, 0.01);
        filter.insert(&"foo");

        let keys_0 = filter.hashers[0].keys();
        let keys_1 = filter.hashers[1].keys();

        let serialized_filter = serialize(&filter).unwrap();
        filter = deserialize(&serialized_filter).unwrap();

        assert!(filter.contains(&"foo"));
        assert_eq!(filter.hasher_count, 7);
        assert_eq!(filter.hashers[0].keys(), keys_0);
        assert_eq!(filter.hashers[1].keys(), keys_1);
    }
}
