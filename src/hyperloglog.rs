//! Space-efficient probabilistic data structure for estimating the number of distinct items in a
//! multiset.

use rand::{Rng, XorShiftRng};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::cmp;
use std::f64;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// A space-efficient probabilitic data structure to count the number of distinct items in a
/// multiset.
///
/// A `HyperLogLog<T>` uses the observation that the cardinality of a multiset of uniformly
/// distributed items can be estimated by calculating the maximum number of leading zeros in the
/// hash of each item in the multiset. It also buckets each item in a register and takes the
/// harmonic mean of the count in order to reduce the variance. Finally, it uses linear counting
/// for small cardinalities and small correction for large cardinalities.
///
/// # Examples
///
/// ```
/// # use std::f64::EPSILON;
/// use probabilistic_collections::hyperloglog::HyperLogLog;
///
/// let mut hhl = HyperLogLog::<u32>::new(0.1);
///
/// assert!(hhl.is_empty());
///
/// for key in &[0, 1, 2, 0, 1, 2] {
///     hhl.insert(&key);
/// }
///
/// assert!((hhl.len().round() - 3.0).abs() < EPSILON);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct HyperLogLog<T> {
    alpha: f64,
    p: usize,
    registers: Vec<u8>,
    hasher: SipHasher,
    _marker: PhantomData<T>,
}

impl<T> HyperLogLog<T> {
    fn get_hasher() -> SipHasher {
        let mut rng = XorShiftRng::new_unseeded();
        SipHasher::new_with_keys(rng.next_u64(), rng.next_u64())
    }

    fn get_alpha(p: usize) -> f64 {
        assert!(4 <= p && p <= 16);
        match p {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            p => 0.7213 / (1.0 + 1.079 / f64::from(1 << p)),
        }
    }

    /// Constructs a new, empty `HyperLogLog<T>` with a given error probability;
    ///
    /// # Panics
    ///
    /// Panics if `error_probability` is not in (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let hhl: HyperLogLog<u32> = HyperLogLog::<u32>::new(0.1);
    /// ```
    pub fn new(error_probability: f64) -> Self {
        assert!(0.0 < error_probability && error_probability < 1.0);
        let p = (1.04 / error_probability).powi(2).ln().ceil() as usize;
        let alpha = Self::get_alpha(p);
        let registers_len = 1 << p;
        HyperLogLog {
            alpha,
            p,
            registers: vec![0; registers_len],
            hasher: Self::get_hasher(),
            _marker: PhantomData,
        }
    }

    /// Inserts an item into the `HyperLogLog<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let mut hhl = HyperLogLog::<u32>::new(0.1);
    ///
    /// hhl.insert(&0);
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let mut sip = self.hasher;
        item.hash(&mut sip);
        let hash = sip.finish();
        let register_index = hash as usize & (self.registers.len() - 1);
        let value = (!hash >> self.p).trailing_zeros() as u8;
        self.registers[register_index] = cmp::max(self.registers[register_index], value + 1);
    }

    /// Merges `self` with `other`.
    ///
    /// # Panics
    ///
    /// Panics if the error probability of `self` is not equal to the error probability of `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::f64::EPSILON;
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let mut hhl1 = HyperLogLog::<u32>::new(0.1);
    /// hhl1.insert(&0);
    /// hhl1.insert(&1);
    ///
    /// let mut hhl2 = HyperLogLog::<u32>::new(0.1);
    /// hhl2.insert(&0);
    /// hhl2.insert(&2);
    ///
    /// hhl1.merge(&hhl2);
    ///
    /// assert!((hhl1.len().round() - 3.0).abs() < EPSILON);
    /// ```
    pub fn merge(&mut self, other: &HyperLogLog<T>) {
        assert_eq!(self.p, other.p);

        for (index, value) in self.registers.iter_mut().enumerate() {
            *value = cmp::max(*value, other.registers[index]);
        }
    }

    fn get_estimate(&self) -> f64 {
        let len = self.registers.len() as f64;
        1.0 / (self.alpha
            * len
            * len
            * self
                .registers
                .iter()
                .map(|value| 1.0 / 2.0f64.powi(i32::from(*value)))
                .sum::<f64>())
    }

    /// Returns the estimated number of distinct items in the `HyperLogLog<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::f64::EPSILON;
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let mut hhl = HyperLogLog::<u32>::new(0.1);
    /// assert!((hhl.len().round() - 0.0).abs() < EPSILON);
    ///
    /// hhl.insert(&1);
    /// assert!((hhl.len().round() - 1.0).abs() < EPSILON);
    /// ```
    pub fn len(&self) -> f64 {
        let len = self.registers.len() as f64;
        match self.get_estimate() {
            x if x <= 2.5 * len => {
                let zeros = self
                    .registers
                    .iter()
                    .map(|value| if *value == 0 { 1 } else { 0 })
                    .sum::<u64>();
                len * (len / zeros as f64).ln()
            }
            x if x <= 1.0 / 3.0 * 2.0f64.powi(32) => x,
            x => -(2.0f64.powi(32)) * (1.0 - x / 2.0f64.powi(32)).ln(),
        }
    }

    /// Returns `true` is the `HyperLogLog<T>` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::f64::EPSILON;
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let mut hhl = HyperLogLog::<u32>::new(0.1);
    /// assert!(hhl.is_empty());
    ///
    /// hhl.insert(&1);
    /// assert!(!hhl.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() < f64::EPSILON
    }

    /// Clears the `HyperLogLog<T>`, removing all items.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::f64::EPSILON;
    /// use probabilistic_collections::hyperloglog::HyperLogLog;
    ///
    /// let mut hhl = HyperLogLog::<u32>::new(0.1);
    /// assert!(hhl.is_empty());
    ///
    /// hhl.insert(&1);
    /// assert!(!hhl.is_empty());
    ///
    /// hhl.clear();
    /// assert!(hhl.is_empty());
    /// ```
    pub fn clear(&mut self) {
        for value in &mut self.registers {
            *value = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HyperLogLog;
    use std::f64::EPSILON;

    #[test]
    #[should_panic]
    fn test_panic_new_invalid_error_probability() {
        let _hhl: HyperLogLog<u32> = HyperLogLog::<u32>::new(0.0);
    }

    #[test]
    #[should_panic]
    fn test_panic_new_mismatch_error_iprobability() {
        let mut hhl1: HyperLogLog<u32> = HyperLogLog::<u32>::new(0.1);
        let hhl2: HyperLogLog<u32> = HyperLogLog::<u32>::new(0.2);
        hhl1.merge(&hhl2);
    }

    #[test]
    fn test_simple() {
        let mut hhl = HyperLogLog::<u32>::new(0.01);
        assert!(hhl.is_empty());
        assert!(hhl.len() < EPSILON);

        for key in &[0, 1, 2, 0, 1, 2] {
            hhl.insert(&key);
        }

        assert!(!hhl.is_empty());
        assert!((hhl.len().round() - 3.0).abs() < EPSILON);

        hhl.clear();
        assert!(hhl.is_empty());
    }

    #[test]
    fn test_merge() {
        let mut hhl1 = HyperLogLog::<u32>::new(0.01);

        for key in &[0, 1, 2, 0, 1, 2] {
            hhl1.insert(&key);
        }

        let mut hhl2 = HyperLogLog::<u32>::new(0.01);

        for key in &[0, 1, 3, 0, 1, 3] {
            hhl2.insert(&key);
        }

        assert!((hhl1.len().round() - 3.0).abs() < EPSILON);
        assert!((hhl2.len().round() - 3.0).abs() < EPSILON);

        hhl1.merge(&hhl2);
        assert!((hhl1.len().round() - 4.0).abs() < EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut hhl = HyperLogLog::<u32>::new(0.01);
        for key in &[0, 1, 2, 0, 1, 2] {
            hhl.insert(&key);
        }

        let serialized_hhl = bincode::serialize(&hhl).unwrap();
        let de_hhl: HyperLogLog<u32> = bincode::deserialize(&serialized_hhl).unwrap();

        assert!((hhl.len() - de_hhl.len()).abs() < EPSILON);
        assert!((hhl.alpha - de_hhl.alpha).abs() < EPSILON);
        assert_eq!(hhl.p, de_hhl.p);
        assert_eq!(hhl.registers, de_hhl.registers);
        // SipHasher doesn't implement PartialEq, but it does implement Debug,
        // and its Debug impl does print all internal state.
        assert_eq!(format!("{:?}", hhl.hasher), format!("{:?}", de_hhl.hasher));
    }
}
