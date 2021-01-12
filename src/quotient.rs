//! Space-efficient probabilistic data structure for approximate membership queries in a set.

use crate::bitstring_vec::BitstringVec;
use crate::util;
use crate::SipHasherBuilder;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::f64::consts;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

const SHIFTED_MASK: u64 = 0b001;
const CONTINUATION_MASK: u64 = 0b010;
const OCCUPIED_MASK: u64 = 0b100;
const METADATA_MASK: u64 = 0b111;
const METADATA_BITS: u8 = 3;

/// A space-efficient probabilistic data structure to test for membership in a set.
///
/// A quotient filter is essentially a compact hash table. Each item is hashed to a 64-bit
/// fingerprint. The top `q` bits is the quotient of the item and the bottom `r` bits is the
/// remainder of the item. The quotient which defines the index of the table to store the
/// remainder. This index is called the "canoncial slot" of the item. When multiple items map to
/// the same location, they are stored in contiguous slots called a run, and the quotient filter
/// maintains that items in the same run are sorted by their remainder. Additionally, all runs are
/// sorted by their canonical slot. If run `r1` has a canonical slot at index `i1` and run `r2` has
/// a canonical slot at index `i2` where `i1` < `i2`, then `r1` occurs to the left of `r2`. Note
/// that a run's first fingerprint may not occupy its canonical slot if the run has been forced
/// right by some run to its left. These invariants are established by maintaining three bits of
/// metadata about a slot: `is_shifted`, `is_continuation`, and `is_occupied`.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::quotient::QuotientFilter;
///
/// let mut filter = QuotientFilter::<String>::new(8, 4);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.quotient_bits(), 8);
/// assert_eq!(filter.remainder_bits(), 4);
/// ```
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct QuotientFilter<T, B = SipHasherBuilder> {
    quotient_bits: u8,
    remainder_bits: u8,
    // Defined as RR...RRMMM where R are remainder bits and M are metadata bits
    // MMM
    // |||
    // ||- is_shifted: is set when remainder is in this slot that is not its canonical slot
    // |-- is_continuation: is set when this slot is occupied by not the first remainder in a run
    // --- is_occupied: is set when this slot is canonical slot for some key in quotient filter
    quotient_mask: u64,
    remainder_mask: u64,
    slot_vec: BitstringVec,
    hash_builder: B,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> QuotientFilter<T> {
    /// Constructs a new, empty `QuotientFilter` with the specified number of quotient and
    /// remainder bits. `quotient_bits` and `remainder_bits` must be positive integers whose sum
    /// cannot exceed `64`.
    ///
    /// # Panics
    ///
    /// Panics if `quotient_bits` is 0, `remainder_bits` is 0, or if `quotient_bits +
    /// remainder_bits` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    /// ```
    pub fn new(quotient_bits: u8, remainder_bits: u8) -> Self {
        Self::with_hasher(
            quotient_bits,
            remainder_bits,
            SipHasherBuilder::from_entropy(),
        )
    }

    /// Constructs a new, empty `QuotientFilter` that can store `capacity` items with an estimated
    /// false positive probability of less than `fpp`. The ideal fullness of quotient filter is
    /// 75%, so the contructed quotient filter will have a maximum capacity of `1.33 * capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0, or if `fpp` is not in the range `(0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::from_fpp(100, 0.05);
    /// ```
    pub fn from_fpp(capacity: usize, fpp: f64) -> Self {
        Self::from_fpp_with_hasher(capacity, fpp, SipHasherBuilder::from_entropy())
    }
}

impl<T, B> QuotientFilter<T, B>
where
    B: BuildHasher,
{
    fn get_mask(size: u8) -> u64 {
        (1u64 << size) - 1
    }

    fn get_quotient_and_remainder(&self, hash: u64) -> (usize, u64) {
        (
            ((hash >> self.remainder_bits) & self.quotient_mask) as usize,
            hash & self.remainder_mask,
        )
    }

    fn increment_index(&self, index: &mut usize) {
        if *index == self.capacity() - 1 {
            *index = 0;
        } else {
            *index += 1;
        }
    }

    fn decrement_index(&self, index: &mut usize) {
        if *index == 0 {
            *index = self.capacity() - 1;
        } else {
            *index -= 1;
        }
    }

    // returns (index of run start, # of runs from cluster start, # of occupied slots)
    fn get_run_start(&self, mut index: usize) -> (usize, usize, usize) {
        // find start of cluster
        let mut occupied_count = 0;
        loop {
            let slot = self.slot_vec.get(index);
            if slot & OCCUPIED_MASK != 0 {
                occupied_count += 1;
            }
            if slot & SHIFTED_MASK == 0 {
                break;
            }

            self.decrement_index(&mut index);
        }

        // find start of run
        let mut runs_count = 0;
        let mut total_occupied_count = 0;
        loop {
            let slot = self.slot_vec.get(index);
            if slot & OCCUPIED_MASK != 0 {
                total_occupied_count += 1;
            }
            if slot & CONTINUATION_MASK == 0 {
                runs_count += 1;
            }
            if occupied_count == runs_count {
                break;
            }

            self.increment_index(&mut index);
        }

        (index, runs_count, total_occupied_count)
    }

    fn insert_and_shift_right(&mut self, mut index: usize, slot: u64) {
        let mut curr_slot = slot;

        loop {
            let mut next_slot = self.slot_vec.get(index);
            let is_empty_slot = next_slot & METADATA_MASK == 0;

            // transfer occupied bit since they stay with the index
            if next_slot & OCCUPIED_MASK != 0 {
                next_slot &= !OCCUPIED_MASK;
                curr_slot |= OCCUPIED_MASK;
            }

            self.slot_vec.set(index, curr_slot);
            curr_slot = next_slot;
            self.increment_index(&mut index);

            if is_empty_slot {
                break;
            }

            // set shifted flag for all slots after since they are all shifted
            curr_slot |= SHIFTED_MASK;
        }
    }

    /// Constructs a new, empty `QuotientFilter` with the specified number of quotient and
    /// remainder bits, and hasher builder. `quotient_bits` and `remainder_bits` must be positive
    /// integers whose sum cannot exceed `64`.
    ///
    /// # Panics
    ///
    /// Panics if `quotient_bits` is 0, `remainder_bits` is 0, or if `quotient_bits +
    /// remainder_bits` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = QuotientFilter::<String>::with_hasher(8, 4, SipHasherBuilder::from_entropy());
    /// ```
    pub fn with_hasher(quotient_bits: u8, remainder_bits: u8, hash_builder: B) -> Self {
        assert!(quotient_bits > 0);
        assert!(remainder_bits > 0);
        assert!(quotient_bits + remainder_bits <= 64);
        let slot_bits = remainder_bits + METADATA_BITS;
        let slot_vec_len = u64::from(slot_bits) * (1u64 << quotient_bits);
        QuotientFilter {
            quotient_bits,
            remainder_bits,
            quotient_mask: Self::get_mask(quotient_bits),
            remainder_mask: Self::get_mask(remainder_bits),
            slot_vec: BitstringVec::new(slot_bits as usize, slot_vec_len as usize),
            hash_builder,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `QuotientFilter` that can store `capacity` items with an estimated
    /// false positive probability of less than `fpp` with a specified hasher builder. The ideal
    /// fullness of quotient filter is 75%, so the contructed quotient filter will have a maximum
    /// capacity of `1.33 * capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0, or if `fpp` is not in the range `(0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = QuotientFilter::<String>::from_fpp_with_hasher(
    ///     100,
    ///     0.05,
    ///     SipHasherBuilder::from_entropy(),
    /// );
    /// ```
    pub fn from_fpp_with_hasher(capacity: usize, fpp: f64, hash_builder: B) -> Self {
        let quotient_bits = (capacity as f64 * 1.33).log2().ceil() as u8;
        let remainder_bits = (1.0 / -2.0 / (1.0 - fpp).ln()).log2().ceil() as u8;
        Self::with_hasher(quotient_bits, remainder_bits, hash_builder)
    }

    /// Inserts an element into the quotient filter.
    ///
    /// # Panics
    ///
    /// Panics if the quotient filter is completely full.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (quotient, remainder) =
            self.get_quotient_and_remainder(util::hash(&self.hash_builder, &item));
        let slot = self.slot_vec.get(quotient);

        // empty slot
        if slot & METADATA_MASK == 0 {
            self.slot_vec
                .set(quotient, (remainder << METADATA_BITS) | OCCUPIED_MASK);
            self.len += 1;
            return;
        }

        // item already exists
        if self.contains(item) {
            return;
        }
        assert!(self.len() < self.capacity());

        // canonical slot not occupied, so insertion will generate a new run
        // we set occupied mask first so `get_run_start` will get the correct position to insert
        // the new run
        let new_run = {
            if slot & OCCUPIED_MASK == 0 {
                self.slot_vec.set(quotient, slot | OCCUPIED_MASK);
                true
            } else {
                false
            }
        };

        // insert into run and maintain sorted order
        let (mut index, ..) = self.get_run_start(quotient);
        let run_start = index;
        let mut new_slot = remainder << METADATA_BITS;
        let mut slot = self.slot_vec.get(index);

        if !new_run {
            // find position to insert
            loop {
                // found position in run to insert
                if remainder < slot >> METADATA_BITS {
                    break;
                }

                self.increment_index(&mut index);
                slot = self.slot_vec.get(index);

                // end of run
                if slot & CONTINUATION_MASK == 0 {
                    break;
                }
            }

            // new position is start of run
            if index == run_start {
                let mut run_start_slot = self.slot_vec.get(run_start);
                run_start_slot |= CONTINUATION_MASK;
                self.slot_vec.set(run_start, run_start_slot);
            } else {
                new_slot |= CONTINUATION_MASK;
            }
        }

        // if we are not at the canonical slot, then set shifted
        if index != quotient {
            new_slot |= SHIFTED_MASK;
        }

        self.len += 1;
        self.insert_and_shift_right(index, new_slot);
    }

    /// Checks if an element is possibly in the quotient filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (quotient, remainder) =
            self.get_quotient_and_remainder(util::hash(&self.hash_builder, &item));
        let slot = self.slot_vec.get(quotient);

        // no such run exists
        if slot & OCCUPIED_MASK == 0 {
            return false;
        }

        // item in canonical slot
        if slot >> METADATA_BITS == remainder
            && slot & CONTINUATION_MASK == 0
            && slot & SHIFTED_MASK == 0
        {
            return true;
        }

        let (mut index, ..) = self.get_run_start(quotient);

        let mut slot = self.slot_vec.get(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => return true,
                // runs are sorted, so further items in run will always be larger
                Ordering::Greater => return false,
                Ordering::Less => {
                    self.increment_index(&mut index);
                    slot = self.slot_vec.get(index);

                    // end of run
                    if slot & CONTINUATION_MASK == 0 {
                        return false;
                    }
                }
            }
        }
    }

    /// Removes an element from the quotient filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// filter.remove("foo");
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn remove<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let (quotient, remainder) =
            self.get_quotient_and_remainder(util::hash(&self.hash_builder, &item));

        // empty slot
        if self.slot_vec.get(quotient) & METADATA_MASK == 0 {
            return;
        }

        let (mut index, mut runs_count, mut occupied_count) = self.get_run_start(quotient);
        let mut slot = self.slot_vec.get(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => break,
                // runs are sorted, so further items in run will always be larger
                Ordering::Greater => return,
                Ordering::Less => {
                    self.increment_index(&mut index);
                    slot = self.slot_vec.get(index);

                    if slot & OCCUPIED_MASK != 0 {
                        occupied_count += 1;
                    }

                    // end of run
                    if slot & CONTINUATION_MASK == 0 {
                        return;
                    }
                }
            }
        }

        // found item, have to delete and shift left
        let mut is_run_start = slot & CONTINUATION_MASK == 0;

        // keep occupied bit only, if it exists
        slot &= OCCUPIED_MASK;
        self.slot_vec.set(index, 0);

        let mut next_index = index;
        self.increment_index(&mut next_index);
        let mut next_slot = self.slot_vec.get(next_index);

        // continue until it is not shifted and not a continuation: the only cases are if it is an
        // item in its canonical position, or if the slot is empty
        while next_slot & CONTINUATION_MASK != 0 || next_slot & SHIFTED_MASK != 0 {
            self.slot_vec.set(next_index, 0);

            // update number of runs since the entire run gets shifted to left
            if next_slot & CONTINUATION_MASK == 0 {
                runs_count += 1;
                // next slot is a new run, so we can delete occupied bit of canonical position
                if is_run_start {
                    let canonical_slot = self.slot_vec.get(quotient) & !OCCUPIED_MASK;
                    self.slot_vec.set(quotient, canonical_slot);
                }
            }
            // if first item is a start of a run, the next item must be a start of a run. The
            // next item is either already a start of a run or the new start of the removed
            // item's run
            else if !is_run_start {
                slot |= CONTINUATION_MASK;
            }
            is_run_start = false;

            // if current slot is occupied and the occupied count is equal to the number of runs,
            // then the shifted item is in its canonical slot
            if slot & OCCUPIED_MASK == 0 || occupied_count != runs_count {
                slot |= SHIFTED_MASK;
            }

            slot |= next_slot & !METADATA_MASK;
            self.slot_vec.set(index, slot);

            // update occupied count since occupied bit does not get shifted
            if next_slot & OCCUPIED_MASK != 0 {
                occupied_count += 1;
            }

            // keep occupied bit only, if it exists
            slot = next_slot & OCCUPIED_MASK;

            index = next_index;
            self.increment_index(&mut next_index);
            next_slot = self.slot_vec.get(next_index);
        }

        self.len -= 1;
    }

    /// Clears the quotient filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.slot_vec.clear();
        self.len = 0;
    }

    /// Returns the number of items in the quotient filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// filter.insert("foo");
    /// assert_eq!(filter.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the quotient filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// assert!(filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the capacity of the quotient filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// assert_eq!(filter.capacity(), 256);
    /// ```
    pub fn capacity(&self) -> usize {
        1 << self.quotient_bits
    }

    /// Returns the number of quotient bits in a fingerprint for a item.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// assert_eq!(filter.quotient_bits(), 8);
    /// ```
    pub fn quotient_bits(&self) -> u8 {
        self.quotient_bits
    }

    /// Returns the number of remainder bits in a fingerprint for a item.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    ///
    /// assert_eq!(filter.remainder_bits(), 4);
    /// ```
    pub fn remainder_bits(&self) -> u8 {
        self.remainder_bits
    }

    /// Returns the estimated false positive probability of the quotient filter. This value will
    /// increase as more items are added.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let mut filter = QuotientFilter::<String>::from_fpp(100, 0.05);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.05);
    /// ```
    pub fn estimated_fpp(&self) -> f64 {
        let fill_ratio = self.len() as f64 / self.capacity() as f64;
        1.0 - consts::E.powf(-fill_ratio / 2.0f64.powf(f64::from(self.remainder_bits)))
    }

    /// Returns a reference to the quotient filter's hasher builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::quotient::QuotientFilter;
    ///
    /// let filter = QuotientFilter::<String>::new(8, 4);
    /// let hasher_builder = filter.hasher();
    /// ```
    pub fn hasher(&self) -> &B {
        &self.hash_builder
    }
}

use std::fmt;
impl<T> fmt::Debug for QuotientFilter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.capacity() {
            let slot = self.slot_vec.get(i);
            write!(f, "{}|{}:{:03b} ", i, slot >> 3, slot & METADATA_MASK)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::QuotientFilter;
    use rand::{seq::SliceRandom, Rng, SeedableRng};

    #[test]
    fn test_new() {
        let mut filter = QuotientFilter::<usize>::new(8, 4);
        assert_eq!(filter.capacity(), 256);
        assert_eq!(filter.quotient_bits(), 8);
        assert_eq!(filter.remainder_bits(), 4);
        assert!(filter.is_empty());

        for i in 0..128 {
            filter.insert(&i);
        }

        assert!(filter.estimated_fpp() < 0.05);
    }

    #[test]
    fn test_from_fpp() {
        let mut filter = QuotientFilter::<usize>::from_fpp(100, 0.05);
        assert_eq!(filter.capacity(), 256);
        assert_eq!(filter.quotient_bits(), 8);
        assert_eq!(filter.remainder_bits(), 4);
        assert!(filter.is_empty());

        for i in 0..128 {
            filter.insert(&i);
        }

        assert!(filter.estimated_fpp() < 0.05);
    }

    #[test]
    fn test_insert() {
        let mut filter = QuotientFilter::<String>::new(8, 4);
        filter.insert("foo");
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_insert_existing_item() {
        let mut filter = QuotientFilter::<String>::new(8, 4);
        filter.insert("foo");
        filter.insert("foo");
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        assert!(filter.contains("foo"));
    }

    #[test]
    fn test_remove() {
        let mut filter = QuotientFilter::<String>::new(8, 4);
        filter.insert("foo");
        filter.remove("foo");

        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert!(!filter.contains("foo"));
    }

    #[test]
    fn test_clear() {
        let mut filter = QuotientFilter::<String>::new(8, 4);

        filter.insert("foobar");
        filter.insert("barfoo");
        filter.insert("baz");
        filter.insert("qux");

        filter.clear();

        assert!(!filter.contains("baz"));
        assert!(!filter.contains("qux"));
        assert!(!filter.contains("foobar"));
        assert!(!filter.contains("barfoo"));
    }

    #[test]
    fn test_stress() {
        let mut rng = rand_xorshift::XorShiftRng::from_entropy();
        let quotient_bits = 16;
        let remainder_bits = 48;
        let n = 18;

        // large remainder to decrease chance of false positives
        let mut filter = QuotientFilter::<u64>::new(quotient_bits, remainder_bits);
        assert!(filter.is_empty());
        assert_eq!(filter.quotient_bits(), quotient_bits);
        assert_eq!(filter.remainder_bits(), remainder_bits);

        let mut items = Vec::new();
        for _ in 0..1 << quotient_bits {
            let mut item = rng.gen_range(1 << n, 1 << 63);
            while filter.contains(&item) {
                item = rng.gen_range(1 << n, 1 << 63);
            }
            filter.insert(&item);
            filter.insert(&item);
            items.push(item);
            assert_eq!(filter.len(), items.len());
        }

        items.shuffle(&mut rng);
        for item in items {
            assert!(filter.contains(&item));
            filter.remove(&item);
            assert!(!filter.contains(&item));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut filter = QuotientFilter::<String>::new(8, 4);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: QuotientFilter<String> = bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.quotient_bits(), de_filter.quotient_bits());
        assert_eq!(filter.remainder_bits(), de_filter.remainder_bits());
        assert_eq!(filter.quotient_mask, de_filter.quotient_mask);
        assert_eq!(filter.remainder_mask, de_filter.remainder_mask);
        assert_eq!(filter.slot_vec, de_filter.slot_vec);
        assert_eq!(filter.len(), de_filter.len());
        assert_eq!(filter.hasher(), de_filter.hasher());
    }
}
