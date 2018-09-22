//! Space-efficient probabilistic data structure to test for membership in a set.

use rand::{Rng, XorShiftRng};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::f64::consts;
use std::hash::{Hash, Hasher};
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
pub struct QuotientFilter<T> {
    quotient_bits: u8,
    remainder_bits: u8,
    // Defined as RR...RRMMM where R are remainder bits and M are metadata bits
    // MMM
    // |||
    // ||- is_shifted: is set when remainder is in this slot that is not its canonical slot
    // |-- is_continuation: is set when this slot is occupied by not the first remainder in a run
    // --- is_occupied: is set when this slot is canonical slot for some key in quotient filter
    slot_bits: u8,
    quotient_mask: u64,
    remainder_mask: u64,
    slot_mask: u64,
    table: Vec<u64>,
    hasher: SipHasher,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> QuotientFilter<T> {
    fn get_hasher() -> SipHasher {
        let mut rng = XorShiftRng::new_unseeded();
        SipHasher::new_with_keys(rng.next_u64(), rng.next_u64())
    }

    fn get_hash<U>(&self, item: &U) -> u64
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        let mut sip = self.hasher;
        item.hash(&mut sip);
        sip.finish()
    }

    fn get_mask(size: u8) -> u64 {
        (1u64 << size) - 1
    }

    fn get_quotient_and_remainder(&self, hash: u64) -> (usize, u64) {
        (((hash >> self.remainder_bits) & self.quotient_mask) as usize, hash & self.remainder_mask)
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
        assert!(quotient_bits > 0);
        assert!(remainder_bits > 0);
        assert!(quotient_bits + remainder_bits <= 64);
        let slot_bits = remainder_bits + METADATA_BITS;
        let table_len = ((slot_bits as u64 * (1u64 << quotient_bits)) as usize + 63) / 64;
        QuotientFilter {
            quotient_bits,
            remainder_bits,
            slot_bits,
            quotient_mask: Self::get_mask(quotient_bits),
            remainder_mask: Self::get_mask(remainder_bits),
            slot_mask: Self::get_mask(slot_bits),
            table: vec![0; table_len],
            hasher: Self::get_hasher(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `QuotientFilter` that can store `capacity` items with an estimated
    /// false positive probability of less than `fpp`. The ideal fullness of quotient filter is
    /// 50%, so the contructed quotient filter will have a maximum capacity of `2 * capacity`.
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
        let quotient_bits = ((capacity * 2) as f64).log2().ceil() as u8;
        let remainder_bits = (1.0 / -2.0 / (1.0 - fpp).ln()).log2().ceil() as u8;
        Self::new(quotient_bits, remainder_bits)
    }

    fn get_slot(&self, index: usize) -> u64 {
        let mut slot = 0;
        let bit_offset = self.slot_bits as usize * index;
        let table_index = bit_offset / 64;
        let bit_index = bit_offset % 64;
        let bits_left = self.slot_bits as isize - (64 - bit_index as isize);
        slot |= (self.table[table_index] >> bit_index) & self.slot_mask;
        if bits_left > 0 {
            let offset = self.slot_bits - bits_left as u8;
            slot |= (self.table[table_index + 1] & Self::get_mask(bits_left as u8)) << offset;
        }
        slot
    }

    fn set_slot(&mut self, index: usize, slot: u64) {
        let bit_offset = self.slot_bits as usize * index;
        let table_index = bit_offset / 64;
        let bit_index = bit_offset % 64;
        let bits_left = self.slot_bits as isize - (64 - bit_index as isize);
        self.table[table_index] &= !(self.slot_mask << bit_index);
        self.table[table_index] |= slot << bit_index;
        if bits_left > 0 {
            let offset = self.slot_bits - bits_left as u8;
            self.table[table_index + 1] &= !Self::get_mask(bits_left as u8);
            self.table[table_index + 1] |= slot >> offset;
        }
    }

    // returns (index of run start, # of runs from cluster start, # of occupied slots)
    fn get_run_start(&self, mut index: usize) -> (usize, usize, usize) {
        // find start of cluster
        let mut occupied_count = 0;
        loop {
            let slot = self.get_slot(index);
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
            let slot = self.get_slot(index);
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
            let mut next_slot = self.get_slot(index);
            let is_empty_slot = next_slot & METADATA_MASK == 0;

            // transfer occupied bit since they stay with the index
            if next_slot & OCCUPIED_MASK != 0 {
                next_slot &= !OCCUPIED_MASK;
                curr_slot |= OCCUPIED_MASK;
            }

            self.set_slot(index, curr_slot);
            curr_slot = next_slot;
            self.increment_index(&mut index);

            if is_empty_slot {
                break;
            }

            // set shifted flag for all slots after since they are all shifted
            curr_slot |= SHIFTED_MASK;
        }
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
        assert!(self.len() < self.capacity());
        let (quotient, remainder) = self.get_quotient_and_remainder(self.get_hash(item));
        let slot = self.get_slot(quotient);

        // empty slot
        if slot & METADATA_MASK == 0 {
            self.set_slot(quotient, (remainder << METADATA_BITS) | OCCUPIED_MASK);
            self.len += 1;
            return;
        }

        // item already exists
        if self.contains(item) {
            return;
        }

        let mut new_run = false;

        // canonical slot not occupied, so insertion will generate a new run
        // we set occupied mask first so `get_run_start` will get the correct position to insert
        // the new run
        if slot & OCCUPIED_MASK == 0 {
            self.set_slot(quotient, slot | OCCUPIED_MASK);
            new_run = true;
        }

        // insert into run and maintain sorted order
        let (mut index, ..) = self.get_run_start(quotient);
        let run_start = index;
        let mut new_slot = remainder << METADATA_BITS;
        let mut slot = self.get_slot(index);

        if !new_run {
            // find position to insert
            loop {
                // found position in run to insert
                if remainder < slot >> METADATA_BITS {
                    break;
                }

                self.increment_index(&mut index);
                slot = self.get_slot(index);

                // end of run
                if slot & CONTINUATION_MASK == 0 {
                    break;
                }
            }

            // new position is start of run
            if index == run_start {
                let mut run_start_slot = self.get_slot(run_start);
                run_start_slot |= CONTINUATION_MASK;
                self.set_slot(run_start, run_start_slot);
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
        let (quotient, remainder) = self.get_quotient_and_remainder(self.get_hash(item));
        let slot = self.get_slot(quotient);

        // empty slot
        if slot & METADATA_MASK == 0 {
            return false;
        }

        // item in canonical slot
        if slot >> METADATA_BITS == remainder {
            return true;
        }

        let (mut index, ..) = self.get_run_start(quotient);

        let mut slot = self.get_slot(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => return true,
                // runs are sorted, so further items in run will always be larger
                Ordering::Greater => return false,
                Ordering::Less => {
                    self.increment_index(&mut index);
                    slot = self.get_slot(index);

                    // end of run
                    if slot & CONTINUATION_MASK == 0 {
                        break;
                    }
                }
            }
        }

        false
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
        let (quotient, remainder) = self.get_quotient_and_remainder(self.get_hash(item));

        // empty slot
        if self.get_slot(quotient) & METADATA_MASK == 0 {
            return;
        }

        let (mut index, mut runs_count, mut occupied_count) = self.get_run_start(quotient);
        let mut slot = self.get_slot(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => break,
                // runs are sorted, so further items in run will always be larger
                Ordering::Greater => return,
                Ordering::Less => {
                    self.increment_index(&mut index);
                    slot = self.get_slot(index);

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
        self.set_slot(index, 0);

        let mut next_index = index;
        self.increment_index(&mut next_index);
        let mut next_slot = self.get_slot(next_index);

        // continue until it is not shifted and not a continuation: the only cases are if it is an
        // item in its canonical position, or if the slot is empty
        while next_slot & CONTINUATION_MASK != 0 || next_slot & SHIFTED_MASK != 0 {
            self.set_slot(next_index, 0);

            // update number of runs since the entire run gets shifted to left
            if next_slot & CONTINUATION_MASK == 0 {
                runs_count += 1;
                // next slow is a new run, so we can delete occupied bit of canonical position
                if is_run_start {
                    let canonical_slot = self.get_slot(quotient) & !OCCUPIED_MASK;
                    self.set_slot(quotient, canonical_slot);
                }
            } else {
                // if first item is a start of a run, the next item must be a start of a run. The
                // next item is either already a start of a run or the new start of the removed
                // item's run
                if !is_run_start {
                    slot |= CONTINUATION_MASK;
                }
            }
            is_run_start = false;

            // if current slot is occupied and the occupied count is equal to the number of runs,
            // then the shifted item is in its canonical slot
            if slot & OCCUPIED_MASK == 0 || occupied_count != runs_count {
                slot |= SHIFTED_MASK;
            }

            slot |= next_slot & !METADATA_MASK;
            self.set_slot(index, slot);

            // update occupied count since occupied bit does not get shifted
            if next_slot & OCCUPIED_MASK != 0 {
                occupied_count += 1;
            }

            // keep occupied bit only, if it exists
            slot = next_slot & OCCUPIED_MASK;

            index = next_index;
            self.increment_index(&mut next_index);
            next_slot = self.get_slot(next_index);
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
        for value in &mut self.table {
            *value = 0;
        }
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
    /// assert!(filter.estimate_fpp() < 1e-15);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimate_fpp() > 1e-15);
    /// assert!(filter.estimate_fpp() < 0.05);
    /// ```
    pub fn estimate_fpp(&self) -> f64 {
        let fill_ratio = self.len() as f64 / self.capacity() as f64;
        1.0 - consts::E.powf(- fill_ratio / 2.0f64.powf(self.remainder_bits as f64))
    }
}

use std::fmt;
impl<T> fmt::Debug for QuotientFilter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.capacity() {
            let slot = self.get_slot(i);
            write!(f, "{}|{}:{:03b} ", i, slot >> 3, slot & METADATA_MASK)?;
        }
        Ok(())
    }
}

mod tests {
    extern crate rand;

    use self::rand::{thread_rng, Rng};
    use super::QuotientFilter;

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

        assert!(filter.estimate_fpp() < 0.05);
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

        assert!(filter.estimate_fpp() < 0.05);
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
        let mut rng: rand::XorShiftRng = rand::SeedableRng::from_seed([1, 1, 1, 1]);
        let quotient_bits = 16;
        let remainder_bits = 48;

        // large remainder to not get false positives
        let mut filter = QuotientFilter::<u64>::new(quotient_bits, remainder_bits);
        assert!(filter.is_empty());
        assert_eq!(filter.quotient_bits(), quotient_bits);
        assert_eq!(filter.remainder_bits(), remainder_bits);

        let mut items = Vec::new();
        for _ in 0..1 << quotient_bits {
            let item = rng.gen_range::<u64>(1 << 32, 1 << 63) as u64;
            if !filter.contains(&item) {
                filter.insert(&item);
                filter.insert(&item);
                items.push(item);
            }
        }

        for i in 0..100 {
            let item = rng.gen_range::<u64>(0, 1 << 32);
            assert!(!filter.contains(&item));
            filter.remove(&item);
        }

        assert_eq!(filter.len(), items.len());

        thread_rng().shuffle(&mut items);
        for item in items {
            assert!(filter.contains(&item));
            filter.remove(&item);
            assert!(!filter.contains(&item));
        }
    }
}
