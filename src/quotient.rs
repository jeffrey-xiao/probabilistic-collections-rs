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

pub struct QuotientFilter<T> {
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
    capacity: usize,
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
        (((hash >> self.remainder_mask) & self.quotient_mask) as usize, hash & self.remainder_mask)
    }

    fn increment_index(&self, index: &mut usize) {
        if *index == self.capacity - 1 {
            *index = 0;
        } else {
            *index += 1;
        }
    }

    fn decrement_index(&self, index: &mut usize) {
        if *index == 0 {
            *index = self.capacity - 1;
        } else {
            *index -= 1;
        }
    }

    pub fn new(quotient_bits: u8, remainder_bits: u8) -> Self {
        assert!(quotient_bits > 0);
        assert!(remainder_bits > 0);
        assert!(quotient_bits + remainder_bits <= 64);
        let slot_bits = remainder_bits + METADATA_BITS;
        let table_len = ((slot_bits * quotient_bits) as usize + 63) / 64;
        QuotientFilter {
            remainder_bits,
            slot_bits,
            quotient_mask: Self::get_mask(quotient_bits),
            remainder_mask: Self::get_mask(remainder_bits),
            slot_mask: Self::get_mask(slot_bits),
            table: vec![0; table_len],
            hasher: Self::get_hasher(),
            len: 0,
            capacity: 1 << quotient_bits,
            _marker: PhantomData,
        }
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
        let mut total_occupied_count = occupied_count - 1;
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

    // pub fn insert<U>(&mut self, item: &U)
    // where
    //     T: Borrow<U>,
    //     U: Hash + ?Sized,
    // {
    //    let (quotient, remainder) = self.get_quotient_and_remainder(self.get_hash(item));
    pub fn insert(&mut self, quotient: usize, remainder: u64) {
        assert!(self.len() < self.capacity());
        let slot = self.get_slot(quotient);

        // empty slot
        if slot & METADATA_MASK == 0 {
            self.set_slot(quotient, (remainder << METADATA_BITS) | OCCUPIED_MASK);
            self.len += 1;
            return;
        }

        // item already exists
        // if self.contains(item) {
        //     return;
        // }

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

    // pub fn remove<U>(&mut self, item: &U)
    // where
    //     T: Borrow<U>,
    //     U: Hash + ?Sized,
    // {
    //    let (quotient, remainder) = self.get_quotient_and_remainder(self.get_hash(item));
    pub fn remove(&mut self, quotient: usize, remainder: u64) {
        let (mut index, mut runs_count, mut occupied_count) = self.get_run_start(quotient);
        let mut slot = self.get_slot(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => break,
                // runs are sorted, so further elements in run will always be larger
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

        // found element, have to delete and shift left
        let mut is_run_start = slot & CONTINUATION_MASK == 0;

        // keep occupied bit only, if it exists
        slot &= OCCUPIED_MASK;
        self.set_slot(index, 0);

        let mut next_index = index;
        self.increment_index(&mut next_index);
        let mut next_slot = self.get_slot(next_index);

        // continue until it is not shifted and not a continuation: the only cases are if it is an
        // element in its canonical position, or if the slot is empty
        while next_slot & CONTINUATION_MASK != 0 || next_slot & SHIFTED_MASK != 0 {
            self.set_slot(next_index, 0);

            // update number of runs since the entire run gets shifted to left
            if next_slot & CONTINUATION_MASK == 0 {
                runs_count += 1;
            } else {
                // if first element is a start of a run, the next element must be a start of a run. The
                // next element is either already a start of a run or the new start of the removed
                // element's run
                if !is_run_start {
                    slot |= CONTINUATION_MASK;
                }
            }
            is_run_start = false;

            // if current slot is occupied and the occupied count is equal to the number of runs,
            // then the shifted element is in its canonical slot
            println!("AT {}, occupied: {}, runs: {}", index, occupied_count, runs_count);
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

        let (mut index, ..) = self.get_run_start(quotient);

        let mut slot = self.get_slot(index);
        loop {
            match (slot >> METADATA_BITS).cmp(&remainder) {
                Ordering::Equal => return true,
                // runs are sorted, so further elements in run will always be larger
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

    pub fn clear(&mut self) {
        for value in &mut self.table {
            *value = 0;
        }
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn estimate_fpp(&self) -> f64 {
        let fill_ratio = self.len() as f64 / self.capacity() as f64;
        1.0 - consts::E.powf(- fill_ratio / 2.0f64.powf(self.remainder_bits as f64))
    }
}

use std;
impl<T> std::fmt::Debug for QuotientFilter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for index in 0..self.capacity {
            let slot = self.get_slot(index);
            let quotient = slot >> METADATA_BITS;
            let remainder = slot & METADATA_MASK;
            write!(f, "{}:{:03b} ", quotient, remainder)?;
        }
        Ok(())
    }
}

mod tests {
    use super::QuotientFilter;

    #[test]
    fn test() {
        let mut qf: QuotientFilter<usize> = QuotientFilter::new(3, 3);
        qf.insert(1, 5);
        qf.insert(4, 6);
        qf.insert(7, 7);
        println!("{:?}", qf);

        qf.insert(1, 2);
        qf.insert(2, 4);
        println!("{:?}", qf);

        qf.insert(1, 3);
        println!("{:?}", qf);
        qf.insert(1, 7);

        qf.remove(7, 7);
        println!("{:?}", qf);

        qf.remove(1, 2);
        println!("{:?}", qf);

        qf.remove(1, 5);
        qf.remove(1, 3);
        println!("{:?}", qf);
        qf.remove(1, 7);
        qf.remove(2, 4);
        qf.remove(4, 6);
        println!("{:?}", qf);

        qf.insert(1, 5);
        qf.insert(1, 4);
        qf.insert(1, 3);
        qf.insert(1, 2);
        qf.insert(1, 1);
        qf.insert(2, 1);
        qf.insert(3, 1);
        qf.insert(4, 1);
        println!("{:?}", qf);
        qf.remove(1, 1);
        qf.remove(1, 2);
        qf.remove(1, 3);
        qf.remove(1, 4);
        qf.remove(1, 5);
        println!("{:?}", qf);
    }
}
