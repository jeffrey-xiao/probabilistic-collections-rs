//! Growable list of bitstrings.

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::iter;
use std::mem;
use std::ops::Range;

#[derive(Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct BitstringVec {
    blocks: Vec<u64>,
    bit_count: usize,
    occupied_len: usize,
    len: usize,
}

const BLOCK_BIT_COUNT: usize = mem::size_of::<u64>() * 8;

impl BitstringVec {
    #[inline]
    fn get_block_count(bit_count: usize, len: usize) -> usize {
        (bit_count * len + BLOCK_BIT_COUNT - 1) / BLOCK_BIT_COUNT
    }

    #[inline]
    fn get_elem_count(bit_count: usize, block_count: usize) -> usize {
        (block_count * BLOCK_BIT_COUNT) / bit_count
    }

    #[inline]
    fn get_mask(size: usize) -> u64 {
        if size == 64 {
            return !0;
        } else {
            (1u64 << size) - 1
        }
    }

    pub fn new(bit_count: usize, len: usize) -> Self {
        BitstringVec {
            blocks: vec![0; Self::get_block_count(bit_count, len)],
            bit_count,
            occupied_len: 0,
            len,
        }
    }

    pub fn from_elem(bit_count: usize, len: usize, bitstring: u64) -> Self {
        let mut ret = BitstringVec {
            blocks: vec![0; Self::get_block_count(bit_count, len)],
            bit_count,
            occupied_len: 0,
            len,
        };
        for index in 0..len {
            ret.set(index, bitstring);
        }
        ret
    }

    pub fn with_capacity(bit_count: usize, len: usize) -> Self {
        BitstringVec {
            blocks: Vec::with_capacity(Self::get_block_count(bit_count, len)),
            bit_count,
            occupied_len: 0,
            len: 0,
        }
    }

    pub fn set(&mut self, index: usize, bitstring: u64) {
        assert!(index < self.len);
        let prev_is_zero = self.get(index) == 0;
        let bit_offset = index * self.bit_count;
        let table_index = bit_offset / 64;
        let bit_index = bit_offset % 64;
        let bits_left = self.bit_count as isize - (64 - bit_index as isize);
        self.blocks[table_index] &= !(Self::get_mask(self.bit_count) << bit_index);
        self.blocks[table_index] |= bitstring << bit_index;
        if bits_left > 0 {
            let offset = self.bit_count - bits_left as usize;
            self.blocks[table_index + 1] &= !Self::get_mask(bits_left as usize);
            self.blocks[table_index + 1] |= bitstring >> offset;
        }
        let curr_is_zero = bitstring == 0;
        if prev_is_zero != curr_is_zero {
            if curr_is_zero {
                self.occupied_len -= 1;
            } else {
                self.occupied_len += 1;
            }
        }
    }

    pub fn get(&self, index: usize) -> u64 {
        assert!(index < self.len);
        let mut bitstring = 0;
        let bit_offset = index * self.bit_count;
        let table_index = bit_offset / 64;
        let bit_index = bit_offset % 64;
        let bits_left = self.bit_count as isize - (64 - bit_index as isize);
        bitstring |= (self.blocks[table_index] >> bit_index) & Self::get_mask(self.bit_count);
        if bits_left > 0 {
            let offset = self.bit_count - bits_left as usize;
            bitstring |=
                (self.blocks[table_index + 1] & Self::get_mask(bits_left as usize)) << offset;
        }
        bitstring
    }

    pub fn truncate(&mut self, len: usize) {
        while len < self.len {
            self.pop();
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap <= Self::get_elem_count(self.bit_count, self.blocks.capacity()) {
            return;
        }

        let target_cap = Self::get_block_count(self.bit_count, desired_cap);
        let additional_blocks = target_cap - self.blocks.len();
        self.blocks.reserve(additional_blocks);
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap <= Self::get_elem_count(self.bit_count, self.blocks.capacity()) {
            return;
        }

        let target_cap = Self::get_block_count(self.bit_count, desired_cap);
        let additional_blocks = target_cap - self.blocks.len();
        self.blocks.reserve_exact(additional_blocks);
    }

    pub fn pop(&mut self) -> u64 {
        assert!(!self.is_empty());
        let ret = self.get(self.len - 1);
        let len = self.len;
        let new_blocks_len = Self::get_block_count(self.bit_count, len - 1);
        self.blocks.truncate(new_blocks_len);
        self.len -= 1;
        if ret != 0 {
            self.occupied_len -= 1;
        }
        ret
    }

    pub fn push(&mut self, bitstring: u64) {
        let new_block_len = Self::get_block_count(self.bit_count, self.len + 1);
        let block_len = self.blocks.len();
        self.blocks
            .extend(iter::repeat(0).take(new_block_len - block_len));
        let len = self.len;
        let occupied_len = self.occupied_len;
        self.len += 1;
        self.set(len, bitstring);
        self.occupied_len = occupied_len;
        if bitstring != 0 {
            self.occupied_len = occupied_len + 1;
        }
    }

    pub fn clear(&mut self) {
        self.occupied_len = 0;
        for byte in &mut self.blocks {
            *byte = 0;
        }
    }

    pub fn iter(&self) -> BitstringVecIter<'_> {
        BitstringVecIter {
            bitstring_vec: self,
            range: 0..self.len,
        }
    }

    pub fn capacity(&self) -> usize {
        Self::get_elem_count(self.bit_count, self.blocks.capacity())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn occupied_len(&self) -> usize {
        self.occupied_len
    }

    pub fn bit_count(&self) -> usize {
        self.bit_count
    }
}

pub struct BitstringVecIter<'a> {
    bitstring_vec: &'a BitstringVec,
    range: Range<usize>,
}

impl<'a> Iterator for BitstringVecIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.range.next().map(|index| self.bitstring_vec.get(index))
    }
}

impl<'a> IntoIterator for &'a BitstringVec {
    type IntoIter = BitstringVecIter<'a>;
    type Item = u64;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct BitstringVecIntoIter {
    bitstring_vec: BitstringVec,
    range: Range<usize>,
}

impl Iterator for BitstringVecIntoIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.range.next().map(|index| self.bitstring_vec.get(index))
    }
}

impl IntoIterator for BitstringVec {
    type IntoIter = BitstringVecIntoIter;
    type Item = u64;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len;
        Self::IntoIter {
            bitstring_vec: self,
            range: 0..len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitstringVec;
    use rand::SeedableRng;

    fn mask(bitstring: u64, bit_count: usize) -> u64 {
        bitstring & BitstringVec::get_mask(bit_count)
    }

    fn gen_bitstring(rng: &mut impl rand::Rng, bit_count: usize) -> u64 {
        let mut bitstring = 0;
        while bitstring == 0 {
            bitstring = mask(rng.gen(), bit_count);
        }
        bitstring
    }

    fn test_with_bit_count(bit_count: usize) {
        let len = 8;
        let mut rng = rand_xorshift::XorShiftRng::from_entropy();
        let mut bsv = BitstringVec::new(bit_count, len);
        let mut vec = vec![0; len];

        assert_eq!(bsv.len(), len);

        bsv.set(0, 0);
        vec[0] = 0;
        assert_eq!(bsv.occupied_len(), 0);

        for i in 0..len {
            let bitstring = gen_bitstring(&mut rng, bit_count);
            bsv.set(i, bitstring);
            vec[i] = bitstring;
            assert_eq!(bsv.occupied_len(), i + 1);
        }

        for i in 0..len {
            assert_eq!(bsv.get(i), vec[i]);
            bsv.set(i, 0);
            assert_eq!(bsv.occupied_len(), 8 - i - 1);
        }

        for i in 0..len {
            let bitstring = gen_bitstring(&mut rng, bit_count);
            bsv.set(i, bitstring);
            vec[i] = bitstring;
        }

        bsv.truncate(4);
        vec.truncate(4);
        assert_eq!(bsv.occupied_len(), 4);
        assert_eq!(bsv.len(), 4);

        assert_eq!(bsv.iter().collect::<Vec<u64>>(), vec,);

        bsv.push(1);
        vec.push(1);
        assert_eq!(bsv.occupied_len(), 5);
        assert_eq!(bsv.len(), 5);

        bsv.push(0);
        vec.push(0);
        assert_eq!(bsv.occupied_len(), 5);
        assert_eq!(bsv.len(), 6);

        assert_eq!(bsv.pop(), 0);
        assert_eq!(bsv.occupied_len(), 5);
        assert_eq!(bsv.len(), 5);

        assert_eq!(bsv.pop(), 1);
        assert_eq!(bsv.occupied_len(), 4);
        assert_eq!(bsv.len(), 4);

        bsv.truncate(0);
        assert!(bsv.is_empty());
        assert_eq!(bsv.occupied_len(), 0);
        assert_eq!(bsv.len(), 0);

        let mut bsv = BitstringVec::new(bit_count, len);
        bsv.reserve(10);
        assert!(bsv.capacity() >= 10);

        let mut bsv = BitstringVec::new(bit_count, len);
        bsv.reserve_exact(10);
        assert!(bsv.capacity() >= 10);
    }

    #[test]
    fn test_with_bit_count_17() {
        test_with_bit_count(17);
    }

    #[test]
    fn test_with_bit_count_32() {
        test_with_bit_count(32);
    }

    #[test]
    fn test_with_bit_count_47() {
        test_with_bit_count(47);
    }

    #[test]
    fn test_with_bit_count_64() {
        test_with_bit_count(64);
    }

    #[test]
    fn test_bit_count() {
        let bsv = BitstringVec::new(8, 10);
        assert_eq!(bsv.bit_count(), 8);
    }

    #[test]
    fn test_from_elem() {
        let bsv = BitstringVec::from_elem(5, 4, 1);
        assert_eq!(bsv.iter().collect::<Vec<u64>>(), vec![1, 1, 1, 1],);
    }

    #[test]
    fn test_with_capacity() {
        let bsv = BitstringVec::with_capacity(5, 4);

        assert_eq!(bsv.len(), 0);
        assert_eq!(bsv.capacity(), 12);
    }

    #[test]
    fn test_into_iter() {
        let mut bsv = BitstringVec::new(5, 0);
        bsv.push(0);
        bsv.push(1);
        bsv.push(2);
        bsv.push(3);

        assert_eq!(bsv.into_iter().collect::<Vec<u64>>(), vec![0, 1, 2, 3],);
    }
}
