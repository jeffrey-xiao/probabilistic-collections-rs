//! Growable list of bits.

use std::mem;
use std::ops::{Index, Range};
use std::slice;

/// A growable list of bits implemented using a `Vec<u8>`
///
/// # Examples
///
///
/// ```
/// use probabilistic_collections::bit_vec::BitVec;
///
/// let mut bv = BitVec::from_elem(5, false);
///
/// bv.set(0, true);
/// bv.set(1, true);
/// bv.set(2, true);
/// assert_eq!(
///     bv.iter().collect::<Vec<bool>>(),
///     vec![true, true, true, false, false],
/// );
///
/// bv.set_all(true);
/// assert_eq!(
///     bv.iter().collect::<Vec<bool>>(),
///     vec![true, true, true, true, true],
/// );
///
/// bv.flip(0);
/// bv.flip_all();
/// assert_eq!(
///     bv.iter().collect::<Vec<bool>>(),
///     vec![true, false, false, false, false],
/// );
///
/// bv.push(true);
/// assert_eq!(
///     bv.iter().collect::<Vec<bool>>(),
///     vec![true, false, false, false, false, true],
/// );
/// assert_eq!(bv.pop(), Some(true));
///
/// let clone = bv.clone();
/// bv.flip_all();
/// bv.union(&clone);
/// assert_eq!(
///     bv.iter().collect::<Vec<bool>>(),
///     vec![true, true, true, true, true],
/// );
/// ```
#[derive(Deserialize, PartialEq, Serialize)]
pub struct BitVec {
    blocks: Vec<u8>,
    len: usize,
    one_count: usize,
}

const BLOCK_BIT_COUNT: usize = mem::size_of::<u8>() * 8;

impl BitVec {
    fn get_block_count(len: usize) -> usize {
        (len + BLOCK_BIT_COUNT - 1) / BLOCK_BIT_COUNT
    }

    fn reverse_byte(byte: u8) -> u8 {
        let mut ret = 0;
        for i in 0..BLOCK_BIT_COUNT {
            ret |= (byte >> i & 1) << (BLOCK_BIT_COUNT - i - 1);
        }
        ret
    }

    fn clear_extra_bits(&mut self) {
        let extra_bits = self.len() % BLOCK_BIT_COUNT;
        if extra_bits > 0 {
            let mask = (1 << extra_bits) - 1;
            let blocks_len = self.blocks.len();
            let block = &mut self.blocks[blocks_len - 1];
            *block &= mask;
        }
    }

    /// Constructs a new `BitVec` with a certain number of bits. All bits are initialized to false.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::new(5);
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![false, false, false, false, false],
    /// );
    /// ```
    pub fn new(len: usize) -> Self {
        Self {
            blocks: vec![0; Self::get_block_count(len)],
            len,
            one_count: 0,
        }
    }

    /// Constructs a new `BitVec` with a certain number of bits. All bits are initialized to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_elem(5, true);
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![true, true, true, true, true],
    /// );
    /// ```
    pub fn from_elem(len: usize, bit: bool) -> Self {
        let mut ret = BitVec {
            blocks: vec![if bit { <u8>::max_value() } else { 0 }; Self::get_block_count(len)],
            len,
            one_count: if bit { len } else { 0 },
        };
        ret.clear_extra_bits();
        ret
    }

    /// Constructs a `BitVec` from a byte slice. Each byte becomes eight bits, with the most
    /// signficant bits of each byte coming first.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b11010000]);
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![true, true, false, true, false, false, false, false],
    /// );
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes.len() * BLOCK_BIT_COUNT;
        BitVec {
            blocks: bytes
                .to_vec()
                .iter()
                .map(|byte| Self::reverse_byte(*byte))
                .collect(),
            len,
            one_count: bytes
                .to_vec()
                .iter()
                .map(|byte| byte.count_ones() as usize)
                .sum(),
        }
    }

    /// Returns the byte-vec representation of the `BitVec` with the first bit in the `BitVec`
    /// becoming the high-order bit of the first byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new(8);
    ///
    /// bv.set(0, true);
    /// bv.set(1, true);
    /// bv.set(3, true);
    ///
    /// assert_eq!(bv.to_bytes(), vec![0b11010000]);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        self.blocks
            .iter()
            .map(|byte| Self::reverse_byte(*byte))
            .collect()
    }

    /// Constructs a new, empty `BitVec` with a certain capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::with_capacity(5);
    /// ```
    pub fn with_capacity(len: usize) -> Self {
        BitVec {
            blocks: Vec::with_capacity(Self::get_block_count(len)),
            len: 0,
            one_count: 0,
        }
    }

    /// Sets the value at index `index` to `bit`.
    ///
    /// # Panics
    ///
    /// Panics if attempt to set an index out-of-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new(5);
    /// bv.set(1, true);
    ///
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// ```
    pub fn set(&mut self, index: usize, bit: bool) {
        assert!(index < self.len);
        let block_index = index / BLOCK_BIT_COUNT;
        let bit_index = index % BLOCK_BIT_COUNT;
        let mask = 1 << bit_index;
        let prev = ((self.blocks[block_index] >> bit_index) & 1) != 0;
        if bit {
            if !prev {
                self.one_count += 1;
            }
            self.blocks[block_index] |= mask;
        } else {
            if prev {
                self.one_count -= 1;
            }
            self.blocks[block_index] &= !mask;
        }
    }

    /// Returns the value at index `index`, or `None` if index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new(5);
    /// bv.set(1, true);
    ///
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// ```
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            None
        } else {
            let block_index = index / BLOCK_BIT_COUNT;
            let bit_index = index % BLOCK_BIT_COUNT;
            self.blocks
                .get(block_index)
                .map(|block| ((block >> bit_index) & 1) != 0)
        }
    }

    /// Sets all values in the `BitVec` to `bit`.
    ///
    /// # Panics
    ///
    /// Panics if attempting to set a bit out-of-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    /// bv.set_all(true);
    ///
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![true, true, true, true, true],
    /// );
    /// ```
    pub fn set_all(&mut self, bit: bool) {
        let mask;
        if bit {
            mask = !0;
            self.one_count = self.len;
        } else {
            mask = 0;
            self.one_count = 0;
        }
        for block in &mut self.blocks {
            *block = mask;
        }
        self.clear_extra_bits();
    }

    /// Flip the value at index `index`.
    ///
    /// # Panics
    ///
    /// Panics if attempt to flip an index out-of-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    ///
    /// bv.flip(0);
    /// assert_eq!(bv.get(0), Some(true));
    ///
    /// bv.flip(1);
    /// assert_eq!(bv.get(0), Some(true));
    /// ```
    pub fn flip(&mut self, index: usize) {
        assert!(index < self.len);
        let block_index = index / BLOCK_BIT_COUNT;
        let bit_index = index % BLOCK_BIT_COUNT;
        let mask = 1 << bit_index;
        if (self.blocks[block_index] >> bit_index) & 1 == 0 {
            self.one_count += 1;
            self.blocks[block_index] |= mask;
        } else {
            self.one_count -= 1;
            self.blocks[block_index] &= !mask;
        }
    }

    /// Flips all values in the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    ///
    /// bv.flip_all();
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![true, true, true, true, true],
    /// );
    ///
    /// bv.flip_all();
    /// assert_eq!(
    ///     bv.iter().collect::<Vec<bool>>(),
    ///     vec![false, false, false, false, false],
    /// );
    /// ```
    pub fn flip_all(&mut self) {
        self.one_count = self.len - self.one_count;
        for block in &mut self.blocks {
            *block = !*block;
        }
    }

    fn apply<F>(&mut self, other: &BitVec, mut op: F)
    where
        F: FnMut(u8, u8) -> u8,
    {
        assert_eq!(self.len(), other.len());
        assert_eq!(self.blocks.len(), other.blocks.len());
        for (x, y) in self.blocks_mut().zip(other.blocks()) {
            *x = op(*x, y);
        }
        self.one_count = 0;
        for index in 0..self.blocks.len() {
            if index == self.blocks.len() - 1 && self.len() % BLOCK_BIT_COUNT != 0 {
                let shift = BLOCK_BIT_COUNT - self.len() % BLOCK_BIT_COUNT;
                self.one_count += (self.blocks[index] << shift).count_ones() as usize;
            } else {
                self.one_count += self.blocks[index].count_ones() as usize;
            }
        }
    }

    /// Sets `self` to the union of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the two `BitVec` are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv1 = BitVec::new(4);
    /// bv1.set(0, true);
    /// bv1.set(1, true);
    ///
    /// let mut bv2 = BitVec::new(4);
    /// bv2.set(0, true);
    /// bv2.set(2, true);
    ///
    /// bv1.union(&bv2);
    /// assert_eq!(
    ///     bv1.iter().collect::<Vec<bool>>(),
    ///     vec![true, true, true, false],
    /// );
    /// ```
    pub fn union(&mut self, other: &Self) {
        self.apply(other, |x, y| x | y)
    }

    /// Sets `self` to the intersection of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the two `BitVec` are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv1 = BitVec::new(4);
    /// bv1.set(0, true);
    /// bv1.set(1, true);
    ///
    /// let mut bv2 = BitVec::new(4);
    /// bv2.set(0, true);
    /// bv2.set(2, true);
    ///
    /// bv1.intersection(&bv2);
    /// assert_eq!(
    ///     bv1.iter().collect::<Vec<bool>>(),
    ///     vec![true, false, false, false],
    /// );
    /// ```
    pub fn intersection(&mut self, other: &Self) {
        self.apply(other, |x, y| x & y)
    }

    /// Sets `self` to the difference of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the two `BitVec` are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv1 = BitVec::new(4);
    /// bv1.set(0, true);
    /// bv1.set(1, true);
    ///
    /// let mut bv2 = BitVec::new(4);
    /// bv2.set(0, true);
    /// bv2.set(2, true);
    ///
    /// bv1.difference(&bv2);
    /// assert_eq!(
    ///     bv1.iter().collect::<Vec<bool>>(),
    ///     vec![false, true, false, false],
    /// );
    /// ```
    pub fn difference(&mut self, other: &Self) {
        self.apply(other, |x, y| x & !y)
    }

    /// Sets `self` to the symmetric difference of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the two `BitVec` are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv1 = BitVec::new(4);
    /// bv1.set(0, true);
    /// bv1.set(1, true);
    ///
    /// let mut bv2 = BitVec::new(4);
    /// bv2.set(0, true);
    /// bv2.set(2, true);
    ///
    /// bv1.symmetric_difference(&bv2);
    /// assert_eq!(
    ///     bv1.iter().collect::<Vec<bool>>(),
    ///     vec![false, true, true, false],
    /// );
    /// ```
    pub fn symmetric_difference(&mut self, other: &Self) {
        self.apply(other, |x, y| (x & !y) | (!x & y))
    }

    /// Truncates a `BitVec`, dropping excess elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    ///
    /// bv.truncate(2);
    /// assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![false, false]);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
            self.blocks.truncate(Self::get_block_count(len));
            self.clear_extra_bits();
        }
    }

    /// Reserves capacity for at least `additional` more bits to be inserted in the given
    /// `BitVec`.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    ///
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 5);
    /// assert!(bv.capacity() >= 16);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap > self.capacity() {
            let additional_blocks = Self::get_block_count(desired_cap) - self.blocks.len();
            self.blocks.reserve(additional_blocks);
        }
    }

    /// Reserves capacity for exactly `additional` more bits to be inserted in the given
    /// `BitVec`.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    /// bv.reserve_exact(10);
    /// assert_eq!(bv.len(), 5);
    /// assert_eq!(bv.capacity(), 16);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap > self.capacity() {
            let additional_blocks = Self::get_block_count(desired_cap) - self.blocks.len();
            self.blocks.reserve_exact(additional_blocks);
        }
    }

    /// Returns and removes the last element of the `BitVec`. Returns `None` if the `BitVec` is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(1, false);
    ///
    /// assert_eq!(bv.pop(), Some(false));
    /// assert_eq!(bv.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<bool> {
        if self.is_empty() {
            None
        } else {
            let index = self.len - 1;
            let ret = self.get(index);
            self.set(index, false);
            self.len -= 1;
            if self.len % BLOCK_BIT_COUNT == 0 {
                self.blocks.pop();
            }
            ret
        }
    }

    /// Pushes an element into the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(1, false);
    ///
    /// bv.push(true);
    /// assert_eq!(bv.get(1), Some(true));
    /// ```
    pub fn push(&mut self, bit: bool) {
        if self.len % BLOCK_BIT_COUNT == 0 {
            self.blocks.push(0);
        }
        let index = self.len;
        self.len += 1;
        self.set(index, bit);
    }

    fn blocks(&self) -> Blocks {
        Blocks {
            iter: self.blocks.iter(),
        }
    }

    fn blocks_mut(&mut self) -> BlocksMut {
        self.blocks.iter_mut()
    }

    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(1, false);
    ///
    /// bv.push(true);
    /// assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![false, true]);
    /// ```
    pub fn iter(&self) -> BitVecIter {
        BitVecIter {
            bit_vec: self,
            range: 0..self.len,
        }
    }

    /// Returns `true` if the `BitVec` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(1, false);
    ///
    /// assert!(!bv.is_empty());
    /// bv.pop();
    /// assert!(bv.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements in the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(1, false);
    ///
    /// assert_eq!(bv.len(), 1);
    /// bv.pop();
    /// assert_eq!(bv.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the capacity of the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new(0);
    ///
    /// bv.reserve_exact(10);
    /// assert_eq!(bv.capacity(), 16);
    /// ```
    pub fn capacity(&self) -> usize {
        self.blocks.capacity() * BLOCK_BIT_COUNT
    }

    /// Returns the number of set bits in the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b11010000]);
    /// assert_eq!(bv.count_ones(), 3);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.one_count
    }

    /// Returns the number of unset bits in the `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b11010000]);
    /// assert_eq!(bv.count_zeros(), 5);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.len - self.one_count
    }
}

impl Clone for BitVec {
    fn clone(&self) -> Self {
        BitVec {
            blocks: self.blocks.clone(),
            len: self.len,
            one_count: self.one_count,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.len = source.len;
        self.blocks.clone_from(&source.blocks);
    }
}

/// An owning iterator for `BitVec`.
///
/// This iterator yields bits in order.
pub struct BitVecIter<'a> {
    bit_vec: &'a BitVec,
    range: Range<usize>,
}

impl<'a> Iterator for BitVecIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        self.range
            .next()
            .map(|index| self.bit_vec.get(index).unwrap())
    }
}

impl<'a> IntoIterator for &'a BitVec {
    type IntoIter = BitVecIter<'a>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator for `BitVec`.
///
/// This iterator yields bits in order.
pub struct BitVecIntoIter {
    bit_vec: BitVec,
    range: Range<usize>,
}

impl Iterator for BitVecIntoIter {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        self.range
            .next()
            .map(|index| self.bit_vec.get(index).unwrap())
    }
}

impl IntoIterator for BitVec {
    type IntoIter = BitVecIntoIter;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len;
        Self::IntoIter {
            bit_vec: self,
            range: 0..len,
        }
    }
}

type BlocksMut<'a> = slice::IterMut<'a, u8>;

struct Blocks<'a> {
    iter: slice::Iter<'a, u8>,
}

impl<'a> Iterator for Blocks<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        self.iter.next().cloned()
    }
}

static TRUE: bool = true;
static FALSE: bool = false;

impl Index<usize> for BitVec {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
        if self.get(index).expect("Error: index out of bounds.") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitVec;

    #[test]
    fn test_new() {
        let bv = BitVec::new(5);
        assert_eq!(
            bv.iter().collect::<Vec<bool>>(),
            vec![false, false, false, false, false]
        );
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 5);
    }

    #[test]
    fn test_from_elem() {
        let bv = BitVec::from_elem(5, true);
        assert_eq!(
            bv.iter().collect::<Vec<bool>>(),
            vec![true, true, true, true, true]
        );
        assert_eq!(bv.count_ones(), 5);
        assert_eq!(bv.count_zeros(), 0);
    }

    #[test]
    fn test_from_bytes() {
        let bv = BitVec::from_bytes(&[0b11010000]);
        assert_eq!(
            bv.iter().collect::<Vec<bool>>(),
            vec![true, true, false, true, false, false, false, false],
        );
        assert_eq!(bv.count_ones(), 3);
        assert_eq!(bv.count_zeros(), 5);
    }

    #[test]
    fn test_to_bytes() {
        let mut bv = BitVec::new(8);
        bv.set(0, true);
        bv.set(1, true);
        bv.set(3, true);

        assert_eq!(bv.to_bytes(), vec![0b11010000]);
    }

    #[test]
    fn test_with_capacity() {
        let bv = BitVec::with_capacity(10);

        assert_eq!(bv.capacity(), 16);
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 0);
    }

    #[test]
    fn test_set_get() {
        let mut bv = BitVec::new(2);
        bv.set(0, true);
        bv.set(1, false);

        assert_eq!(bv[0], true);
        assert_eq!(bv[1], false);
        assert_eq!(bv.get(2), None);
        assert_eq!(bv.count_ones(), 1);
        assert_eq!(bv.count_zeros(), 1);
    }

    #[test]
    fn test_set_all() {
        let mut bv = BitVec::new(3);

        bv.set_all(true);
        assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![true, true, true]);

        bv.set_all(false);
        assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![false, false, false]);
    }

    #[test]
    fn test_flip() {
        let mut bv = BitVec::new(1);

        bv.flip(0);
        assert_eq!(bv.get(0), Some(true));

        bv.flip(0);
        assert_eq!(bv.get(0), Some(false));
    }

    #[test]
    fn test_flip_all() {
        let mut bv = BitVec::new(3);

        bv.flip_all();
        assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![true, true, true]);
        assert_eq!(bv.count_ones(), 3);
        assert_eq!(bv.count_zeros(), 0);

        bv.flip_all();
        assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![false, false, false]);
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 3);
    }

    #[test]
    fn test_union() {
        let mut bv1 = BitVec::new(4);
        bv1.set(0, true);
        bv1.set(1, true);

        let mut bv2 = BitVec::new(4);
        bv2.set(0, true);
        bv2.set(2, true);

        bv1.union(&bv2);
        assert_eq!(
            bv1.iter().collect::<Vec<bool>>(),
            vec![true, true, true, false]
        );
        assert_eq!(bv1.count_ones(), 3);
        assert_eq!(bv1.count_zeros(), 1);
    }

    #[test]
    fn test_intersection() {
        let mut bv1 = BitVec::new(4);
        bv1.set(0, true);
        bv1.set(1, true);

        let mut bv2 = BitVec::new(4);
        bv2.set(0, true);
        bv2.set(2, true);

        bv1.intersection(&bv2);
        assert_eq!(
            bv1.iter().collect::<Vec<bool>>(),
            vec![true, false, false, false]
        );
        assert_eq!(bv1.count_ones(), 1);
        assert_eq!(bv1.count_zeros(), 3);
    }

    #[test]
    fn test_difference() {
        let mut bv1 = BitVec::new(4);
        bv1.set(0, true);
        bv1.set(1, true);

        let mut bv2 = BitVec::new(4);
        bv2.set(0, true);
        bv2.set(2, true);

        bv1.difference(&bv2);
        assert_eq!(
            bv1.iter().collect::<Vec<bool>>(),
            vec![false, true, false, false]
        );
        assert_eq!(bv1.count_ones(), 1);
        assert_eq!(bv1.count_zeros(), 3);
    }

    #[test]
    fn test_symmetric_difference() {
        let mut bv1 = BitVec::new(4);
        bv1.set(0, true);
        bv1.set(1, true);

        let mut bv2 = BitVec::new(4);
        bv2.set(0, true);
        bv2.set(2, true);

        bv1.symmetric_difference(&bv2);
        assert_eq!(
            bv1.iter().collect::<Vec<bool>>(),
            vec![false, true, true, false]
        );
        assert_eq!(bv1.count_ones(), 2);
        assert_eq!(bv1.count_zeros(), 2);
    }

    #[test]
    fn test_truncate() {
        let mut bv = BitVec::from_elem(9, false);

        bv.truncate(1);
        assert_eq!(bv.iter().collect::<Vec<bool>>(), vec![false]);
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 1);
    }

    #[test]
    fn test_reserve() {
        let mut bv = BitVec::from_elem(1, false);

        bv.reserve(9);
        assert_eq!(bv.len(), 1);
        assert!(bv.capacity() >= 16);
    }

    #[test]
    fn test_reserve_exact() {
        let mut bv = BitVec::from_elem(1, false);

        bv.reserve_exact(9);
        assert_eq!(bv.len(), 1);
        assert!(bv.capacity() == 16);
    }

    #[test]
    fn test_push_pop() {
        let mut bv = BitVec::new(0);
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 0);

        bv.push(true);
        assert_eq!(bv.count_ones(), 1);
        assert_eq!(bv.count_zeros(), 0);

        bv.push(false);
        assert_eq!(bv.count_ones(), 1);
        assert_eq!(bv.count_zeros(), 1);

        assert_eq!(bv.pop(), Some(false));
        assert_eq!(bv.count_ones(), 1);
        assert_eq!(bv.count_zeros(), 0);

        assert_eq!(bv.pop(), Some(true));
        assert_eq!(bv.count_ones(), 0);
        assert_eq!(bv.count_zeros(), 0);

        assert_eq!(bv.pop(), None);
    }

    #[test]
    fn test_push() {
        let mut bv = BitVec::from_elem(1, false);

        bv.push(true);
        assert_eq!(bv.get(1), Some(true));
        assert_eq!(bv.count_ones(), 1);
        assert_eq!(bv.count_zeros(), 1);
    }

    #[test]
    fn test_is_empty() {
        let mut bv = BitVec::new(0);

        assert!(bv.is_empty());

        bv.push(true);
        assert!(!bv.is_empty());
    }

    #[test]
    fn test_len() {
        let mut bv = BitVec::new(0);

        assert_eq!(bv.len(), 0);

        bv.push(true);
        assert_eq!(bv.len(), 1);
    }

    #[test]
    fn test_clone() {
        let bv = BitVec::from_bytes(&[0b11010000]);
        let mut cloned = BitVec::new(0);

        assert_eq!(
            bv.clone().iter().collect::<Vec<bool>>(),
            vec![true, true, false, true, false, false, false, false],
        );

        cloned.clone_from(&bv);
        assert_eq!(
            cloned.iter().collect::<Vec<bool>>(),
            vec![true, true, false, true, false, false, false, false],
        );
    }

    #[test]
    fn test_iter() {
        let bv = BitVec::from_bytes(&[0b11010000]);

        assert_eq!(
            (&bv).into_iter().collect::<Vec<bool>>(),
            vec![true, true, false, true, false, false, false, false],
        );

        assert_eq!(
            bv.into_iter().collect::<Vec<bool>>(),
            vec![true, true, false, true, false, false, false, false],
        );
    }
}
