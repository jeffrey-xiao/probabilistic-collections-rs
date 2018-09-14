//! Growable list of bit arrays.

use std::cmp;
use std::iter;
use std::mem;
use std::ops::Range;

/// A growable list of bit arrays implemented using a `Vec<u8>`.
///
/// The bit arrays contained in the `BitArrayVec` must all be the same size. `BitArrayVec` is very
/// memory efficient for small bit arrays for a small time tradeoff.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bit_array_vec::BitArrayVec;
///
/// let mut bav = BitArrayVec::new(4, 4);
///
/// bav.set(0, &[0]);
/// bav.set(1, &[1]);
/// bav.set(2, &[2]);
/// bav.set(3, &[3]);
///
/// assert_eq!(
///     bav.iter().collect::<Vec<Vec<u8>>>(),
///     vec![vec![0], vec![1], vec![2], vec![3]],
/// );
/// ```
#[derive(Clone)]
pub struct BitArrayVec {
    blocks: Vec<u8>,
    bit_count: usize,
    occupied_len: usize,
    len: usize,
}

const BLOCK_BIT_COUNT: usize = mem::size_of::<u8>() * 8;

impl BitArrayVec {
    fn get_block_count(bit_count: usize, len: usize) -> usize {
        (bit_count * len + BLOCK_BIT_COUNT - 1) / BLOCK_BIT_COUNT
    }

    fn get_elem_count(bit_count: usize, block_count: usize) -> usize {
        (block_count * BLOCK_BIT_COUNT) / bit_count
    }

    /// Constructs a new `BitArrayVec` with a certain number of bit arrays. All bit arrays are
    /// initialized to all zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let bav = BitArrayVec::new(5, 4);
    /// assert_eq!(
    ///     bav.iter().collect::<Vec<Vec<u8>>>(),
    ///     vec![vec![0], vec![0], vec![0], vec![0]],
    /// );
    /// ```
    pub fn new(bit_count: usize, len: usize) -> Self {
        BitArrayVec {
            blocks: vec![0; Self::get_block_count(bit_count, len)],
            bit_count,
            occupied_len: 0,
            len,
        }
    }

    /// Constructs a new `BitArrayVec` with a certain number of bit arrays. All bit arrays are
    /// initialized to `bytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let bav = BitArrayVec::from_elem(5, 4, &[1]);
    /// assert_eq!(
    ///     bav.iter().collect::<Vec<Vec<u8>>>(),
    ///     vec![vec![1], vec![1], vec![1], vec![1]],
    /// );
    /// ```
    pub fn from_elem(bit_count: usize, len: usize, bytes: &[u8]) -> Self {
        let mut ret = BitArrayVec {
            blocks: vec![0; Self::get_block_count(bit_count, len)],
            bit_count,
            occupied_len: 0,
            len,
        };
        for index in 0..len {
            ret.set(index, bytes);
        }
        ret
    }

    /// Constructs a new, empty `BitArrayVec` with a certain capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let bav = BitArrayVec::with_capacity(5, 4);
    /// ```
    pub fn with_capacity(bit_count: usize, len: usize) -> Self {
        BitArrayVec {
            blocks: Vec::with_capacity(Self::get_block_count(bit_count, len)),
            bit_count,
            occupied_len: 0,
            len: 0,
        }
    }

    /// Sets the value at index `index` to `bytes`.
    ///
    /// # Panics
    ///
    /// Panics if attempt to set an index out-of-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 4);
    /// bav.set(1, &[1]);
    ///
    /// assert_eq!(bav.get(0), vec![0]);
    /// assert_eq!(bav.get(1), vec![1]);
    /// ```
    pub fn set(&mut self, index: usize, bytes: &[u8]) {
        assert!(index < self.len);
        let prev_is_zero = self.get(index).iter().all(|byte| *byte == 0);
        let mut bits_left = self.bit_count;
        let mut bits_offset = index * self.bit_count;
        let mut byte_offset = 0;

        while bits_left > 0 {
            let curr_bits = cmp::min(bits_left, 8 - bits_offset % 8);
            bits_left -= curr_bits;

            let mut new_bits = {
                if byte_offset % 8 == 0 {
                    bytes[byte_offset / 8]
                } else if (8 - byte_offset % 8) >= bits_left + curr_bits {
                    (bytes[byte_offset / 8]) >> (byte_offset % 8)
                } else {
                    let curr = bytes[byte_offset / 8] >> (byte_offset % 8);
                    let next = bytes[byte_offset / 8 + 1] << (8 - byte_offset % 8);
                    curr | next
                }
            };

            new_bits = new_bits << (8 - curr_bits) >> (8 - curr_bits);

            self.blocks[bits_offset / 8] &= !(!0 >> (8 - curr_bits) << (bits_offset % 8));
            self.blocks[bits_offset / 8] |= new_bits << (bits_offset % 8);

            byte_offset += curr_bits;
            bits_offset += curr_bits;
        }
        let curr_is_zero = self.get(index).iter().all(|byte| *byte == 0);
        if prev_is_zero != curr_is_zero {
            if curr_is_zero {
                self.occupied_len -= 1;
            } else {
                self.occupied_len += 1;
            }
        }
    }

    /// Returns the value at index `index`.
    ///
    /// # Panics
    ///
    /// Panics if attempt to get an index out-of-bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 4);
    /// bav.set(1, &[1]);
    ///
    /// assert_eq!(bav.get(0), vec![0]);
    /// assert_eq!(bav.get(1), vec![1]);
    /// ```
    pub fn get(&self, index: usize) -> Vec<u8> {
        assert!(index < self.len);
        let mut ret = vec![0; (self.bit_count + 7) / 8];
        let mut bits_left = self.bit_count;
        let mut bits_offset = index * self.bit_count;
        let mut ret_offset = 0;

        while bits_left > 0 {
            let curr_bits = cmp::min(bits_left, 8);
            bits_left -= curr_bits;

            let old_bits = {
                if bits_offset % 8 == 0 {
                    self.blocks[bits_offset / 8]
                } else if 8 - bits_offset % 8 >= curr_bits {
                    self.blocks[bits_offset / 8] >> (bits_offset % 8)
                } else {
                    let curr = self.blocks[bits_offset / 8] >> (bits_offset % 8);
                    let next = self.blocks[bits_offset / 8 + 1] << (8 - bits_offset % 8);
                    curr | next
                }
            };

            ret[ret_offset / 8] = old_bits & (!0u8 >> (8 - curr_bits));

            bits_offset += curr_bits;
            ret_offset += curr_bits;
        }
        ret
    }

    /// Truncates a `BitArrayVec`, dropping excess elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 4);
    ///
    /// bav.truncate(2);
    /// assert_eq!(bav.iter().collect::<Vec<Vec<u8>>>(), vec![vec![0], vec![0]]);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        while len < self.len {
            self.pop();
        }
    }

    /// Reserves capacity for at least `additional` more bit arrays to be inserted in the given
    /// `BitArrayVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 4);
    ///
    /// bav.reserve(10);
    /// assert!(bav.capacity() >= 14);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap <= Self::get_elem_count(self.bit_count, self.blocks.capacity()) {
            return;
        }

        let target_cap = Self::get_block_count(self.bit_count, desired_cap);
        let additional_blocks = target_cap - self.blocks.len();
        self.blocks.reserve(additional_blocks);
    }

    /// Reserves capacity for at least `additional` more bit arrays to be inserted in the given
    /// `BitArrayVec`. Allocates exactly enough space in the underlying `Vec<u8>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 4);
    ///
    /// bav.reserve_exact(10);
    /// assert_eq!(bav.capacity(), 14);
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self.len + additional;
        if desired_cap <= Self::get_elem_count(self.bit_count, self.blocks.capacity()) {
            return;
        }

        let target_cap = Self::get_block_count(self.bit_count, desired_cap);
        let additional_blocks = target_cap - self.blocks.len();
        self.blocks.reserve_exact(additional_blocks);
    }

    /// Returns and removes the last element of the `BitVecArray`.
    ///
    /// # Panics
    ///
    /// Panics if the `BitVecArray` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 2);
    ///
    /// bav.set(1, &[1]);
    ///
    /// assert_eq!(bav.pop(), &[1]);
    /// assert_eq!(bav.pop(), &[0]);
    /// ```
    pub fn pop(&mut self) -> Vec<u8> {
        assert!(!self.is_empty());
        let ret = self.get(self.len - 1);
        let len = self.len;
        let new_blocks_len = Self::get_block_count(self.bit_count, len - 1);
        self.blocks.truncate(new_blocks_len);
        self.len -= 1;
        if !ret.iter().all(|byte| *byte == 0) {
            self.occupied_len -= 1;
        }
        ret
    }

    /// Pushes an element into the `BitArrayVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 1);
    ///
    /// bav.push(&[1]);
    ///
    /// assert_eq!(bav.pop(), &[1]);
    /// assert_eq!(bav.pop(), &[0]);
    /// ```
    pub fn push(&mut self, bytes: &[u8]) {
        let new_block_len = Self::get_block_count(self.bit_count, self.len + 1);
        let block_len = self.blocks.len();
        self.blocks
            .extend(iter::repeat(0).take(new_block_len - block_len));
        let len = self.len;
        let occupied_len = self.occupied_len;
        self.len += 1;
        self.set(len, bytes);
        self.occupied_len = occupied_len;
        if !bytes.iter().all(|byte| *byte == 0) {
            self.occupied_len = occupied_len + 1;
        }
    }

    /// Clears all elements in the `BitVecArray`, setting all bit arrays to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::from_elem(5, 4, &[1]);
    ///
    /// bav.clear();
    ///
    /// assert_eq!(
    ///     bav.iter().collect::<Vec<Vec<u8>>>(),
    ///     vec![vec![0], vec![0], vec![0], vec![0]],
    /// );
    /// ```
    pub fn clear(&mut self) {
        self.occupied_len = 0;
        for byte in &mut self.blocks {
            *byte = 0;
        }
    }

    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 1);
    ///
    /// bav.push(&[1]);
    ///
    /// assert_eq!(bav.iter().collect::<Vec<Vec<u8>>>(), vec![vec![0], vec![1]]);
    /// ```
    pub fn iter(&self) -> BitArrayVecIter {
        BitArrayVecIter {
            bit_array_vec: self,
            range: 0..self.len,
        }
    }

    /// Returns the capacity of the `BitArrayVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 0);
    ///
    /// bav.reserve_exact(10);
    /// assert_eq!(bav.capacity(), 11);
    /// ```
    pub fn capacity(&self) -> usize {
        Self::get_elem_count(self.bit_count, self.blocks.capacity())
    }

    /// Returns the number of elements in the `BitArrayVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 1);
    /// bav.push(&[1]);
    ///
    /// assert_eq!(bav.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the `BitArrayVec` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 0);
    /// assert!(bav.is_empty());
    ///
    /// bav.push(&[1]);
    /// assert!(!bav.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of non-zero elements in the `BitArrayVec`;
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let mut bav = BitArrayVec::new(5, 1);
    /// bav.push(&[1]);
    ///
    /// assert_eq!(bav.occupied_len(), 1);
    /// ```
    pub fn occupied_len(&self) -> usize {
        self.occupied_len
    }

    /// Returns the number of bits in each bit array stored by the `BitArrayVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bit_array_vec::BitArrayVec;
    ///
    /// let bav = BitArrayVec::new(5, 0);
    ///
    /// assert_eq!(bav.bit_count(), 5);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }
}

/// An owning iterator for `BitArrayVec`.
///
/// This iterator yields elements in order.
pub struct BitArrayVecIter<'a> {
    bit_array_vec: &'a BitArrayVec,
    range: Range<usize>,
}

impl<'a> Iterator for BitArrayVecIter<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        self.range.next().map(|index| self.bit_array_vec.get(index))
    }
}

impl<'a> IntoIterator for &'a BitArrayVec {
    type IntoIter = BitArrayVecIter<'a>;
    type Item = Vec<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator for `BitArrayVec`.
///
/// This iterator yields bits in order.
pub struct BitArrayVecIntoIter {
    bit_array_vec: BitArrayVec,
    range: Range<usize>,
}

impl Iterator for BitArrayVecIntoIter {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        self.range.next().map(|index| self.bit_array_vec.get(index))
    }
}

impl IntoIterator for BitArrayVec {
    type IntoIter = BitArrayVecIntoIter;
    type Item = Vec<u8>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len;
        Self::IntoIter {
            bit_array_vec: self,
            range: 0..len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitArrayVec;

    #[test]
    fn test_bit_count_5() {
        let mut bav = BitArrayVec::new(5, 8);
        assert_eq!(bav.len(), 8);

        bav.set(0, &[0]);
        assert_eq!(bav.occupied_len(), 0);

        for i in 0..8 {
            bav.set(i, &[((i + 1) as u8)]);
            assert_eq!(bav.occupied_len(), i + 1);
        }

        bav.set(0, &[1]);
        assert_eq!(bav.occupied_len(), 8);

        for i in 0..8 {
            assert_eq!(bav.get(i), vec![((i + 1) as u8)]);
            bav.set(i, &[0]);
            assert_eq!(bav.occupied_len(), 8 - i - 1);
        }

        for i in 0..7 {
            bav.set(i, &[((i + 1) as u8)]);
        }

        bav.truncate(4);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        assert_eq!(
            bav.iter().collect::<Vec<Vec<u8>>>(),
            vec![vec![1], vec![2], vec![3], vec![4]],
        );

        bav.push(&[5]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        bav.push(&[0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 6);

        assert_eq!(bav.pop(), vec![0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        assert_eq!(bav.pop(), vec![5]);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        bav.truncate(0);
        assert!(bav.is_empty());
        assert_eq!(bav.occupied_len(), 0);
        assert_eq!(bav.len(), 0);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve(10);
        assert!(bav.capacity() >= 18);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve_exact(10);
        assert_eq!(bav.capacity(), 18);
    }

    #[test]
    fn test_bit_count_13() {
        let mut bav = BitArrayVec::new(13, 8);
        assert_eq!(bav.len(), 8);

        bav.set(0, &[0, 0]);
        assert_eq!(bav.occupied_len(), 0);

        for i in 0..8 {
            bav.set(i, &[((i + 1) as u8), !0]);
            assert_eq!(bav.occupied_len(), i + 1);
        }

        bav.set(0, &[1, !0]);
        assert_eq!(bav.occupied_len(), 8);

        for i in 0..8 {
            assert_eq!(bav.get(i), vec![((i + 1) as u8), 0b11111]);
            bav.set(i, &[0, 0]);
            assert_eq!(bav.occupied_len(), 8 - i - 1);
        }

        for i in 0..7 {
            bav.set(i, &[((i + 1) as u8), !0]);
        }

        bav.truncate(4);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        assert_eq!(
            bav.iter().collect::<Vec<Vec<u8>>>(),
            vec![
                vec![1, 0b11111],
                vec![2, 0b11111],
                vec![3, 0b11111],
                vec![4, 0b11111],
            ],
        );

        bav.push(&[5, !0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        bav.push(&[0, 0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 6);

        assert_eq!(bav.pop(), vec![0, 0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        assert_eq!(bav.pop(), vec![5, 0b11111]);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        bav.truncate(0);
        assert!(bav.is_empty());
        assert_eq!(bav.occupied_len(), 0);
        assert_eq!(bav.len(), 0);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve(10);
        assert!(bav.capacity() >= 18);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve_exact(10);
        assert_eq!(bav.capacity(), 18);
    }

    #[test]
    fn test_bit_count_21() {
        let mut bav = BitArrayVec::new(21, 8);
        assert_eq!(bav.len(), 8);

        bav.set(0, &[0, 0, 0]);
        assert_eq!(bav.occupied_len(), 0);

        for i in 0..8 {
            bav.set(i, &[((i + 1) as u8), !0, !0]);
            assert_eq!(bav.occupied_len(), i + 1);
        }

        bav.set(0, &[1, !0, !0]);
        assert_eq!(bav.occupied_len(), 8);

        for i in 0..8 {
            assert_eq!(bav.get(i), vec![((i + 1) as u8), !0, 0b11111]);
            bav.set(i, &[0, 0, 0]);
            assert_eq!(bav.occupied_len(), 8 - i - 1);
        }

        for i in 0..7 {
            bav.set(i, &[((i + 1) as u8), !0, !0]);
        }

        bav.truncate(4);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        assert_eq!(
            bav.iter().collect::<Vec<Vec<u8>>>(),
            vec![
                vec![1, !0, 0b11111],
                vec![2, !0, 0b11111],
                vec![3, !0, 0b11111],
                vec![4, !0, 0b11111],
            ],
        );

        bav.push(&[5, !0, !0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        bav.push(&[0, 0, 0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 6);

        assert_eq!(bav.pop(), vec![0, 0, 0]);
        assert_eq!(bav.occupied_len(), 5);
        assert_eq!(bav.len(), 5);

        assert_eq!(bav.pop(), vec![5, !0, 0b11111]);
        assert_eq!(bav.occupied_len(), 4);
        assert_eq!(bav.len(), 4);

        bav.truncate(0);
        assert!(bav.is_empty());
        assert_eq!(bav.occupied_len(), 0);
        assert_eq!(bav.len(), 0);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve(10);
        assert!(bav.capacity() >= 18);

        let mut bav = BitArrayVec::new(21, 8);
        bav.reserve_exact(10);
        assert_eq!(bav.capacity(), 18);
    }

    #[test]
    fn test_bit_count() {
        let bav = BitArrayVec::new(8, 10);
        assert_eq!(bav.bit_count(), 8);
    }

    #[test]
    fn test_from_elem() {
        let bav = BitArrayVec::from_elem(5, 4, &[1]);
        assert_eq!(
            bav.iter().collect::<Vec<Vec<u8>>>(),
            vec![vec![1], vec![1], vec![1], vec![1]],
        );
    }

    #[test]
    fn test_with_capacity() {
        let bav = BitArrayVec::with_capacity(5, 4);

        assert_eq!(bav.len(), 0);
        assert_eq!(bav.capacity(), 4);
    }

    #[test]
    fn test_into_iter() {
        let mut bav = BitArrayVec::new(5, 0);
        bav.push(&[0]);
        bav.push(&[1]);
        bav.push(&[2]);
        bav.push(&[3]);

        assert_eq!(
            bav.into_iter().collect::<Vec<Vec<u8>>>(),
            vec![vec![0], vec![1], vec![2], vec![3]],
        );
    }
}
