//! Space-efficient probabilistic data structure for estimating the number of item occurrences.
use rand::{Rng, XorShiftRng};
use std::marker::PhantomData;
use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};

/// Trait for types that have the logic for estimating the number of item occurrences.
pub trait CountStrategy {
    /// Returns the estimated number of item occurrences given the number of items in the
    /// `CountMinSketch`, the rows in the grid, the columns in the grid, and the values
    /// corresponding to the item in the grid
    fn get_estimate(items: i64, rows: usize, cols: usize, iter: ItemValueIter) -> i64;
}

/// A count strategy that uses the minimum value to estimate the number of item occurrences. This
/// strategy is suspectible to overestimation, but never underestimation.
pub struct CountMinStrategy;

impl CountStrategy for CountMinStrategy {
    fn get_estimate(_items: i64, _rows: usize, _cols: usize, iter: ItemValueIter) -> i64 {
        iter.min().expect("Expected `CountMinSketch` to be non-empty")
    }
}

/// A count strategy that uses the mean value to estimate the number of item occurrences. This
/// strategy performs well when there are item removals or if some values are negative.
pub struct CountMeanStrategy;

impl CountStrategy for CountMeanStrategy {
    fn get_estimate(_items: i64, rows: usize, _cols: usize, iter: ItemValueIter) -> i64 {
        (iter.sum::<i64>() as f64 / rows as f64).round() as i64
    }
}

/// A count strategy that uses the median value after substracting a bias from all values to
/// mitigate the effects of overestimation. This strategy may also overestimate, so we take the
/// minimum of the `CountMedianBiasStrategy` and the `CountMinStrategy` to get a more accurate
/// estimate.
pub struct CountMedianBiasStrategy;

impl CountStrategy for CountMedianBiasStrategy {
    fn get_estimate(items: i64, rows: usize, cols: usize, iter: ItemValueIter) -> i64 {
        let min_count = CountMinStrategy::get_estimate(items, rows, cols, iter.clone());
        let mut items_with_bias: Vec<i64> = iter
            .map(|value| value - ((items - value) as f64 / (cols - 1) as f64).ceil() as i64)
            .collect();
        items_with_bias.sort();
        let median_count = items_with_bias[(items_with_bias.len() - 1) / 2];
        i64::min(min_count, median_count)
    }
}

/// A space-efficient probabilistic data structure that serves as a frequency table of events in a
/// stream of data.
///
/// `CountMinSketch` uses hash functions to maps items to columns in a grid for every column. It
/// uses sublinear space at the expense of overestimating items due to collisions.
///
/// This implementation provides three distinct counting strategies: `CountMinStrategy`,
/// `CountMeanStrategy`, and `CountMedianBiasStrategy`.
///
/// # Examples
/// ```
/// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
///
/// let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::new(3, 28);
///
/// count_min_sketch.add(&"foo", 3);
/// count_min_sketch.add(&"bar", 5);
/// assert_eq!(count_min_sketch.count(&"foo"), 3);
/// assert_eq!(count_min_sketch.count(&"bar"), 5);
///
/// count_min_sketch.remove(&"foo", 2);
/// assert_eq!(count_min_sketch.count(&"foo"), 1);
///
/// count_min_sketch.clear();
/// assert_eq!(count_min_sketch.count(&"foo"), 0);
/// assert_eq!(count_min_sketch.count(&"bar"), 0);
///
/// assert_eq!(count_min_sketch.cols(), 28);
/// assert_eq!(count_min_sketch.rows(), 3);
/// assert!(count_min_sketch.confidence() <= 0.1);
/// assert!(count_min_sketch.error() <= 0.05);
/// ```
pub struct CountMinSketch<T> {
    // A 2D grid represented as a 1D vector of signed 64-bit integers to support removals and
    // negatives
    rows: usize,
    cols: usize,
    items: i64,
    grid: Vec<i64>,
    hashers: [SipHasher; 2],
    _marker: PhantomData<T>,
}

impl<T> CountMinSketch<T>
where
    T: CountStrategy,
{
    fn get_hashers() -> [SipHasher; 2] {
        let mut rng = XorShiftRng::new_unseeded();
        [
            SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
            SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
        ]
    }

    /// Constructs a new, empty `CountMinSketch` with a specific number of rows and columns.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::new(3, 28);
    ///
    /// assert_eq!(count_min_sketch.rows(), 3);
    /// assert_eq!(count_min_sketch.cols(), 28);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        CountMinSketch {
            rows,
            cols,
            items: 0,
            grid: vec![0; rows * cols],
            hashers: Self::get_hashers(),
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `CountMinSketch` with a upper bound on the confidence (`epsilon`)
    /// and the error (`delta`).
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    ///
    /// assert!(count_min_sketch.confidence() <= 0.1);
    /// assert!(count_min_sketch.error() <= 0.05);
    /// ```
    pub fn from_error(epsilon: f64, delta: f64) -> Self {
        let rows = (1.0 / delta).ln().ceil() as usize;
        let cols = ((1.0_f64).exp() / epsilon).ceil() as usize;
        CountMinSketch {
            rows,
            cols,
            items: 0,
            grid: vec![0; rows * cols],
            hashers: Self::get_hashers(),
            _marker: PhantomData,
        }
    }

    fn get_hashes<U>(&self, item: &U) -> [u64; 2]
    where
        U: Hash,
    {
        let mut ret = [0; 2];
        for (index, hash) in ret.iter_mut().enumerate() {
            let mut sip = self.hashers[index];
            item.hash(&mut sip);
            *hash = sip.finish();
        }
        ret
    }

    /// Inserts an element into the count-min sketch `value` times.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// count_min_sketch.add(&"foo", 3);
    /// assert_eq!(count_min_sketch.count(&"foo"), 3);
    /// ```
    pub fn add<U>(&mut self, item: &U, value: i64)
    where
        U: Hash,
    {
        self.items += value;
        let hashes = self.get_hashes(item);
        for row in 0..self.rows {
            let mut offset = (row as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
            offset = hashes[0].wrapping_add(offset);
            offset %= self.cols as u64;
            self.grid[row * self.cols + offset as usize] += value;
        }
    }

    /// Removes an element from the count-min sketch `value` times.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// count_min_sketch.add(&"foo", 3);
    /// count_min_sketch.remove(&"foo", 2);
    /// assert_eq!(count_min_sketch.count(&"foo"), 1);
    /// ```
    pub fn remove<U>(&mut self, item: &U, value: i64)
    where
        U: Hash,
    {
        self.add(item, -value);
    }

    /// Returns the estimated number of times `item` is in the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// count_min_sketch.add(&"foo", 3);
    /// assert_eq!(count_min_sketch.count(&"foo"), 3);
    /// ```
    pub fn count<U>(&mut self, item: &U) -> i64
    where
        U: Hash,
    {
        let iter = ItemValueIter {
            row: 0,
            rows: self.rows,
            cols: self.cols,
            hashes: self.get_hashes(item),
            grid: &self.grid,
        };
        T::get_estimate(self.items, self.rows, self.cols, iter)
    }

    /// Clears all items from the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// count_min_sketch.add(&"foo", 3);
    /// count_min_sketch.clear();
    /// assert_eq!(count_min_sketch.count(&"foo"), 0);
    /// ```
    pub fn clear(&mut self) {
        for value in &mut self.grid {
            *value = 0
        }
    }

    /// Returns the number of rows in the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::new(3, 28);
    /// assert_eq!(count_min_sketch.rows(), 3);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::new(3, 28);
    /// assert_eq!(count_min_sketch.cols(), 28);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the approximate confidence of the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// assert!(count_min_sketch.confidence() <= 0.1);
    /// ```
    pub fn confidence(&self) -> f64 {
        1.0_f64.exp() / self.cols as f64
    }

    /// Returns the approximate error of the count-min sketch.
    ///
    /// # Examples
    /// ```
    /// use probabilistic_collections::count_min_sketch::{CountMinStrategy, CountMinSketch};
    ///
    /// let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
    /// assert!(count_min_sketch.error() <= 0.05);
    /// ```
    pub fn error(&self) -> f64 {
        1.0_f64 / (self.rows as f64).exp()
    }
}

/// An iterator that yields values corresponding to an item in the count-min sketch.
#[derive(Clone)]
pub struct ItemValueIter<'a> {
    row: usize,
    rows: usize,
    cols: usize,
    grid: &'a Vec<i64>,
    hashes: [u64; 2],
}

impl<'a> Iterator for ItemValueIter<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row == self.rows {
            return None;
        }

        let mut offset = (self.row as u64).wrapping_mul(self.hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
        offset = self.hashes[0].wrapping_add(offset);
        offset %= self.cols as u64;
        offset += (self.row * self.cols) as u64;
        self.row += 1;
        Some(self.grid[offset as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::{CountMinSketch, CountMeanStrategy, CountMedianBiasStrategy, CountMinStrategy};

    #[test]
    fn test_new() {
        let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::new(3, 28);

        assert_eq!(count_min_sketch.cols(), 28);
        assert_eq!(count_min_sketch.rows(), 3);
        assert!(count_min_sketch.confidence() <= 0.1);
        assert!(count_min_sketch.error() <= 0.05);
    }

    #[test]
    fn test_from_error() {
        let count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);

        assert_eq!(count_min_sketch.cols(), 28);
        assert_eq!(count_min_sketch.rows(), 3);
        assert!(count_min_sketch.confidence() <= 0.1);
        assert!(count_min_sketch.error() <= 0.05);
    }

    #[test]
    fn test_add() {
        let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 3);

        let mut count_min_sketch: CountMinSketch<CountMeanStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 3);

        let mut count_min_sketch: CountMinSketch<CountMedianBiasStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 3);
    }

    #[test]
    fn test_remove() {
        let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        count_min_sketch.remove(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 0);

        let mut count_min_sketch: CountMinSketch<CountMeanStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        count_min_sketch.remove(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 0);

        let mut count_min_sketch: CountMinSketch<CountMedianBiasStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        count_min_sketch.remove(&"foo", 3);
        assert_eq!(count_min_sketch.count(&"foo"), 0);
    }

    #[test]
    fn test_clear() {
        let mut count_min_sketch: CountMinSketch<CountMinStrategy> = CountMinSketch::from_error(0.1, 0.05);
        count_min_sketch.add(&"foo", 3);
        count_min_sketch.clear();
        assert_eq!(count_min_sketch.count(&"foo"), 0);
    }
}
