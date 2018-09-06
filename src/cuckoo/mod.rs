//! Space-efficient probabilistic data structure to test for membership in a set with the ability
//! to remove items.

mod cuckoo_filter;
mod scalable_cuckoo_filter;

const DEFAULT_ENTRIES_PER_INDEX: usize = 4;
const DEFAULT_FINGERPRINT_BIT_COUNT: usize = 8;
const DEFAULT_MAX_KICKS: usize = 512;

pub use self::cuckoo_filter::CuckooFilter;
pub use self::scalable_cuckoo_filter::ScalableCuckooFilter;
