# Changelog

## 0.7.0 - 2020-05-10

### Added

- Add `DoubleHasher` and `HashIter` to abstract away double hashing. A cubic was also added to
  double hashing to improve the distribution (enhanced double hashing).
- Add `SipHasherBuilder` as the default hash builder for all collections.
- Add `with_hasher` methods to all collections to specify a custom hasher.

### Changed

- Rename `add` to `insert` in `CountMinSketch`.
- Move insertion assertion for unique items in `QuotientFilter`. Previously, an assertion would fire
  if a duplicate item is inserted into a full `QuotientFilter`. The new behavior is that the
  assertion would only fire if an unique item is inserted into a full `QuotientFilter`.
- Change RNGs to seed from entropy.

### Fixed

- Update minimal versions of dependencies so `cargo check --all-targets --all-features` passes after
  running `cargo update -Zminimal-versions`.
- Properly check that an item is in its canonical slot in `QuotientFilter`. Previously, only the
  remainder was checked. The continuation and shifted bits must also be checked to ensure that the
  item is in its canonical slot.
- Fix issue with `ScalableCuckooFilter` where an overflow item is inserted into a new filter with
  the wrong fingerprint. To maintain the false positive probability, the new filter may have a
  larger item fingerprint than the old filter. It is incorrect to use the fingerprint of the old
  filter in the new filter. The overflow item remains in the old filter.

## 0.6.0 - 2020-04-24

### Added

- `serde` support under the `serde` feature.

### Changed

- Relax `count` in `count_min_sketch` to take `&self` instead of `&mut self`.

## 0.5.0 - 2018-11-03

### Added

- `quotient` module with `QuotientFilter`.

### Changed

- Refactor tests to use macros.
- More consistent error messages.
- Rename `estimate_fpp` to `estimated_fpp`.

## 0.4.0 - 2018-09-08

### Added

- `similarity` module with `MinHash`, `SimHash`, and `get_jaccard_similarity`.

### Changed

- Abstract out common hash utility functions.

## 0.3.0 - 2018-09-08

### Changed

- All types are now generic over a single type.
- API uses `Borrow` instead of `&` where applicable.

## 0.2.0 - 2018-09-07

### Added

- `CountMinSketch` with four different counting strategies: `CountMinStrategy`, `CountMeanStrategy`,
  and `CountMedianBiasStrategy`.

## 0.1.0 - 2018-09-06

### Added

- `hyperloglog` module with `HyperLogLog`.
- `bloom` module with `BloomFilter`, `BSBloomFilter`, `BSSDBloomFilter`, `RLBSBloomFilter`,
  `PartitionedBloomFilter`, and `ScalableBloomFilter`.
- `cuckoo` module with `CuckooFilter`, and `ScalableCuckooFilter`.
