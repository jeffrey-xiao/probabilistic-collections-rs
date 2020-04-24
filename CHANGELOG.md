# Changelog

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
