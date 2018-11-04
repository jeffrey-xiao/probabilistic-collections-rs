# Changelog

## 0.4.0

### Added

 - `similarity` module with `MinHash`, `SimHash`, and `get_jaccard_similarity`.

### Changed

 - Abstract out common hash utility functions.

## 0.3.0

### Changed

 - All types are now generic over a single type.
 - API uses `Borrow` instead of `&` where applicable.

## 0.2.0

### Added

 - `CountMinSketch` with four different counting strategies: `CountMinStrategy`, `CountMeanStrategy`
   and `CountMedianBiasStrategy`. 

## 0.1.0

### Added

 - `hyperloglog` module with `HyperLogLog`.
 - `bloom` module with `BloomFilter`, `BSBloomFilter`, `BSSDBloomFilter`, `RLBSBloomFilter`,
   `PartitionedBloomFilter`, and `ScalableBloomFilter`.
 - `cuckoo` module with `CuckooFilter`, and `ScalableCuckooFilter`.
