//! # probabilistic-collections-rs
//!
//! [![probabilistic-collections](http://meritbadge.herokuapp.com/probabilistic-collections)](https://crates.io/crates/probabilistic-collections)
//! [![Documentation](https://docs.rs/probabilistic-collections/badge.svg)](https://docs.rs/probabilistic-collections)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//! [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
//! [![Pipeline Status](https://gitlab.com/jeffrey-xiao/probabilistic-collections-rs/badges/master/pipeline.svg)](https://gitlab.com/jeffrey-xiao/probabilistic-collections-rs/-/commits/master)
//! [![Coverage Report](https://gitlab.com/jeffrey-xiao/probabilistic-collections-rs/badges/master/coverage.svg)](https://gitlab.com/jeffrey-xiao/probabilistic-collections-rs/-/commits/master)
//!
//! `probabilistic-collections` contains various implementations of collections that use
//! approximations to improve on running time or memory, but introduce a certain amount of error.
//! The error can be controlled under a certain threshold which makes these data structures
//! extremely useful for big data and streaming applications.
//!
//! The following types of collections are implemented:
//!
//! - Approximate Membership in Set: `BloomFilter`, `PartitionedBloomFilter`, `CuckooFilter`,
//!   `QuotientFilter`
//! - Scalable Approximate Membership in Set: `ScalableBloomFilter`, `ScalableCuckooFilter`
//! - Approximate Membership in Stream: `BSBloomFilter`, `BSSDBloomFilter`, `RLBSBloomFilter`
//! - Approximate Item Count: `CountMinSketch`
//! - Approximate Distinct Item Count: `HyperLogLog`
//! - Set similarity: `MinHash`, `SimHash`
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! probabilistic-collections = "*"
//! ```
//!
//! For [`serde`](https://github.com/serde-rs/serde) support, include the `serde` feature:
//!
//! ```toml
//! [dependencies]
//! probabilistic-collections = { version = "*", features = ["serde"] }
//! ```
//!
//! Add this to your crate root if you are using Rust 2015:
//!
//! ## Caveats
//!
//! If you are using this crate to create collections to be used across different platforms, you
//! must be careful not to use keys of type `[T]`. The Rust standard library implementation of
//! `Hash` for `[T]` first hashes the length of the slice, then the contents of the slice. The
//! length is platform specific because it is of type `usize`. Therefore, a collection with keys of
//! type `[T]` will have unexpected results when used across platforms. For example, if you were to
//! generate and serilize a `BloomFilter` on i686 compiled code where the keys are of type `[u8]`,
//! the filter will not have the correct results when deserialized on x86_64 compiled code.
//!
//! The recommended work-around to this problem is to a define a wrapper struct around a `[T]`
//! with a `Hash` implementation that only hashes the contents of the slice.
//!
//! ```rust
//! extern crate probabilistic_collections;
//! ```
//!
//! ## Changelog
//!
//! See [CHANGELOG](CHANGELOG.md) for more details.
//!
//! ## References
//!
//! - [Advanced Bloom Filter Based Algorithms for Efficient Approximate Data De-Duplication in Streams](https://arxiv.org/abs/1212.3964)
//!   > Bera, Suman K., Sourav Dutta, Ankur Narang, and Souvik Bhattacherjee. 2012. "Advanced Bloom Filter Based Algorithms for Efficient Approximate Data de-Duplication in Streams." _CoRR_ abs/1212.3964. <http://arxiv.org/abs/1212.3964>.
//! - [An improved data stream summary: the count-min sketch and its applications](https://dl.acm.org/citation.cfm?id=1073718)
//!   > Cormode, Graham, and S. Muthukrishnan. 2005. "An Improved Data Stream Summary: The Count-Min Sketch and Its Applications." _J. Algorithms_ 55 (1). Duluth, MN, USA: Academic Press, Inc.: 58--75. <https://doi.org/10.1016/j.jalgor.2003.12.001>.
//! - [Cuckoo Filter: Practically Better Than Bloom](https://dl.acm.org/citation.cfm?id=2674994)
//!   > Fan, Bin, Dave G. Andersen, Michael Kaminsky, and Michael D. Mitzenmacher. 2014. "Cuckoo Filter: Practically Better Than Bloom." In _Proceedings of the 10th Acm International on Conference on Emerging Networking Experiments and Technologies_, 75--88. CoNEXT '14. New York, NY, USA: ACM. <https://doi.org/10.1145/2674005.2674994>.
//! - [Don't thrash: how to cache your hash on flash](https://dl.acm.org/citation.cfm?id=2350275)
//!   > Bender, Michael A., Martin Farach-Colton, Rob Johnson, Russell Kraner, Bradley C. Kuszmaul, Dzejla Medjedovic, Pablo Montes, Pradeep Shetty, Richard P. Spillane, and Erez Zadok. 2012. "Don'T Thrash: How to Cache Your Hash on Flash." _Proc. VLDB Endow._ 5 (11). VLDB Endowment: 1627--37. <https://doi.org/10.14778/2350229.2350275>.
//! - [HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm](https://dl.acm.org/citation.cfm?id=2452456)
//!   > Heule, Stefan, Marc Nunkesser, and Alexander Hall. 2013. "HyperLogLog in Practice: Algorithmic Engineering of a State of the Art Cardinality Estimation Algorithm." In _Proceedings of the 16th International Conference on Extending Database Technology_, 683--92. EDBT '13. New York, NY, USA: ACM. <https://doi.org/10.1145/2452376.2452456>.
//! - [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
//!   > Flajolet, Philippe, Éric Fusy, Olivier Gandouet, and Frédéric Meunier. 2007. "Hyperloglog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm." In _IN Aofa '07: PROCEEDINGS of the 2007 International Conference on Analysis of Algorithms_.
//! - [Less hashing, same performance: Building a better Bloom filter](https://dl.acm.org/citation.cfm?id=1400125)
//!   > Kirsch, Adam, and Michael Mitzenmacher. 2008. "Less Hashing, Same Performance: Building a Better Bloom Filter." _Random Struct. Algorithms_ 33 (2). New York, NY, USA: John Wiley & Sons, Inc.: 187--218. <https://doi.org/10.1002/rsa.v33:2>.
//! - [Min-wise independent permutations (extended abstract)](https://dl.acm.org/citation.cfm?id=276781)
//!   > Broder, Andrei Z., Moses Charikar, Alan M. Frieze, and Michael Mitzenmacher. 1998. "Min-Wise Independent Permutations (Extended Abstract)." In _Proceedings of the Thirtieth Annual Acm Symposium on Theory of Computing_, 327--36. STOC '98. New York, NY, USA: ACM. <https://doi.org/10.1145/276698.276781>.
//! - [Probabilistic near-duplicate detection using simhash](https://dl.acm.org/citation.cfm?id=2063737)
//!   > Sood, Sadhan, and Dmitri Loguinov. 2011. "Probabilistic Near-Duplicate Detection Using Simhash." In _Proceedings of the 20th Acm International Conference on Information and Knowledge Management_, 1117--26. CIKM '11. New York, NY, USA: ACM. <https://doi.org/10.1145/2063576.2063737>.
//! - [Scalable Bloom Filters](https://dl.acm.org/citation.cfm?id=1224501)
//!   > Almeida, Paulo Sérgio, Carlos Baquero, Nuno Preguiça, and David Hutchison. 2007. "Scalable Bloom Filters." _Inf. Process. Lett._ 101 (6). Amsterdam, The Netherlands, The Netherlands: Elsevier North-Holland, Inc.: 255--61. <https://doi.org/10.1016/j.ipl.2006.10.007>.
//!
//! ## License
//!
//! `probabilistic-collections-rs` is dual-licensed under the terms of either the MIT License or the
//! Apache License (Version 2.0).
//!
//! See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for more details.

#![warn(missing_docs)]

mod bit_vec;
mod bitstring_vec;
pub mod bloom;
pub mod count_min_sketch;
pub mod cuckoo;
pub mod hyperloglog;
pub mod quotient;
pub mod similarity;
mod util;

pub use self::util::SipHasherBuilder;
use self::util::{DoubleHasher, HashIter};
