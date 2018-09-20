//! # probabilistic-collections-rs
//!
//! [![probabilistic-collections](http://meritbadge.herokuapp.com/probabilistic-collections)](https://crates.io/crates/probabilistic-collections)
//! [![Documentation](https://docs.rs/probabilistic-collections/badge.svg)](https://docs.rs/probabilistic-collections)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//! [![Build Status](https://travis-ci.org/jeffrey-xiao/probabilistic-collections-rs.svg?branch=master)](https://travis-ci.org/jeffrey-xiao/probabilistic-collections-rs)
//! [![codecov](https://codecov.io/gh/jeffrey-xiao/probabilistic-collections-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/jeffrey-xiao/probabilistic-collections-rs)
//!
//! `probabilistic-collections` contains various implementations of collections that use randomization
//! to improve on running time or memory, but introduce a certain amount of error. The error can be
//! controlled under a certain threshold which makes these data structures extremely useful for big data
//! and streaming applications.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! probabilistic-collections = "*"
//! ```
//! and this to your crate root:
//! ```rust
//! extern crate probabilistic_collections;
//! ```
//!
//! ## References
//!
//!  - [Scalable Bloom Filters](https://dl.acm.org/citation.cfm?id=1224501)
//!  > Almeida, Paulo Sérgio, Carlos Baquero, Nuno Preguiça, and David Hutchison. 2007. “Scalable Bloom Filters.” *Inf. Process. Lett.* 101 (6). Amsterdam, The Netherlands, The Netherlands: Elsevier North-Holland, Inc.: 255–61. doi:[10.1016/j.ipl.2006.10.007](https://doi.org/10.1016/j.ipl.2006.10.007).
//!  - [Advanced Bloom Filter Based Algorithms for Efficient Approximate Data De-Duplication in Streams](https://arxiv.org/abs/1212.3964)
//!  > Bera, Suman K., Sourav Dutta, Ankur Narang, and Souvik Bhattacherjee. 2012. “Advanced Bloom Filter Based Algorithms for Efficient Approximate Data de-Duplication in Streams.” *CoRR* abs/1212.3964. <http://arxiv.org/abs/1212.3964>.
//!  - [Cuckoo Filter: Practically Better Than Bloom](https://dl.acm.org/citation.cfm?id=2674994)
//!  > Fan, Bin, Dave G. Andersen, Michael Kaminsky, and Michael D. Mitzenmacher. 2014. “Cuckoo Filter: Practically Better Than Bloom.” In *Proceedings of the 10th Acm International on Conference on Emerging Networking Experiments and Technologies*, 75–88. CoNEXT ’14. New York, NY, USA: ACM. doi:[10.1145/2674005.2674994](https://doi.org/10.1145/2674005.2674994).
//!  - [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
//!  > Flajolet, Philippe, Éric Fusy, Olivier Gandouet, and Frédéric Meunier. 2007. “Hyperloglog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm.” In *IN Aofa ’07: PROCEEDINGS of the 2007 International Conference on Analysis of Algorithms*.
//!  - [HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm](https://dl.acm.org/citation.cfm?id=2452456)
//!  > Heule, Stefan, Marc Nunkesser, and Alexander Hall. 2013. “HyperLogLog in Practice: Algorithmic Engineering of a State of the Art Cardinality Estimation Algorithm.” In *Proceedings of the 16th International Conference on Extending Database Technology*, 683–92. EDBT ’13. New York, NY, USA: ACM. doi:[10.1145/2452376.2452456](https://doi.org/10.1145/2452376.2452456).
//!  - [Less hashing, same performance: Building a better Bloom filter](https://dl.acm.org/citation.cfm?id=1400125)
//!  > Kirsch, Adam, and Michael Mitzenmacher. 2008. “Less Hashing, Same Performance: Building a Better Bloom Filter.” *Random Struct. Algorithms* 33 (2). New York, NY, USA: John Wiley & Sons, Inc.: 187–218. doi:[10.1002/rsa.v33:2](https://doi.org/10.1002/rsa.v33:2).

#![warn(missing_docs)]

extern crate bincode;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate siphasher;

pub mod bit_array_vec;
pub mod bit_vec;
pub mod bloom;
pub mod count_min_sketch;
pub mod cuckoo;
pub mod hyperloglog;
pub mod quotient;
pub mod similarity;
mod util;
