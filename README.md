# probabilistic-collections-rs

[![probabilistic-collections](http://meritbadge.herokuapp.com/probabilistic-collections)](https://crates.io/crates/probabilistic-collections)
[![Documentation](https://docs.rs/probabilistic-collections/badge.svg)](https://docs.rs/probabilistic-collections)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.org/jeffrey-xiao/probabilistic-collections-rs.svg?branch=master)](https://travis-ci.org/jeffrey-xiao/probabilistic-collections-rs)
[![codecov](https://codecov.io/gh/jeffrey-xiao/probabilistic-collections-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/jeffrey-xiao/probabilistic-collections-rs)

`probabilistic-collections` contains various implementations of collections that use randomization
to improve on running time or memory, but introduce a certain amount of error. The error can be
controlled under a certain threshold which makes these data structures extremely useful for big data
and streaming applications.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
probabilistic-collections = "*"
```
and this to your crate root if you are using Rust 2015:
```rust
extern crate probabilistic_collections;
```

## Changelog

See [CHANGELOG](CHANGELOG.md) for more details.

## References

 - [Advanced Bloom Filter Based Algorithms for Efficient Approximate Data De-Duplication in Streams](https://arxiv.org/abs/1212.3964)
 > Bera, Suman K., Sourav Dutta, Ankur Narang, and Souvik Bhattacherjee. 2012. "Advanced Bloom Filter Based Algorithms for Efficient Approximate Data de-Duplication in Streams." *CoRR* abs/1212.3964. <http://arxiv.org/abs/1212.3964>.
 - [An improved data stream summary: the count-min sketch and its applications](https://dl.acm.org/citation.cfm?id=1073718)
 > Cormode, Graham, and S. Muthukrishnan. 2005. "An Improved Data Stream Summary: The Count-Min Sketch and Its Applications." *J. Algorithms* 55 (1). Duluth, MN, USA: Academic Press, Inc.: 58--75. <https://doi.org/10.1016/j.jalgor.2003.12.001>.
 - [Cuckoo Filter: Practically Better Than Bloom](https://dl.acm.org/citation.cfm?id=2674994)
 > Fan, Bin, Dave G. Andersen, Michael Kaminsky, and Michael D. Mitzenmacher. 2014. "Cuckoo Filter: Practically Better Than Bloom." In *Proceedings of the 10th Acm International on Conference on Emerging Networking Experiments and Technologies*, 75--88. CoNEXT '14. New York, NY, USA: ACM. <https://doi.org/10.1145/2674005.2674994>.
 - [Don't thrash: how to cache your hash on flash](https://dl.acm.org/citation.cfm?id=2350275)
 > Bender, Michael A., Martin Farach-Colton, Rob Johnson, Russell Kraner, Bradley C. Kuszmaul, Dzejla Medjedovic, Pablo Montes, Pradeep Shetty, Richard P. Spillane, and Erez Zadok. 2012. "Don'T Thrash: How to Cache Your Hash on Flash." *Proc. VLDB Endow.* 5 (11). VLDB Endowment: 1627--37. <https://doi.org/10.14778/2350229.2350275>.
 - [HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm](https://dl.acm.org/citation.cfm?id=2452456)
 > Heule, Stefan, Marc Nunkesser, and Alexander Hall. 2013. "HyperLogLog in Practice: Algorithmic Engineering of a State of the Art Cardinality Estimation Algorithm." In *Proceedings of the 16th International Conference on Extending Database Technology*, 683--92. EDBT '13. New York, NY, USA: ACM. <https://doi.org/10.1145/2452376.2452456>.
 - [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
 > Flajolet, Philippe, Éric Fusy, Olivier Gandouet, and Frédéric Meunier. 2007. "Hyperloglog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm." In *IN Aofa '07: PROCEEDINGS of the 2007 International Conference on Analysis of Algorithms*.
 - [Less hashing, same performance: Building a better Bloom filter](https://dl.acm.org/citation.cfm?id=1400125)
 > Kirsch, Adam, and Michael Mitzenmacher. 2008. "Less Hashing, Same Performance: Building a Better Bloom Filter." *Random Struct. Algorithms* 33 (2). New York, NY, USA: John Wiley & Sons, Inc.: 187--218. <https://doi.org/10.1002/rsa.v33:2>.
 - [Min-wise independent permutations (extended abstract)](https://dl.acm.org/citation.cfm?id=276781)
 > Broder, Andrei Z., Moses Charikar, Alan M. Frieze, and Michael Mitzenmacher. 1998. "Min-Wise Independent Permutations (Extended Abstract)." In *Proceedings of the Thirtieth Annual Acm Symposium on Theory of Computing*, 327--36. STOC '98. New York, NY, USA: ACM. <https://doi.org/10.1145/276698.276781>.
 - [Probabilistic near-duplicate detection using simhash](https://dl.acm.org/citation.cfm?id=2063737)
 > Sood, Sadhan, and Dmitri Loguinov. 2011. "Probabilistic Near-Duplicate Detection Using Simhash." In *Proceedings of the 20th Acm International Conference on Information and Knowledge Management*, 1117--26. CIKM '11. New York, NY, USA: ACM. <https://doi.org/10.1145/2063576.2063737>.
 - [Scalable Bloom Filters](https://dl.acm.org/citation.cfm?id=1224501)
 > Almeida, Paulo Sérgio, Carlos Baquero, Nuno Preguiça, and David Hutchison. 2007. "Scalable Bloom Filters." *Inf. Process. Lett.* 101 (6). Amsterdam, The Netherlands, The Netherlands: Elsevier North-Holland, Inc.: 255--61. <https://doi.org/10.1016/j.ipl.2006.10.007>.

## License

`probabilistic-collections-rs` is dual-licensed under the terms of either the MIT License or the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for more details.
