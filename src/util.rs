use rand::{Rng, XorShiftRng};
use siphasher::sip::SipHasher;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

pub fn get_hashers() -> [SipHasher; 2] {
    let mut rng = XorShiftRng::new_unseeded();
    [
        SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
        SipHasher::new_with_keys(rng.next_u64(), rng.next_u64()),
    ]
}

pub fn get_hashes<T, U>(hashers: &[SipHasher; 2], item: &U) -> [u64; 2]
where
    T: Borrow<U>,
    U: Hash + ?Sized,
{
    let mut ret = [0; 2];
    for (index, hash) in ret.iter_mut().enumerate() {
        let mut sip = hashers[index];
        item.hash(&mut sip);
        *hash = sip.finish();
    }
    ret
}

pub fn get_hash(index: usize, hashes: &[u64; 2]) -> u64 {
    let mut hash = (index as u64).wrapping_mul(hashes[1]) % 0xFFFF_FFFF_FFFF_FFC5;
    hash = hashes[0].wrapping_add(hash);
    hash
}
