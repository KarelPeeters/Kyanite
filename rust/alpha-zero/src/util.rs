use std::cmp::Ordering;

use rand::{Error, Rng, RngCore};

/// An Rng implementation that panics as soon as it is called.
/// Useful to assert that something doesn't actually use any randomness.
#[derive(Debug)]
pub struct PanicRng;

impl RngCore for PanicRng {
    fn next_u32(&mut self) -> u32 {
        panic!("Tried to use PanicRng")
    }

    fn next_u64(&mut self) -> u64 {
        panic!("Tried to use PanicRng")
    }

    fn fill_bytes(&mut self, _: &mut [u8]) {
        panic!("Tried to use PanicRng")
    }

    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Error> {
        panic!("Tried to use PanicRng")
    }
}

/// Similar to [rand::seq::IteratorRandom::choose] but will only pick items with the maximum key.
/// Equivalent to first finding the max key, then filtering items matching that key and then choosing a random element,
/// but implemented in a single pass over the iterator.
pub fn choose_max_by_key<T, I: IntoIterator<Item=T>, K: Ord, F: FnMut(&T) -> K>(
    iter: I,
    mut key: F,
    rng: &mut impl Rng,
) -> Option<T> {
    let mut iter = iter.into_iter();

    let mut curr = iter.next()?;
    let mut max_key = key(&curr);
    let mut i = 1;

    for next in iter {
        let next_key = key(&next);
        match next_key.cmp(&max_key) {
            Ordering::Less => continue,
            Ordering::Equal => {
                i += 1;
                if rng.gen_range(0..i) == 0 {
                    curr = next;
                }
            }
            Ordering::Greater => {
                i = 1;
                curr = next;
                max_key = next_key;
            }
        }
    }

    Some(curr)
}
