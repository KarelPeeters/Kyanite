use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

use itertools::zip;
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

pub trait IndexOf<T> {
    fn index_of(self, element: T) -> Option<usize>;
}

impl<T: PartialEq, I: Iterator<Item=T>> IndexOf<T> for I {
    fn index_of(mut self, element: I::Item) -> Option<usize> {
        self.position(|cand| cand == element)
    }
}

pub fn display_option<T: Display>(value: Option<T>) -> impl Display {
    struct Wrapper<T>(Option<T>);
    impl<T: Display> Display for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match &self.0 {
                None => write!(f, "None"),
                Some(value) => write!(f, "Some({})", value),
            }
        }
    }
    Wrapper(value)
}

/// Calculates `D_KDL(P || Q)`, read as _the divergence of P from Q_,
/// a measure of how different two probability distributions are.
/// See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
pub fn kdl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len());

    zip(p, q)
        .map(|(&p, &q)| p * (p / q).ln())
        .sum()
}
