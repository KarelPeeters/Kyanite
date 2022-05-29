use core::clone::Clone;
use std::cmp::Ordering;
use std::iter::Zip;

use itertools::Itertools;
use rand::Rng;

/// Similar to [rand::seq::IteratorRandom::choose] but will only pick items with the maximum key.
/// Equivalent to first finding the max key, then filtering items matching that key and then choosing a random element,
/// but implemented in a single pass over the iterator.
pub fn choose_max_by_key<T, I: IntoIterator<Item = T>, K: Ord, F: FnMut(&T) -> K>(
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

pub fn zip_eq_exact<L, R, LI, RI>(left: L, right: R) -> Zip<LI, RI>
where
    L: IntoIterator<IntoIter = LI>,
    R: IntoIterator<IntoIter = RI>,
    LI: ExactSizeIterator,
    RI: ExactSizeIterator,
{
    let left = left.into_iter();
    let right = right.into_iter();
    assert_eq!(left.len(), right.len(), "Both iterators must have the same length");
    left.zip(right)
}

/// Get the indices of the highest `k` values. The indices themselves are sorted from high to low as well.
/// `NaN` values are allowed but considered higher then any others, to ensure they don't go unnoticed.
pub fn top_k_indices_sorted(values: &[f32], k: usize) -> Vec<usize> {
    fn compare(a: f32, b: f32) -> Ordering {
        let ord = a.partial_cmp(&b);
        let eq = a == b || (a.is_nan() && b.is_nan());
        let first_nan = a.is_nan();

        match (ord, eq, first_nan) {
            (Some(ord), _, _) => ord,
            (None, true, _) => Ordering::Equal,
            (None, false, true) => Ordering::Greater,
            (None, false, false) => Ordering::Less,
        }
    }

    let compare_index = |&i: &usize, &j: &usize| compare(values[i], values[j]).reverse();

    let n = values.len();
    let mut result = (0..n).collect_vec();
    if k < n {
        result.select_nth_unstable_by(k, compare_index);
        result.truncate(k);
    }
    result.sort_by(compare_index);

    result
}

#[cfg(test)]
mod test {
    use crate::sequence::top_k_indices_sorted;

    #[test]
    fn top_k() {
        assert_eq!(top_k_indices_sorted(&[0.0, 2.0, 1.0], 2), vec![1, 2]);
        assert_eq!(top_k_indices_sorted(&[1.0, 2.0, 3.0], 20), vec![2, 1, 0]);
        assert_eq!(top_k_indices_sorted(&[1.0, 2.0, 3.0], 0), vec![]);
        assert_eq!(top_k_indices_sorted(&[f32::NAN, 2.0, 1.0], 2), vec![0, 1]);

        let result = top_k_indices_sorted(&[f32::NAN, 2.0, f32::NAN], 2);
        assert!(result == vec![0, 2] || result == vec![2, 0]);
    }
}

pub trait IndexOf<T> {
    fn index_of(self, element: T) -> Option<usize>;
}

impl<T: PartialEq, I: Iterator<Item = T>> IndexOf<T> for I {
    fn index_of(mut self, element: I::Item) -> Option<usize> {
        self.position(|cand| cand == element)
    }
}

pub trait VecExtPad {
    type T;
    fn pad(&mut self, result_size: usize, value: Self::T);
}

#[derive(Debug)]
pub struct SingleErr {
    pub len: usize,
}

pub trait VecExtSingle {
    type T;
    fn single(self) -> Result<Self::T, SingleErr>;
}

impl<T: Clone> VecExtPad for Vec<T> {
    type T = T;

    fn pad(&mut self, result_size: usize, value: T) {
        assert!(
            result_size >= self.len(),
            "Cannot pad to smaller size, curr {} target {}",
            self.len(),
            result_size
        );
        self.resize(result_size, value)
    }
}

impl<T> VecExtSingle for Vec<T> {
    type T = T;

    fn single(mut self) -> Result<T, SingleErr> {
        if self.len() == 1 {
            Ok(self.pop().unwrap())
        } else {
            Err(SingleErr { len: self.len() })
        }
    }
}
