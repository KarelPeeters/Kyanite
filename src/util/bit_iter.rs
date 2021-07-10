use num::{PrimInt, Unsigned};

pub struct BitIter<N: PrimInt + Unsigned> {
    left: N,
}

impl<N: PrimInt + Unsigned> BitIter<N> {
    pub fn new(left: N) -> Self {
        BitIter { left }
    }
}

impl<N: PrimInt + Unsigned> Iterator for BitIter<N> {
    type Item = u32;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.left.is_zero() {
            None
        } else {
            let index = self.left.trailing_zeros();
            self.left = self.left & (self.left - N::one());
            Some(index)
        }
    }
}