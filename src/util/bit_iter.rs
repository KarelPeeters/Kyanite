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
    type Item = u8;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        //TODO report bug to intel-rust that self.left.is_zero() complains about a missing trait
        if self.left == N::zero() {
            None
        } else {
            let index = self.left.trailing_zeros() as u8;
            self.left = self.left & (self.left - N::one());
            Some(index)
        }
    }
}