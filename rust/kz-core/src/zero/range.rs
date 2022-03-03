use std::convert::TryInto;
use std::num::NonZeroUsize;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct IdxRange {
    pub start: NonZeroUsize,
    pub length: u8,
}

impl IdxRange {
    pub fn new(start: usize, end: usize) -> IdxRange {
        assert!(end > start, "IdxRange must be non-empty");
        IdxRange {
            start: NonZeroUsize::new(start).expect("IdxRange start cannot be 0"),
            length: (end - start).try_into().expect("IdxRange length too high"),
        }
    }

    pub fn iter(&self) -> std::ops::Range<usize> {
        self.start.get()..(self.start.get() + self.length as usize)
    }

    pub fn get(&self, index: usize) -> usize {
        assert!(
            index < self.length as usize,
            "Index {} out of bounds for {:?}",
            index,
            self
        );
        self.start.get() + index
    }
}

impl IntoIterator for IdxRange {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
