use std::ops::Index;

#[derive(Debug)]
pub struct BitBuffer {
    storage: Vec<u32>,
    capacity: usize,
    len: usize,
}

impl BitBuffer {
    pub fn new(capacity: usize) -> Self {
        BitBuffer {
            storage: vec![0; split(capacity - 1).0 + 1],
            capacity,
            len: 0,
        }
    }

    pub fn push(&mut self, b: bool) {
        assert!(self.len < self.capacity);

        let (index, bit) = split(self.len);
        self.len += 1;

        let block = &mut self.storage[index];
        if b {
            *block |= 1 << bit;
        } else {
            *block &= !(1 << bit);
        }
    }

    pub fn clear(&mut self) {
        self.storage.fill(0);
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn storage(&self) -> &[u32] {
        &self.storage
    }
}

fn split(index: usize) -> (usize, u32) {
    (index / 32, (index % 32) as u32)
}

const TRUE: bool = true;
const FALSE: bool = false;

impl Index<usize> for BitBuffer {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        let (index, bit) = split(index);
        match (self.storage[index] >> bit) != 0 {
            true => &TRUE,
            false => &FALSE,
        }
    }
}

impl Extend<bool> for BitBuffer {
    fn extend<T: IntoIterator<Item=bool>>(&mut self, iter: T) {
        iter.into_iter().for_each(|b| self.push(b))
    }
}

#[cfg(test)]
mod test {
    use crate::mapping::bit_buffer::BitBuffer;

    #[test]
    fn short() {
        let mut buf = BitBuffer::new(8);
        buf.push(true);
        buf.push(false);
        buf.push(true);

        assert_eq!(&[0b101], buf.storage());
    }

    #[test]
    fn edge_length() {
        let mut buf = BitBuffer::new(32);
        buf.extend(std::iter::repeat(false).take(32));
        assert_eq!(&[0], buf.storage());

        let mut buf = BitBuffer::new(33);
        buf.extend(std::iter::repeat(false).take(33));
        assert_eq!(&[0, 0], buf.storage());
    }

    #[test]
    fn longer() {
        let mut buf = BitBuffer::new(60);
        for i in 0..40 {
            buf.push(i == 1 || i == 10 || i == 36);
        }
        assert_eq!(&[0b10000000010, 0b10000], buf.storage());
    }

    #[test]
    #[should_panic]
    fn overflow() {
        let mut buf = BitBuffer::new(32);
        for _ in 0..33 { buf.push(false); }
    }
}