use std::fmt::{Debug, Formatter};
use std::ops::Index;

pub struct BitBuffer {
    storage: Vec<u8>,
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
        assert!(self.len < self.capacity, "Not enough space left");

        let (index, bit) = split(self.len);
        self.len += 1;

        let block = &mut self.storage[index];
        if b {
            *block |= 1 << bit;
        } else {
            *block &= !(1 << bit);
        }
    }

    pub fn push_block(&mut self, i: u64) {
        assert!(self.len + 64 <= self.capacity, "Not enough space left");
        assert_eq!(self.len % 8, 0, "Can only push aligned blocks of bits");

        let index = self.len / 8;
        self.storage[index..index + 8].copy_from_slice(&i.to_le_bytes());

        self.len += 64;
    }

    pub fn clear(&mut self) {
        self.storage.fill(0);
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn storage(&self) -> &[u8] {
        &self.storage
    }
}

fn split(index: usize) -> (usize, u8) {
    (index / 8, (index % 8) as u8)
}

const TRUE: bool = true;
const FALSE: bool = false;

impl Index<usize> for BitBuffer {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        let (index, bit) = split(index);
        match ((self.storage[index] >> bit) & 1) != 0 {
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

impl Debug for BitBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_list();
        for i in 0..self.capacity {
            if i < self.len() {
                d.entry(&(self[i] as u8));
            } else {
                d.entry(&'-');
            }
        }
        d.finish()
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
        let mut buf = BitBuffer::new(8);
        buf.extend(std::iter::repeat(true).take(8));
        assert_eq!(&[0b1111_1111], buf.storage());

        let mut buf = BitBuffer::new(9);
        buf.extend(std::iter::repeat(true).take(9));
        assert_eq!(&[0b1111_1111, 0b1], buf.storage());
    }

    #[test]
    fn longer() {
        let mut buf = BitBuffer::new(16);
        for i in 0..16 {
            buf.push(i == 1 || i == 5 || i == 12);
        }
        assert_eq!(&[0b0010_0010, 0b1_0000], buf.storage());
    }

    #[test]
    #[should_panic]
    fn overflow() {
        let mut buf = BitBuffer::new(32);
        for _ in 0..33 { buf.push(false); }
    }

    #[test]
    fn block() {
        let mut buf = BitBuffer::new(64);
        buf.push_block(0b1_0000_0001);

        assert_eq!(&[0b1, 0b1], &buf.storage[0..2]);
        assert_eq!(&vec![0; 6], &buf.storage[2..]);
        assert_eq!(64, buf.len());
    }
}