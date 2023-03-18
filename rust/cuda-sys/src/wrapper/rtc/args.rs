use std::cmp::max;
use std::slice;

pub unsafe trait KernelArg {}

#[derive(Debug)]
pub struct KernelArgs {
    buffer: Vec<u8>,
    max_alignment: usize,
}

impl KernelArgs {
    pub fn new() -> Self {
        Self {
            buffer: vec![],
            max_alignment: 1,
        }
    }

    pub fn push<T: KernelArg>(&mut self, value: T) {
        // handle alignment
        let alignment = std::mem::align_of::<T>();
        self.max_alignment = max(self.max_alignment, alignment);
        self.pad_to(alignment);

        // append bytes
        unsafe {
            let bytes = slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>());
            self.buffer.extend_from_slice(bytes);
        }
    }

    pub fn push_int(&mut self, value: i32) {
        self.push(value)
    }

    pub fn finish(self) -> Vec<u8> {
        // we're not supposed to pad until alignment here
        self.buffer
    }

    fn pad_to(&mut self, alignment: usize) {
        while self.buffer.len() % alignment != 0 {
            self.buffer.push(0);
        }
    }
}

unsafe impl KernelArg for u8 {}

unsafe impl KernelArg for u16 {}

unsafe impl KernelArg for u32 {}

unsafe impl KernelArg for u64 {}

unsafe impl KernelArg for i8 {}

unsafe impl KernelArg for i16 {}

unsafe impl KernelArg for i32 {}

unsafe impl KernelArg for i64 {}

unsafe impl KernelArg for f32 {}

unsafe impl KernelArg for f64 {}

unsafe impl<T> KernelArg for *const T {}

unsafe impl<T> KernelArg for *mut T {}
