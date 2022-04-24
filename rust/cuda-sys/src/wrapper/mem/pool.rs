use crate::wrapper::handle::Device;
use crate::wrapper::mem::device::DevicePtr;

#[derive(Debug)]
pub struct DevicePool {
    total_size_bytes: usize,
    buffer: DevicePtr,
    next_offset: usize,
}

impl DevicePool {
    pub fn new(device: Device, total_size_bytes: usize) -> Self {
        DevicePool {
            total_size_bytes,
            buffer: device.alloc(total_size_bytes),
            next_offset: 0,
        }
    }

    pub fn alloc(&mut self, size_bytes: usize) -> DevicePtr {
        assert!(
            self.next_offset + size_bytes <= self.total_size_bytes,
            "Not enough space left, used {}/{} and requested {}",
            self.next_offset,
            self.total_size_bytes,
            size_bytes
        );

        let result = self.buffer.clone().offset_bytes(self.next_offset as isize);
        self.next_offset += size_bytes;
        result
    }

    pub fn clear(&mut self) {
        assert_eq!(
            self.buffer.shared_count(),
            1,
            "Can only clear buffer without any outstanding users"
        );
        self.next_offset = 0;
    }

    pub fn total_size_bytes(&self) -> usize {
        self.total_size_bytes
    }

    pub fn size_left_bytes(&self) -> usize {
        self.total_size_bytes - self.next_offset
    }

    pub fn buffer(&self) -> &DevicePtr {
        &self.buffer
    }
}
