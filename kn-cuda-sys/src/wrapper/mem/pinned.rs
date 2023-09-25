use std::ffi::c_void;
use std::ptr::null_mut;
use std::slice;

use crate::bindings::{cudaFreeHost, cudaHostAlloc, cudaHostAllocDefault, cudaHostAllocWriteCombined};
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct PinnedMem {
    ptr: *mut c_void,
    len_bytes: usize,
}

impl Drop for PinnedMem {
    fn drop(&mut self) {
        unsafe {
            cudaFreeHost(self.ptr()).unwrap_in_drop();
        }
    }
}

impl PinnedMem {
    // TODO should we take a device param here?
    pub fn alloc(len_bytes: usize, write_combined: bool) -> Self {
        //TODO should we set cudaHostAllocPortable/Mapped here?
        let flags = if write_combined {
            cudaHostAllocWriteCombined
        } else {
            cudaHostAllocDefault
        };

        unsafe {
            let mut result = null_mut();
            cudaHostAlloc(&mut result as *mut _, len_bytes, flags).unwrap();
            PinnedMem { ptr: result, len_bytes }
        }
    }

    pub unsafe fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Safety: ensure no other slice currently exists and no device is writing to this memory.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_slice(&self) -> &mut [u8] {
        // TODO this function (and maybe others like it, check) is terrible:
        //     * it can expose uninitialized memory as a slice
        //     * we need a hack to get empty slices to work at all (they can't be null)
        //     * more generally it's too easy to abuse
        //   consider replacing this with https://crates.io/crates/presser
        // TODO this is probably just unsafe, see https://rust-lang.github.io/rust-clippy/master/index.html#mut_from_ref
        if self.len_bytes == 0 {
            &mut []
        } else {
            slice::from_raw_parts_mut(self.ptr as *mut u8, self.len_bytes)
        }
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}
