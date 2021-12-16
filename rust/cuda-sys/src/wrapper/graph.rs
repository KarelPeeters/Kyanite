use std::ptr::null_mut;

use crate::bindings::{cudaGraph_t, cudaGraphCreate, cudaGraphDestroy, cudaGraphExec_t, cudaGraphExecDestroy, cudaGraphInstantiate, cudaGraphLaunch};
use crate::wrapper::handle::CudaStream;
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct CudaGraph {
    inner: cudaGraph_t,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            cudaGraphDestroy(self.inner).unwrap_in_drop()
        }
    }
}

impl CudaGraph {
    pub fn new() -> Self {
        unsafe {
            let mut inner = null_mut();
            cudaGraphCreate(&mut inner as *mut _, 0).unwrap();
            CudaGraph { inner }
        }
    }

    pub unsafe fn new_from_inner(inner: cudaGraph_t) -> CudaGraph {
        CudaGraph { inner }
    }

    pub unsafe fn instantiate(&self) -> CudaGraphExec {
        //TODO try printing error string for fun
        let mut inner = null_mut();
        cudaGraphInstantiate(&mut inner as *mut _, self.inner(), null_mut(), null_mut(), 0).unwrap();
        CudaGraphExec { inner }
    }

    pub unsafe fn inner(&self) -> cudaGraph_t {
        self.inner
    }
}

#[derive(Debug)]
pub struct CudaGraphExec {
    inner: cudaGraphExec_t,
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe {
            cudaGraphExecDestroy(self.inner).unwrap_in_drop()
        }
    }
}

impl CudaGraphExec {
    pub unsafe fn inner(&self) -> cudaGraphExec_t {
        self.inner
    }

    pub unsafe fn launch(&self, stream: &CudaStream) {
        cudaGraphLaunch(self.inner(), stream.inner()).unwrap();
    }
}
