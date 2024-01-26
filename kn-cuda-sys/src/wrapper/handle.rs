use std::mem::MaybeUninit;
use std::ptr::null_mut;

use bytemuck::cast_slice;

use crate::bindings::{
    cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasLtCreate, cublasLtDestroy, cublasLtHandle_t,
    cublasSetStream_v2, cudaDeviceAttr, cudaDeviceGetAttribute, cudaDeviceProp, cudaEventRecord, cudaGetDevice,
    cudaGetDeviceCount, cudaSetDevice, cudaStream_t, cudaStreamBeginCapture, cudaStreamCaptureMode,
    cudaStreamCreate, cudaStreamDestroy, cudaStreamEndCapture, cudaStreamSynchronize, cudaStreamWaitEvent, cudnnCreate,
    cudnnDestroy, cudnnHandle_t, cudnnSetStream,
};
// TODO fix this annoying v2 import once https://github.com/rust-lang/rust-bindgen/issues/2544 is fixed
use crate::bindings::cudaGetDeviceProperties_v2 as cudaGetDeviceProperties;
use crate::wrapper::event::CudaEvent;
use crate::wrapper::graph::CudaGraph;
use crate::wrapper::mem::device::DevicePtr;
use crate::wrapper::status::Status;

/// A cuda device index.
///
/// This crate tries to eliminate the global "current device" cuda state:
/// Every cuda call that depends on the device should be preceded by `device.switch_to()`,
/// which corresponds to [cudaSetDevice].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct CudaDevice(i32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct CudaDeviceNotAvailable;

impl CudaDevice {
    pub fn new(device: i32) -> Result<Self, CudaDeviceNotAvailable> {
        if 0 <= device && device <= cuda_device_count() {
            Ok(CudaDevice(device))
        } else {
            Err(CudaDeviceNotAvailable)
        }
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (0..cuda_device_count()).map(CudaDevice)
    }

    pub unsafe fn current() -> CudaDevice {
        let mut inner = 0;
        cudaGetDevice(&mut inner as *mut _).unwrap();
        CudaDevice(inner)
    }

    pub fn inner(self) -> i32 {
        self.0
    }

    // Set the current cuda device to this device.
    //TODO is this enough when there are multiple threads running?
    pub fn switch_to(self) {
        unsafe { cudaSetDevice(self.inner()).unwrap() }
    }

    pub fn alloc(self, len_bytes: usize) -> DevicePtr {
        DevicePtr::alloc(self, len_bytes)
    }

    pub fn properties(self) -> cudaDeviceProp {
        unsafe {
            self.switch_to();
            let mut properties = MaybeUninit::uninit();
            cudaGetDeviceProperties(properties.as_mut_ptr(), self.inner()).unwrap();
            properties.assume_init()
        }
    }

    pub fn attribute(self, attribute: cudaDeviceAttr) -> i32 {
        unsafe {
            let mut value: i32 = 0;
            cudaDeviceGetAttribute(&mut value as *mut _, attribute, self.inner()).unwrap();
            value
        }
    }

    pub fn compute_capability(self) -> ComputeCapability {
        ComputeCapability {
            major: self.attribute(cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor),
            minor: self.attribute(cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor),
        }
    }

    pub fn name(self) -> String {
        let properties = self.properties();
        let name = &properties.name;

        let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
        std::str::from_utf8(cast_slice::<i8, u8>(&name[..len]))
            .unwrap()
            .to_owned()
    }
}

fn cuda_device_count() -> i32 {
    unsafe {
        let mut count = 0;
        cudaGetDeviceCount(&mut count as *mut _).unwrap();
        count
    }
}

//TODO copy? clone? default stream?
#[derive(Debug)]
pub struct CudaStream {
    device: CudaDevice,
    inner: cudaStream_t,
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.inner).unwrap_in_drop();
        }
    }
}

impl CudaStream {
    pub fn new(device: CudaDevice) -> Self {
        unsafe {
            let mut inner = null_mut();
            device.switch_to();
            cudaStreamCreate(&mut inner as *mut _).unwrap();
            CudaStream { device, inner }
        }
    }

    pub fn synchronize(&self) {
        unsafe { cudaStreamSynchronize(self.inner()).unwrap() }
    }

    pub fn device(&self) -> CudaDevice {
        self.device
    }

    pub unsafe fn inner(&self) -> cudaStream_t {
        self.inner
    }

    pub fn record_event(&self) -> CudaEvent {
        let event = CudaEvent::new();
        self.record_existing_event(&event);
        event
    }

    pub fn record_existing_event(&self, event: &CudaEvent) {
        unsafe { cudaEventRecord(event.inner(), self.inner()).unwrap() }
    }

    pub fn wait_for_event(&self, event: &CudaEvent) {
        unsafe {
            cudaStreamWaitEvent(self.inner, event.inner(), 0).unwrap();
        }
    }

    pub unsafe fn begin_capture(&self) {
        cudaStreamBeginCapture(self.inner(), cudaStreamCaptureMode::cudaStreamCaptureModeGlobal).unwrap()
    }

    pub unsafe fn end_capture(&self) -> CudaGraph {
        let mut graph = null_mut();
        cudaStreamEndCapture(self.inner(), &mut graph as *mut _).unwrap();
        CudaGraph::new_from_inner(graph)
    }
}

#[derive(Debug)]
pub struct CudnnHandle {
    inner: cudnnHandle_t,
    stream: CudaStream,
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            self.device().switch_to();
            cudnnDestroy(self.inner).unwrap_in_drop()
        }
    }
}

impl CudnnHandle {
    pub fn new(device: CudaDevice) -> Self {
        CudnnHandle::new_with_stream(CudaStream::new(device))
    }

    pub fn new_with_stream(stream: CudaStream) -> Self {
        unsafe {
            let mut inner = null_mut();
            stream.device.switch_to();
            cudnnCreate(&mut inner as *mut _).unwrap();
            cudnnSetStream(inner, stream.inner()).unwrap();
            CudnnHandle { inner, stream }
        }
    }

    pub fn device(&self) -> CudaDevice {
        self.stream.device()
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub unsafe fn inner(&self) -> cudnnHandle_t {
        self.inner
    }
}

#[derive(Debug)]
pub struct CublasHandle {
    inner: cublasHandle_t,
    stream: CudaStream,
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.inner).unwrap_in_drop() }
    }
}

impl CublasHandle {
    pub fn new(device: CudaDevice) -> Self {
        CublasHandle::new_with_stream(CudaStream::new(device))
    }

    pub fn new_with_stream(stream: CudaStream) -> Self {
        unsafe {
            let mut inner = null_mut();
            stream.device.switch_to();
            cublasCreate_v2(&mut inner as *mut _).unwrap();
            cublasSetStream_v2(inner, stream.inner()).unwrap();
            CublasHandle { inner, stream }
        }
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub unsafe fn inner(&self) -> cublasHandle_t {
        self.inner
    }
}

#[derive(Debug)]
pub struct CublasLtHandle {
    inner: cublasLtHandle_t,
}

impl Drop for CublasLtHandle {
    fn drop(&mut self) {
        unsafe { cublasLtDestroy(self.inner).unwrap_in_drop() }
    }
}

impl CublasLtHandle {
    pub fn new(device: CudaDevice) -> Self {
        unsafe {
            let mut inner = null_mut();
            device.switch_to();
            cublasLtCreate(&mut inner as *mut _).unwrap();
            CublasLtHandle { inner }
        }
    }

    pub unsafe fn inner(&self) -> cublasLtHandle_t {
        self.inner
    }
}
