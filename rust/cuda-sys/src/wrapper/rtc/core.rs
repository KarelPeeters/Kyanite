use itertools::Itertools;
use std::ffi::{c_void, CString};
use std::ptr::{null, null_mut};
use std::rc::Rc;

use crate::bindings::{
    cuLaunchKernel, cuModuleGetFunction, cuModuleLoadDataEx, cuModuleUnload, cudaError_enum as CUresult,
    nvrtcCompileProgram, nvrtcCreateProgram, nvrtcDestroyProgram, nvrtcGetPTX, nvrtcGetPTXSize, nvrtcGetProgramLog,
    nvrtcGetProgramLogSize, nvrtcResult, CU_LAUNCH_PARAM_BUFFER_POINTER, CU_LAUNCH_PARAM_BUFFER_SIZE,
    CU_LAUNCH_PARAM_END,
};
use crate::wrapper::handle::{CudaStream, Device};
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct CuModule {
    inner: Rc<CuModuleInner>,
}

#[derive(Debug)]
struct CuModuleInner {
    inner: crate::bindings::CUmodule,
}

#[derive(Debug)]
pub struct CuFunction {
    // field is never used, but is present to keep module from being dropped
    _module: Rc<CuModuleInner>,
    function: crate::bindings::CUfunction,
}

#[must_use]
#[derive(Debug)]
pub struct CompileResult {
    pub log: String,
    pub module: Result<CuModule, nvrtcResult>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Drop for CuModuleInner {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.inner).unwrap_in_drop();
        }
    }
}

impl CuModule {
    pub unsafe fn from_ptx(ptx: &[u8]) -> CuModule {
        let mut inner = null_mut();
        cuModuleLoadDataEx(
            &mut inner as *mut _,
            ptx.as_ptr() as *const _,
            0,
            null_mut(),
            null_mut(),
        )
        .unwrap();
        CuModule {
            inner: Rc::new(CuModuleInner { inner }),
        }
    }

    pub unsafe fn from_source(src: &str, name: Option<&str>, device: Device) -> CompileResult {
        let mut program = null_mut();

        let src_c = CString::new(src.as_bytes()).unwrap();
        let name_c = name.map(|name| CString::new(name.as_bytes()).unwrap());

        nvrtcCreateProgram(
            &mut program as *mut _,
            src_c.as_ptr() as *const i8,
            name_c.map_or(null(), |name_c| name_c.as_ptr() as *const i8),
            0,
            null(),
            null(),
        )
        .unwrap();

        let props = device.properties();

        let args = vec![
            format!("--gpu-architecture=compute_{}{}", props.major, props.minor),
            // "-G".to_string(),
            // "--generate-line-info".to_string(),
            "--define-macro=NVRTC".to_string(),
        ];

        let args = args
            .into_iter()
            .map(CString::new)
            .collect::<Result<Vec<CString>, _>>()
            .unwrap();
        let args = args.iter().map(|s| s.as_ptr() as *const i8).collect_vec();

        let result = nvrtcCompileProgram(program, args.len() as i32, args.as_ptr());

        let mut log_size: usize = 0;
        nvrtcGetProgramLogSize(program, &mut log_size as *mut _).unwrap();

        let mut log_bytes = vec![0u8; log_size];
        nvrtcGetProgramLog(program, log_bytes.as_mut_ptr() as *mut _).unwrap();

        let log_c = CString::from_vec_with_nul(log_bytes).unwrap();
        let log = log_c.to_str().unwrap().to_owned();

        if result != nvrtcResult::NVRTC_SUCCESS {
            return CompileResult {
                log,
                module: Err(result),
            };
        }

        let mut ptx_size = 0;
        nvrtcGetPTXSize(program, &mut ptx_size as *mut _).unwrap();

        let mut ptx = vec![0u8; ptx_size];
        nvrtcGetPTX(program, ptx.as_mut_ptr() as *mut _);

        nvrtcDestroyProgram(&mut program as *mut _).unwrap();

        let module = CuModule::from_ptx(&ptx);

        CompileResult {
            log,
            module: Ok(module),
        }
    }

    pub unsafe fn get_function(&self, name: &str) -> Option<CuFunction> {
        let name_c = CString::new(name.as_bytes()).unwrap();
        let mut function = null_mut();

        let result = cuModuleGetFunction(&mut function as *mut _, self.inner.inner, name_c.as_ptr());

        if result == CUresult::CUDA_ERROR_NOT_FOUND {
            None
        } else {
            result.unwrap();
            Some(CuFunction {
                _module: Rc::clone(&self.inner),
                function,
            })
        }
    }
}

impl CuFunction {
    pub unsafe fn launch_kernel_pointers(
        &self,
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem_bytes: u32,
        stream: &CudaStream,
        args: &[*mut c_void],
    ) {
        cuLaunchKernel(
            self.function,
            grid_dim.x,
            grid_dim.y,
            grid_dim.z,
            block_dim.x,
            block_dim.y,
            block_dim.z,
            shared_mem_bytes,
            stream.inner(),
            args.as_ptr() as *mut _,
            null_mut(),
        )
        .unwrap()
    }

    pub unsafe fn launch_kernel(
        &self,
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem_bytes: u32,
        stream: &CudaStream,
        args: &[u8],
    ) -> CUresult {
        let mut config = [
            CU_LAUNCH_PARAM_BUFFER_POINTER,
            args.as_ptr() as *mut c_void,
            CU_LAUNCH_PARAM_BUFFER_SIZE,
            &mut args.len() as *mut usize as *mut c_void,
            CU_LAUNCH_PARAM_END,
        ];

        cuLaunchKernel(
            self.function,
            grid_dim.x,
            grid_dim.y,
            grid_dim.z,
            block_dim.x,
            block_dim.y,
            block_dim.z,
            shared_mem_bytes,
            stream.inner(),
            null_mut(),
            config.as_mut_ptr(),
        )
    }
}

impl Dim3 {
    pub fn single(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}
