use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use itertools::Itertools;

use crate::bindings::{
    cuLaunchKernel, cuModuleGetFunction, cuModuleLoadDataEx, cuModuleUnload, nvrtcAddNameExpression,
    nvrtcCompileProgram, nvrtcCreateProgram, nvrtcDestroyProgram, nvrtcGetLoweredName, nvrtcGetPTX, nvrtcGetPTXSize,
    nvrtcGetProgramLog, nvrtcGetProgramLogSize, nvrtcResult, CUresult, CU_LAUNCH_PARAM_BUFFER_POINTER,
    CU_LAUNCH_PARAM_BUFFER_SIZE, CU_LAUNCH_PARAM_END,
};
use crate::wrapper::handle::{ComputeCapability, CudaStream};
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct CuModule {
    inner: Arc<CuModuleInner>,
}

#[derive(Debug)]
struct CuModuleInner {
    inner: crate::bindings::CUmodule,
}

#[derive(Debug, Clone)]
pub struct CuFunction {
    // field is never used, but is present to keep module from being dropped
    _module: Arc<CuModuleInner>,
    function: crate::bindings::CUfunction,
}

unsafe impl Send for CuModuleInner {}

unsafe impl Send for CuFunction {}

#[must_use]
#[derive(Debug)]
pub struct CompileResult {
    pub log: String,
    pub module: Result<CuModule, nvrtcResult>,
    pub lowered_names: HashMap<String, String>,
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
            inner: Arc::new(CuModuleInner { inner }),
        }
    }

    pub fn from_source(
        cap: ComputeCapability,
        src: &str,
        name: Option<&str>,
        expected_names: &[&str],
    ) -> CompileResult {
        unsafe {
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

            // add requested names
            for &expected_name in expected_names {
                let expected_name_c = CString::new(expected_name.as_bytes()).unwrap();
                nvrtcAddNameExpression(program, expected_name_c.as_ptr()).unwrap();
            }

            // figure out the arguments
            let args = vec![
                format!("--gpu-architecture=compute_{}{}", cap.major, cap.minor),
                // "-G".to_string(),
                // "--generate-line-info".to_string(),
                // "--define-macro=NVRTC".to_string(),
            ];

            let args = args
                .into_iter()
                .map(CString::new)
                .collect::<Result<Vec<CString>, _>>()
                .unwrap();
            let args = args.iter().map(|s| s.as_ptr() as *const i8).collect_vec();

            // actually compile the program
            let result = nvrtcCompileProgram(program, args.len() as i32, args.as_ptr());

            let mut log_size: usize = 0;
            nvrtcGetProgramLogSize(program, &mut log_size as *mut _).unwrap();

            let mut log_bytes = vec![0u8; log_size];
            nvrtcGetProgramLog(program, log_bytes.as_mut_ptr() as *mut _).unwrap();

            let log_c = CString::from_vec_with_nul(log_bytes).unwrap();
            let log = log_c.to_str().unwrap().trim().to_owned();

            if result != nvrtcResult::NVRTC_SUCCESS {
                return CompileResult {
                    log,
                    module: Err(result),
                    lowered_names: Default::default(),
                };
            }

            // extract the lowered names
            let lowered_names: HashMap<String, String> = expected_names
                .iter()
                .map(|&expected_name| {
                    let expected_name_c = CString::new(expected_name.as_bytes()).unwrap();

                    let mut lowered_name = null();
                    nvrtcGetLoweredName(program, expected_name_c.as_ptr(), &mut lowered_name as *mut _).unwrap();
                    let lowered_name = CStr::from_ptr(lowered_name).to_str().unwrap().to_owned();

                    (expected_name.to_owned(), lowered_name)
                })
                .collect();

            // get the resulting assembly
            let mut ptx_size = 0;
            nvrtcGetPTXSize(program, &mut ptx_size as *mut _).unwrap();

            let mut ptx = vec![0u8; ptx_size];
            nvrtcGetPTX(program, ptx.as_mut_ptr() as *mut _);

            nvrtcDestroyProgram(&mut program as *mut _).unwrap();

            let module = CuModule::from_ptx(&ptx);

            CompileResult {
                log,
                module: Ok(module),
                lowered_names,
            }
        }
    }

    pub fn get_function(&self, name: &str) -> Option<CuFunction> {
        unsafe {
            let name_c = CString::new(name.as_bytes()).unwrap();
            let mut function = null_mut();

            let result = cuModuleGetFunction(&mut function as *mut _, self.inner.inner, name_c.as_ptr());

            if result == CUresult::CUDA_ERROR_NOT_FOUND {
                None
            } else {
                result.unwrap();
                Some(CuFunction {
                    _module: Arc::clone(&self.inner),
                    function,
                })
            }
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
