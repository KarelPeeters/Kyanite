use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Mutex;

use lazy_static::lazy_static;

use cuda_sys::wrapper::handle::{ComputeCapability, Device};
use cuda_sys::wrapper::rtc::core::{CuFunction, CuModule};

pub mod scalar;

lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<KernelKey, CuFunction>> = Mutex::new(HashMap::new());
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct KernelKey {
    capability: ComputeCapability,
    source: String,
    func_name: String,
}

fn compile_cached_kernel(key: KernelKey) -> CuFunction {
    // keep locked for the duration of compilation
    let mut cache = KERNEL_CACHE.lock().unwrap();

    let func = cache.entry(key).or_insert_with_key(|key| {
        let module = CuModule::from_source(key.capability, &key.source, None, &[&key.func_name]);

        if !module.log.is_empty() {
            eprintln!("Log while compiling kernel:\n{}\n\n{}", module.log, key.source);
        }

        let lowered_names = module.lowered_names;

        let module = module.module.unwrap();
        let lowered_name = lowered_names.get(&key.func_name).unwrap();
        let func = module.get_function(lowered_name).unwrap();

        func
    });

    func.clone()
}
