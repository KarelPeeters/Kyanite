use std::collections::HashMap;
use std::sync::Mutex;

use itertools::Itertools;
use lazy_static::lazy_static;

use kn_cuda_sys::wrapper::handle::CudaDevice;
use kn_cuda_sys::wrapper::rtc::core::{CuFunction, CuModule};

// TODO cache kernel compilation on disk
//   * make sure to invalidate old files?
//   * user-configurable cache dir, either env var or actual code?
//   * disabled by default to ensure it always works, even on read-only fs
lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<KernelKey, CuFunction>> = Mutex::new(HashMap::new());
    static ref HEADERS: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        map.insert("util.cu", include_str!("util.cu"));
        map
    };
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct KernelKey {
    pub device: CudaDevice,
    pub source: String,
    pub func_name: String,
}

pub fn compile_cached_kernel(key: KernelKey) -> CuFunction {
    // keep locked for the duration of compilation
    let mut cache = KERNEL_CACHE.lock().unwrap();

    let func = cache.entry(key).or_insert_with_key(|key| {
        let module = CuModule::from_source(key.device, &key.source, None, &[&key.func_name], &HEADERS);

        if !module.log.is_empty() {
            let source_numbered = module.source_with_line_numbers();
            eprintln!("Kernel source:\n{}\nLog:\n{}\n", source_numbered, module.log);
        }

        module.get_function_by_name(&key.func_name).unwrap().unwrap()
    });

    func.clone()
}

pub fn fill_replacements(src: &str, replacements: &[(&str, String)]) -> String {
    let result = replacements.iter().fold(src.to_owned(), |src, (key, value)| {
        assert!(
            key.starts_with('$') && key.ends_with('$'),
            "Key '{}' should start and end with '$'",
            key
        );
        assert!(src.contains(key), "Source does not contain key '{}'", key);
        src.replace(key, value)
    });

    if result.contains('$') {
        eprintln!("Source after replacements:\n{}", result);
        panic!("Source still contains $");
    }

    result
}

pub fn c_nested_array_string(values: &[Vec<isize>]) -> String {
    assert!(values.len() > 0, "C array cannot be empty");
    format!("{{{}}}", values.iter().map(|a| c_array_string(a)).join(", "))
}

pub fn c_array_string(values: &[isize]) -> String {
    assert!(values.len() > 0, "C array cannot be empty");
    format!("{{{}}}", values.iter().map(|v| v.to_string()).join(", "))
}

pub fn ceil_div(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}
