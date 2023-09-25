use std::collections::HashMap;
use std::fmt::Write;
use std::fmt::{Display, Formatter};
use std::num::FpCategory;
use std::sync::Mutex;

use itertools::Itertools;
use lazy_static::lazy_static;

use kn_cuda_sys::wrapper::handle::Device;
use kn_cuda_sys::wrapper::rtc::core::{CuFunction, CuModule};

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
    pub device: Device,
    pub source: String,
    pub func_name: String,
}

pub fn compile_cached_kernel(key: KernelKey) -> CuFunction {
    // keep locked for the duration of compilation
    let mut cache = KERNEL_CACHE.lock().unwrap();

    let func = cache.entry(key).or_insert_with_key(|key| {
        let module = CuModule::from_source(key.device, &key.source, None, &[&key.func_name], &HEADERS);

        if !module.log.is_empty() {
            let source_numbered = prefix_line_numbers(&key.source);
            eprintln!("Kernel source:\n{}\nLog:\n{}\n", source_numbered, module.log);
        }

        let lowered_names = module.lowered_names;

        let module = module.module.unwrap();
        let lowered_name = lowered_names.get(&key.func_name).unwrap();
        let func = module.get_function(lowered_name).unwrap();

        func
    });

    func.clone()
}

fn prefix_line_numbers(s: &str) -> String {
    let line_count = s.lines().count();
    let max_number_size = (line_count + 1).to_string().len();

    let mut result = String::new();

    for (i, line) in s.lines().enumerate() {
        let line_number = i + 1;
        let line_number = format!("{}", line_number);

        result.extend(std::iter::repeat(' ').take(max_number_size - line_number.len()));
        writeln!(&mut result, "{}| {}", line_number, line).unwrap();
    }

    result
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

#[derive(Debug)]
pub struct DisplayCFloat(pub f32);

impl Display for DisplayCFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = if self.0.is_sign_negative() { "-" } else { "" };

        match self.0.classify() {
            FpCategory::Nan => write!(f, "({s}(0.0/0.0))"),
            FpCategory::Infinite => write!(f, "({s}(1.0/0.0))"),
            FpCategory::Zero => write!(f, "({s}0.0)"),
            FpCategory::Subnormal | FpCategory::Normal => write!(f, "{}", self.0),
        }
    }
}
