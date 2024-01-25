use std::ops::Deref;

use itertools::Itertools;
use numpy::{PyArrayDyn, PyUntypedArray};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3::exceptions::PyTypeError;

use kn_cuda_eval::{Device, runtime::{PreparedToken, Runtime}};
use kn_graph::{graph::Graph, onnx::load_graph_from_onnx_bytes};
use kn_graph::dtype::{DBool, DTensor};
use kn_graph::optimizer::optimize_graph;
use kn_graph::shape::{ConcreteShape, infer_batch_size};

#[pymodule]
fn kyanite(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyPrepared>()?;

    Ok(())
}

#[pyclass(name = "Graph")]
struct PyGraph {
    inner: Graph,
}

// TODO rename to "compiled"?
#[pyclass(name = "Prepared")]
struct PyPrepared {
    runtime: Runtime,
    token: PreparedToken,
}

#[pymethods]
impl PyGraph {
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        match load_graph_from_onnx_bytes(bytes) {
            Ok(inner) => Ok(PyGraph { inner }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    fn optimize(&self) -> PyGraph {
        PyGraph {
            inner: optimize_graph(&self.inner, Default::default()),
        }
    }

    fn infer_batch_size(&self, inputs: Vec<&PyUntypedArray>) -> PyResult<Option<usize>> {
        let expected = self.inner.input_shapes();
        let actual = inputs.iter().map(|a| ConcreteShape::new(a.shape().to_owned())).collect_vec();

        infer_batch_size(&expected, &actual)
            .map_err(|e| PyRuntimeError::new_err(format!("Could not infer batch size: {:?}", e)))
    }

    fn prepare(&self, device: &str, batch_size: usize) -> PyResult<PyPrepared> {
        let core = parse_device(device).map_err(|_| PyRuntimeError::new_err(format!("Invalid device: '{}'", device)))?;
        let mut runtime = Runtime::new(core);
        let token = runtime.prepare(self.inner.clone(), batch_size);
        Ok(PyPrepared { runtime, token })
    }
}

#[pymethods]
impl PyPrepared {
    fn eval<'py>(&mut self, inputs: Vec<&PyUntypedArray>, py: Python<'py>) -> PyResult<Vec<&'py PyUntypedArray>> {
        let inputs_dtensor: Vec<_> = inputs.into_iter().map(|t| array_to_tensor(py, t)).try_collect()?;
        let outputs_dtensor = self.runtime.eval(self.token, &inputs_dtensor);
        let outputs = outputs_dtensor.into_iter().map(|t| tensor_to_array(py, t)).collect_vec();
        Ok(outputs)
    }
}

fn array_to_tensor(py: Python, array: &PyUntypedArray) -> Result<DTensor, PyErr> {
    macro_rules! branch {
        ($ty:ident, $wrap: ident) => {
            if array.dtype().is_equiv_to(numpy::dtype::<$ty>(py)) {
                let array: &numpy::PyArrayDyn<$ty> = array.downcast().unwrap();
                let view = unsafe { array.as_array() };
                return Ok(DTensor::$wrap(view.into_owned().into_shared()));
            }
        };
    }

    branch!(f32, F32);
    branch!(f64, F64);
    branch!(i8, I8);
    branch!(i16, I16);
    branch!(i32, I32);
    branch!(i64, I64);
    branch!(u8, U8);
    branch!(u16, U16);
    branch!(u32, U32);
    branch!(u64, U64);

    // bool needs slightly different code
    if array.dtype().is_equiv_to(numpy::dtype::<bool>(py)) {
        let array: &PyArrayDyn<bool> = array.downcast().unwrap();
        let view = unsafe { array.as_array() };
        let owned = view.mapv(|x| DBool(x));
        return Ok(DTensor::Bool(owned.into_shared()));
    }

    Err(PyTypeError::new_err(format!("Unsupported dtype {:?} for input tensor", array.dtype())))
}

fn tensor_to_array(py: Python, tensor: DTensor) -> &PyUntypedArray {
    match tensor {
        DTensor::F32(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::F64(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::I8(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::I16(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::I32(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::I64(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::U8(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::U16(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::U32(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::U64(tensor) => PyArrayDyn::from_array(py, &tensor).deref(),
        DTensor::Bool(tensor) => {
            let tensor = tensor.mapv(|b| b.0);
            PyArrayDyn::from_array(py, &tensor).deref()
        }
    }
}

fn parse_device(device: &str) -> Result<Option<Device>, ()> {
    let device = device.to_lowercase();

    if device == "cpu" {
        return Ok(None);
    }
    if device == "cuda" {
        return Ok(Some(Device::new(0)));
    }

    if let Some(("cuda", index)) = device.split_once(':') {
        if let Ok(index) = index.parse() {
            return Ok(Some(Device::new(index)));
        }
    }

    Err(())
}
