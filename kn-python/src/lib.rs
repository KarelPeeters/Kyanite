use kn_cuda_eval::{runtime::{PreparedToken, Runtime}, Device};
use kn_graph::{graph::Graph, onnx::load_graph_from_onnx_bytes};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pymodule]
fn kyanite(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyPreparedToken>()?;

    Ok(())
}

#[pyclass(name = "Runtime")]
struct PyRuntime {
    inner: Runtime,
}

#[pyclass(name = "Graph")]
struct PyGraph {
    inner: Graph,
}
#[pyclass(name = "Prepared")]
struct PyPreparedToken {
    inner: PreparedToken,
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
}
    
#[pymethods]
impl PyRuntime {
    #[new]
    pub fn new(device: &str) -> PyResult<Self> {
        let core = parse_device(device).map_err(|_| PyRuntimeError::new_err(format!("Invalid device: '{}'", device)))?;
        let inner = kn_cuda_eval::runtime::Runtime::new(core);
        Ok(PyRuntime { inner })
    }

    pub fn prepare(&mut self, graph: &PyGraph, batch_size: usize) -> PyPreparedToken {
        let inner = self.inner.prepare(graph.inner.clone(), batch_size);
        PyPreparedToken { inner }
    }

    pub fn eval(&mut self, token: &PyPreparedToken) {
        let info = self.inner.info(token.inner);
        let inputs = info.graph.dummy_zero_inputs(info.batch_size);
        let _ = self.inner.eval(token.inner, &inputs);
    }
}

fn parse_device(device: &str) -> Result<Option<Device>, ()> {
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
