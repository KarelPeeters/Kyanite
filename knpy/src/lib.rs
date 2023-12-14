use kn_cuda_eval::runtime::{Runtime, PreparedToken};
use kn_graph::{onnx::load_graph_from_onnx_bytes, graph::Graph};
use pyo3::{prelude::*, exceptions::PyRuntimeError};

#[pyclass(name="Runtime")]
struct PyRuntime {
    inner: Runtime,
}

#[pyclass(name="Graph")]
struct PyGraph {
    inner: Graph,
}
#[pyclass(name="Prepared")]
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
    pub fn new(device: String) -> PyResult<Self> {
        // TODO use device
        let core = None;
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

/// A Python module implemented in Rust.
#[pymodule]
fn knpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyPreparedToken>()?;

    Ok(())
}
