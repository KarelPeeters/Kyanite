use crate::onnx::result::{OnnxError, OnnxResult, ToOnnxLoadResult};
use itertools::Itertools;
use rand::Rng;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub trait ExternalDataLoader {
    fn load_external_data(
        &mut self,
        location: &Path,
        offset: usize,
        length: Option<usize>,
        length_guess: usize,
    ) -> OnnxResult<Vec<u8>>;
}

#[derive(Debug)]
pub struct NoExternalData;

#[derive(Debug)]
pub struct DummyExternalData<R: Rng>(pub R);

#[derive(Debug)]
pub struct PathExternalData(pub PathBuf);

impl ExternalDataLoader for NoExternalData {
    fn load_external_data(&mut self, location: &Path, _: usize, _: Option<usize>, _: usize) -> OnnxResult<Vec<u8>> {
        panic!(
            "External data not allowed, trying to read from '{}'",
            location.display()
        );
    }
}

impl<R: Rng> ExternalDataLoader for DummyExternalData<R> {
    fn load_external_data(&mut self, _: &Path, _: usize, _: Option<usize>, length_guess: usize) -> OnnxResult<Vec<u8>> {
        Ok((0..length_guess).map(|_| self.0.gen()).collect_vec())
    }
}

impl ExternalDataLoader for PathExternalData {
    fn load_external_data(
        &mut self,
        location: &Path,
        offset: usize,
        length: Option<usize>,
        _: usize,
    ) -> OnnxResult<Vec<u8>> {
        if !path_is_normal(location) {
            return Err(OnnxError::NonNormalExternalDataPath(location.to_owned()));
        }

        let path = self.0.join(location);

        let mut file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open file {:?}", path));
        file.seek(SeekFrom::Start(offset as u64)).to_onnx_result(&path)?;

        let mut buffer = vec![];

        if let Some(length) = length {
            buffer.resize(length, 0);
            file.read_exact(&mut buffer).to_onnx_result(&path)?;
        } else {
            file.read_to_end(&mut buffer).to_onnx_result(&path)?;
        }

        Ok(buffer)
    }
}

fn path_is_normal(path: &Path) -> bool {
    path.components().all(|c| matches!(c, Component::Normal(_)))
}
