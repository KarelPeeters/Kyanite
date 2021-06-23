use std::time::Instant;

use ndarray::Array;
use onnxruntime::environment::Environment;
use onnxruntime::LoggingLevel;
use onnxruntime::tensor::OrtOwnedTensor;

fn main() -> onnxruntime::Result<()> {
    let env = Environment::builder()
        .with_name("test_env")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = env.new_session_builder()?
        // .with_optimization_level(GraphOptimizationLevel::All)?
        // .with_number_threads(1)?
        .with_model_from_file("../data/onnx/small.onnx")?;

    let batch_size = 1000;
    let start = Instant::now();
    let mut total = 0;

    loop {
        let input = Array::from_elem((100, 5, 9, 9), 0.0f32);
        let input = vec![input];
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)?;

        total += batch_size;
        let delta = (Instant::now() - start).as_secs_f32();
        let throughput = (total as f32) / delta;
        println!("Throughput: {:.2} boards/s", throughput);
    }

    // Ok(())
}