use std::path::PathBuf;
use std::process::Command;

pub fn convert_pt_to_onnx(path: &str, game: &str) {
    let mut output_path = PathBuf::from(path);
    output_path.set_extension("onnx");

    let status = Command::new("python")
        .env("PYTHONPATH", "../python")
        .arg("../python/main/convert_network.py")
        .args(["--game", game])
        .arg(path)
        .arg(output_path)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    if !status.success() {
        panic!("Failed to convert network file, status: {}", status);
    }
}
