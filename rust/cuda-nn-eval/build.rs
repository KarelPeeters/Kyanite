fn main() -> std::io::Result<()> {
    prost_build::compile_protos(
        &["proto/onnx.proto3"],
        &["proto/"]
    )?;

    Ok(())
}