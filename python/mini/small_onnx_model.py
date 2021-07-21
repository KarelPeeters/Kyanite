import timeit

import onnx
import onnxruntime
import torch
from torch.onnx import TrainingMode

from models import GoogleModel


def main():
    model = GoogleModel(64, 5, 2, 32, 8, True, None, False)

    batch_size = 1000
    input = torch.randn(batch_size, 5, 9, 9)
    outputs = model(input)
    model.eval()

    # model(input)[0][0, 0].item()
    # print("torch CPU:", timeit.timeit(lambda: model(input)[0][0, 0].item(), number=20))

    # model.cuda()
    # input = input.cuda()
    # model(input)[0][0, 0].item()
    # print("torch CUDA:", timeit.timeit(lambda: model(input)[0][0, 0].item(), number=20))

    print("Exporting to onnx")
    torch.onnx.export(
        model,
        input,
        "../../data/onnx/small.onnx",
        example_outputs=outputs,
        opset_version=12,
        input_names=["input"],
        output_names=["value", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "value": {0: "batch_size"}, "policy": {0: "batch_size"}},
        training=TrainingMode.EVAL,
    )

    # print("Quantizing")
    # quantize_dynamic("../../data/onnx/small.onnx", "../../data/onnx/small_quant.onnx", weight_type=QuantType.QUInt8)

    print("Loading onnx")
    model = onnx.load("../../data/onnx/small.onnx")
    onnx.checker.check_model(model)
    model = onnx.load("../../data/onnx/small_quant.onnx")
    onnx.checker.check_model(model)

    print("Building profile")
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = True
    # sess_options.log_severity_level = 0
    session = onnxruntime.InferenceSession("../../data/onnx/small.onnx", sess_options=sess_options)

    print("Running model")
    onnx_input = input.cpu().numpy()
    _ = session.run(None, {"input": onnx_input})
    _ = session.run(None, {"input": onnx_input})
    rounds = 100
    delta = timeit.timeit(lambda: session.run(None, {"input": onnx_input}), number=rounds)
    throughput = batch_size * rounds / delta
    print(f"onnx cuda (?): {throughput} boards/s")


if __name__ == '__main__':
    main()
