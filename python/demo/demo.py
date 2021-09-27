import torch


def f(device: str, affine: bool):
    print(f"Trying combination {device} {affine}")
    network = torch.nn.BatchNorm2d(10, affine=affine)
    network.to(device)

    input = torch.zeros(1, 10, 4, 4, device=device)
    torch.onnx.export(network, input, "test.onnx")


f("cpu", True)
f("cpu", False)
f("cuda", True)
f("cuda", False)
