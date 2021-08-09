import os.path

import onnx
import onnxoptimizer
import torch.jit
from torch.fx._experimental.fuser import fuse
from torch.optim import Adam

from models import TowerModel, ResBlock
from train import train_model, TrainSettings, WdlTarget, TrainState
from util import save_fused_params, load_data_multiple


def save_multiple(model, path: str):
    model.eval()

    os.makedirs(path, exist_ok=True)
    pt_path = os.path.join(path, "model.pt")
    onnx_path = os.path.join(path, "model.onnx")
    onnx_opt_path = os.path.join(path, "model_opt.onnx")

    torch.jit.script(model).save(pt_path)

    example_input = torch.zeros(100, 3, 7, 7)
    torch.onnx.export(
        model, example_input, onnx_path,
        input_names=["input"],
        output_names=["wdl", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "wdl": {0: "batch_size"}, "policy": {0: "batch_size"}},
    )

    model_onnx = onnx.load_model(onnx_path)
    model_onnx_opt = onnxoptimizer.optimize(model_onnx)
    onnx.save_model(model_onnx_opt, onnx_opt_path)

    params_path = os.path.join(path, "params.npz")
    save_fused_params(model, params_path)


def print_outputs(model):
    all_zero = torch.zeros(1, 3, 7, 7)

    # wdl, policy = model(all_zero)
    # torch.set_printoptions(threshold=100000)
    # print("policy", policy)
    # print("wdl", wdl)

    model_fused = fuse(model)
    wdl, policy = model_fused(all_zero)
    print("policy", policy)
    print("wdl", wdl.detach())


def train(model):
    train_data, test_data = load_data_multiple(
        [
            "../../data/ataxx/loop2/gen_53/games_from_prev.bin",
            "../../data/ataxx/loop2/gen_54/games_from_prev.bin",
            "../../data/ataxx/loop2/gen_55/games_from_prev.bin",
            "../../data/ataxx/loop2/gen_56/games_from_prev.bin",
        ],
        0.1
    )

    settings = TrainSettings(
        epochs=1,
        wdl_target=WdlTarget.Final,
        policy_weight=1.0,
        batch_size=128,
        plot_points=100,
        plot_smooth_points=50
    )

    state = TrainState(
        settings=settings,
        output_path="../../data/derp/basic_res_model/training/",
        train_data=train_data,
        test_data=test_data,
        optimizer=Adam(model.parameters()),
        scheduler=None,
    )

    train_model(model, state)


def main():
    model = TowerModel(
        32, 8, 16, True, True, True, lambda: ResBlock(32, 32, True, False, None)
    )

    model.print = False
    train(model)

    model.eval()

    model.print = False
    print_outputs(model)
    save_multiple(model, "../../data/derp/basic_res_model/")


if __name__ == '__main__':
    main()
