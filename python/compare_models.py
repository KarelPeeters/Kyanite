import os
import time

import numpy as np
from matplotlib import pyplot
from torch.optim import Adam

from models import TowerModel, ResBlock, MobileV2Block
from train import TrainSettings, WdlTarget, train_model, TrainState, TRAIN_PLOT_TITLES
from util import load_data_multiple, print_param_count, uniform_window_filter


def gen_data(root: str):
    train_data, test_data = load_data_multiple(
        [
            "../data/ataxx/loop2/gen_53/games_from_prev.bin",
            "../data/ataxx/loop2/gen_54/games_from_prev.bin",
            "../data/ataxx/loop2/gen_55/games_from_prev.bin",
            "../data/ataxx/loop2/gen_56/games_from_prev.bin",
        ],
        0.1
    )

    settings = TrainSettings(
        epochs=4,
        wdl_target=WdlTarget.Final,
        policy_weight=2.0,
        batch_size=128,
        plot_points=100,
        plot_smooth_points=50
    )

    models = {
        "res": TowerModel(16, 8, 32, False, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act": TowerModel(16, 8, 32, True, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_vconv": TowerModel(16, 8, 32, False, True, True, lambda: ResBlock(16, 16, True, False, None)),

        # "res_deep": TowerModel(16, 16, 32, False, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_wide": TowerModel(32, 8, 32, False, False, True, lambda: ResBlock(32, 32, True, False, None)),
        # "res_wide_bottle": TowerModel(32, 8, 32, False, False, True, lambda: ResBlock(32, 16, True, False, None)),

        # "base_2p": TowerModel(16, 8, 32, lambda: ResBlock(16, 16, False, False, None)),

        # "res": TowerModel(16, 8, 32, False, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act": TowerModel(16, 8, 32, True, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_vconv": TowerModel(16, 8, 32, False, True, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act_vconv": TowerModel(16, 8, 32, True, True, True, lambda: ResBlock(16, 16, True, False, None)),

        # "res_16": TowerModel(16, 16, 32, False, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act_16": TowerModel(16, 16, 32, True, False, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_vconv_16": TowerModel(16, 16, 32, False, True, True, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act_vconv_16": TowerModel(16, 16, 32, True, True, True, lambda: ResBlock(16, 16, True, False, None)),

        # "res_vconv_16_bot": TowerModel(16, 16, 32, False, True, True, lambda: ResBlock(16, 8, True, False, None)),
        # "res_vconv_16_sep": TowerModel(16, 16, 32, False, True, True, lambda: ResBlock(16, 8, True, True, None)),

        # "res_pfc": TowerModel(16, 8, 32, False, False, False, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act_pfc": TowerModel(16, 8, 32, True, False, False, lambda: ResBlock(16, 16, True, False, None)),
        # "res_vconv_pfc": TowerModel(16, 8, 32, False, True, False, lambda: ResBlock(16, 16, True, False, None)),
        # "res_act_vconv_pfc": TowerModel(16, 8, 32, True, True, False, lambda: ResBlock(16, 16, True, False, None)),

        # "mob": TowerModel(16, 8, 32, True, True, True, lambda: MobileV2Block(16, 2, True))

        # "res_deeper_bigger": TowerModel(32, 32, 32, lambda: ResBlock(32, 8, True, False, None)),


        # "res_bottle": TowerModel(16, depth, 32, lambda: ResBlock(16, 8, True, False, None)),

        # "sep": TowerModel(16, depth, 32, lambda: ResBlock(16, True, True, None)),

        # "sq": TowerModel(16, depth, 32, lambda: ResBlock(16, True, False, 8)),
        # "v2": TowerModel(16, depth, 32, lambda: MobileV2Block(16, 4, True))

        # "v2_2": TowerModel(16, depth, 32, lambda: MobileV2Block(16, 2, True)),
        # "v2_4": TowerModel(16, depth, 32, lambda: MobileV2Block(16, 2, True)),
        # "v2_32": TowerModel(16, depth, 32, lambda: MobileV2Block(16, 32, True)),

        # "v2_8_32": TowerModel(8, depth, 32, lambda: MobileV2Block(8, 32, True)),
    }

    for name, model in models.items():
        os.makedirs(f"{root}/{name}", exist_ok=True)

        with open(f"{root}/{name}/info.txt", "w") as f:
            print(f"Model {name}:", file=f)
            print_param_count(model, f)
            f.flush()

            state = TrainState(
                settings=settings,
                output_path=f"{root}/{name}/",
                train_data=train_data,
                test_data=test_data,
                optimizer=Adam(model.parameters()),
                scheduler=None,
            )

            start = time.perf_counter()
            train_model(model, state)
            end = time.perf_counter()
            print(f"training took {end - start:.2}s", file=f)


def plot_data(root: str):
    for i in range(3):
        title = TRAIN_PLOT_TITLES[i]

        for model in os.listdir(root):
            if not os.path.isdir(f"{root}/{model}"):
                continue

            print(model)

            axis = np.load(f"{root}/{model}/plot_axis.npy")
            data = np.load(f"{root}/{model}/plot_data.npy")

            data = uniform_window_filter(data[:, 3 + i], 3)

            pyplot.plot(axis, data, label=model)

        pyplot.title(title)
        pyplot.legend()
        pyplot.savefig(f"{root}/all_{title}.png")
        pyplot.show()


def main():
    root = "../data/ataxx/compare_models_less"
    gen_data(root)
    plot_data(root)


if __name__ == '__main__':
    main()
