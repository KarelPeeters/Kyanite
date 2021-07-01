import os

import torch.jit

from loop import LoopSettings, SelfplaySettings, run_loop
from models import GoogleModel
from train import TrainSettings, ValueTarget


def main():
    selfplay_settings = SelfplaySettings(
        game_count=10_000,
        inf_temp_move_count=20,
        keep_tree=False,
        dirichlet_alpha=1.0,
        dirichlet_eps=0.25,
        iterations=100,
        exploration_weight=2.0,
        batch_size=256,
        threads_per_device=2,
    )

    train_settings = TrainSettings(
        epochs=2,
        value_target=ValueTarget.FinalValue,
        policy_weight=2.0,
        batch_size=256,
        plot_points=100,
        plot_smooth_points=50,
    )

    root_path = "data/loop_scratch"
    os.makedirs(root_path, exist_ok=True)
    initial_path = os.path.join(root_path, "initial_network.pt")

    model = GoogleModel(32, 8, 1, 32, 4, True, None, False)
    model = torch.jit.script(model)
    torch.jit.save(model, initial_path)

    settings = LoopSettings(
        root_path=root_path,
        initial_network=initial_path,
        generations=100,
        buffer_gen_count=10,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
        train_weight_decay=1e-5,
        test_fraction=0.05,
    )

    run_loop(settings)


if __name__ == '__main__':
    main()
