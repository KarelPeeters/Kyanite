import os
import shutil

import torch.jit

from loop import LoopSettings, SelfplaySettings, run_loop
from train import TrainSettings, WdlTarget


def main():
    selfplay_settings = SelfplaySettings(
        game="ataxx",
        game_count=256,
        temperature=1.0,
        zero_temp_move_count=20,
        max_game_length=300,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        full_iterations=500,
        part_iterations=500,
        exploration_weight=2.0,
        random_symmetries=True,
        batch_size=128,
        threads_per_device=2,
    )

    train_settings = TrainSettings(
        epochs=1,
        wdl_target=WdlTarget.Final,
        policy_weight=2.0,
        batch_size=128,
        plot_points=100,
        plot_smooth_points=50,
    )

    root_path = "data/ataxx/loop1"
    # TODO remove this again, pretty dangerous!
    shutil.rmtree(root_path, ignore_errors=True)
    os.makedirs(root_path, exist_ok=True)
    initial_path = os.path.join(root_path, "initial_network.pt")

    model = torch.jit.load("data/ataxx/supervised_4/model_2_epochs.pt")
    torch.jit.save(model, initial_path)

    settings = LoopSettings(
        root_path=root_path,
        initial_network=initial_path,
        generations=100,
        buffer_gen_count=5,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
        train_weight_decay=1e-5,
        test_fraction=0.05,
    )

    run_loop(settings)


if __name__ == '__main__':
    main()
