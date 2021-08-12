import os
import shutil

from loop import LoopSettings, SelfplaySettings, run_loop
from models import TowerModel, ResBlock
from selfplay_client import StartupSettings
from train import TrainSettings, WdlTarget


def main():
    startup_settings = StartupSettings(
        game="ataxx",
        output_folder="",
        threads_per_device=1,
        batch_size=256,
        games_per_file=100,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=20,
        max_game_length=400,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        full_iterations=500,
        part_iterations=500,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        epochs=1,
        wdl_target=WdlTarget.Final,
        policy_weight=2.0,
        batch_size=128,
        plot_points=100,
        plot_smooth_points=50,
    )

    root_path = "data/derp/test_loop"
    assert not os.path.exists(root_path), f"{root_path} already exists"
    os.makedirs(root_path, exist_ok=True)

    def initial_network():
        return TowerModel(32, 8, 16, True, True, True, lambda: ResBlock(32, 32, True, False, None))

    settings = LoopSettings(
        root_path=root_path,
        initial_network=initial_network,
        buffer_gen_count=1,
        startup_settings=startup_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
        train_weight_decay=1e-5,
        test_fraction=0.05,
    )

    run_loop(settings)


if __name__ == '__main__':
    main()
