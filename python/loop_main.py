from games import find_game
from loop import LoopSettings, SelfplaySettings, run_loop
from models import TowerModel, ResBlock
from selfplay_client import FixedSelfplaySettings
from train import TrainSettings, WdlTarget, WdlLoss


def main():
    game = find_game("chess")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=512,
        games_per_gen=100,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=10,
        max_game_length=500,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.1,
        full_search_prob=1.0,
        full_iterations=1000,
        part_iterations=1000,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        epochs=1,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.CrossEntropy,
        policy_weight=1.0,
        batch_size=128,
        plot=False,
        plot_points=100,
        plot_smooth_points=50,
    )

    def initial_network():
        return TowerModel(game, 32, 8, 16, True, True, True, lambda: ResBlock(32, 32, True, False, False, None))

    settings = LoopSettings(
        root_path="data/var_game/test_loop",
        initial_network=initial_network,
        buffer_gen_count=20,
        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
        train_weight_decay=1e-5,
        test_fraction=0.05,
    )

    run_loop(settings)


if __name__ == '__main__':
    main()
