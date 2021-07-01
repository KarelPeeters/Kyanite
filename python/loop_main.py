from loop import LoopSettings, SelfplaySettings, run_loop
from train import TrainSettings, ValueTarget


def main():
    selfplay_settings = SelfplaySettings(
        game_count=100,
        inf_temp_move_count=20,
        keep_tree=False,
        dirichlet_alpha=1.0,
        dirichlet_eps=0.25,
        iterations=100,
        exploration_weight=2.0,
        batch_size=64,
        threads_per_device=2,
    )

    train_settings = TrainSettings(
        epochs=1,
        value_target=ValueTarget.FinalValue,
        policy_weight=2.0,
        batch_size=256,
        plot_points=100,
        plot_smooth_points=50,
    )

    settings = LoopSettings(
        root_path="data/proper_loop",
        initial_network="data/loop2/modest_cont22_cont/model_7_epochs.pt",
        generations=10,
        buffer_gen_count=2,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
        train_weight_decay=1e-5,
        test_fraction=0.05,
    )

    run_loop(settings)


if __name__ == '__main__':
    main()
