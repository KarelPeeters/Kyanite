from torch.optim import AdamW

from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.model.post_act import PostActNetwork
from lib.model.simple import DenseNetwork
from lib.selfplay_client import SelfplaySettings
from lib.train import TrainSettings, ValueTarget


def main():
    game = Game.find("sttt")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=64,
        games_per_gen=10,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=300,
        use_value=False,
        max_game_length=300,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        full_iterations=20,
        part_iterations=20,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=0.2,
        policy_weight=1.0,
        clip_norm=20.0,
        value_target=ValueTarget.Final,
        train_in_eval_mode=False,
    )

    def initial_network():
        return PostActNetwork(game, 8, 32, 8, 64)

    # TODO implement retain setting, maybe with a separate training folder even
    settings = LoopSettings(
        root_path=f"data/loop/{game.name}/",
        initial_network=initial_network,

        target_buffer_size=1_000_000,
        train_steps_per_gen=2,
        train_batch_size=256,

        optimizer=lambda params: AdamW(params, weight_decay=1e-5),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
    )

    print_expected_buffer_behaviour(settings, game.estimate_moves_per_game)

    settings.run_loop()


def print_expected_buffer_behaviour(settings: LoopSettings, average_game_length: int):
    games_in_buffer = settings.target_buffer_size / average_game_length
    gens_in_buffer = games_in_buffer / settings.fixed_settings.games_per_gen

    positions_per_gen = settings.train_steps_per_gen * settings.train_batch_size
    visits_per_position = gens_in_buffer * positions_per_gen / settings.target_buffer_size
    visits_per_game = visits_per_position * average_game_length

    print("Expected numbers:")
    print(f"  Positions in buffer: {settings.target_buffer_size}")
    print(f"  Games in buffer: {games_in_buffer}")
    print(f"  Generations in buffer: {gens_in_buffer}")
    print(f"  Positions per gen: {positions_per_gen}")
    print(f"  Visits per position: {visits_per_position}")
    print(f"  Visits per game: {visits_per_game}")


if __name__ == '__main__':
    main()
