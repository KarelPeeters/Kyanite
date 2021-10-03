from torch.optim import AdamW, SGD

from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model_blocks import TowerModel, ResBlock
from lib.selfplay_client import SelfplaySettings
from lib.train import TrainSettings, ValueTarget, ValueLoss


# TODO fix startup behaviour, only release games in the order they were started to ensure a constant
#  distribution of game lengths

def main():
    game = Game.find("chess")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=256,
        games_per_gen=64,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        # TODO alphazero uses value 30 (plies, 15 moves)
        zero_temp_move_count=30,
        # TODO give this information to zero tree search too! now it might be stalling without a good reason
        max_game_length=300,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        # TODO increase this again?
        full_iterations=400,
        part_iterations=400,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        game=game,
        wdl_target=ValueTarget.Final,
        wdl_loss=ValueLoss.MSE,
        policy_weight=4,
        batch_size=256,
        batches=4,
    )

    def initial_network():
        return TowerModel(game, 32, 8, 16, True, True, True, lambda: ResBlock(32, 32, True, False, False, None))

    # TODO implement retain setting, maybe with a separate training folder even
    settings = LoopSettings(
        root_path="data/new_loop/full_pov",
        initial_network=initial_network,
        target_buffer_size=100_000,
        test_fraction=1 / fixed_settings.games_per_gen,

        optimizer=lambda params: AdamW(params, weight_decay=1e-5),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
    )

    print_expected_buffer_behaviour(settings, 100)
    settings.run_loop()


def print_expected_buffer_behaviour(settings: LoopSettings, average_game_length: int):
    games_in_buffer = settings.target_buffer_size / average_game_length
    gens_in_buffer = games_in_buffer / settings.fixed_settings.games_per_gen

    positions_per_gen = settings.train_settings.batches * settings.train_settings.batch_size
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