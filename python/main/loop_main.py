import sys

import torch
from torch.optim import SGD

from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.selfplay_client import SelfplaySettings
from lib.train import TrainSettings, ValueTarget


def main():
    game = Game.find("chess")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=512,
        games_per_gen=1000,
        reorder_games=False,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=30,
        use_value=False,
        max_game_length=300,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        full_iterations=600,
        part_iterations=20,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=600,
    )

    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        clip_norm=20.0,
        value_target=ValueTarget.Final,
        train_in_eval_mode=False,
    )

    def initial_network():
        return torch.jit.load("data/large_att_cs_continue_1_network_8320.pb")

    # TODO implement retain setting, maybe with a separate training folder even
    settings = LoopSettings(
        gui=sys.platform == "win32",
        root_path=f"data/loop/{game.name}/continue",
        initial_network=initial_network,
        only_generate=True,

        min_buffer_size=500_000,
        max_buffer_size=1_000_000,

        train_batch_size=1024,
        samples_per_position=0.1,

        optimizer=lambda params: SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-5),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
    )

    # settings.run_loop()
    settings.calc_batch_count_per_gen()


if __name__ == '__main__':
    main()
