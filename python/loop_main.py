from torch.optim import AdamW

from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model import TowerModel, ResBlock
from lib.selfplay_client import SelfplaySettings
from lib.train import TrainSettings, WdlTarget, WdlLoss


def main():
    game = Game.find("chess")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=64,
        games_per_gen=16,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=10,
        max_game_length=100,
        keep_tree=False,
        dirichlet_alpha=0.2,
        dirichlet_eps=0.1,
        full_search_prob=1.0,
        full_iterations=20,
        part_iterations=20,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        game=game,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.MSE,
        policy_weight=1.0,
        batch_size=32,
        batches=10,
    )

    def initial_network():
        return TowerModel(game, 32, 1, 16, True, True, True, lambda: ResBlock(32, 32, True, False, False, None))

    # TODO implement retain setting, maybe with a separate training folder even
    settings = LoopSettings(
        root_path="data/new_loop/test",
        initial_network=initial_network,
        target_buffer_size=1024,

        optimizer=lambda params: AdamW(params, weight_decay=1e-5),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
    )

    settings.run_loop()


if __name__ == '__main__':
    main()
