import glob
import sys

from torch.optim import AdamW

from lib.data.file import DataFile
from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model.post_act import PredictionHeads, ResTower, ScalarHead, ConvPolicyHead
from lib.selfplay_client import SelfplaySettings, UctWeights
from lib.train import TrainSettings, ScalarTarget


def main():
    game = Game.find("go-9")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        muzero=False,
        start_pos="default",

        simulations_per_gen=200,

        cpu_threads_per_device=4,
        gpu_threads_per_device=1,
        gpu_batch_size=1024,
        gpu_batch_size_root=0,
        search_batch_size=16,

        saved_state_channels=0,
        eval_random_symmetries=True,
    )

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=30,
        q_mode="wdl+0.0",
        max_game_length=400,
        dirichlet_alpha=0.03,
        dirichlet_eps=0.25,
        search_policy_temperature_root=1.4,
        search_policy_temperature_child=1.0,
        search_fpu_root="fixed+1",
        search_fpu_child="relative+0",
        search_virtual_loss_weight=1.0,
        full_search_prob=1.0,
        full_iterations=200,
        part_iterations=20,
        weights=UctWeights.default(),
        cache_size=200,
        top_moves=100,
    )

    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        sim_weight=0.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=5.0,
        scalar_target=ScalarTarget.Final,
        train_in_eval_mode=False,
        mask_policy=True,
    )

    def build_network(depth: int, channels: int):
        return PredictionHeads(
            common=ResTower(depth, game.full_input_channels, channels),
            scalar_head=ScalarHead(game.board_size, channels, 4, 32),
            policy_head=ConvPolicyHead(game, channels, extra_moves=1),
        )

    def initial_network():
        return build_network(16, 1024)

    initial_files_pattern = ""

    # TODO implement retain setting, maybe with a separate training folder even
    settings = LoopSettings(
        gui=sys.platform == "win32",
        root_path=f"data/loop/{game.name}/first/",
        port=63105,
        wait_for_new_network=True,

        dummy_network=None,
        initial_network=initial_network,
        initial_data_files=[DataFile.open(game, path) for path in glob.glob(initial_files_pattern)],

        only_generate=False,

        min_buffer_size=500_000,
        max_buffer_size=1_000_000,

        train_batch_size=128,
        samples_per_position=0.3,
        test_fraction=0.05,

        optimizer=lambda params: AdamW(params, weight_decay=1e-3),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,

        sample_muzero_steps=None,
        # TODO should alphazero training include the final position?
        sample_include_final=False,
        sample_random_symmetries=True,
    )

    settings.calc_batch_count_per_gen(game.estimate_moves_per_game, do_print=True)
    settings.run_loop()


if __name__ == '__main__':
    main()
