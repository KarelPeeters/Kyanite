import glob
import sys

from torch import nn
from torch.optim import AdamW

from lib.data.file import DataFile
from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model.layers import Flip
from lib.model.post_act import ScalarHead, PredictionHeads, ResTower, ConcatInputsChannelwise, \
    ResBlock, AttentionPolicyHead
from lib.networks import MuZeroNetworks
from lib.selfplay_client import SelfplaySettings, UctWeights
from lib.train import TrainSettings, ScalarTarget


def main():
    game = Game.find("chess")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        muzero=True,
        games_per_gen=200,

        cpu_threads_per_device=2,
        gpu_threads_per_device=1,
        cpu_batch_size=512,
        gpu_batch_size=512,
        gpu_batch_size_root=8,
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
        full_iterations=400,
        part_iterations=20,
        weights=UctWeights.default(),
        random_symmetries=False,
        cache_size=200,
        top_moves=100,
    )

    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=5.0,
        scalar_target=ScalarTarget.Final,
        train_in_eval_mode=False,
        mask_policy=False,
    )

    def build_network(depth: int, channels: int, saved_channels: int):
        representation = nn.Sequential(
            ResTower(depth, game.full_input_channels, channels, final_affine=False),
            nn.Hardtanh(-1.0, 1.0),
        )
        dynamics = ConcatInputsChannelwise(nn.Sequential(
            ResTower(depth, saved_channels + game.input_mv_channels, channels, final_affine=False),
            nn.Hardtanh(-1.0, 1.0),
            Flip(dim=2),
        ))
        prediction = PredictionHeads(
            common=ResBlock(channels),
            scalar_head=ScalarHead(game.board_size, channels, 8, 128),
            policy_head=AttentionPolicyHead(game, channels, channels)
            # policy_head=ConvPolicyHead(game, channels)
        )

        return MuZeroNetworks(
            state_channels=channels,
            state_channels_saved=saved_channels,
            state_quant_bits=8,
            representation=representation,
            dynamics=dynamics,
            prediction=prediction,
        )

    def dummy_network():
        return build_network(1, 32, 32)

    def initial_network():
        return build_network(16, 256, 64)

    initial_files_pattern = ""

    settings = LoopSettings(
        gui=sys.platform == "win32",
        root_path=f"data/loop_mu/{game.name}/profile/",

        dummy_network=None,
        initial_network=initial_network,
        initial_data_files=[DataFile.open(game, path) for path in glob.glob(initial_files_pattern)],

        only_generate=True,

        min_buffer_size=500_000,
        max_buffer_size=1_000_000,

        train_batch_size=128,
        samples_per_position=0.3,

        optimizer=lambda params: AdamW(params, weight_decay=1e-3),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,

        muzero_steps=5,
    )

    # settings.calc_batch_count_per_gen()
    settings.run_loop()


if __name__ == '__main__':
    main()
