from lib.games import Game


class GameDataView:
    def __init__(self, game: Game, full):
        self.game = game
        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.game_id = take(1)
        self.position_id = take(1)

        self.wdl_final = take(3)
        self.wdl_est = take(3)

        self.policy_mask = take(game.policy_size).view(-1, *game.policy_shape)
        self.policy = take(game.policy_size).view(-1, *game.policy_shape)

        self.input = take(game.input_size).view(-1, *game.input_shape)

        assert i == game.data_width

    def __len__(self):
        return len(self.full)
