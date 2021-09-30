from lib.games import Game


class GameDataView:
    def __init__(self, game: Game, full, includes_history: bool):
        self.game = game
        self.full = full
        self.includes_history = includes_history

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

        if self.includes_history:
            input_size = game.input_size_history
            input_shape = game.input_shape_history
        else:
            input_size = game.input_size
            input_shape = game.full_input_shape

        self.input = take(input_size).reshape(-1, *input_shape)

        if not includes_history:
            assert i == game.data_width

    def __len__(self):
        return len(self.full)
