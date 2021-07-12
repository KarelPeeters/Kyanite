import torch
from torch import nn


class MixLayer(nn.Module):
    def __init__(self, message_size: int):
        super().__init__()

        # TODO try a simple fully connected network (maybe with repeated board input) and see if that performs good
        #   to see if we're onto something here or literally anything works

        # TODO look into implementing this with convolutions instead, maybe that's easier/faster
        #   convolutions? doesn't that imply weight sharing? maybe we want that, maybe not
        #   probably using transpose to flip the messages?
        # TODO weight sharing between corners and edges?
        # TODO batch norm? where? on the from side or on the to side?
        # TODO more complicated internal network? right now we only have a single layer per message!
        #   meh, things send messages to themself as well so they can emulate multiple layers already
        # TODO think about ways to reduce the number of parameters!
        #   but more importantly the size of intermediate activations, for compute time
        # TODO maybe don't include all macros as the input for every tile? just the current macro may be enough
        #   bah, these kinds of interactions could be very important!

        self.message_size = message_size
        self.linears = nn.ModuleList(
            [nn.Linear(9 + 2 * (9 + 9) + 9 * message_size, 9 * message_size) for _ in range(9)])

    def forward(self, input_mask, input_board, input_messages):
        # input_mask: Bx9x9
        # input_board: Bx2x10x9
        # input_messages: Bx9x9xM, from->to
        # output_messages: Bx9x9xM, from->to

        all_output_messages = []
        for om, linear in enumerate(self.linears):
            macro_input = torch.cat([
                input_mask[:, om, :],
                input_board[:, 0, om, :],
                input_board[:, 1, om, :],
                input_board[:, 0, -1, :],
                input_board[:, 1, -1, :],
                input_messages[:, :, om, :].reshape(-1, 9 * self.message_size),
            ], dim=1)

            output_messages = nn.functional.relu(linear(macro_input))
            all_output_messages.append(output_messages.view(-1, 1, 9, self.message_size))

        return torch.cat(all_output_messages, dim=1)


class MixModel(nn.Module):
    def __init__(self, depth: int, message_size: int, wdl_size: int, same_weight: bool):
        super().__init__()

        self.message_size = message_size

        if same_weight:
            layers = [MixLayer(message_size)] * depth
        else:
            layers = [MixLayer(message_size) for _ in range(depth)]
        self.common_tower = nn.ModuleList(layers)

        self.common_linear = nn.Linear(9 * 9 * message_size, 9 * 9 + wdl_size)
        self.wdl_linear = nn.Linear(wdl_size, 3)

    def forward(self, input_mask, input_board):
        prev_messages = torch.zeros(len(input_board), 9, 9, self.message_size, device=input_board.device)

        for layer in self.common_tower:
            prev_messages = layer(input_mask, input_board, prev_messages)

        common = self.common_linear(prev_messages.view(-1, 9 * 9 * self.message_size))
        policy = common[:, :81]
        wdl_hidden = common[:, 81:]
        wdl = self.wdl_linear(nn.functional.relu(wdl_hidden))

        return wdl, policy
