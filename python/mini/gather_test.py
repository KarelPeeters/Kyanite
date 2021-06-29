import torch

batch_size = 2
board_size = 9

input = torch.randn(batch_size, 5, board_size)
policy = torch.randn(batch_size, 81)
indices = torch.randint(board_size, size=(batch_size, board_size))
indices[0, 0:2] = 1

print(input.shape)
print(indices.shape)
print(policy.shape)

view = indices.view(-1, 1, board_size).expand(-1, 5, -1)
input_sym = torch.gather(input, 2, view)
policy_sym = torch.gather(policy, 1, indices)

print(view.shape)
print(input_sym.shape)
print(policy_sym.shape)

print(input)
print(policy)
print(indices)
print(input_sym)
print(policy_sym)
