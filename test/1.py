import torch

batch_size = 10

mask = torch.rand(batch_size) <= 0.5
print(mask)
selected_samples = torch.nonzero(mask)[:, 0]
print(selected_samples)
