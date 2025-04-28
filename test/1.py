import torch

inputs = torch.stack(
    [
        torch.randn(3, 4, 4, device="cuda"),
        torch.randn(3, 4, 4, device="cuda"),
        torch.randn(3, 4, 4, device="cuda"),
    ]
)
inputs[0] = 0.0

print(inputs)
