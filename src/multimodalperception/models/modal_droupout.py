from __future__ import annotations

import torch
import torch.nn as nn


class ModalDropout(nn.Module):
    def __init__(self, probability=0.1, device="cuda") -> None:
        super().__init__()
        self.probability = probability
        self.device = device

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        # 模拟模态信息随机缺失
        inputs = torch.stack(inputs)

        modal_length = len(inputs)
        if torch.rand(1) <= self.probability:
            print(1)
            dropout_indices = torch.argmin(torch.rand(modal_length))
            inputs[dropout_indices] = 0.0

            return list(inputs)

        return list(inputs)
