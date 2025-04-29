from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
import torch.functional as F


# 模态丢弃模块
class ModalDropoutBlock(nn.Module):
    def __init__(self, probability=0.1) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, input_modals: list[torch.Tensor]) -> list[torch.Tensor]:
        if not input_modals:
            return input_modals

        num_modals = len(input_modals)
        batch_size = input_modals[0].size(0)
        device = input_modals[0].device

        # 1. 确定哪些样本需要丢弃模态
        mask = torch.rand(batch_size, device=device) <= self.probability
        selected_samples = torch.nonzero(mask)[:, 0]

        if len(selected_samples) > 0:
            # 2. 为每个被选中的样本随机选择一个模态
            modal_indices = torch.randint(0, num_modals, (len(selected_samples),), device=device)

            # 3. 对选中的样本-模态组合进行置零
            for sample_idx, modal_idx in zip(selected_samples, modal_indices):
                input_modals[modal_idx][sample_idx] = 0.0

        return input_modals


# 自注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        attn = torch.bmm(q, k) * (C**-0.5)
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        return x + self.proj_out(h)  # 在数据传播过程中保留原始信息并增强全局依赖


# 交叉注意力模块
class CrossAttentionBlock(nn.Module):
    """交叉注意力模块"""

    def __init__(self, embed_dim: Any, num_heads: Any):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # 输入形状: (B, C, H, W)
        B, C, H, W = query.shape

        # 转换为MultiheadAttention需要的形状 (S, B, C)
        query = query.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        key = key.view(B, C, -1).permute(2, 0, 1)
        value = value.view(B, C, -1).permute(2, 0, 1)

        # 计算注意力
        attn_output, _ = self.attention(query, key, value)

        # 恢复原始形状
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        return attn_output
