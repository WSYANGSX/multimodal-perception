from __future__ import annotations

import torch
import torch.nn as nn

from torchvision import transforms

# 加载数据
data = data_parse("./src/machine_learning/data/minist")

# 数据处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),
    ]
)

# 模型搭建与初始化


# 模型训练


# 模型保存


# 模型验证
