from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn

from multimodal_perception.models.base import BaseNet
from multimodal_perception.models.blocks import AttentionBlock, CrossAttentionBlock, ModalDropoutBlock, ResidualBlock


# UNet模块
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.output_size = None

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.SiLU()

        # 下采样部分
        self.down_blocks = nn.ModuleList(
            [
                # 第1层
                ResidualBlock(self.in_channels, 64),  # 不改变图像大小,改变通道数
                AttentionBlock(64),  # 不改变图像大小
                nn.Conv2d(64, 64, 3, 2, 1),  # 下采样,不改变通道数,改变图像大小
                # 第2层
                ResidualBlock(64, 128),  # 不改变图像大小,改变通道数
                AttentionBlock(128),  # 不改变图像大小
                nn.Conv2d(128, 128, 3, 2, 1),  # 下采样，改变图像大小
                # 第3层
                ResidualBlock(128, 256),  # 不改变图像大小,改变通道数
                AttentionBlock(256),  # 不改变图像大小
                nn.Conv2d(256, 256, 3, 2, 1),  # 下采样，改变图像大小
                # 第4层
                ResidualBlock(256, 512),  # 不改变图像大小,改变通道数
                AttentionBlock(512),  # 不改变图像大小
                nn.Conv2d(512, 512, 3, 2, 1),  # 下采样，改变图像大小
            ]
        )

        # 中间部分，不改变大小和通道数
        self.mid_block1 = ResidualBlock(512, 1024)
        self.mid_attn = AttentionBlock(1024)
        self.mid_block2 = ResidualBlock(1024, 1024)

        # 上采样部分
        self.up_blocks = nn.ModuleList(
            [
                # 第1层
                nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),  # 上采样，改变图像大小和通道数
                ResidualBlock(1024, 512),
                AttentionBlock(512),
                # 第2层
                nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),  # 上采样，改变图像大小和通道数
                ResidualBlock(512, 256),
                AttentionBlock(256),
                # 第3层
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # 上采样，改变图像大小和通道数
                ResidualBlock(256, 128),
                AttentionBlock(128),
                # 第4层
                nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),  # 上采样，改变图像大小和通道数
                ResidualBlock(128, 64),
                AttentionBlock(64),
            ]
        )

        # 输出层
        self.out_norm = nn.GroupNorm(32, 64)
        self.out_conv = nn.Conv2d(64, self.out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        # 存储跳跃连接
        skips = []

        # 下采样过程
        for block, attn, downsample in self.down_blocks:
            x = block(x)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        skips_copied = deepcopy(skips)

        # 中间处理
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # 上采样过程
        for upsample, block, attn in self.up_blocks:
            x = upsample(x)
            x = torch.cat([x, skips_copied.pop()], dim=1)
            x = block(x)
            x = attn(x)

        # 输出
        x = self.out_norm(x)
        x = self.act(x)

        return self.out_conv(x), skips


# 分割网络
class SegCUnet(BaseNet):
    def __init__(self, rgb_size: torch.Size, thermal_size: torch.Size):
        super().__init__()
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size

        self.modal_droupout = ModalDropoutBlock(0.1)

        self.rgb_unet = UNet(3, 1)
        self.thermal_unet = UNet(1, 1)

        self.cross_attentation = CrossAttentionBlock()

        self.post_process = nn.Conv2d(2, 1, 3, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.modal_droupout(inputs)

        # 模态信息拆分
        rgb_raw, theraml_raw = inputs["rgb"], inputs["theraml"]

        rgb_feature, rgb_skips = self.rgb_unet(rgb_raw)
        thermal_feature, thermal_skips = self.thermal_unet(theraml_raw)

        # 特征交叉融合（拼接/相加）
        rgb_attn = self.cross_attn_rgb(rgb_feature, thermal_feature, thermal_feature)
        thermal_attn = self.cross_attn_thermal(thermal_feature, rgb_feature, rgb_feature)

        fused_attn = torch.cat([rgb_attn, thermal_attn], dim=1)  # (B,512,28,28)

        output = self.post_process(fused_attn)

        return output

    def view_structure(self):
        pass


# # 目标检测网络
# class DetCUnet(BaseNet):
#     def __init__(self):
#         super().__init__()
#         self.modal_droupout = ModalDropoutBlock(0.1)

#         self.rgb_unet = UNet(3, 1)
#         self.thermal_unet = UNet(1, 1)

#         self.cross_attentation = CrossAttentionBlock()

#     def forward(self, inputs: list[torch.Tensor]):
#         inputs = self.modal_droupout(inputs)

#         # 模态信息拆分
#         rgb_raw, theraml_raw = inputs[0], inputs[1]

#         rgb_feature, rgb_skips = self.rgb_unet(rgb_raw)
#         thermal_feature, thermal_skips = self.thermal_unet(theraml_raw)

#         # 特征交叉融合（拼接/相加）
#         rgb_attn = self.cross_attn_rgb(rgb_feature, thermal_feature, thermal_feature)
#         thermal_attn = self.cross_attn_thermal(thermal_feature, rgb_feature, rgb_feature)

#         fused_attn = torch.cat([rgb_attn, thermal_attn], dim=1)  # (B,512,28,28)

#     def view_structure(self):
#         pass
