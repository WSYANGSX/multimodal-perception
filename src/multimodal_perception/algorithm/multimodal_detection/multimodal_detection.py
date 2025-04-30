from typing import Literal, Mapping

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from multimodal_perception.models import BaseNet
from multimodal_perception.algorithm.base import AlgorithmBase


class MultiModalDetection(AlgorithmBase):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str = "multimodal_detection",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """
        多模态感知算法实现

        parameters:
        - cfg (str): 配置文件路径 (YAML格式).
        - models (Mapping[str, BaseNet]): 多模态感知模型算法所需模型.{"cunet":model}.
        - name (str): 算法名称. Default to "multimodal_detection".
        - device (str): 运行设备 (auto自动选择).
        """
        super().__init__(cfg, models, name, device)

        # -------------------- 配置优化器 --------------------
        self._configure_optimizers()
        self._configure_schedulers()

    def _configure_optimizers(self) -> None:
        opt_config = self.cfg["optimizer"]

        if opt_config["type"] == "Adam":
            self._optimizers.update(
                {
                    "cunet": torch.optim.Adam(
                        params=self.models["cunet"].parameters(),
                        lr=opt_config["learning_rate"],
                        betas=(opt_config["beta1"], opt_config["beta2"]),
                        eps=opt_config["eps"],
                        weight_decay=opt_config["weight_decay"],
                    )
                }
            )
        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_schedulers(self) -> None:
        sch_config = self.cfg["scheduler"]

        if sch_config and sch_config.get("type") == "ReduceLROnPlateau":
            self._schedulers.update(
                {
                    "cunet": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizers["cunet"],
                        mode="min",
                        factor=sch_config.get("factor", 0.1),
                        patience=sch_config.get("patience", 10),
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10) -> float:
        """训练单个epoch"""
        self._models["cunet"].train()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            prediction = self._models["cunet"](*data)

            loss = criterion(prediction, labels)

            self._optimizers["cunet"].zero_grad()
            loss.backward()  # 反向传播计算各权重的梯度
            torch.nn.utils.clip_grad_norm_(self.models["cunet"].parameters(), self.cfg["training"]["grad_clip"])
            self._optimizers["cunet"].step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                writer.add_scalar(
                    "loss/train_batch", loss.item(), epoch * len(self.train_loader) + batch_idx
                )  # batch loss

        avg_loss = total_loss / len(self.train_loader)

        return {"cunet": avg_loss}

    def validate(self) -> float:
        """验证步骤"""
        self._models["cunet"].eval()

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                prediction = self._models["cunet"](*data)

                loss = criterion(prediction, labels)
                total_loss += loss

        avg_loss = total_loss / len(self.val_loader)

        return {"cunet": avg_loss, "save_metric": avg_loss}

    # @torch.no_grad()
    # def eval(self, num_samples: int = 5) -> None:
    #     """可视化重构结果"""
    #     self._models["noise_predictor"].eval()

    #     data = torch.randn(num_samples, *self.batch_data_shape[1:], device=self.device)

    #     print("[INFO] Start sampling...")

    #     epoch = 0
    #     for time_step in trange(self.time_steps, 0, -1, desc="Processing: "):
    #         time_step = torch.full((num_samples,), time_step, device=self.device)
    #         data = self.sample(data, time_step)
    #         epoch += 1

    #     plot_figures(data, cmap="gray")
