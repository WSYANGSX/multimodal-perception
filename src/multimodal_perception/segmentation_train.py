import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from multimodal_perception.trainer import Trainer
from multimodal_perception.models import SegCUnet
from multimodal_perception.algorithm import MultiModalSegmentation
from multimodal_perception.utils import data_parse

# 模型定义


def main():
    rgb_size = torch.Size((3, 640, 512))
    thremal_size = torch.Size((1, 640, 512))

    cunet = SegCUnet(rgb_size, thremal_size)
    models = {"cunet": cunet}

    multimodal_segmentation = MultiModalSegmentation(
        "./src/multimodal_perception/algorithm/multimodal_segmentation/config/config.yaml",
        models,
    )

    rgb_transform = Compose([ToTensor(), Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])
    thermal_transform = Compose([ToTensor(), Normalize(mean=0, std=1)])
    transforms = {"rgb": rgb_transform, "thermal": thermal_transform}

    data = data_parse("./data/flir_aligned")

    train_cfg = {
        "epochs": 100,
        "log_dir": "./logs/multimodal_segmentation/",
        "model_dir": "./checkpoints/multimodal_segmentation/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transforms, multimodal_segmentation)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
