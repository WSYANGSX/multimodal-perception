from torchvision import transforms

from multimodal_perception.trainer import Trainer
from multimodal_perception.models import CUnet
from multimodal_perception.algorithm import MultiModalSegmentation
from multimodal_perception.utils import data_parse

# 模型定义


def main():
    rgb_image_size = (3, 640, 512)
    thremal_image_size = (1, 640, 512)

    cunet = CUnet()
    models = {"cunet": cunet}

    multimodal_segmentation = MultiModalSegmentation(
        "./src/multimodal_perception/algorithm/multimodal_segmentation/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1),
        ]
    )
    data = data_parse("./data/minist")

    train_cfg = {
        "epochs": 100,
        "log_dir": "./logs/vq_vae/",
        "model_dir": "./checkpoints/vq_vae/",
        "log_interval": 10,
        "save_interval": 10,
        "batch_size": 256,
        "data_num_workers": 4,
    }

    trainer = Trainer(train_cfg, data, transform, multimodal_segmentation)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
