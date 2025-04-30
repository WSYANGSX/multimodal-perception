import torch
import torch.nn as nn
from torchvision import transforms

from machine_learning.trainer import Trainer
from machine_learning.algorithms import VQ_VAE


# 模型定义


def main():
    image_size = (1, 28, 28)
    latent_size = (64, 7, 7)

    encoder = Encoder(input_size=image_size, output_size=latent_size)
    decoder = Decoder(input_size=latent_size, output_size=image_size)
    models = {"encoder": encoder, "decoder": decoder}

    vq_vae = VQ_VAE(
        "./src/machine_learning/algorithms/vq_vae/config/config.yaml",
        models,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.1307, std=0.3081),
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

    trainer = Trainer(train_cfg, data, transform, vq_vae)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
