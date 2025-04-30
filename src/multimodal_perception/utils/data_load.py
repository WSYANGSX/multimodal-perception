import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, image_data, thermal_data, labels=None, image_transform=None, thermal_transform=None) -> None:
        super().__init__()

        if len(image_data) != len(thermal_data):
            raise ValueError(
                f"The length of image data ({len(image_data)}) is not equal to the length of thermal data ({len(thermal_data)})."
            )

        self.image_data = image_data
        self.thermal_data = thermal_data
        self.labels = labels

        self.image_transform = image_transform
        self.thermal_transform = thermal_transform

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, index):
        image_sample = self.image_data[index]
        thermal_sample = self.thermal_data[index]

        if self.labels is not None:
            labels_sample = self.labels[index]

        if self.image_transform:
            image_sample = self.image_transform(image_sample)
        if self.thermal_transform:
            thermal_sample = self.thermal_transform(thermal_sample)

        return {"image": image_sample, "thermal": thermal_sample}, labels_sample


def load_data_ids(file_path: str) -> list[str]:
    """加载数据ID列表"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Missing ID file: {file_path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load IDs from {file_path}: {e}")
        raise


def load_modality_data(data_ids: list, data_path: str, annotations_path: str) -> tuple:
    """加载多模态数据"""
    rgb_images = []
    thermal_images = []
    masks = []

    for idx, data_id in enumerate(data_ids, 1):
        # 构建文件路径
        thermal_filename = f"{data_id}.jpeg"
        rgb_filename = f"{data_id.replace('PreviewData', 'RGB')}.jpg"
        mask_filename = f"{data_id.replace('PreviewData', 'mask')}.jpg"

        thermal_path = os.path.join(data_path, thermal_filename)
        rgb_path = os.path.join(data_path, rgb_filename)
        mask_path = os.path.join(annotations_path, mask_filename)

        try:
            # 加载并校验thermal图像
            with Image.open(thermal_path) as thermal_img:
                thermal_array = np.array(thermal_img)

            # 加载并校验RGB图像
            with Image.open(rgb_path) as rgb_img:
                rgb_array = np.array(rgb_img)
                if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                    raise ValueError("Invalid RGB image dimensions")

            # 加载并校验mask
            with Image.open(mask_path) as mask_img:
                mask_array = np.array(mask_img)

            # 校验尺寸一致性
            if thermal_array.shape[:2] != mask_array.shape[:2]:
                raise ValueError("Thermal-mask dimension mismatch")
            if thermal_array.shape[:2] != rgb_array.shape[:2]:
                raise ValueError("Thermal-RGB dimension mismatch")

            # 收集有效数据
            thermal_images.append(thermal_array)
            rgb_images.append(rgb_array)
            masks.append(mask_array)

        except Exception as e:
            print(f"[WARNING] Skipping invalid data {data_id}: {str(e)}")
            continue

    return np.array(rgb_images), np.array(thermal_images), np.array(masks)


def data_parse(file_path: str) -> tuple:
    """解析多模态数据集"""
    try:
        # 构建完整路径
        base_path = os.path.abspath(file_path)
        train_ids = load_data_ids(os.path.join(base_path, "train.txt"))
        val_ids = load_data_ids(os.path.join(base_path, "validation.txt"))
        data_path = os.path.join(base_path, "JPEGImages")
        annotations_path = os.path.join(base_path, "Annotations")

        # 加载训练数据
        print("[INFO] Loading training data...")
        rgb_train, thermal_train, mask_train = load_modality_data(train_ids, data_path, annotations_path)

        # 加载验证数据
        print("\n[INFO] Loading validation data...")
        rgb_val, thermal_val, mask_val = load_modality_data(val_ids, data_path, annotations_path)

        # 打印最终数据信息
        print("\n[SUCCESS] Data loading completed:")
        print(f"Training:  RGB {rgb_train.shape}  Thermal {thermal_train.shape}  Masks {mask_train.shape}")
        print(f"Validation: RGB {rgb_val.shape}  Thermal {thermal_val.shape}  Masks {mask_val.shape}")

        return rgb_train, thermal_train, mask_train, rgb_val, thermal_val, mask_val

    except Exception as e:
        print(f"[ERROR] Data loading failed: {str(e)}")
        raise
