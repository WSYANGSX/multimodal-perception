from torchvision import transforms

target_size = 224  # 目标正方形边长

transform = transforms.Compose(
    [
        # 缩放到短边 = target_size，长边按比例缩放
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        # 计算填充量，将长边补足到 target_size
        transforms.Pad(
            padding=(0, 0, target_size - img.width, target_size - img.height),  # 左、上、右、下填充
            fill=0,  # 填充黑色（可自定义颜色）
        ),
        transforms.ToTensor(),
    ]
)
