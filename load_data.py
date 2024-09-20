import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# 设置图像转换：调整大小、转换为张量并归一化
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载数据集
classified_dataset = ImageFolder(root='output', transform=transform)

# 创建 DataLoader 来批量加载数据
classified_loader = DataLoader(classified_dataset, batch_size=64, shuffle=True)

# 获取类别名称
class_names = classified_dataset.classes
print("classname: ", class_names)

# test
for batch_idx, (images, labels) in enumerate(classified_loader):
    print(f'Batch {batch_idx + 1}')
