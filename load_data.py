import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class CIFAR10DataLoader:
    def __init__(self, batch_size=64, image_size=(96, 96)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.class_names = None

    def load_data(self, root='./data'):
        # 加载 CIFAR-10 数据集
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=self.transform)

        # 创建 DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 获取类别名称
        self.class_names = train_dataset.classes

        return self.train_loader, self.test_loader

    def show_batch_info(self, dataloader):
        # 测试 DataLoader，输出一个批次的图像信息
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}')
            print(f'Images batch shape: {images.size()}')
            print(f'Labels batch shape: {labels.size()}')
            break  # 仅输出一个批次

    def show_augmented_images(self, dataloader):
        # 显示增强后的图像
        for images, labels in dataloader:
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                img = images[i].permute(1, 2, 0).numpy()  # 转换为 numpy 格式以进行可视化
                img = img * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])  # 反归一化
                img = np.clip(img, 0, 1)
                axs[i].imshow(img)
                axs[i].set_title(f'Label: {self.class_names[labels[i]]}')
                axs[i].axis('off')
            plt.show()
            break  # 只显示一个批次

# 示例用法：
# from load_data import CIFAR10DataLoader
# data_loader = CIFAR10DataLoader()
# train_loader, val_loader = data_loader.load_data()
