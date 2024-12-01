import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class CIFAR10DataLoader:
    def __init__(self, batch_size=32, image_size=(128, 128)):
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # 调整图像大小
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
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

# 你可以这样在其他文件中引用此类
# from load_data import CIFAR10DataLoader


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataLoader:
    def __init__(self, batch_size=32, image_size=(128, 128)):
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # 调整图像大小
            transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化，单通道图像
        ])
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.class_names = [str(i) for i in range(10)]  # MNIST 类别为0到9

    def load_data(self, root='./data'):
        # 加载 MNIST 数据集
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)

        # 创建 DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

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
                img = images[i].squeeze(0).numpy()  # 转换为 numpy 格式以进行可视化
                img = img * 0.5 + 0.5  # 反归一化
                img = np.clip(img, 0, 1)
                axs[i].imshow(img, cmap='gray')
                axs[i].set_title(f'Label: {self.class_names[labels[i]]}')
                axs[i].axis('off')
            plt.show()
            break  # 只显示一个批次

# 使用示例
# mnist_loader = MNISTDataLoader(batch_size=32, image_size=(28, 28))
# train_loader, test_loader = mnist_loader.load_data()
# mnist_loader.show_batch_info(train_loader)
# mnist_loader.show_augmented_images(train_loader)
