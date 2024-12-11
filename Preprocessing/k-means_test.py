import numpy as np
import os
import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import joblib  # 导入joblib库
import torch.nn as nn
# 设置图像数据的转换

class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_c=3, 
                 embed_dim=768, 
                 norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载整个数据集
dataset = datasets.ImageFolder('/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/data/training_data', transform=transform)

# 将数据集加载到内存中
data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
data_iter = iter(data_loader)

# 使用内置函数 next() 获取迭代器的下一个元素
images, labels = next(data_iter)

# 将图像数据转换为numpy数组
images_np = images.numpy()
images_np = images_np.reshape(images_np.shape[0], -1)  # 展平每个图像
print(images_np.shape)

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=3, random_state=0)  # 分成3类
kmeans.fit(images_np)
embedding = PatchEmbedding(img_size=256)
outputs=embedding(images_np)
print(outputs.shape)


# 保存训练好的模型
joblib.dump(kmeans, 'K-means_model.pth')

# 获取聚类标签
labels = kmeans.labels_

# 输出每个图像的聚类标签
for idx, label in enumerate(labels):
    print(f"Image {idx} is in cluster {label}")