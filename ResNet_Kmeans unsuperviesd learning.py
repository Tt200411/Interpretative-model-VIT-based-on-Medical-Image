import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 ResNet-18 模型并指定预训练权重
resnet16 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用指定的预训练权重
# 移除最后的全连接层
resnet16 = torch.nn.Sequential(*list(resnet16.children())[:-1])
resnet16 = resnet16.to(device)  # 将模型移至 GPU

print("load successful")

# 设置数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以适配ResNet输入
    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
])

# 下载并加载MNIST数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=False)

print("start to train")
# 编码MNIST数据集并使用进度条
encoded_features = []
with torch.no_grad():  # 关闭梯度计算，节省内存
    for images, labels in tqdm(mnist_loader, desc="Encoding MNIST Images", unit="batch"):
        images = images.to(device)  # 将数据移至 GPU
        features = resnet16(images)  # 通过ResNet编码
        features = features.view(features.size(0), -1)  # 展平为一维向量
        encoded_features.append(features)

encoded_features = torch.cat(encoded_features, dim=0)  # 合并所有特征

# 转换为 NumPy 数组以便与K-means兼容
encoded_features_np = encoded_features.cpu().numpy()

# 使用 K-means 进行聚类，设置类别数量为 10（MNIST有10类）
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(encoded_features_np)

# 获取聚类结果
predicted_labels = kmeans.labels_

# 可视化部分：展示一些分类结果
# 随机选择10张图片，展示其聚类标签
# 可视化部分：展示一些分类结果
# 随机选择10张图片，展示其聚类标签
for i in range(10):
    plt.subplot(2, 5, i+1)
    
    # 将图像从 CxHxW 转换为 HxWxC 格式
    img = mnist_dataset[i][0].permute(1, 2, 0).cpu().numpy()
    
    # 显示图像
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.show()

