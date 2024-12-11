import os
import shutil
import numpy as np
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
import joblib
from PIL import Image

# 加载训练好的K-means模型
kmeans = joblib.load('Preprocessing/K-means_model.pth')

# 设置图像数据的转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载指定文件夹中的图像数据
input_folder = '/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/data/unclasssified-pic'  # 替换为你的输入文件夹路径
image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.endswith(('png', 'jpg', 'jpeg'))]

# 预处理图像并转换为numpy数组
images_np = []
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    images_np.append(image.numpy().flatten())

images_np = np.array(images_np)

# 使用K-means模型进行分类
labels = kmeans.predict(images_np)

# 创建输出文件夹
output_folder = '/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/data/classified-pic'  # 替换为你的输出文件夹路径
os.makedirs(output_folder, exist_ok=True)
for i in range(3):  # 假设分成3类
    os.makedirs(os.path.join(output_folder, str(i+1)), exist_ok=True)

# 复制图像到对应的分类文件夹中
for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
    dst_folder = os.path.join(output_folder, str(label+1))
    shutil.copy(image_path, dst_folder)
    print(f"Copied {image_path} to {dst_folder}")