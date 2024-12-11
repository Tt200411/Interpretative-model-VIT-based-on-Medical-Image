import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

# 指定图像的文件夹路径
image_folder = '/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/data/unclasssified-pic'
output_folder = '/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/data/test'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取图像文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 初始化一个空的列表用于存放图像特征
image_features = []
image_paths = []

# 读取图像并提取特征
for file in image_files:
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path)

    # 调整图像大小以降低计算开销
    img_resized = cv2.resize(img, (64, 64))
    
    # 将图像展平为一维特征向量（R, G, B像素值）
    img_flattened = img_resized.flatten()
    
    image_features.append(img_flattened)
    image_paths.append(img_path)

# 将图像特征转为numpy数组
image_features = np.array(image_features)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(image_features)

# 获取每张图像的聚类标签
labels = kmeans.labels_

# 创建用于存储分类图像的文件夹
for i in range(3):
    cluster_folder = os.path.join(output_folder, f'cluster_{i}')
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

# 根据聚类标签将图像复制到相应的文件夹中
for idx, label in enumerate(labels):
    src_path = image_paths[idx]
    dest_path = os.path.join(output_folder, f'cluster_{label}', os.path.basename(src_path))
    shutil.copy(src_path, dest_path)

print("图像已成功分为三类并复制到相应文件夹。")
