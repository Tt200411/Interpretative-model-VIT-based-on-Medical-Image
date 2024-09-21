import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 设置图像数据的转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv layer: input channels = 3, output channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second conv layer: input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
        # Dropout layer
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 64 * 32 * 32)           # Flatten
        x = F.relu(self.fc1(x))               # Fully connected -> ReLU
        x = F.relu(self.fc2(x))               # Fully connected -> ReLU
        x = self.dropout(x)                   # Apply Dropout
        x = self.fc3(x)                       # Output layer
        return x

# 加载已经训练好的模型
model = SimpleCNN()
model.load_state_dict(torch.load('/Users/luhaoran/Interpretative-model-VIT-based-on-Medical-Image/Preprocessing/model_simpleCNN.pth'))  # 替换为你实际的模型路径
model.eval()  # 切换为评估模式

# 创建分类文件夹
output_dirs = ['output/vertical', 'output/sagittal', 'output/frontal']
for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)

def classify_and_copy_images(model, image_folder, output_dirs, transform):
    # 遍历所有图片
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 过滤图像文件
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')  # 打开图像
            image_tensor = transform(image).unsqueeze(0)  # 应用预处理并添加batch维度
            
            # 使用模型进行预测
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output.data, 1)
            
            # 根据预测结果确定分类并复制文件
            predicted_class = predicted.item()  # 获取分类结果
            class_name = output_dirs[predicted_class]  # 获取对应的类别文件夹
            
            # 复制文件到对应的文件夹
            shutil.copy(image_path, os.path.join(class_name, filename))
            print(f"Copied {filename} to {class_name}")

# 调用分类和复制函数
image_folder = '/Users/luhaoran/Machine Learning/peoject_part2'  # 替换为你的图片目录
classify_and_copy_images(model, image_folder, output_dirs, transform)
