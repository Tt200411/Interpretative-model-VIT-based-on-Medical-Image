import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 预处理步骤要和训练时的一致
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义模型结构（和训练时的一致）
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)  # 输出3个类别
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载已经训练好的模型
model = CNNClassifier()
model.load_state_dict(torch.load('/Users/luhaoran/Machine Learning/model.pth'))  # 替换为你实际的模型路径
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
