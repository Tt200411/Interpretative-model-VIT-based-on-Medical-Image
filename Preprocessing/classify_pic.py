import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 预处理步骤要和训练时的一致
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义模型结构（与训练时一致）
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),  # 输入维度需要保持一致
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 动态展平
        x = self.classifier(x)
        return x

# 加载已经训练好的模型
model = AlexNet()
model.load_state_dict(torch.load('/Users/han/Desktop/解释性模型/model_alexnet.pth'))  # 替换为实际模型路径
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
image_folder = '/Users/han/Desktop/解释性模型/project_part2'  # 替换为你的图片目录
classify_and_copy_images(model, image_folder, output_dirs, transform)
