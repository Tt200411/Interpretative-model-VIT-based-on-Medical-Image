import os
import shutil
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# 创建文件夹的函数
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 加载模型
def load_model(model_path, num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练的权重
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 替换最后一层全连接层
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型
    model.eval()  # 设置为评估模式
    return model

# 定义图片预处理方式
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 的均值和标准差
])

# 预测并保存图像到对应文件夹
def classify_and_save_images(model, input_dir, output_dir, num_classes):
    # 遍历输入文件夹中的所有图片
    for class_id in range(num_classes):
        create_folder(os.path.join(output_dir, str(class_id)))  # 创建类别文件夹

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据实际情况修改
            # 打开图片并进行预处理
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)  # 添加 batch 维度

            # 推断类别
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)

            # 保存图片到相应的文件夹
            class_folder = os.path.join(output_dir, str(predicted.item()))
            shutil.copy(img_path, class_folder)  # 复制图片到对应的类别文件夹

# 设置参数
model_path = '/Users/han/Desktop/year3/MLW/Interpretative-model-VIT-based-on-Medical-Image/resnet18_finetuned.pth'  
input_dir = '/Users/han/Desktop/data/test_val'  # 替换为未分类的图片文件夹路径
output_dir = '/Users/han/Desktop/data/output'  # 替换为保存分类图片的文件夹路径
num_classes = 3  # 类别数，与你的训练模型一致

# 加载模型
model = load_model(model_path, num_classes)
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 对未分类的图片进行分类并保存
classify_and_save_images(model, input_dir, output_dir, num_classes)

print("Images have been classified and saved.")
