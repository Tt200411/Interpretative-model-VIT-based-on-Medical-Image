import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 ResNet-18 模型并指定预训练权重
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练的权重
# 修改全连接层的输出维度（假设类别数为 num_classes）
num_classes = 3  # 假设类别数为3，可以根据你的数据集调整
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)  # 替换最后一层全连接层
resnet18 = resnet18.to(device)  # 将模型移至 GPU

print("Model loaded successfully")

# 设置数据预处理（包含数据增强）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),     # 随机旋转
    transforms.Resize((224, 224)),     # 调整大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
])

# 定义本地数据集路径
dataset_dir = '/Users/han/Desktop/data'

# 加载训练数据和验证数据
train_dataset = datasets.ImageFolder(root=f'{dataset_dir}/train', transform=transform)
val_dataset = datasets.ImageFolder(root=f'{dataset_dir}/train_val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于分类问题
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)  # 使用Adam优化器

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_loss_list = []
    val_acc_list = []
    
    try:
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # 每个epoch都有训练和验证两个阶段
            model.train()  # 设定为训练模式
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            
            # 训练阶段
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()  # 清零梯度
                
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                
                running_loss += loss.item() * inputs.size(0)  # 累加损失
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                correct_preds += torch.sum(preds == labels.data)  # 计算正确预测数
                total_preds += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct_preds.double() / total_preds
            
            train_loss_list.append(epoch_loss)
            print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
            
            # 验证阶段
            model.eval()  # 设定为评估模式
            correct_preds = 0
            total_preds = 0
            
            with torch.no_grad():  # 不计算梯度
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)  # 前向传播
                    _, preds = torch.max(outputs, 1)  # 获取预测类别
                    correct_preds += torch.sum(preds == labels.data)  # 计算正确预测数
                    total_preds += labels.size(0)
            
            epoch_val_acc = correct_preds.double() / total_preds
            val_acc_list.append(epoch_val_acc)
            print(f'Validation Accuracy: {epoch_val_acc:.4f}')
            
            # 保存最佳模型权重
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_wts = model.state_dict()
            
            # 学习率调度
            scheduler.step()

        print(f'Best Validation Accuracy: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)  # 加载最佳模型权重
        
        # 可视化训练过程
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss_list)), train_loss_list, label='Training Loss')
        plt.plot(range(len(val_acc_list)), val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        plt.show()

        return model
    except Exception as e:
        print(f"An error occurred during training: {e}")

# 训练模型
resnet18 = train_model(resnet18, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# 保存训练后的模型
model_save_path = f'resnet18_finetuned_{time.strftime("%Y%m%d_%H%M%S")}.pth'
torch.save(resnet18.state_dict(), model_save_path)
print(f"Model saved successfully at {model_save_path}")