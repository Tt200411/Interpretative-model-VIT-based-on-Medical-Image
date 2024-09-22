import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

# 设置图像数据的转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载整个数据集
dataset = ImageFolder('/Users/luhaoran/Downloads/data', transform=transform)

# 获取类名
class_names = dataset.classes
num_classes = len(class_names)
print("Classes: ", class_names)

# 划分数据集为训练集（80%）和测试集（20%）
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义VGGNetModified模型
class VGGNetModified(nn.Module):
    def __init__(self, num_classes=3):
        super(VGGNetModified, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1 - 不使用池化层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Block 2 - 保持池化层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3 - 减小池化影响
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 修改为步幅1
            
            # Conv Block 4 - 保持原设计
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 5 - 减小池化影响
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)  # 修改为步幅1
        )
        
        # 计算卷积层输出的维度
        self.fc_layers = nn.Sequential(
            nn.Linear(460800, 4096),  # 调整输入维度
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # 输出维度应为类别数
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 初始化模型、损失函数和优化器
model = VGGNetModified(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练参数
num_epochs = 10

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # 将梯度归零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 统计损失
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
   
    # 训练完成后保存模型
    torch.save(model.state_dict(), 'model_VGGNetModified.pth')
    print("模型已保存为 model_VGGNetModified.pth")

train_model(model, train_loader, criterion, optimizer, num_epochs)

def test_model(model, test_loader):
    model.eval()  # 评估测试集
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时关闭梯度计算
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Accuracy: " + str(accuracy))

test_model(model, test_loader)

# 定义你的模型架构
model = VGGNetModified()

# 加载保存好的模型权重
model.load_state_dict(torch.load('model_VGGNetModified.pth'))

# 将模型设置为评估模式
model.eval()

# 定义测试模型的函数
def test_model(model, test_loader):
    model.eval()  # 评估测试集
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时关闭梯度计算
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Accuracy: " + str(accuracy))

# 调用测试函数，测试保存的模型在测试集上的性能
test_model(model, test_loader)
