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
dataset = ImageFolder('/Users/han/Desktop/解释性模型/data', transform=transform)

# 获取类名
class_names = dataset.classes
print("Classes: ", class_names)

# 划分数据集为训练集（80%）和测试集（20%）
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):  # 3类分类
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
        # 修改此处的Linear层输入大小为 256 * 3 * 3 = 2304
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096), 
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


# 初始化模型、损失函数和优化器
model = AlexNet(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 设置训练参数
num_epochs = 20

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
    torch.save(model.state_dict(), 'model_alexnet.pth')
    print("模型已保存为 model_alexnet.pth")

train_model(model, train_loader, criterion, optimizer, num_epochs)

# 测试模型
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

# 调用测试函数，测试模型在测试集上的性能
test_model(model, test_loader)

# 加载保存好的模型权重
model.load_state_dict(torch.load('model_alexnet.pth'))

# 将模型设置为评估模式
model.eval()

# 再次测试保存的模型
test_model(model, test_loader)
