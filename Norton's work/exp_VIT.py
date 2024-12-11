import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from VIT import ViT
from load_data import CIFAR10DataLoader
from utils import train_epoch, evaluate

# 超参数设置
batch_size = 32
image_size = (128, 128)
learning_rate = 0.01
epoch_num = 120
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化数据加载类
data_loader = CIFAR10DataLoader(batch_size=batch_size, image_size=image_size)
train_loader, val_loader = data_loader.load_data()

# 定义ViT模型，增加drop_path_ratio用于正则化
model = ViT(num_classes=num_classes, drop_path_ratio=0.1).to(device)

# 优化器和学习率调度器
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# 使用Cosine Annealing学习率调度器
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

# 用于记录损失和准确率
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# 训练和验证
for epoch in range(epoch_num):
    # 训练模型
    train_loss, train_acc = train_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device,
                                        epoch=epoch)
    # 验证模型
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

    # 记录损失和准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # 更新学习率
    scheduler.step()

# 绘制损失和准确率图表
epochs = range(1, epoch_num + 1)

plt.figure(figsize=(12, 5))

# 绘制损失图
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率图
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
