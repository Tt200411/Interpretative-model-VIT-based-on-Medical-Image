import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from MedViT import MedViT
from load_data import CIFAR10DataLoader
from utils import train_epoch, evaluate
from advanced_visualization import run_advanced_visualization
import wandb
from torch.amp import autocast, GradScaler  # 新的导入方式
import torch.nn as nn

# 启动wandb记录
wandb.init(project='VIT', entity='renxuanhao290-bnu-hkbu-united-international-college')

# 超参数设置
batch_size = 64  # 增加批量大小以提高训练效率
image_size = (96, 96)  # 减小图像尺寸以降低计算量
learning_rate = 3e-4  # 调整学习率
epoch_num = 1
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化数据加载类
data_loader = CIFAR10DataLoader(batch_size=batch_size, image_size=image_size)
train_loader, val_loader = data_loader.load_data()

# 定义ViT模型，增加drop_path_ratio用于正则化
# Example initialization of MedViT

model = MedViT(
    stem_chs=[3, 32, 64],  # Set stem_chs[0] to 3 for RGB input
    depths=[3, 4, 10, 3],
    path_dropout=0.1,
    attn_drop=0.1,
    drop=0.1,
    num_classes=10,
    strides=[1, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    head_dim=32,
    mix_block_ratio=0.75
)

model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

# 优化器和学习率调度器
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(pg, lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

# 使用更复杂的学习率调度器
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=epoch_num,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

# 用于记录损失和准确率
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


# 添加早停
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(patience=10)

# 创建梯度缩放器
scaler = GradScaler()

# 训练和验证
for epoch in range(epoch_num):
    # 训练模型
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # 使用混合精度训练
        with autocast(device_type='cuda'   ):
            output = model(data)
            loss = criterion(output, target)  # 使用交叉熵损失函数

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total

    # 验证模型
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

    # 记录损失和准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # 更新学习率
    scheduler.step()

    if early_stopping(val_loss, model):
        print("Early stopping")
        break

run_advanced_visualization(model, train_loader, val_loader, device, data_loader.class_names)

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
