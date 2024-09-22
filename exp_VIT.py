import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
from VIT import ViT
from load_data import CIFAR10DataLoader  # 引入你写的CIFAR10DataLoader类

# 超参数设置
batch_size = 64
image_size = (224, 224)
learning_rate = 0.01
epoch_num = 50
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化数据加载类
data_loader = CIFAR10DataLoader(batch_size=batch_size, image_size=image_size)
train_loader, val_loader = data_loader.load_data()

# 定义ViT模型
model = ViT(num_classes=num_classes).to(device)

# 优化器和学习率调度器
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=learning_rate)

# Cosine annealing learning rate function
lf = lambda x: ((1 + math.cos(x * math.pi / epoch_num )) / 2) * (1 - 0.001) + 0.001  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# 训练和验证
for epoch in range(epoch_num):
    train_loss, train_acc = train_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)
    scheduler.step()
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
