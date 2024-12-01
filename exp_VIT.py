import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
from VIT import ViT
from load_data import CIFAR10DataLoader 
from utils import train_epoch
from utils import evaluate
import numpy as np

# 超参数设置
batch_size = 32
image_size = (128, 128)
learning_rate = 0.01
epoch_num = 3
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化数据加载类
# mnist_loader = MNISTDataLoader(batch_size=32, image_size=(28, 28))
# train_loader, test_loader = mnist_loader.load_data()

# data_loader = CIFAR10DataLoader(batch_size=batch_size, image_size=image_size)
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
# 训练和验证
for epoch in range(epoch_num):
    train_loss, train_acc, train_attention_weights = train_epoch(
        model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, return_attention_weights=True)
    
    scheduler.step()
    
    val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
    
    # 在每个epoch结束时处理注意力权重
    if len(train_attention_weights) > 0:
        # 这里可以处理train_attention_weights，比如保存到文件或者进行可视化
        print(f"Epoch {epoch}: Training attention weights obtained.")
        print(np.array(train_attention_weights).shape)

    # 清空attention_weights以便下一个epoch使用
    train_attention_weights = []

    # 进行验证等其他操作



