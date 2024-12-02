import os
import sys
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
def read_split_data(root:str,val_rate:float=0.3, plot = True):
    random.seed(0)
    class_list = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root,cls))] #each folder represent one class
    class_list.sort()
    class_idx = dict((k,v) for v,k in enumerate(class_list)) #generate indexs of every class

    train_data = [] #record paths for training_datas
    train_label = [] #record paths for training_labels
    val_data = [] #record paths for validation_datas
    val_label = [] #record paths for validation_labels
    every_class_num = [] #record number for each class
    supported_ext = [".jpg",".JPG",".png",".PNG",".jpeg",".JPEG"]
    # traverse each folder
    for cls in class_list:
        cls_path = os.path.join(root,cls)
        images = [os.path.join(root,cls,i) for i in os.listdir(cls_path) if os.path.splitext(i)[-1] in supported_ext]
        images.sort()
        image_class = class_idx[cls]
        every_class_num.append(len(images))
        val_set = random.sample(images, k=int(len(images)*val_rate))
        # traverse every image
        for img in images:
            if img in val_set:
                val_data.append(img)
                val_label.append(image_class)
            else:
                train_data.append(img)
                train_label.append(image_class)
    print(f"{sum(every_class_num)} images were found. {len(train_data)} for training and {len(val_data)} for validation.")
    # draw the class distribution
    if plot:
        plt.bar(range(len(class_list)), every_class_num, align = 'center', color='g')
        plt.title("class distribution")
        plt.show()
    
    return train_data,train_label,val_data,val_label


def train_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累加损失
    accu_num = torch.zeros(1).to(device)  # 累加正确预测数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for stp, data in enumerate(data_loader):
        img, labels = data  # 解包数据和标签
        print(f"Epoch {epoch}, Step {stp}:")
        print(f"Training data shape: {img.shape}")
        print(f"Target shape: {labels.shape}")

        # 确保数据和标签都在同一设备上
        img = img.to(device)
        labels = labels.to(device)

        sample_num += img.shape[0]

        # 前向传播
        pred = model(img)

        # 获取预测类别
        pred_classes = torch.max(pred, dim=1)[1]

        # 累加正确预测数
        accu_num += torch.eq(pred_classes, labels).sum()

        # 计算损失并反向传播
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 更新显示进度条
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (stp + 1),
            accu_num.item() / sample_num
        )

        # 优化器更新
        optimizer.step()
        optimizer.zero_grad()

        # 使用wandb记录训练过程中的指标
        wandb.log({
            "train_loss": accu_loss.item() / (stp + 1),
            "train_accuracy": accu_num.item() / sample_num,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    return accu_loss.item() / (stp + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()  # 设置为评估模式
    accu_loss = torch.zeros(1).to(device)  # 累加损失
    accu_num = torch.zeros(1).to(device)  # 累加正确预测数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for stp, data in enumerate(data_loader):
        imgs, labels = data  # 解包数据和标签
        imgs = imgs.to(device)  # 将图像数据移到正确的设备
        labels = labels.to(device)  # 将标签移到正确的设备

        sample_num += imgs.shape[0]

        # 前向传播
        pred = model(imgs)

        # 获取预测类别
        pred_classes = torch.max(pred, dim=1)[1]

        # 累加正确预测数
        accu_num += torch.eq(pred_classes, labels).sum()

        # 计算损失
        loss = loss_function(pred, labels)
        accu_loss += loss

        # 更新进度条显示
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (stp + 1),
            accu_num.item() / sample_num
        )

    # 返回平均损失和准确率
    return accu_loss.item() / (stp + 1), accu_num.item() / sample_num

    