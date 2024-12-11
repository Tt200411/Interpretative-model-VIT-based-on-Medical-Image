import os
import sys
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb


def read_split_data(root: str, val_rate: float = 0.3, plot=True):
    random.seed(0)
    class_list = [cls for cls in os.listdir(root) if
                  os.path.isdir(os.path.join(root, cls))]  # each folder represents one class
    class_list.sort()
    class_idx = dict((k, v) for v, k in enumerate(class_list))  # generate indexes for each class

    train_data = []  # record paths for training data
    train_label = []  # record paths for training labels
    val_data = []  # record paths for validation data
    val_label = []  # record paths for validation labels
    every_class_num = []  # record number for each class
    supported_ext = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    # Traverse each folder
    for cls in class_list:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path) if os.path.splitext(i)[-1] in supported_ext]
        images.sort()
        image_class = class_idx[cls]
        every_class_num.append(len(images))
        val_set = random.sample(images, k=int(len(images) * val_rate))

        # Traverse every image
        for img in images:
            if img in val_set:
                val_data.append(img)
                val_label.append(image_class)
            else:
                train_data.append(img)
                train_label.append(image_class)

    print(
        f"{sum(every_class_num)} images were found. {len(train_data)} for training and {len(val_data)} for validation.")

    # Draw the class distribution
    if plot:
        plt.bar(range(len(class_list)), every_class_num, align='center', color='g')
        plt.title("Class Distribution")
        plt.show()

    return train_data, train_label, val_data, val_label


def train_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # Accumulate loss
    accu_num = torch.zeros(1).to(device)  # Accumulate correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for stp, data in enumerate(data_loader):
        img, labels = data  # Unpack data and labels
        print(f"Epoch {epoch}, Step {stp}:")
        print(f"Training data shape: {img.shape}")
        print(f"Target shape: {labels.shape}")

        # Ensure data and labels are on the same device
        img = img.to(device)
        labels = labels.to(device)

        sample_num += img.shape[0]

        # Forward pass
        pred = model(img)

        # Get predicted classes
        pred_classes = torch.max(pred, dim=1)[1]

        # Accumulate correct predictions
        accu_num += torch.eq(pred_classes, labels).sum()

        # Compute loss and backpropagate
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # Update progress bar
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (stp + 1),
            accu_num.item() / sample_num
        )

        # Optimizer update
        optimizer.step()
        optimizer.zero_grad()

        # Log the training process in wandb
        wandb.log({
            "train_loss": accu_loss.item() / (stp + 1),
            "train_accuracy": accu_num.item() / sample_num,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    return accu_loss.item() / (stp + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()  # Set to evaluation mode
    accu_loss = torch.zeros(1).to(device)  # Accumulate loss
    accu_num = torch.zeros(1).to(device)  # Accumulate correct predictions
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for stp, data in enumerate(data_loader):
        imgs, labels = data  # Unpack data and labels
        imgs = imgs.to(device)  # Move image data to the correct device
        labels = labels.to(device)  # Move labels to the correct device

        sample_num += imgs.shape[0]

        # Forward pass
        pred = model(imgs)

        # Get predicted classes
        pred_classes = torch.max(pred, dim=1)[1]

        # Accumulate correct predictions
        accu_num += torch.eq(pred_classes, labels).sum()

        # Compute loss
        loss = loss_function(pred, labels)
        accu_loss += loss

        # Update progress bar
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (stp + 1),
            accu_num.item() / sample_num
        )

    # Return average loss and accuracy
    return accu_loss.item() / (stp + 1), accu_num.item() / sample_num


# New function to merge BN layers
def merge_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre-BN to reduce inference runtime. """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight * (
                    pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias
