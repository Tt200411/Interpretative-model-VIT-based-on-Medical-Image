from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images_path:list,images_class:list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        try:
            img = Image.open(self.images_path[item])
        except Exception as e:
            print(f"Error opening image: {self.images_path[item]}, error: {e}")
            return
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert('RGB')
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label