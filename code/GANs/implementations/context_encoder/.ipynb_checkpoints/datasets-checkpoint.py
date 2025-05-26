import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    # def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train",  phase=""):
    def __init__(self, root, transforms_=None, img_size=128, mode="train",  phase=""):
        self.img_size = img_size
        self.phase = phase
        # self.mask_size = mask_size
        self.mode = mode
        # self.transform_resize =  transforms.Compose([
        #     transforms.Resize((self.mask_size, self.mask_size)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5]),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     # transforms.Grayscale(num_output_channels=1),
        # ])
        self.transform = transforms.Compose(transforms_)
        self.files = []
        if isinstance(root, str):
            self.files = sorted(glob.glob("%s/*.png" % root))
        elif isinstance(root, list):
            for i in range(len(root)):
                self.files = self.files + sorted(glob.glob("%s/*.png" % root[i]))
        self.totalnum = len(self.files)
        self.files = self.files[:int(self.totalnum*0.6)] if mode == "train" else self.files[int(self.totalnum*0.6):]
        self.num = len(self.files)
        # print('TOTAL NUM:',len(self.files))
        
    def do_reverse(self, A: torch.Tensor, B: torch.Tensor):
        assert A.shape == B.shape, "输入张量形状必须一致"

        # 生成掩码（A中值为1的位置为True）
        mask = (A == 1)

        # 创建全1张量（与B同类型、同设备）
        ones_tensor = torch.ones_like(B)

        # 组合结果：满足掩码的位置取B值，其他位置取1
        result = torch.where(mask, B, ones_tensor)
        return result
    
    
    def apply_random_mask(self, img):
        """Randomly masks image"""
        num_holes = 3
        hole_size_range=(10, 80)
        masked_img = img.clone()
        masked_part = img.clone()
        for _ in range(num_holes):
            x1 = random.randint(0, self.img_size - hole_size_range[1])
            y1 = random.randint(0, self.img_size - hole_size_range[1])
            w = random.randint(hole_size_range[0], hole_size_range[1])
            h = random.randint(hole_size_range[0], hole_size_range[1])
            y2, x2 = y1 + h, x1 + w
            # 在指定位置挖空
            masked_img[:, y1:y2, x1:x2] = 1
        masked_part = self.do_reverse(masked_img, masked_part)
        
        # y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        # y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        # masked_part = img[:, y1:y2, x1:x2]
        # masked_img = img.clone()
        # masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

#     def apply_center_mask(self, img):
#         """Mask center part of image"""
#         # Get upper-left pixel coordinate
#         i = (self.img_size - self.mask_size) // 2
#         masked_img = img.clone()
#         masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

#         return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')
        if "normal" in self.files[index % len(self.files)]:
            ill_state = False
        else:
            ill_state = True
        if self.phase == "2-2":
            # img_resized = self.transform_resize(img)
            img = self.transform(img)
            masked_img, masked_part = self.apply_random_mask(img)
            return img, masked_img, masked_part, ill_state
        else:
            img = self.transform(img)
            
            # if self.phase == "1" and self.mode == "val":
            #     masked_img, aux = self.apply_center_mask(img)
            # else:
            #     masked_img, aux = self.apply_random_mask(img)
            masked_img, masked_part = self.apply_random_mask(img)
            return img, masked_img, masked_part, ill_state

    def __len__(self):
        return len(self.files)
