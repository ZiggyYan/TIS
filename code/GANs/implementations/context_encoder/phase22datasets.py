import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train", phase=""):
        self.img_size = img_size
        self.phase = phase
        self.mask_size = mask_size
        self.transform = transforms.Compose(transforms_)
        self.transform_resize =  transforms.Compose([
            transforms.Resize((self.mask_size, self.mask_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Grayscale(num_output_channels=1),
        ])
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.png" % root))
        self.totalnum = len(self.files)
        self.files = self.files[:int(self.totalnum*0.6)] if mode == "train" else self.files[int(self.totalnum*0.6):]
        self.num = len(self.files)
        # print('TOTAL NUM:',len(self.files))
    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')
        img_resized = self.transform_resize(img)
        img = self.transform(img)
        
            
        # print(img_resized.shape)
        # if self.mode == "train":
            # For training data perform random mask
        masked_img, aux = self.apply_random_mask(img)
        # else:
            # For test data mask the center of the image
            # masked_img, aux = self.apply_center_mask(img)
        return img_resized, masked_img, aux

    def __len__(self):
        return len(self.files)
