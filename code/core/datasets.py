import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.datasets
class ImageDataset(Dataset):
    def __init__(self, root, transform_initial=None, mode=''):
        # self.mask_size = mask_size
        self.mode = mode
        self.transform_initial = transform_initial
        self.transforms_1channel = transforms.Compose(
                            [
                                transforms.Grayscale(1),
                            ]
                        )
        self.files = []
        if isinstance(root, str):
            self.files = sorted(glob.glob("%s/*.png" % root))
        elif isinstance(root, list):
            for i in range(len(root)):
                self.files = self.files + sorted(glob.glob("%s/*.png" % root[i]))
        if mode == 'eval':
            self.gt_adr = root.replace('images','GT')
            self.gt_files = sorted(glob.glob("%s/*.png" % self.gt_adr))
        self.totalnum = len(self.files)
        # self.files = self.files[:int(self.totalnum*0.6)] if mode == "train" else self.files[int(self.totalnum*0.6):]
        self.num = len(self.files)
        
    

    def __getitem__(self, index):
        
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        # gray_img = Image.open(self.files[index % len(self.files)]).convert('L')
        if self.mode == 'train':
            mutichannels_img = self.transform_initial(img)
            singlechannel_img = self.transforms_1channel(mutichannels_img)
            return mutichannels_img, singlechannel_img
        elif self.mode == 'eval':
            mutichannels_img = self.transform_initial(img)
            gt = Image.open(self.gt_files[index % len(self.files)]).convert('L')
            return mutichannels_img, gt
        # elif self.mode == 'infer':
        #     mutichannels_img = self.transform_initial(img)
        #     return mutichannels_img
        else:
            assert False, "MODE ERROR"
            return None
            
            
            

    def __len__(self):
        return len(self.files)
# class ImageDataset(torchvision.datasets.folder.ImageFolder):
#     def __init__(self, root, transforms_=None):
#         super().__init__(root=root) # 调用了 父类的 初始化函数，就拥有了以下的 self 属性
#         classes = self.classes # list 每个类的文件名
#         class_to_idx = self.class_to_idx # 字典 每个类的文件名，类别标签(数字)
#         samples = self.samples # list 图像路径，标签(0,1,2...)
#         targets = self.targets # list 类别标签 数字：0,1,2...
#         self.transform = transforms_
#         self.files = []
#         if isinstance(root, str):
#             self.files = sorted(glob.glob("%s/0/*.png" % root))
#         elif isinstance(root, list):
#             for i in range(len(root)):
#                 self.files = self.files + sorted(glob.glob("%s/*.png" % root[i]))
#         self.totalnum = len(self.files)
#         # self.files = self.files[:int(self.totalnum*0.6)] if mode == "train" else self.files[int(self.totalnum*0.6):]
#         self.num = len(self.files)
        
    

#     def __getitem__(self, index):
#         img = Image.open(self.files[index % len(self.files)]).convert('L')
#         img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.files)