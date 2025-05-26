"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""
from datetime import datetime

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=12000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=80, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="busb", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
# parser.add_argument("--mask_size", type=int, default=128, help="size of random mask")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--scale_min", type=float, default=0.9)
parser.add_argument("--scale_max", type=float, default=1.0)
parser.add_argument("--save_frequency", type=int, default=200, help="save generated images every 5 epochs")
parser.add_argument("--para_save_frequency", type=int, default=400, help="save generated images every 5 epochs")

opt = parser.parse_args()
print(opt)
os.makedirs("./code/GANs/model_weights/discriminator/", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_size / 2 ** 3), int(opt.img_size / 2 ** 3)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)
discriminator.load_state_dict(torch.load("./code/GANs/model_weights/discriminator/phase2-1_best_weights.pth"))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


transforms_ =  [
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(
                opt.img_size, scale=(opt.scale_min, opt.scale_max), interpolation=3
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Grayscale(num_output_channels=1),
        ]
transforms_train =  [
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(
                opt.img_size, scale=(opt.scale_min, opt.scale_max), interpolation=3
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Grayscale(num_output_channels=1),
        ]
# Dataset loader
# transforms_ = [
#     transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
fused_train_dataset =  ImageDataset(["./Dataset/BUSI_normal/images","./Dataset/BUSB/images"], transforms_=transforms_,
                 img_size=opt.img_size, phase = '2-1', mode = "train")
fused_test_dataset =  ImageDataset(["./Dataset/BUSI_normal/images","./Dataset/BUSB/images"], transforms_=transforms_,
                 img_size=opt.img_size, phase = '2-1', mode = "test")

train_fused_dataloader = DataLoader(
    fused_train_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

test_fused_dataloader = DataLoader(
    fused_test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)
# os.makedirs("./Dataset/BUSI_normal/generated/Context_Encoder/", exist_ok=True)

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    
# 定义文件名和内容
record_file = "./code/GANs/model_weights/discriminator/phase2extra_test.txt"
content = "This is the record file for extra check of phase 2."

# 使用 'w' 模式写入（覆盖已有内容）
with open(record_file, "w", encoding="utf-8") as f:
    f.write(content)


current_loss = 0
print('-------------EXTRA TEST FOR PHASE 2-1-------------')
for i, (imgs, masked_imgs, masked_parts, ill_states) in enumerate(test_fused_dataloader):
    # Adversarial ground truths
    judge_mask = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
    diseased = Variable(Tensor(*patch).fill_(1.0), requires_grad=False)
    healthy = Variable(Tensor(*patch).fill_(0.0), requires_grad=False)
    for j in range(imgs.shape[0]):
        if ill_states[j]:
            judge_mask[j] = diseased
        else:
            judge_mask[j] = healthy
    # print(discriminator(masked_parts.cuda()).shape)
    # Measure discriminator's ability to classify real from generated samples
    fused_loss = adversarial_loss(discriminator(imgs.cuda()), judge_mask)
    current_loss = current_loss + fused_loss  
print('CURRENT LOSS:',current_loss)
# 定义要追加的内容
append_content1 = "\nBest performance is updated:"+str(current_loss)
append_content3 = "\nCurrent Time:"+str(datetime.now())
# 使用 'a' 模式追加写入
with open(record_file, "a", encoding="utf-8") as f:
    f.write(append_content1)
    f.write(append_content3)


    
