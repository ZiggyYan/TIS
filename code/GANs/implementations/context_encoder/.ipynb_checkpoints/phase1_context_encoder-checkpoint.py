"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""

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
parser.add_argument("--batch_size", type=int, default=60, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="busb", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
# parser.add_argument("--mask_size", type=int, default=256, help="size of random mask")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--scale_min", type=float, default=0.9)
parser.add_argument("--scale_max", type=float, default=1.0)
parser.add_argument("--save_frequency", type=int, default=200, help="save generated images every 5 epochs")
parser.add_argument("--para_save_frequency", type=int, default=400, help="save generated images every 5 epochs")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_size / (2 ** 3)), int(opt.img_size / (2 ** 3))
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

# Dataset loader
# transforms_ = [
#     transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
dataloader = DataLoader(
    ImageDataset("./Dataset/BUSI_normal/images", transforms_=transforms_,
                 img_size=opt.img_size, phase = '1'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = DataLoader(
    ImageDataset("./Dataset/BUSI_normal/images", transforms_=transforms_, mode="val", 
                 img_size=opt.img_size, phase = '1'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)
os.makedirs("./Dataset/BUSI_normal/generated/Context_Encoder/", exist_ok=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# def save_sample(batches_done):
#     samples, masked_samples, i = next(iter(test_dataloader))
#     samples = Variable(samples.type(Tensor))
#     masked_samples = Variable(masked_samples.type(Tensor))
#     i = i[0].item()  # Upper-left coordinate of mask
#     # Generate inpainted image
#     gen_mask = generator(masked_samples)
#     filled_samples = masked_samples.clone()
#     filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
#     # Save sample
#     sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
#     save_image(sample, "./Dataset/BUSI_normal/generated/Context_Encoder/%s.png" % batches_done, nrow=6, normalize=True)
    


# ----------
#  Training
# ----------
print('NUM OF TRAIN LOADER:',len(dataloader))
print('NUM OF TEST LOADER:',len(test_dataloader))
best_loss = 1000000
no_progress_record = 0
for epoch in range(opt.n_epochs):
    # -----------------
    #  Training Phase
    # -----------------
    print("------TRAINING------")
    for i, (imgs, masked_imgs, masked_parts, states) in enumerate(dataloader):
        
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        
        gen_parts = generator(masked_imgs)
    
        target_position_sign = (masked_imgs == 1)
        gen_images = torch.where(target_position_sign, gen_parts, masked_imgs)
        
        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_images), valid)
        g_pixel = pixelwise_loss(gen_images, imgs)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_images.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )
        
        # if epoch%opt.save_frequency == 0:
        #     # for n in range(int(gen_imgs.shape[0]/4)):
        #     saved_name = "ep"+str(epoch)+"_"+str(i)
        #     save_image(gen_imgs.data[:16], "./Dataset/BUSB/generated/DCGAN/%s.png" % saved_name, nrow=4,normalize=True)
        # Generate sample at sample interval
        # batches_done = "ep_"+str(epoch)
        # if epoch % opt.save_frequency == 0:
        #     save_sample(batches_done)
        # if epoch % opt.para_save_frequency == 0:
        #     torch.save(generator.state_dict(), "./code/PyTorch-GAN-master/model_weights/context_encoder/"
        #                +"ep"+str(epoch)+'_weights.pth')

    # -----------------
    #  Testing Phase
    # -----------------
    current_loss = 0
    print("------TESTING------")
    for i, (imgs, masked_imgs, masked_parts, _) in enumerate(test_dataloader):
        
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        target_position_sign = (masked_imgs == 1)
        gen_images = torch.where(target_position_sign, gen_parts, masked_imgs)
        
        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_images), valid)
        g_pixel = pixelwise_loss(gen_images, imgs)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel
        current_loss = current_loss + g_loss
       
        print(
            "[Epoch %d/%d] [Batch %d/%d] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader),  g_adv.item(), g_pixel.item())
        )
        
    if current_loss<best_loss:
        torch.save(generator.state_dict(), "./code/GANs/model_weights/context_encoder/"
                   +'best_weights.pth')
        best_loss = current_loss
        print('SAVING WEIGHTS:',current_loss)
    else:
        no_progress_record  = no_progress_record + 1
    if no_progress_record > 2500:
        break