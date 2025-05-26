import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import sys
sys.path.append("/root/autodl-tmp/unsupervised/code")  

from GANs.implementations.context_encoder.models import * 

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

adversarial_loss = torch.nn.MSELoss()
cuda = True if torch.cuda.is_available() else False

class block_loss(nn.Module):
    def __init__(self, generator_pth_adr, discriminator_pth_adr, channels):
        super().__init__()
        self.channels = channels
        self.generator = Generator(channels=self.channels)
        self.discriminator = Discriminator(channels=self.channels)

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
        # Initialize weights
        self.generator.load_state_dict(torch.load(generator_pth_adr))
        self.discriminator.load_state_dict(torch.load(discriminator_pth_adr))
        self.generator.eval()
        self.discriminator.eval()
    # Segmentation loss between predicted masks and masks generated with HOG images
    def generate_loss(self, pred_masks, images):
        # Initialize key elements
        assert images.shape[2] == images.shape[3], "SHAPE ERROR!"
        img_size = images.shape[2]
        # diseased = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        # healthy = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
        patch_h, patch_w = int(img_size / (2 ** 3)), int(img_size / (2 ** 3))
        patch = (1, patch_h, patch_w)
        diseased = torch.ones((images.shape[0], *patch), dtype=torch.float32, requires_grad=False)
        healthy = torch.zeros((images.shape[0], *patch), dtype=torch.float32, requires_grad=False)

        # Crop masks from images
        pred_masks_signs = pred_masks.clone()
        pred_masks_signs[pred_masks_signs>=0.4] = 1
        mask_position_sign = (pred_masks_signs == 1)
        ones_tensor = torch.ones_like(pred_masks)

        masked_imgs = torch.where(mask_position_sign, ones_tensor, images)

        # Generate "FAKE" "HEALTHY" parts
        gen_parts = self.generator(masked_imgs)
        # Composite masked_images with generated parts
        gen_images = torch.where(mask_position_sign, gen_parts, images)

        # If the results is correct, the entire images is supposed to be healthy in discriminator' eyes.
        # Do judge
        result = adversarial_loss(self.discriminator(gen_images.detach()), healthy.cuda())

        return result

# def generator_loss(pred_masks,images):
#     # Initialize discriminator and generator
#     generator = Generator(channels=opt.channels)
#     discriminator = Discriminator(channels=opt.channels)

#     if cuda:
#         generator.cuda()
#         discriminator.cuda()
#     # Initialize weights
#     generator.load_state_dict(torch.load("./code/PyTorch-GAN-master/model_weights/context_encoder/ep2345_weights.pth"))
#     discriminator.load_state_dict(torch.load("./code/PyTorch-GAN-master/model_weights/",
#                                              "discriminator/phase2-1_best_weights.pth"))
    
#     # Initialize key elements
#     diseased = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
#     healthy = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
    
    
#     # Crop masks from images
#     pred_masks_signs = pred_masks.clone()
#     pred_masks_signs[pred_masks_signs>=xxxxx] = 1
#     mask_position_sign = (pred_masks_signs == 1)
#     ones_tensor = torch.ones_like(pred_masks)
    
#     masked_imgs = torch.where(mask_position_sign, ones_tensor, images)
    
#     # Generate "FAKE" "HEALTHY" parts
#     gen_parts = generator(masked_imgs)
#     gen_images = torch.where(mask_position_sign, gen_parts, images)
#     # Composite masked_images with generated parts
#     composite_images = 
    
    
#     # If the results is correct, the entire images is supposed to be healthy in discriminator' eyes.
#     # Do judge
#     result = adversarial_loss(discriminator(gen_parts.detach()), healthy)
    
    
#     return result
    
    