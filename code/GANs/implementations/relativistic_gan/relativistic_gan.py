import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision as tv

# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=8000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=80, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
parser.add_argument("--save_frequency", type=int, default=100, help="save generated images every 5 epochs")
parser.add_argument("--scale_min", type=float, default=0.9)
parser.add_argument("--scale_max", type=float, default=1.0)
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
parser.set_defaults(pin_mem=True)
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_datasets(data_path, transform_train):
    """
    Get the datasets from the data_path
    Possible concatenation of datasets, separated by comma
    """
    data_paths = data_path.split(",")
    if len(data_paths) == 1:
        data_paths = [data_paths[0]]
    datasets = []
    for data_path in data_paths:
        dataset_train = tv.datasets.ImageFolder(data_path, transform=transform_train)
        datasets.append(dataset_train)
    if len(datasets) == 1:
        dataset_train = datasets[0]
    else:
        dataset_train = torch.utils.data.ConcatDataset(datasets)

    return dataset_train


def build_dataloaders(args, dataset_train):
    """
    Get the dataloaders from the dataset_train
    """
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_mem,
        num_workers=args.num_workers,
        drop_last=True,
    )
    return dataloader_train


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Configure data loader
os.makedirs("./Dataset/BUSB/generated/RELATIVSTIC_GAN/", exist_ok=True)
transform_train = transforms.Compose(
        [
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(
                opt.img_size, scale=(opt.scale_min, opt.scale_max), interpolation=3
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(num_output_channels=1),
        ]
    )
# transform_train = transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         )
dataset = build_datasets("./Dataset/BUSB/original", transform_train)
dataloader = build_dataloaders(opt,dataset)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        real_pred = discriminator(real_imgs).detach()
        fake_pred = discriminator(gen_imgs)

        if opt.rel_avg_gan:
            g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
        else:
            g_loss = adversarial_loss(fake_pred - real_pred, valid)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Predict validity
        real_pred = discriminator(real_imgs)
        fake_pred = discriminator(gen_imgs.detach())

        if opt.rel_avg_gan:
            real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
            fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
        else:
            real_loss = adversarial_loss(real_pred - fake_pred, valid)
            fake_loss = adversarial_loss(fake_pred - real_pred, fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if epoch%opt.save_frequency == 0:
            # for n in range(int(gen_imgs.shape[0]/4)):
            saved_name = "ep"+str(epoch)+"_"+str(i)
            save_image(gen_imgs.data[:16], "./Dataset/BUSB/generated/RELATIVSTIC_GAN/%s.png" % saved_name, nrow=4,normalize=True)
        