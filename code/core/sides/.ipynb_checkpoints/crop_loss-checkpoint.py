import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
torch.autograd.set_detect_anomaly(True)
def random_block(img_shape, block_shape):
    """Randomly masks image"""
    y1, x1 = np.random.randint(0, img_shape - block_shape, 2)
    y2 = int(y1 + block_shape)
    x2 = int(x1 + block_shape)
    return [y1, x1, y2, x2]

def dice_loss(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def crop_loss(model, modifiers_gen, images, num_blocks):    
    assert images.shape[2] == images.shape[3], "SHAPE ERROR!"
    assert images.shape[2] > 0 and images.shape[3] > 0, "SIZE(<=0) ERROR!"
    img_shape = images.shape[2]
    crop_losses = 0
    blocks = [] 
    crop_images = images.clone()
    preds = model(crop_images)
    preds = modifiers_gen(
        preds
    ) 
    crop_preds = preds['mask']
    # print('-------START-------')
    # print(images.shape)
    # Randomly initialize coordinates of croppoed blocks
    for i in range(num_blocks):
        block_shape = np.random.randint(10, img_shape, 1)
        # print(block_shape)
        single_block = random_block(img_shape, block_shape)
        # print(single_block)
        blocks.append(single_block)
    # Transform methods for cropped blcok to match the previous size
    resize_transform = transforms.Resize((img_shape, img_shape))
    # Randomly crop some parts from images and masks
    for i in range(num_blocks):
        single_block = blocks[i]
        # 3 channel images B,C,H,W
        images_s = crop_images[:, :, single_block[0]:single_block[2], single_block[1]:single_block[3]]
        # print('-------------------')
        # print(images_s.shape)
        # print('-------------------')
        resized_images_s = resize_transform(images_s)
        preds_cropped = crop_preds[:, :, single_block[0]:single_block[2], single_block[1]:single_block[3]]
        resized_preds_cropped = resize_transform(preds_cropped)
        # preds_s = model(imges_s)[]
        with torch.no_grad():
            return_dict_s = model(resized_images_s)
            return_dict_s_modified = modifiers_gen(
                return_dict_s
            )  
        preds_s = return_dict_s_modified['mask']
        crop_loss = dice_loss(preds_s, resized_preds_cropped)
        crop_losses = crop_losses + crop_loss
    # print('-------END-------')
    # print(crop_losses/num_blocks)
    return crop_losses/num_blocks

# import torch.nn.functional as F

