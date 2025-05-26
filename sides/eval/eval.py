import torch
import torch.nn.functional as F
import os
import cv2
import fnmatch
from PIL import Image
import torchvision.transforms as transforms


def dice_coeff(pred, gt):
    target = torch.zeros_like(gt)
    target[gt > 0.5] = 1
    target = gt
    #?
    preddd = torch.zeros_like(pred)
    preddd[pred > 0.4] = 1
    pred = preddd
    
    
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2)
    dice = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    avg_dice = dice.sum() / num
    return avg_dice


 
# 假设你有一些真实的图像和预测的图像
# y_true = torch.tensor(...)  # 真实标签
# y_pred = torch.tensor(...)  # 预测标签
 
# 计算损失
# loss = dice_loss(y_true, y_pred)
# autodl-tmp/unsupervised/code/move-seg/output_generator+hog/pred_original_448_keepAR_checkpoint-21
pred_root = './code/move-seg/output_hog^block_back=1.8/pred_original_448_keepAR_checkpoint-20/0'
gt_root = './Dataset/BUSB/GT/'
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
])
 
# 应用转换

file_list = [f for f in os.listdir(gt_root) if os.path.isfile(os.path.join(gt_root, f)) and fnmatch.fnmatch(f.lower(), '*.png')]
print('NUM OF IMAGES:',len(file_list))
dice_values = 0
for file in file_list:
    print('PROCESSING:',file)
    pred_dir = os.path.join(pred_root,file)
    gt_dir = os.path.join(gt_root,file)
    new_size = (224, 224)
    pred = Image.open(pred_dir).convert('L')
    gt = Image.open(gt_dir).convert('L')
    pred = pred.resize(new_size)
    gt = gt.resize(new_size)
    # 应用转换
    pred_tensor = transform(pred)
    gt_tensor = transform(gt)
    dice_value = dice_coeff(pred_tensor,gt_tensor)
    dice_values = dice_values+dice_value
dice_all = dice_values/len(file_list)
print('FINAL DICE VALUE:',dice_all)