import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import sys
# from hog_descriptor import hog
from sides.hog_descriptor import hog

import torch.nn as nn
import torch.nn.functional as F
import torch

def tensor_to_image(tensor, color_map=None):
    """
    将数值为{0,1,2,3}的二维Tensor转为RGB图像
    Args:
        tensor: 二维Tensor (H, W), 取值范围{0,1,2,3}
        color_map: 自定义颜色映射，格式为{0: (R,G,B), 1: (R,G,B), ...}
                  默认为 {0:黑, 1:红, 2:绿, 3:蓝}
    Returns:
        PIL.Image: 可显示或保存的图像
        np.ndarray: (H, W, 3)的RGB数组
    """
    # 默认颜色映射（R, G, B格式）
    default_colors = {
        0: (255, 255, 255),      # 黑
        1: (255, 0, 0),    # 红
        2: (0, 255, 0),    # 绿
        3: (0, 0, 0)      # 蓝
    }
    color_map = default_colors
    
    # 检查输入合法性
    assert isinstance(tensor, torch.Tensor), "输入必须是torch.Tensor"
    assert tensor.dim() == 2, "输入必须是二维Tensor"
    unique_vals = torch.unique(tensor)
    # assert all(v in color_map for v in unique_vals), f"Tensor包含未定义的颜色值（支持的键: {list(color_map.keys())}）"
    
    # 转为numpy数组
    np_array = tensor.cpu().numpy().astype(np.uint8)
    
    # 创建RGB图像
    h, w = np_array.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in color_map.items():
        rgb_array[np_array == val] = color
    
    # 转为PIL.Image
    image = Image.fromarray(rgb_array)
    
    return image


class hog_loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon  # 防止除零的小常数
        self.pixels_per_cell_size = (4,4)
    def hog_judge(self, hog_feature):
        row = hog_feature.shape[0]
        col = hog_feature.shape[1]
        # value to filter out useless background
        background_threshold = 1.6
        # scale to indicate the obvious direction in hog
        # 2.5 or 1.5
        obvious_threshold = 2.5

        # Judge
        filtered_hog = torch.zeros((row, col))
        for i in range(row):
            for j in range(col):
                hog_values = hog_feature[i][j]
                left = hog_values[0]
                right = hog_values[1]
                if left<=background_threshold and right<=background_threshold:
                    filtered_hog[i][j] = 3
                    # filtered_hog[i][j]=3
                else:
                    if left>=right*obvious_threshold:
                        # filtered_hog[i][j]=2
                        filtered_hog[i][j]=2
                    elif right>=left*obvious_threshold:
                        filtered_hog[i][j]=1
                    else:
                        filtered_hog[i][j]=0
        # Select out activated 3s
        # activated_hog = check_activation(filtered_hog)
        # filtered_hog = filtered_hog
        # return activated_hog
        return filtered_hog
    

    
    def find_regions(self, matrix):
        if not matrix:
            return []

        rows = len(matrix)
        cols = len(matrix[0])
        regions = []
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        # 定义8个方向的移动
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        def dfs(i, j, region):
            if i < 0 or i >= rows or j < 0 or j >= cols or matrix[i][j] == 3 or visited[i][j]:
                return

            current_value = matrix[i][j]
            visited[i][j] = True
            region.append((i, j))

            for dx, dy in directions:
                x, y = i + dx, j + dy
                if x < 0 or x >= rows or y < 0 or y >= cols:
                    continue
                neighbor_value = matrix[x][y]

                # 0 只能与 1 或 2 连接，不能与 0 连接
                if current_value == 0 and neighbor_value in {1, 2}:
                    dfs(x, y, region)

                # 1可以与1按照左上和右下的方式相连，与0、2任意方向相连
                elif current_value == 1:
                    if neighbor_value == 1:
                        if (dx == -1 and dy == -1) or (dx == 1 and dy == 1):
                            dfs(x, y, region)
                    elif neighbor_value in {0, 2}:
                        dfs(x, y, region)

                # 2可以与2按照右上和左下的方式相连，与0、1任意方向相连
                elif current_value == 2:
                    if neighbor_value == 2:
                        if (dx == -1 and dy == 1) or (dx == 1 and dy == -1):
                            dfs(x, y, region)
                    elif neighbor_value in {0, 1}:
                        dfs(x, y, region)

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] != 3 and not visited[i][j]:
                    region = []
                    dfs(i, j, region)
                    if len(region) > 6:  # 只保留长度大于6的区域
                        regions.append(region)

        return regions

    def match_count(self, A, B):
        # 1. 创建掩码：标记A中所有1和2的位置
        mask = (A == 1) | (A == 2)
        
        # 2. 计算A中1和2的总数量
        total = torch.sum(mask).float()
        
        # 3. 定义成功条件：(A==B) 或 (B==0)
        success_mask = ((A == B) | (B == 0)) & mask
        # success_mask = (A == B) & mask
        
        # 4. 计算匹配成功的位置数
        correct_matches = torch.sum(success_mask).float()
        # 5. 计算匹配成功率
        success_rate = correct_matches / (total + self.epsilon)
        # print(correct_matches)
        # print(total)
        # 5. 返回损失：1 - 成功率（损失越低越好）
        return 1.0 - success_rate
    
    # Segmentation loss between predicted masks and masks generated with HOG images
    def generate_loss(self,pred_masks,images):
        if isinstance(pred_masks, list) and isinstance(images, list): 
            matched_losses = 0
            assert len(pred_masks) == len(images), "Num of images and masks should be the same!" 
            nums = len(pred_masks)
            for i in range(nums):
                pred_mask = pred_masks[i]
                image = images[i]
                # Generate hog tensor for pred_masks
                pred_hog_feature = hog(pred_mask, orientations=2, pixels_per_cell=self.pixels_per_cell_size,
                                      cells_per_block=(1, 1), visualize=True, feature_vector=False)
                pred_hog_tensor = torch.from_numpy(pred_hog_feature)
                pred_hog =  self.hog_judge(pred_hog_tensor)
                # Generate hog tensor for images
                img_hog_feature = hog(image, orientations=2, pixels_per_cell=self.pixels_per_cell_size,
                                          cells_per_block=(1, 1), visualize=True, feature_vector=False)
                img_hog_tensor = torch.from_numpy(img_hog_feature)
                img_hog =  self.hog_judge(img_hog_tensor)
                
                matched_loss = self.match_count(pred_hog, img_hog)
                # print("single:",matched_loss)
                matched_losses = matched_loss+matched_losses
            # print("total:",matched_losses)
            
            matched_losses = matched_losses/nums
            # print("avg:",matched_losses)
            return matched_losses
        elif isinstance(pred_masks, np.ndarray) and isinstance(images, np.ndarray):
            # Generate hog tensor for pred_masks
            pred_hog_feature = hog(pred_masks, orientations=2, pixels_per_cell=self.pixels_per_cell_size,
                                      cells_per_block=(1, 1), visualize=True, feature_vector=False)
            pred_hog_tensor = torch.from_numpy(pred_hog_feature)

            pred_hog =  self.hog_judge(pred_hog_tensor)

            # Generate hog tensor for images
            img_hog_feature = hog(images, orientations=2, pixels_per_cell=self.pixels_per_cell_size,
                                      cells_per_block=(1, 1), visualize=True, feature_vector=False)
            img_hog_tensor = torch.from_numpy(img_hog_feature)

            img_hog =  self.hog_judge(img_hog_tensor)
            # Here, the principle we use for loss calculation：
            # we believe HOG can encompass all potential lesion regions in a very tolerant manner. 
            # In other words, the HOG generated from the correct mask must be included within the HOG generated from the original image, 
            # even if the original image's HOG contains many other non-lesion parts.

            matched_loss = self.match_count(pred_hog, img_hog)
            return matched_loss
        else:
            assert False, "Masks and images should be either single tensors or a list made of many ndarrays." 
            return None

if __name__ == "__main__":
    import cv2
    loss_fn = hog_loss()
#     # Example1：Perfect Match
#     A = torch.tensor([[1, 2], [3, 0]])
#     B = torch.tensor([[1, 2], [3, 0]])
#     print(loss_fn.match_count(A, B))  # Output 0 

#     # Example2：Partially Match
#     A = torch.tensor([[1, 2], [2, 1]])
#     B = torch.tensor([[1, 3], [0, 1]])
#     print(loss_fn.match_count(A, B))  # output 0.5

#     # Example3：No Match
#     A = torch.tensor([[1, 2], [2, 1]])
#     B = torch.tensor([[0, 0], [0, 0]])
#     print(loss_fn.match_count(A, B))  # output 1

#     # Example4：No Match
#     A = torch.tensor([[0, 3], [0, 3]])
#     B = torch.tensor([[1, 2], [3, 0]])
#     print(loss_fn.match_count(A, B))  # output 1
    #定义文件名和内容
#     record_file = "./code/move-seg/utils/record4-4BACK1.txt"
#     content = "This is the record file for hog."
#     # content1 = "\nStart Time:"+str(datetime.now())

#     # 使用 'w' 模式写入（覆盖已有内容）
#     with open(record_file, "w", encoding="utf-8") as f:
#         f.write(content)
    
#     directory = './Dataset/BUSB/images/'
#     losses = 0
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         # 检查文件是否为PNG文件
#         if filename.lower().endswith('.png'):
#             image_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
#             mask_gray = cv2.imread(file_path.replace('images','GT'),cv2.IMREAD_GRAYSCALE)
#             hog_loss = loss_fn.generate_loss(mask_gray,image_gray)
#             losses = losses+hog_loss
#             print(hog_loss)
#             with open(record_file, "a", encoding="utf-8") as f:
#                 f.write('\n')
#                 f.write(str(hog_loss))
#     losses = losses/len(os.listdir(directory))
#     print('AVERAGE:',losses)
#     with open(record_file, "a", encoding="utf-8") as f:
#             f.write('\n')
#             f.write('AVERAGE:'+str(losses))



 
    # losses = 0
    image_gray1 = cv2.imread('./Dataset/BUSB/images/000005.png',cv2.IMREAD_GRAYSCALE)
    image_gray2 = cv2.imread('./Dataset/BUSB/images/000006.png',cv2.IMREAD_GRAYSCALE)
    mask_gray1 = cv2.imread('./Dataset/BUSB/images/000006.png'.replace('images','GT'),cv2.IMREAD_GRAYSCALE)
    mask_gray2 = cv2.imread('./Dataset/BUSB/images/000006.png'.replace('images','GT'),cv2.IMREAD_GRAYSCALE)
    print(image_gray1.shape)
    print(image_gray2.shape)
    print(mask_gray1.shape)
    print(mask_gray2.shape)
    
    print(image_gray1.max())
    print(image_gray1.min())
    print(mask_gray1.max())
    print(mask_gray1.min())
    hog_loss = loss_fn.generate_loss([mask_gray1,mask_gray2],[image_gray1,image_gray2])
    print('AVERAGE:',hog_loss)
                  
   