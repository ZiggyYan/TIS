# from skimage import color
from hog_descriptor import hog
from skimage import data, exposure, io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

def check_activation(hog_feature):
    row = hog_feature.shape[0]
    col = hog_feature.shape[1]
    hog_judged = hog_feature.clone()
    for i in range(row):
        for j in range(col):
            if hog_feature[i][j] == 0:
                if i == 0:
                    if j == 0:
                        if hog_feature[i+1][j+1] == 1 or hog_feature[i+1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                    elif j == (col-1):
                        if hog_feature[i+1][j-1] == 1 or hog_feature[i+1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                    else:
                        if hog_feature[i+1][j-1] == 1 or hog_feature[i+1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                        if hog_feature[i+1][j+1] == 1 or hog_feature[i+1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                elif i == (row-1):
                    if j == 0:
                        if hog_feature[i-1][j+1] == 1 or hog_feature[i-1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                    elif j == (col-1):
                        if hog_feature[i-1][j-1] == 1 or hog_feature[i-1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                    else:
                        if hog_feature[i-1][j-1] == 1 or hog_feature[i-1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                        if hog_feature[i-1][j+1] == 1 or hog_feature[i-1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                else:
                    if j == 0:
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                        if hog_feature[i-1][j+1] == 1 or hog_feature[i-1][j+1] == 2:
                            continue
                        if hog_feature[i+1][j+1] == 1 or hog_feature[i+1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                    elif j == (col-1):
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                        if hog_feature[i-1][j-1] == 1 or hog_feature[i-1][j-1] == 2:
                            continue
                        if hog_feature[i+1][j-1] == 1 or hog_feature[i+1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                    else:
                        if hog_feature[i+1][j] == 1 or hog_feature[i+1][j] == 2:
                            continue
                        if hog_feature[i-1][j] == 1 or hog_feature[i-1][j] == 2:
                            continue
                        if hog_feature[i-1][j-1] == 1 or hog_feature[i-1][j-1] == 2:
                            continue
                        if hog_feature[i+1][j-1] == 1 or hog_feature[i+1][j-1] == 2:
                            continue
                        if hog_feature[i][j-1] == 1 or hog_feature[i][j-1] == 2:
                            continue
                        if hog_feature[i-1][j+1] == 1 or hog_feature[i-1][j+1] == 2:
                            continue
                        if hog_feature[i+1][j+1] == 1 or hog_feature[i+1][j+1] == 2:
                            continue
                        if hog_feature[i][j+1] == 1 or hog_feature[i][j+1] == 2:
                            continue
                # print('DEEEEEEEEEEEEEEEEACTIVATED')
                hog_judged[i][j] = 3
    return hog_judged

def hog_judge(hog_feature):
    row = hog_feature.shape[0]
    col = hog_feature.shape[1]
    # value to filter out useless background
    background_threshold = 0
    # scale to indicate the obvious direction in hog
    obvious_threshold = 5
    
    # Judge
    filtered_hog = torch.zeros((row, col))
    for i in range(row):
        for j in range(col):
            hog_values = hog_feature[i][j]
            left = hog_values[0]
            right = hog_values[1]
            if left<=background_threshold and right<=background_threshold:
                filtered_hog[i][j]=3
            else:
                if left>=right*obvious_threshold:
                    filtered_hog[i][j]=2
                elif right>=left*obvious_threshold:
                    filtered_hog[i][j]=1
                else:
                    filtered_hog[i][j]=0
    # Select out activated 3s
    activated_hog = check_activation(filtered_hog)
    return activated_hog
    
def find_regions(matrix):
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
    
    
# def find_regions(matrix):
#     if not matrix:
#         return []

#     rows = len(matrix)
#     cols = len(matrix[0])
#     regions = []
#     visited = [[False for _ in range(cols)] for _ in range(rows)]

#     # 定义8个方向的移动
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),          (0, 1),
#                   (1, -1),  (1, 0), (1, 1)]

#     def dfs(i, j, region):
#         if i < 0 or i >= rows or j < 0 or j >= cols or matrix[i][j] == 3 or visited[i][j]:
#             return
        
#         current_value = matrix[i][j]
#         visited[i][j] = True
#         region.append((i, j))

#         for dx, dy in directions:
#             x, y = i + dx, j + dy
#             if x < 0 or x >= rows or y < 0 or y >= cols:
#                 continue
#             neighbor_value = matrix[x][y]
            
#             # 0可以与0、1、2任意方向相连
#             if current_value == 0 and neighbor_value in {0, 1, 2}:
#                 dfs(x, y, region)
            
#             # 1可以与1按照左上和右下的方式相连，与0、2任意方向相连
#             elif current_value == 1:
#                 if neighbor_value == 1:
#                     if (dx == -1 and dy == -1) or (dx == 1 and dy == 1):
#                         dfs(x, y, region)
#                 elif neighbor_value in {0, 2}:
#                     dfs(x, y, region)
            
#             # 2可以与2按照右上和左下的方式相连，与0、1任意方向相连
#             elif current_value == 2:
#                 if neighbor_value == 2:
#                     if (dx == -1 and dy == 1) or (dx == 1 and dy == -1):
#                         dfs(x, y, region)
#                 elif neighbor_value in {0, 1}:
#                     dfs(x, y, region)

#     for i in range(rows):
#         for j in range(cols):
#             if matrix[i][j] != 3 and not visited[i][j]:
#                 region = []
#                 dfs(i, j, region)
#                 if len(region)>6:
#                     regions.append(region)

#     return regions



if __name__ == '__main__':
    adr = '../../Dataset/BUSB/images/000005.png'
    # adr = './hog/a.png'
    image_gray = cv2.imread(adr,cv2.IMREAD_GRAYSCALE)
    # gt_gray = cv2.imread(adr.replace('images','GT'),cv2.IMREAD_GRAYSCALE)
    # image_gray = np.power(image/float(np.max(image)), 1.5)
    # image = cv2.imread('hog/test.jpg')
    # Extract HOG features
    # print(image_gray.shape)
    hog_feature = hog(image_gray, orientations=2, pixels_per_cell=(5, 5),
                              cells_per_block=(1, 1), visualize=True, feature_vector=False)

    hog_tensor = torch.from_numpy(hog_feature)
    # print(hog_tensor.shape)
    # print(image_gray.shape)
    hog_judged = hog_judge(hog_tensor)
    hog_image = torch.zeros_like(torch.from_numpy(image_gray))
    # hog_image = torch.zeros_like(hog_judged)
    hog_regions = find_regions(hog_judged.tolist())
    print('BEFORE',hog_image.sum())
    for i, region in enumerate(hog_regions):
        print(f"Region {i+1}: {region}")
        for element in hog_regions[i]:
            pos_row = element[0]
            pos_col = element[1]
            hog_image[pos_row*12:(pos_row+1)*12,pos_col*12:(pos_col+1)*12] = 1
    print('AFTER',hog_image.sum())
    # hog_image[pos_row*12:(pos_row+1)*12][pos_col*12:(pos_col+1)*12] = 255
    # tensor = torch.rand((100, 100))  # 创建一个100x100的灰度图Tensor
    # for a in range(hog_image.shape[0]):
    #     print(hog_image[a])
    tensor = hog_image.mul(255).byte().numpy()  # 转换为uint8并获取numpy数组
    image = Image.fromarray(tensor, mode='L')  # mode='L'表示灰度图
    # new_size = (501, 440)
    # image = image.resize(new_size)
    image.save('gray_image.png')  # 保存图像
        
        
    
#     # 将tensor转换为numpy数组并转换为uint8类型
#     hog_image = hog_image.numpy().astype(np.uint8)

#     # 使用matplotlib显示图像
#     plt.imshow(hog_image, cmap='gray')  # 对于单通道图像使用灰度图cmap='gray'
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
  
