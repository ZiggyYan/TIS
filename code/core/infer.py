import os
import argparse
import numpy as np

from easydict import EasyDict
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F_tv
from PIL import Image
import shutil
from train import TrainerLite
import bilateral_solver
import fnmatch
device = torch.device("cuda")

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

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

class ImageFolderWPaths(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        size = np.array(sample.size).tolist()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset",  required=True)
    parser.add_argument("--bilateral", action="store_true")
    parser.add_argument("--size", type=int, default=448)  # size of the shorter side
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument(
        "--resize_to_square", action="store_true"
    )  # allows batching but loses aspect ratio
    parser.add_argument(
        "--bs", type=int, default=32
    )  # Set to 1 anyway unless resize_to_square
    parser.add_argument(
        "--out_dir", type=str, default=None
    )  # If None it will be put in the same folder as the model
    args = parser.parse_args()

    dice_open = True

    if not args.resize_to_square:
        transform = [
            transforms.Resize(args.size, max_size=args.max_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        transform = transforms.Compose(transform)
        args.bs = 1
    else:
        transform = [
            transforms.Resize((args.size, args.size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        transform = transforms.Compose(transform)

    db_folder = args.dataset
    imgset = ImageFolderWPaths(db_folder, transform=transform)
    all_paths_cat = [imgset.samples[idx][0] for idx in range(len(imgset))]
    dataloader = DataLoader(imgset, batch_size=args.bs, drop_last=False, shuffle=False)

    relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]
    
    models = []
    # 递归遍历目录
    for root, dirs, files in os.walk(args.model_path):
        for file in files:
            if file.endswith(".pth"):
                # 拼接完整路径
                full_path = os.path.join(root, file)
                models.append(full_path)
    models.sort()
    best_dice = 0
    best_adr = ""
    for model in models:
        MODEL_PATH = model
        ARGS_PATH = MODEL_PATH[:-4] + ".args"

        loaded = torch.load(ARGS_PATH)
        model_args = loaded["args"] 
    
    
        out_dir = args.out_dir
        if out_dir is None:
            db_name = db_folder.rstrip("/").split("/")[-1]
            size_suffix = f"_{args.size}"
            size_suffix += f"_maxsize{args.max_size}" if args.max_size is not None else ""
            size_suffix = "_square" if args.resize_to_square else (size_suffix + "_keepAR")
            out_dir = os.path.join(
                os.path.dirname(MODEL_PATH),
                "pred_"
                + db_name
                + size_suffix
                + "_"
                + os.path.basename(MODEL_PATH).split(".")[0],
            )

        os.makedirs(out_dir, exist_ok=True)

        outpaths = [os.path.join(out_dir, relpath) for relpath in relpaths]
        outpaths = [os.path.splitext(path)[0] + ".png" for path in outpaths]

        do_inference = False
        for outpath in outpaths:
            if ".ipynb_checkpoints" in outpath:
                continue
            if not os.path.exists(outpath):
                print(outpath)
                do_inference = True
                break

        if do_inference:
            print("LOADING MODEL")
            trainer = TrainerLite(
                model_args, None, strategy="dp", gpus=1, precision=model_args.precision
            )
            trainer.model = trainer.setup(trainer.model)
            ckpt = trainer.load(MODEL_PATH)

            trainer.model.load_state_dict(ckpt["model"], strict=False)

            segmenter = trainer.model.segmenter
            segmenter.eval()

            patch_size = (
                segmenter.feature_extractor.patch_size
                if segmenter.feature_extractor is not None
                else 16
            )
            all_masks = []
            all_paths = []
            with torch.no_grad():
                for data in tqdm(dataloader):
                    imgs, _, paths = data[:3]
                    if not args.resize_to_square:
                        h, w = torch.tensor(imgs.shape[-2:]).numpy()
                        imgs = F_tv.pad(
                            imgs,
                            padding_mode="reflect",
                            padding=(
                                0,
                                0,
                                (patch_size - w % patch_size) % patch_size,
                                (patch_size - h % patch_size) % patch_size,
                            ),
                        )  # pad for ViT

                    masks = segmenter(imgs.cuda().float())

                    if not args.resize_to_square:
                        masks = masks[:, :, :h, :w]

                    all_masks.append(masks.cpu())
                    all_paths.append(paths)

            all_masks_cat = []
            all_paths_cat = []
            for paths, masks in zip(all_paths, all_masks):
                all_paths_cat.extend(paths)
                all_masks_cat.extend([m.squeeze(0) for m in masks])

            relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]
            outpaths = [os.path.join(out_dir, relpath) for relpath in relpaths]
            outpaths = [os.path.splitext(path)[0] + ".png" for path in outpaths]

            outdirs = set([os.path.dirname(path) for path in outpaths])
            for out_dir_ in outdirs:
                if ".ipynb_checkpoints" in out_dir_:
                    continue
                print(out_dir_)
                os.makedirs(out_dir_, exist_ok=True)

            for mask, outpath in tqdm(
                zip(all_masks_cat, outpaths), total=len(all_masks_cat)
            ):
                if ".ipynb_checkpoints" in outpath:
                    continue
                img = Image.fromarray((mask.numpy() * 255).astype(np.uint8), "L")
                img.save(outpath)
                
        all_masks = []
        for outpath in tqdm(outpaths):
            if ".ipynb_checkpoints" in outpath:
                continue
            img = np.array(Image.open(outpath)).astype(np.float32) / 255
            all_masks.append(img)
        if dice_open:
            pred_root =out_dir+'/0'
            gt_root = './Dataset/BUSB/GT/'
            transform = transforms.Compose([
                transforms.ToTensor(),  # 将PIL图像转换为Tensor
            ])
            # 应用转换
            file_list = [f for f in os.listdir(gt_root) if os.path.isfile(os.path.join(gt_root, f)) and fnmatch.fnmatch(f.lower(), '*.png')]
            dice_values = 0
            for file in file_list:
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
            print('NOW:',dice_all)
            if dice_all <= best_dice:
                print('NO BEST DICE PERFORMANCE UPDATE:',best_dice)
                print('CURRENT BEST DICE PERFORMANCE IN:',best_adr)
                if os.path.isdir(out_dir):
                    shutil.rmtree(out_dir, ignore_errors=True)

            else:
                best_dice = dice_all
                if best_adr == "":
                    best_adr = out_dir
                else:
                    if os.path.isdir(best_adr):
                        shutil.rmtree(best_adr, ignore_errors=True)
                    best_adr = out_dir
                print('NEW BEST DICE PERFORMANCE:',best_dice)
                print('NEW BEST DICE PERFORMANCE IN:',out_dir)
        del all_masks
        del outpaths
        del file_list
        if args.bilateral:
            print("Computing bilateral solver")
            
            bil_args = EasyDict()
            bil_args.sigma_spatial = 16
            bil_args.sigma_luma = 16
            bil_args.sigma_chroma = 8

            relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]

            db_name_ = db_name + "-bilateral"
            out_dir_bilateral = out_dir + "_bilateral"
            os.makedirs(out_dir_bilateral, exist_ok=True)
            outpaths_bilateral = [
                os.path.join(out_dir_bilateral, relpath) for relpath in relpaths
            ]
            outpaths_bilateral = [
                os.path.splitext(path)[0] + ".png" for path in outpaths_bilateral
            ]

            bilateral_masks = []
            for path, mask, outpath in tqdm(
                zip(all_paths_cat, all_masks, outpaths_bilateral), total=len(all_masks)
            ):
                if os.path.exists(outpath):
                    continue
                output_solver, binary_solver = bilateral_solver.bilateral_solver_output(
                    path,
                    mask,
                    sigma_spatial=bil_args.sigma_spatial,
                    sigma_luma=bil_args.sigma_luma,
                    sigma_chroma=bil_args.sigma_chroma,
                )
                bilateral_masks.append(output_solver)

            outdirs = set([os.path.dirname(path) for path in outpaths_bilateral])
            for out_dir_ in outdirs:
                if ".ipynb_checkpoints" in out_dir_:
                    continue
                print(out_dir_)
                os.makedirs(out_dir_, exist_ok=True)

            for mask, outpath in tqdm(
                zip(bilateral_masks, outpaths_bilateral), total=len(bilateral_masks)
            ):
                if ".ipynb_checkpoints" in outpath:
                    continue
                if os.path.exists(outpath):
                    continue
                img = Image.fromarray((mask * 255).astype(np.uint8), "L")
                img.save(outpath)
            
            
                
                
        
    
