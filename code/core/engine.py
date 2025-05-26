import math
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils_move import AvgLogDict

from sides.hog_loss import *
from sides.crop_loss import *
from sides.block_loss import *

import torch
import numpy as np

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

def tensor_to_numpy(tensor):
    # 确保tensor在CPU上且无梯度
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    assert len(tensor.shape) == 4, "Tensor is supposed to be in the shape of (B,C,H,W)"
    
    batch_size = tensor.shape[0]
    # 初始化一个足够大的numpy数组来保存所有结果
    arrays = np.zeros(tensor.shape, dtype=np.uint8)
    
    for i in range(batch_size):
        single_tensor = tensor[i]  # 获取第i个样本(C,H,W)
        
        # 归一化到[0,1]范围
        if single_tensor.max() > 1.0 or single_tensor.min() < 0.0:
            min_val = single_tensor.min()
            max_val = single_tensor.max()
            # 避免除零错误
            if (max_val - min_val) == 0:
                single_tensor = torch.zeros_like(single_tensor)
            else:
                single_tensor = (single_tensor - min_val) / (max_val - min_val)
        # 缩放到0-255并转为uint8
        arrays[i] = (single_tensor * 255).clamp(0, 255).to(torch.uint8).numpy()
    squeezed_arrays = np.squeeze(arrays)
    return squeezed_arrays


def train_one_epoch(
    args,
    data_loader,
    model,
    discriminator,
    optimizer,
    optimizer_d,
    epoch,
    modifiers_gen,
    modifiers_disc,
    losses,
    train_module,
    gen_keys=["composed"],
    fake_keys=["composed", "real_ae_shift_cp"],
    real_keys=["composed_noshift", "real_ae"],
    accum_iter=1,
    log_every=10,
):
    assert log_every > 1  # here one iteration is either segmenter or discriminator
    model.train(True)
    discriminator.train(True)
    # INITIALIZE
    generator_adr = "./code/GANs/model_weights/context_encoder/best_weights.pth"
    discriminator_adr = "./code/GANs/model_weights/discriminator/phase2-2_best_weights.pth"
    hog_fn = hog_loss()
    block_fn = block_loss(generator_pth_adr = generator_adr, discriminator_pth_adr = discriminator_adr, channels = 1)
    dl_len = len(data_loader)

    phases = [
        "discriminator",
        "segmenter",
    ]  # this is the order in which we train the models
    cur_phase_id = 1

    log_dict = AvgLogDict()

    pbar = tqdm(enumerate(data_loader), total=dl_len)

    # for data_iter_step, (samples, _) in pbar:
    for data_iter_step, (samples, samples_gray) in pbar:
        # print(samples.max())
        # print(samples.min())
        # print(samples_gray.max())
        # print(samples_gray.min())
        # samples_gray = tensor_to_numpy(samples_gray)
        # print(samples_gray.shape)
        # print(samples_gray.max())
        # print(samples_gray.min())
        global_iter = epoch * dl_len + data_iter_step
        if data_iter_step % accum_iter == 0:
            cur_phase_id = (cur_phase_id + 1) % len(phases)

        cur_phase = phases[cur_phase_id]

        if cur_phase == "discriminator":
            with torch.no_grad():
                return_dict = model(samples)
                # print(type(return_dict))
                # print(return_dict.keys())
                return_dict = modifiers_disc(
                    return_dict
                )  # apply modifiers to the output; should be a ComposeModifier
            if "real" in real_keys:
                return_dict["real"] = samples

            # Get the fake and real inputs for the discriminator
            fake_inputs = [return_dict[k] for k in fake_keys]
            real_inputs = [return_dict[k] for k in real_keys]

            # Run the discriminator
            fake_disc_outputs = {
                k: discriminator.run_D(fake_input)
                for k, fake_input in zip(fake_keys, fake_inputs)
            }
            real_disc_outputs = {
                k: discriminator.run_D(real_input)
                for k, real_input in zip(real_keys, real_inputs)
            }

            # Get the loss for the discriminator
            disc_losses, disc_log_dict = discriminator.loss_d(
                fake_disc_outputs, real_disc_outputs
            )
            loss = sum(disc_losses.values())

            disc_losses["loss_d"] = loss

            if global_iter // len(phases) % log_every == 0:
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in disc_losses.items()
                ]
                [
                    log_dict.__setitem__(
                        k, v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    )
                    for k, v in disc_log_dict.items()
                ]

        elif cur_phase == "segmenter":
            # PART FOR CROP LOSS [1]
            # loss_crop = crop_loss(model = model, modifiers_gen = modifiers_gen,
            #                       images = samples, num_blocks = 6)
            # print(loss_crop)
            # loss = loss + loss_crop
            # loss_dict["loss_crop"] = loss_crop
            
            return_dict = model(samples)
            return_dict = modifiers_gen(
                return_dict
            )  # apply modifiers to the output; should be a ComposeModifier

            # Compute mask-related losses
            scaled_loss_dict, loss_dict = losses(return_dict)

            # Adversarial loss
            gen_disc_inputs = [return_dict[k] for k in gen_keys]
            gen_disc_outputs = {
                k: discriminator.run_D(gen_disc_input)
                for k, gen_disc_input in zip(gen_keys, gen_disc_inputs)
            }
            gen_losses, gen_log_dict = discriminator.loss_g(gen_disc_outputs)
           
            loss_g = sum(gen_losses.values())
            gen_losses["loss_g"] = loss_g
            loss_dict.update(gen_losses)
            
            loss = sum(scaled_loss_dict.values()) + loss_g
            
            # # PART FOR CROP LOSS [2]
            # loss = loss + loss_crop.detach()
            # loss_dict["loss_crop"] = loss_crop
            
            
#             # PART FOR HOG LOSS
            samples_hog = tensor_to_numpy(samples_gray)
            pred_mask = tensor_to_numpy(return_dict['mask'])
            pred_list = [samples_hog[i] for i in range(samples_hog.shape[0])]
            image_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
            loss_hog = hog_fn.generate_loss(pred_list,image_list)
            loss = loss + loss_hog
            loss_dict["loss_hog"] = loss_hog
            
            
            
            # PART FOR BLOCK LOSS
#             loss_block = block_fn.generate_loss(return_dict['mask'], samples_gray)
#             loss = loss + loss_block
#             loss_dict["loss_block"] = loss_block
            
#             loss_dict["loss_seg"] = loss
            
            # print('---------------------------')
            # print('scaled_loss_dict:',scaled_loss_dict.keys())
            # print('loss_dict:',loss_dict.keys())
            # print('gen_losses:',gen_losses.keys())  
            # print('---------------------------')
            
            if global_iter // len(phases) % log_every == 0:
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in loss_dict.items()
                ]
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in gen_log_dict.items()
                ]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss/accum_iter
        train_module.backward(loss)
        if (data_iter_step + 1) % accum_iter == 0:
            if cur_phase == "segmenter":
                optimizer.step()
            elif cur_phase == "discriminator":
                optimizer_d.step()

            optimizer.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

        if train_module.is_global_zero:
            # Use your favorite logger to log losses from log_dict
            if (
                (global_iter // len(phases) % log_every == 0)
                and ("loss_g" in log_dict.dict_)
                and ("loss_d" in log_dict.dict_)
            ):
                pbar.set_description(
                    # f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}'
                    # f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}, LossCROP: {log_dict["loss_crop"]:.4f}'
                    f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}, Loss_HOG: {log_dict["loss_hog"]:.4f}'
                    # f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}, Loss_BLOCK: {log_dict["loss_block"]:.4f}'
                    # f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}, Loss_HOG:{log_dict["loss_hog"]:.4f},Loss_BLOCK: {log_dict["loss_block"]:.4f}'
                )
                
                log_dict = AvgLogDict()
