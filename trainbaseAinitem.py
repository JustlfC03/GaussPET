import time
from typing import List, Optional, Tuple, Union

import torch
import os
import sys

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import os


from model.VAE import VAE
from model.lastPIX import Gauss3D,IntensityMapper3D,ModelDir
#from model.lastPIX import verify_solution
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

import math
from typing import List, Optional, Tuple, Union

import torch


def save_tensor_to_nii(tensor, name, dir=None):
    # 设置保存路径的基目录
    base_path = ''
    if dir is not None:
        base_path = dir

    # 确保目录存在
    os.makedirs(base_path, exist_ok=True)

    # 将 tensor 移动到 CPU 并转为 numpy
    data = tensor.squeeze().cpu().numpy()

    datas = []
    # 检查 batch 维度是否存在
    if tensor.shape[0] == 1:
        # 单个样本的情况
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_path, name)
        nib.save(img, save_path)
    else:

        data = tensor[:, 1, :, :].cpu().numpy()
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_path, name)
        nib.save(img, save_path)

def calculate_psnr(gt: torch.Tensor, recon: torch.Tensor, max_val: float = 1.0):
    assert gt.shape == recon.shape, "gt 和 recon 的形状必须相同"

    mse = torch.mean((gt - recon) ** 2, dim=[1, 2, 3])  # 每张图的 MSE
    psnr = 10 * torch.log10(max_val ** 2 / mse)  # 每张图的 PSNR
    return psnr.mean().item()


def calculate_nmse_nmae(gt, recon):
    y = np.asarray(gt, dtype=np.float64)
    y_hat = np.asarray(recon, dtype=np.float64)

    if y.shape != y_hat.shape:
        raise ValueError(f"形状不匹配: gt {y.shape} vs recon {y_hat.shape}")

    y_flat = y.ravel()
    y_hat_flat = y_hat.ravel()

    numerator_nmse = np.sum((y_flat - y_hat_flat) ** 2)
    denominator_nmse = np.sum(y_flat ** 2)

    if denominator_nmse == 0:
        nmse = 0.0 if numerator_nmse == 0 else np.inf
    else:
        nmse = numerator_nmse / denominator_nmse

    numerator_nmae = np.sum(np.abs(y_flat - y_hat_flat))
    denominator_nmae = np.sum(np.abs(y_flat))

    if denominator_nmae == 0:
        nmae = 0.0 if numerator_nmae == 0 else np.inf
    else:
        nmae = numerator_nmae / denominator_nmae

    return nmse, nmae

def normalize(x):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min == 0:
        return x - x_min
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def trainVAE(device, resume, train_loader, val_loader, epochs=None, VAE_save_path=None, input_mode='mri',onlyDecoder=False,Decoder_para = None):
    model = VAE().to(device)

    Encoder = VAE().encoder.to(device)

    if VAE_save_path is not None and onlyDecoder:
        checkpoint = torch.load(VAE_save_path, map_location=device)

        encoder_state_dict = {k.replace('encoder.', ''): v
                              for k, v in checkpoint.items()
                              if k.startswith('encoder.')}

        Encoder.load_state_dict(encoder_state_dict)


    if onlyDecoder:
        model = VAE().decoder.to(device)
        VAE_save_path = Decoder_para

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_panr = 0.0


    if resume:
        if VAE_save_path is not None:
            print(f"Resuming from checkpoint: {VAE_save_path}")
            model.load_state_dict(torch.load(VAE_save_path, map_location=device))
        else:
            raise ValueError("resume=True but VAE_save_path is None. Please provide a valid checkpoint path.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loader_with_progress = tqdm(train_loader,
                                          desc=f'Epoch [{epoch + 1}/{epochs}]',
                                          leave=False)  # leave=False 表示结束后清除进度条
        for (MRI, PET, MCimages, wenli, mri_wenli, name) in train_loader_with_progress:
            MRI, PET, MCimages, wenli, mri_wenli = MRI.to(device), PET.to(device), MCimages.to(
                device), wenli.to(device), mri_wenli.to(device)
            with torch.no_grad():
                if input_mode == 'mri':
                    input_data = MRI
                    GT = MRI
                elif input_mode == 'pet':
                    input_data = PET
                    GT = PET
                else:
                    print(input_mode)
                    raise ValueError(f"Invalid input_mode: {input_mode}. Expected 'mri' or 'pet'.")
                if onlyDecoder:
                    input_data = Encoder(input_data)
                    input_data = normalize(input_data)
            optimizer.zero_grad()
            # mask = input_data != 0
            recon = model(input_data)
            # recon = recon * mask
            loss = F.l1_loss(recon, GT)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader_with_progress.set_postfix(loss=f"{loss.mean().item():.4f}")

        model.eval()
        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        val_loader_with_progress = tqdm(val_loader,
                                        desc=f'Epoch [{epoch + 1}/{epochs}] (Val)',
                                        leave=False)
        with torch.no_grad():
            for (MRI, PET, MCimages, wenli, mri_wenli, name) in val_loader_with_progress:
                MRI, PET = MRI.to(device), PET.to(device)

                if input_mode == 'mri':
                    input_data = MRI
                elif input_mode == 'pet':
                    input_data = PET
                else:
                    print(input_mode)
                    raise ValueError(f"Invalid input_mode: {input_mode}. Expected 'mri' or 'pet'.")
                if onlyDecoder:
                    input_data = Encoder(input_data)
                    input_data = normalize(input_data)

                recon = model(input_data)

                mask = PET != 0
                recon = mask * recon

                
                if onlyDecoder:
                    if input_mode == 'mri':
                        input_data = MRI
                    elif input_mode == 'pet':
                        input_data = PET
                gtpet_numpy = input_data.clone().cpu().numpy()
                out_numpy = recon.clone().cpu().numpy()
                psnr_scale = psnr(gtpet_numpy, out_numpy)
                ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                save_tensor_to_nii(recon, name[0], dir='')

                nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                nmse_all += nmse
                nmae_all += nmae

                ssim_all += ssim_scale
                psnr_all += psnr_scale

                val_loader_with_progress.set_postfix({'val_loss': f"{psnr_scale.item():.4f}"})
            if psnr_all > max_panr:
                max_panr = psnr_all
                torch.save(model.state_dict(), VAE_save_path)


def add_noise(x1, x2, max_noise_std=0.5, generator=None):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    device = x1.device
    choice = torch.randint(0, 2, (), generator=generator, device=device)  # scalar
    selected = x1 if choice == 0 else x2
    noise_std = torch.rand((), generator=generator, device=device) * max_noise_std  # scalar
    noise = torch.randn_like(selected) * noise_std
    return selected + noise


import torch
import torch.nn.functional as F


def diffuse_mix_random_blur(
        mri: torch.Tensor,
        pet: torch.Tensor,
        sigma_range=(0.0, 3.0),  # 随机模糊强度范围（各轴独立采样）
        blur_prob: float = 1.0,  # 以一定概率执行模糊，<1可加速
):
    """
    扩散式插值 + 独立随机模糊（高效版，GPU 友好）
    输入:
        mri, pet: (B, C, H, W, D)
        sigma_range: (min_sigma, max_sigma)，各轴独立均匀采样
        blur_prob: 执行模糊的概率（0~1），可用于提速
    输出:
        fused_blurred: (B, C, H, W, D)
        w: float, 插值权重，MRI 0 --- 1 PET
    说明:
        - 模糊强度与 w 无关；每次调用随机采样 σx, σy, σz
        - 使用三次可分离 1D depthwise conv3d，计算高效
    """
    assert mri.shape == pet.shape and mri.ndim == 5, "输入需为 (B,C,H,W,D) 且形状一致"
    device, dtype = mri.device, mri.dtype
    B, C, H, W, D = mri.shape

    # 1) 随机 w ∈ [0,1]
    w = 0.6 if torch.rand(()).item() > 0.33 else 0.0

    # 2) 线性插值（MRI→PET）
    fused = (1.0 - w) * mri + w * pet

    # 3) 随机决定是否模糊
    if torch.rand(()) > blur_prob or w == 0.0:
        return fused, w

    # 4) 为每个轴独立采样 sigma（与 w 无关）
    smin, smax = sigma_range
    if smax <= 0 or smax < smin:
        return fused, w

    sigmas = torch.empty(3, device=device).uniform_(smin, smax)  # [σx, σy, σz]
    # 若全部 σ 极小则直接返回
    if (sigmas < 1e-3).all():
        return fused, w

    # 生成 1D 高斯核
    def _gauss1d(sigma: torch.Tensor):
        # kernel_size 取 6σ 向上取奇数，限制上限避免超长卷积核
        k = int(max(3, min(65, 2 * round(float(3.0 * sigma)) + 1)))
        x = torch.arange(k, device=device, dtype=dtype) - k // 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / (g.sum() + 1e-8)
        return g  # (k,)

    # depthwise 3D 可分离卷积：沿 H、W、D 三个轴分别做 1D 卷积
    def _blur1d(x: torch.Tensor, kernel_1d: torch.Tensor, axis: int):
        """
        axis: 2->H, 3->W, 4->D
        权重形状: (C,1,k,1,1)/(C,1,1,k,1)/(C,1,1,1,k), groups=C
        """
        Cx = x.shape[1]
        k = kernel_1d.numel()
        if axis == 2:
            weight = kernel_1d.view(1, 1, k, 1, 1).repeat(Cx, 1, 1, 1, 1)
            pad = (k // 2, 0, 0)
        elif axis == 3:
            weight = kernel_1d.view(1, 1, 1, k, 1).repeat(Cx, 1, 1, 1, 1)
            pad = (0, k // 2, 0)
        elif axis == 4:
            weight = kernel_1d.view(1, 1, 1, 1, k).repeat(Cx, 1, 1, 1, 1)
            pad = (0, 0, k // 2)
        else:
            raise ValueError("axis 必须是 2/3/4 对应 H/W/D")
        return F.conv3d(x, weight, padding=pad, groups=Cx)

    x = fused
    # 仅对 sigma > 1e-3 的轴执行卷积
    if sigmas[0] > 1e-3:
        x = _blur1d(x, _gauss1d(sigmas[0]), axis=2)
    if sigmas[1] > 1e-3:
        x = _blur1d(x, _gauss1d(sigmas[1]), axis=3)
    if sigmas[2] > 1e-3:
        x = _blur1d(x, _gauss1d(sigmas[2]), axis=4)

    return x, w


def get_3d_sincos_pos_embed(shape, device='cpu'):
    """
    生成 3D 正余弦位置编码，形状与输入张量一致（用于直接相加）。

    Args:
        shape: tuple (D, H, W) —— 空间维度
        device: 张量所在设备

    Returns:
        pos_embed: Tensor of shape (1, 1, D, H, W)
    """
    D, H, W = shape

    # 创建位置索引
    z = torch.arange(D, device=device).float()
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()

    # 计算每个维度的频率（按 Transformer 方式）
    # 总通道数视为 1，但我们仍用标准频率公式（隐式通道=1）
    # 为保持编码多样性，我们将位置信息投影到一个“虚拟”高维再取平均/投影回标量

    # 频率基（假设隐式通道 dim=64，最后取平均得到标量）
    dim = 64  # 隐式编码维度（越大越精细，64 足够）
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))  # (dim//2,)

    # 对每个维度进行编码
    def encode(pos, inv_freq):
        # pos: (L,), inv_freq: (dim//2,)
        sinusoid_inp = torch.outer(pos, inv_freq)  # (L, dim//2)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)  # (L, dim)
        return emb

    z_emb = encode(z, inv_freq)  # (D, dim)
    y_emb = encode(y, inv_freq)  # (H, dim)
    x_emb = encode(x, inv_freq)  # (W, dim)

    # 广播相加：(D, 1, 1, dim) + (1, H, 1, dim) + (1, 1, W, dim) -> (D, H, W, dim)
    pos_emb = z_emb[:, None, None, :] + y_emb[None, :, None, :] + x_emb[None, None, :, :]

    # 投影回标量（取平均，或可选线性层；这里用平均以保持无参数）
    pos_emb = pos_emb.mean(dim=-1, keepdim=True)  # (D, H, W, 1)

    # 转为 (1, 1, D, H, W)
    pos_emb = pos_emb.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 1, D, H, W)
    return pos_emb


class PatchL1Loss3D(nn.Module):
    def __init__(self, patch_size=16, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, fake, real):
        B, C, D, H, W = fake.shape

        # 计算能提取多少个 patch（避免越界）
        D_out = (D - self.patch_size) // self.stride + 1
        H_out = (H - self.patch_size) // self.stride + 1
        W_out = (W - self.patch_size) // self.stride + 1

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            return F.l1_loss(fake, real)  # 回退到全局 L1

        # 用 unfold 展开成 patches
        # fake: [B, C, D, H, W] → 先 reshape 成 [B*C, 1, D, H, W] 以便 unfold
        fake_reshaped = fake.view(B * C, 1, D, H, W)
        real_reshaped = real.view(B * C, 1, D, H, W)

        # 使用 unfold 沿每个空间维度展开
        # 输出: [B*C, 1 * ps*ps*ps, N_patches]
        fake_patches = fake_reshaped.unfold(2, self.patch_size, self.stride) \
                                  .unfold(3, self.patch_size, self.stride) \
                                  .unfold(4, self.patch_size, self.stride)
        real_patches = real_reshaped.unfold(2, self.patch_size, self.stride) \
                                  .unfold(3, self.patch_size, self.stride) \
                                  .unfold(4, self.patch_size, self.stride)

        # 合并最后三个维度 (ps, ps, ps) → [B*C, ps³, N]
        ps = self.patch_size
        N = fake_patches.size(-1) * fake_patches.size(-2) * fake_patches.size(-3)
        fake_patches = fake_patches.contiguous().view(B * C, ps*ps*ps, -1)
        real_patches = real_patches.contiguous().view(B * C, ps*ps*ps, -1)

        # 计算每个 patch 的 L1 Loss
        # [B*C, ps³, N] → mean over ps³ → [B*C, N] → mean over N
        l1_per_patch = torch.mean(torch.abs(fake_patches - real_patches), dim=1)  # [B*C, N]
        loss = torch.mean(l1_per_patch)  # scalar

        return loss
    
def main_train(train_dataloader, val_dataloader, epochs, device, resume=True, only_test=False, save_nii=False,
                mohu=False,model_save_path=None):
    
    #the_path = model_save_path.replace('.pth', 'new.pth')

    # patch_size = (16,32,32)
    # stride = (16,32,32)
    patch_size = (32,64,64)
    stride = (32,64,64)
    model = Gauss3D(16,patch_size,stride, 3).to(device)
    pre_model = ModelDir(16).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

    if resume:
        if model_save_path is not None:
            model.load_state_dict(torch.load(model_save_path, map_location=device),
    strict=False)
            
    mapping = IntensityMapper3D(10).to(device)
    full_state_dict = torch.load(
    '',
    map_location=device
    )
    mapping_state_dict = {
        k[len("mapping."):]: v
        for k, v in full_state_dict.items()
        if k.startswith("mapping.")
    }
    mapping.load_state_dict(mapping_state_dict)
    mapping.eval()

    pre_model.load_state_dict(torch.load(''))
    pre_model.eval()

    pl1_loss_fn = PatchL1Loss3D(patch_size=16, stride=8)

    max_psnr = 0
    max_ssim = 0
    min_nmse = float('inf')
    min_nmae = float('inf')
    for epoch in range(epochs):
        if not only_test:
            # ---------- Training phase ----------
            train_loader_with_progress = tqdm(train_dataloader,
                                              desc=f'Epoch [{epoch + 1}/{epochs}]',
                                              leave=False)

            total_train_loss = 0.0
            num_train_batches = 0
            model.train()
            for (MRI, PET, name) in train_loader_with_progress:
                MRI, PET= MRI.to(device), PET.to(device)
                mask = PET != 0

                with torch.no_grad():
                    mappingPET = mapping(MRI)
                    pre_result = pre_model(MRI)
            
                b_input = MRI #mappingPET
                input_data = mappingPET#+pre_result #新方法
                GT = PET  #新方法
                if mohu:
                    with torch.no_grad():
                        input_data = diffuse_mix_random_blur(MRI, PET, sigma_range=(0.5, 1.5), blur_prob=1.0)[0]
                
                optimizer.zero_grad()

                out, b_recon= model(input_data, b_input)
                out = out * mask
                b_recon = b_recon * mask
                loss =  F.l1_loss(out, GT)*5+pl1_loss_fn(out, GT)*5# +
                loss += F.l1_loss(b_recon, b_input)*2

                total_train_loss += loss.item()
                num_train_batches += 1
                loss.backward()
                optimizer.step()
                train_loader_with_progress.set_postfix(loss=f"{loss.item():.4f}")
            avg_train_loss = total_train_loss / num_train_batches
            print(avg_train_loss)

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0
        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            model.eval()
            len_val = 0

            with torch.no_grad():
                for (MRI, PET, name) in val_loader_with_progress:
                    # if num_val_batches <= 59 and epoch < 20:
                    #     num_val_batches += 1
                    #     continue
                    MRI, PET = MRI.to(device), PET.to(device)
                    mask = PET != 0

                    mappingPET = mapping(MRI)
                    pre_result = pre_model(MRI)

                    b_input = MRI#mappingPET 
                    input_data =mappingPET#+pre_result#mappingPET
                                        
                    out,b_recon = model(input_data, b_input)
                    #out = model(MRI)
                    
                    #b_recon = b_recon * mask
                    out = out * mask
                 
                    gtpet_numpy = PET.clone().cpu().numpy()
                    out_numpy = out.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                    nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                    nmse_all += nmse
                    nmae_all += nmae

                    ssim_all += ssim_scale
                    psnr_all += psnr_scale

                    len_val += 1

                    val_loader_with_progress.set_postfix(
                        psnr=f"{psnr_scale:.4f}"
                    )
                    if save_nii:
                        save_tensor_to_nii(out, name[0],
                                           dir='')
                if not save_nii:
                    save_tensor_to_nii(out, f"epoch{epoch + 1}_val.nii.gz",
                                       dir='')
                    save_tensor_to_nii(mappingPET , f"epoch{epoch + 1}_PET.nii.gz",
                                         dir='')



            
            if max_psnr < psnr_all:
                max_psnr = psnr_all
                print(
                    f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
                if model_save_path is not None:
                    the_path = model_save_path.replace('.pth', 'new.pth')
                    torch.save(model.state_dict(), the_path)
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved to {model_save_path}")
            if max_ssim < ssim_all:
                max_ssim = ssim_all
                print(
                    f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")

            if nmse_all < min_nmse:
                min_nmse = nmse_all
                print(
                    f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
            if nmae_all < min_nmae:
                min_nmae = nmae_all
                print(
                    f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
            if epoch == epochs - 1:
                print(
                    f"Final model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
            




