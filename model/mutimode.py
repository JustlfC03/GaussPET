import torch
from torch import nn
from torch.nn import functional as F
import math

from model.ModeAfuse import PlainConvUNet


# 32*64*64
class ModelD(nn.Module):
    def __init__(self, channels):
        super().__init__()
        d = channels
        self.model = PlainConvUNet(2, 6, (d, 2 * d, 4 * d, 8 * d,8 * d,8 * d), nn.Conv3d, 3, (1, 1, 1, 1,2,2), (2, 2, 2, 2,2,2), 1,
                                   (2, 2, 2,2,2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0)



    def forward(self, x, b):
        '''
        x:上一个猜测
        b:初始值
        '''
        x = torch.cat([x, b], 1)
        x = self.model(x)
        return x  # a_ij
    
class ModelDir(nn.Module):
    def __init__(self, channels):
        super().__init__()
        d = channels
        self.model = PlainConvUNet(1, 6, (d, 2 * d, 4 * d, 8 * d,8 * d,8 * d), nn.Conv3d, 3, (1, 1, 1, 1,2,2), (2, 2, 2, 2,2,2), 1,
                                   (2, 2, 2,2,2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0)

    def forward(self, x):
        return self.model(x) 
    
class ModelA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        d = channels
        self.model = PlainConvUNet(2, 4, (d, 2 * d, 4 * d, 8 * d), nn.Conv3d, 3, (1, 1,2,2), (2, 2,2,2), 1,
                                   (2,2,2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0)

    def forward(self, x, b):
        '''
        x:上一个猜测
        b:初始值
        '''
        x = torch.cat([x, b], 1)
        x = self.model(x)
        return x  # a_ij


import torch
import torch.nn.functional as F

def extract_3d_patches_overlap(x, patch_size, stride):
    """
    提取带重叠的3D patches，支持梯度传播。
    Args:
        x: [B, C, D, H, W]
        patch_size: int 或 (pd, ph, pw)
        stride: int 或 (sd, sh, sw)
    Returns:
        patches: [B, num_patches, C, pd, ph, pw]
    """
    if isinstance(patch_size, int):
        pd = ph = pw = patch_size
    else:
        pd, ph, pw = patch_size
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride
    #print(x.shape)
    B, C, D, H, W = x.shape

    nd = (D - pd) // sd + 1
    nh = (H - ph) // sh + 1
    nw = (W - pw) // sw + 1

    patches = x.as_strided(
        size=(B, C, nd, nh, nw, pd, ph, pw),
        stride=(
            x.stride(0),
            x.stride(1),
            x.stride(2) * sd,
            x.stride(3) * sh,
            x.stride(4) * sw,
            x.stride(2),
            x.stride(3),
            x.stride(4),
        )
    ).contiguous()

    patches = patches.view(B, C, nd * nh * nw, pd, ph, pw)
    patches = patches.permute(0, 2, 1, 3, 4, 5).contiguous()
    return patches
def reconstruct_from_patches_overlap(patches, original_size, patch_size, stride):
    """
    从带重叠patch重建3D体数据，自动加权融合重叠区域。
    Args:
        patches: [B, num_patches, C, pd, ph, pw]
        original_size: (D, H, W)
        patch_size: int 或 (pd, ph, pw)
        stride: int 或 (sd, sh, sw)
    Returns:
        x_recon: [B, C, D, H, W]
    """
    if isinstance(patch_size, int):
        pd = ph = pw = patch_size
    else:
        pd, ph, pw = patch_size
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride

    B, N, C, _, _, _ = patches.shape
    D, H, W = original_size

    nd = (D - pd) // sd + 1
    nh = (H - ph) // sh + 1
    nw = (W - pw) // sw + 1

    x = torch.zeros((B, C, D, H, W), device=patches.device)
    w = torch.zeros_like(x)

    idx = 0
    for d in range(nd):
        for i in range(nh):
            for j in range(nw):
                patch = patches[:, idx]
                ds, hs, ws = d * sd, i * sh, j * sw
                x[:, :, ds:ds+pd, hs:hs+ph, ws:ws+pw] += patch
                w[:, :, ds:ds+pd, hs:hs+ph, ws:ws+pw] += 1
                idx += 1

    x /= w.clamp_min(1e-6)
    return x



def muti(a, b):
    return a * b


class SimpleFusion3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=16):
        super(SimpleFusion3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    
class SimpleStake3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=16):
        super(SimpleStake3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class IntensityMapper3D(nn.Module):
    def __init__(self, num_bins=10):
        super().__init__()
        self.num_bins = num_bins
        self.weights = nn.Parameter(torch.ones(num_bins)) 

    def forward(self, x):
        bin_idx = (x * self.num_bins).long()              
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        w = self.weights[bin_idx]                         
        y = w * x                                         
        return y

class Gauss3D(nn.Module):
    def __init__(self, channels, patch_size,stride, item):
        super().__init__()
        channels = int(channels)
        self.patch_size = patch_size
        self.stride = stride
        self.d, self.w, self.h = patch_size
        self.stride = stride
        self.modelL = ModelA(channels)
        self.modelD = ModelD(channels)

        self.item = item

        self.last_conv = SimpleStake3D(in_channels=1, out_channels=1)


    def forward(self, x, b, A=None):
        shape = x.shape[2:]

        x_patches = extract_3d_patches_overlap(x, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]
        b_patches = extract_3d_patches_overlap(b, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]

        x_patches = x_patches.permute(0, 2, 1, 3, 4, 5)
        b_patches = b_patches.permute(0, 2, 1, 3, 4, 5)
        N = x_patches.shape[2]

        if A is None:
            A = [[None for _ in range(N)] for _ in range(N)]
            for i in range(N):
                x_i = b_patches[:, :, i, :, :, :]
                mapping_i = x_patches[:, :, i, :, :, :]
                A[i][i] = self.modelD(x_i, mapping_i)
                for j in range(i):
                    x_j = b_patches[:, :, j, :, :, :]
                    val = self.modelL(x_j, mapping_i)  
                    A[i][j] = val
                    A[j][i] = val 
                    
        for item in range(self.item):
            new_patches = []
            for i in range(N):
                b_i = b_patches[:, :, i, :, :, :]        
                A_ii = A[i][i]                          
                coupling_sum = 0.0
                for j in range(N):
                    
                    if i == j:
                        continue
                    A_ij = A[i][j]                      
                    if j < i:
                        x_j = new_patches[j]             
                    else:
                        x_j = x_patches[:, :, j, :, :, :]  
                    coupling_sum = coupling_sum + A_ij * x_j
                A_ii_pos = F.softplus(A_ii) + 1e-8 # 为了保持正定，去除之后无法收敛
                x_new = (b_i - coupling_sum) / A_ii_pos

                new_patches.append(x_new)

            x_new_cat = torch.stack(new_patches, dim=2)          
            x_new_cat = x_new_cat.permute(0, 2, 1, 3, 4, 5)  
            x_last = reconstruct_from_patches_overlap(
                x_new_cat, shape, (self.d, self.w, self.h), stride=self.stride
            )
            x_last = self.last_conv(x_last)             
            x_patches = extract_3d_patches_overlap(x_last, (self.d, self.w, self.h),stride=self.stride)  # [B, N, C, pd, ph, pw]
            x_patches = x_patches.permute(0, 2, 1, 3, 4, 5) 
            
        #x_last = self.last_conv(x_last) 
        return x_last, A

def verify_solution(patch_size,out, A,stride):
    x_last = out

    #mask = b!=0

    B, C, D, H, W = x_last.shape

    x_patches = extract_3d_patches_overlap(
        x_last, patch_size, stride=stride
    )  
    x_patches = x_patches.permute(0, 2, 1, 3, 4, 5)  # [B, C, N, pd, ph, pw]
    N = x_patches.shape[2]
    
    Ax_patches = torch.zeros_like(x_patches)  # [B, C, N, pd, ph, pw]
    for i in range(N):
        Ax_i = torch.zeros(B, C, patch_size[0], patch_size[1], patch_size[2], 
                            device=x_last.device)
        
        for j in range(N):
            x_j = x_patches[:, :, j, :, :, :] 
            A_ij = A[i][j] 
            if A_ij is None:
                continue
            if i == j:
                A_ij = F.softplus(A_ij) + 1e-6  
            Ax_i += A_ij * x_j
        
        Ax_patches[:, :, i, :, :, :] = Ax_i
    
    Ax_patches = Ax_patches.permute(0, 2, 1, 3, 4, 5)  
    Ax_vol = reconstruct_from_patches_overlap(
        Ax_patches, (D, H, W), patch_size, stride=stride
    )
    return Ax_vol




# class Gauss3D(nn.Module):
#     def __init__(self, channels, patch_size,stride, item):
#         super().__init__()
#         channels = int(channels)
#         self.stride = stride
#         self.model = ModelDir(channels)

#         self.last_conv = SimpleStake3D(in_channels=1, out_channels=1)


#     def forward(self, x, b, A=None):
#         A = self.model(x)
#         out = self.last_conv(A*b)
#         return out,A
