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

        self.mapping = IntensityMapper3D(num_bins=10)
        self.premodel = ModelDir(channels)

        self.item = item

        self.last_conv = SimpleStake3D(in_channels=1, out_channels=1)


    def forward(self, x, b, A=None):

        mapping = self.mapping(x)

        pre = self.premodel(x)
        pre_final = pre + mapping
        shape = x.shape[2:]

        x_patches = extract_3d_patches_overlap(mapping, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]
        b_patches = extract_3d_patches_overlap(b, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]
        pre_patches = extract_3d_patches_overlap(pre_final, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]

        x_patches = x_patches.permute(0, 2, 1, 3, 4, 5)
        b_patches = b_patches.permute(0, 2, 1, 3, 4, 5)
        pre_patches = pre_patches.permute(0, 2, 1, 3, 4, 5)
        N = x_patches.shape[2]


        A_D = self.modelD(mapping,pre_final)
        A_D = extract_3d_patches_overlap(A_D, (self.d, self.w, self.h), stride=self.stride)  # [B, N, C, pd, ph, pw]
        A_D = A_D.permute(0, 2, 1, 3, 4, 5)
        A_D = F.softplus(A_D) + 1e-8


        if A is None:
            A = [[None for _ in range(N)] for _ in range(N)]
            for i in range(N):
                b_i = x_patches[:, :, i, :, :, :]
                A[i][i] = A_D[:, :, i, :, :, :]
                for j in range(i):
                    if j < i:
                        A[i][j] = torch.zeros_like(A[i][i])
                        A[j][i] = torch.zeros_like(A[i][i])
                        continue
                    x_j = pre_patches[:, :, j, :, :, :]
                    val = self.modelL(x_j, b_i)  
                    A[i][j] = val
                    A[j][i] = val 
                    
        for item in range(self.item):
            new_patches = []
            for i in range(N):
                b_i = b_patches[:, :, i, :, :, :]        
                A_ii = A[i][i]                          
                coupling_sum = 0.0
                for j in range(N):
                    if j < i-1 or j > i+1:
                        continue
                    if i == j:
                        continue
                    A_ij = A[i][j]                      
                    if j < i:
                        x_j = new_patches[j]             
                    else:
                        x_j = x_patches[:, :, j, :, :, :]  
                    coupling_sum = coupling_sum + A_ij * x_j
                x_new = (b_i - coupling_sum) / A_ii

                new_patches.append(x_new)

            x_new_cat = torch.stack(new_patches, dim=2)          
            x_new_cat = x_new_cat.permute(0, 2, 1, 3, 4, 5)  
            gauss_recon = reconstruct_from_patches_overlap(
                x_new_cat, shape, (self.d, self.w, self.h), stride=self.stride
            )
            #gauss_recon = self.gauss_conv(gauss_recon)             
            x_patches = extract_3d_patches_overlap(gauss_recon, (self.d, self.w, self.h),stride=self.stride)  # [B, N, C, pd, ph, pw]
            x_patches = x_patches.permute(0, 2, 1, 3, 4, 5) 

        b_recon = self.verify_solution(self.patch_size, x, A, self.stride)
        #b_recon = self.last_conv(b_recon) 
        return b_recon,gauss_recon,mapping,pre,A
    

    def verify_solution(self,patch_size, out, A, stride):
        # A = sharpen_A_with_LoG(
        # A,
        # patch_size=patch_size,
        # sigma=1.0,        # 控制边缘尺度（小 sigma → 细边缘，大 sigma → 粗边缘）
        # strength=10000.0,     # 锐化强度（从 1.0 开始试）
        # mix_ratio=0.3     # 保留 30% 原始 A 信息（若 A 是学习的）
        # )
        A = amplify_A_gradient(
            A,
            strength=0.06,
            mix_ratio=0.1
        )


        x_last = out  # Shape: [B, C, D, H, W]
        B, C, D, H, W = x_last.shape

        # Extract 3D patches with overlap
        x_patches = extract_3d_patches_overlap(
            x_last, patch_size, stride=stride
        )  # Shape: [B, num_patches, C, pd, ph, pw]
        
        # Permute to [B, C, num_patches, pd, ph, pw]
        x_patches = x_patches.permute(0, 2, 1, 3, 4, 5)
        N = x_patches.shape[2]  # Number of patches

        # Reshape x_patches for batched operations: [N, B, C, S] where S = pd*ph*pw
        pd, ph, pw = patch_size
        S = pd * ph * pw
        x_patches_flat = x_patches.permute(2, 0, 1, 3, 4, 5).reshape(N, B, C, S)

        # Build A_tensor from list of lists: [N, N, B, C, pd, ph, pw]
        A_tensor = torch.stack([
            torch.stack([A[i][j] for j in range(N)], dim=0)
            for i in range(N)
        ], dim=0).to(x_patches.device)

        # Apply softplus + epsilon to diagonal elements
        #diag_indices = torch.arange(N, device=x_patches.device)
        #A_tensor[diag_indices, diag_indices] = F.softplus(A_tensor[diag_indices, diag_indices]) + 1e-6

        # Flatten spatial dimensions of A_tensor: [N, N, B, C, S]
        A_tensor_flat = A_tensor.reshape(N, N, B, C, S)

        # Compute Ax = A * x using batched operations
        # Expand x_patches_flat to [1, N, B, C, S] for broadcasting
        x_expanded = x_patches_flat.unsqueeze(0)  # Shape: [1, N, B, C, S]
        # Element-wise multiply and sum over j dimension (dim=1)
        Ax_flat = (A_tensor_flat * x_expanded).sum(dim=1)  # Shape: [N, B, C, S]

        # Reshape back to patch format: [N, B, C, pd, ph, pw]
        Ax_patches_flat = Ax_flat.reshape(N, B, C, pd, ph, pw)
        # Permute to [B, C, N, pd, ph, pw]
        Ax_patches = Ax_patches_flat.permute(1, 2, 0, 3, 4, 5)
        # Final permute to match reconstruction input format: [B, N, C, pd, ph, pw]
        Ax_patches = Ax_patches.permute(0, 2, 1, 3, 4, 5)

        # Reconstruct volume from patches
        Ax_vol = reconstruct_from_patches_overlap(
            Ax_patches, (D, H, W), patch_size, stride=stride
        )
        return Ax_vol


import torch
import torch.nn.functional as F
import math

def create_3d_log_kernel(patch_size, sigma=1.0):
    """
    创建 3D 高斯-拉普拉斯 (LoG) 核，支持任意 patch_size（奇数或偶数）。
    输出形状: [pd, ph, pw]
    """
    pd, ph, pw = patch_size

    # 创建以 patch 中心为原点的坐标网格（支持偶数：中心在 voxel 之间）
    z = torch.linspace(-(pd // 2), (pd - 1) // 2, pd, dtype=torch.float32)
    y = torch.linspace(-(ph // 2), (ph - 1) // 2, ph, dtype=torch.float32)
    x = torch.linspace(-(pw // 2), (pw - 1) // 2, pw, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

    # 计算 r² = z² + y² + x²
    r2 = Z**2 + Y**2 + X**2

    # LoG 公式: (r² - 3σ²) / σ⁴ * exp(-r² / (2σ²))
    sigma2 = sigma ** 2
    sigma4 = sigma2 ** 2
    log_kernel = (r2 - 3 * sigma2) / sigma4 * torch.exp(-r2 / (2 * sigma2))

    # 可选：减去均值使核总和为 0（避免直流偏移）
    log_kernel = log_kernel - log_kernel.mean()

    return log_kernel


def sharpen_A_with_LoG(A, patch_size, sigma=1.0, strength=1.0, mix_ratio=0.0):
    """
    对 A 的对角块应用 LoG 锐化，支持任意 patch_size（奇/偶）。
    """
    # 移除对奇数的断言！
    # assert all(p % 2 == 1 for p in patch_size), "patch_size must be odd..."  # ← 删除这行

    A_sharp = [row[:] for row in A]  # 浅拷贝列表
    N = len(A)

    # 生成 LoG 核（现在支持偶数）
    log_kernel = create_3d_log_kernel(patch_size, sigma=sigma)  # [pd, ph, pw]
    log_kernel = log_kernel * strength
    device = A[0][0].device
    log_kernel = log_kernel.to(device)

    for i in range(N):
        orig = A[i][i]
        if orig.dim() == 3:
            new_block = (1 - mix_ratio) * log_kernel + mix_ratio * orig
        elif orig.dim() == 5:
            log_bc = log_kernel[None, None, :, :, :].expand_as(orig)
            new_block = (1 - mix_ratio) * log_bc + mix_ratio * orig
        else:
            raise ValueError(f"Unsupported A[i][i] shape: {orig.shape}")
        A_sharp[i][i] = new_block

    return A_sharp

# class Gauss3D(nn.Module):
#     def __init__(self, channels, patch_size,stride, item):
#         super().__init__()
#         channels = int(channels)
#         self.stride = stride
#         self.model = ModelDir(channels)

#         #self.last_conv = SimpleStake3D(in_channels=1, out_channels=1)


#     def forward(self, x, b, A=None):
#         A = self.model(b)
#         #out = self.last_conv(A*b)
#         out = A*x
#         return out,A


def amplify_A_gradient(
    A,
    strength=0.3,
    mix_ratio=0.3
):
    """
    通过 3D Laplacian 放大 A 的坡度（边界更陡）
    A[i][j]: [B, C, D, H, W]
    """

    # 3D Laplacian kernel（中心差分，尺寸不变）
    lap_kernel = torch.tensor(
        [[[
            [0,  0,  0],
            [0, -1,  0],
            [0,  0,  0]
        ], [
            [0, -1,  0],
            [-1,  6, -1],
            [0, -1,  0]
        ], [
            [0,  0,  0],
            [0, -1,  0],
            [0,  0,  0]
        ]]],
        dtype=torch.float32
    )  # shape [1,1,3,3,3]

    A_new = []

    for i in range(len(A)):
        row = []
        for j in range(len(A[i])):
            Aij = A[i][j]  # [B, C, D, H, W]
            B, C, _, _, _ = Aij.shape
            device = Aij.device

            # group conv：每个 channel 独立
            kernel = lap_kernel.to(device).repeat(C, 1, 1, 1, 1)

            # Laplacian（padding=1 → 尺寸严格不变）
            lap = F.conv3d(Aij, kernel, padding=1, groups=C)

            # 放大坡度（注意是 +lap，而不是替换）
            Aij_sharp = Aij + strength * lap

            # 与原始 A 融合，防止过冲
            Aij_final = mix_ratio * Aij + (1 - mix_ratio) * Aij_sharp

            row.append(Aij_final)

        A_new.append(row)

    return A_new