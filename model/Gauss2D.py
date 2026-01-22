import torch
from torch import nn
from torch.nn import functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        # (1, 1280) -> (1, 1280)
        x = F.silu(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

            # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)B,hw,c
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)B,hw,c2

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output


import torch
import math


def patchify(x, num_patches_h, num_patches_w):
    B, C, H, W = x.shape
    assert H % num_patches_h == 0 and W % num_patches_w == 0, "H/W 必须能被划分"

    patch_h = H // num_patches_h
    patch_w = W // num_patches_w

    x = x.unfold(2, patch_h, patch_h)  # (B, C, num_patches_h, W, patch_h)
    x = x.unfold(3, patch_w, patch_w)  # (B, C, num_patches_h, num_patches_w, patch_h, patch_w)

    x = x.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_h, num_patches_w, C, patch_h, patch_w)

    x = x.flatten(1, 2)  # (B, patch_num, C, patch_h, patch_w)

    x = x.permute(1, 0, 2, 3, 4)  # (patch_num, B, C, patch_h, patch_w)

    return x


def unpatchify(patches, num_patches_h, num_patches_w):
    patch_num, B, C, patch_h, patch_w = patches.shape
    assert patch_num == num_patches_h * num_patches_w

    # 恢复空间结构 (B, Ph, Pw, C, h, w)
    patches = patches.permute(1, 0, 2, 3, 4)  # (B, patch_num, C, h, w)
    patches = patches.view(B, num_patches_h, num_patches_w, C, patch_h, patch_w)

    # 重组图像
    x = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, Ph, h, Pw, w)
    x = x.view(B, C, num_patches_h * patch_h, num_patches_w * patch_w)
    return x


class Gauss(nn.Module):
    def __init__(self, channels, patch_num):
        super().__init__()
        self.A = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(channels, channels)
                for _ in range(patch_num)
            ])
            for _ in range(patch_num)
        ])
        self.conv_input1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_input2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.patch_num_hw = math.isqrt(patch_num)
        self.path_num = patch_num

    def forward(self, x, b):
        x = patchify(x, self.patch_num_hw, self.patch_num_hw)
        b = patchify(b, self.patch_num_hw, self.patch_num_hw)

        X_new = torch.zeros_like(x)

        for i in range(self.path_num):
            A_i = self.A[i]
            x_i = x[i]
            b_i = b[i]

            sum_before = torch.zeros_like(x_i)
            sum_after = torch.zeros_like(x_i)

            for j in range(self.path_num):
                A_ij = A_i[j]
                if j == i:
                    continue
                if j < i:
                    x_j = X_new[j]
                    sum_before += A_ij(x_j)
                if j > i:
                    x_j = x[j]
                    sum_after += A_ij(x_j)

            Xnew = A_i[i](b_i - sum_before - sum_after)
            X_new[i] = Xnew
        x = unpatchify(X_new, self.patch_num_hw, self.patch_num_hw)
        return x









