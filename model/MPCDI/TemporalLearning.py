import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
class TL(nn.Module):
    def __init__(self, in_features, hidden_features, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.icb = ICB(in_features, hidden_features)
        self.asb = Adaptive_Spectral_Block(dim)
        self.timeAttn = TimeAttnLayer(in_features,1)


    def forward(self, x, base_shape):
        # x_org = x
        x1 = self.timeAttn(x, base_shape)
        x2 = x.permute(0, 2, 1)
        x2 = self.asb(x2)
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 2, 1)
        x = x1 + x2
        return x



# 卷积模块
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02) # 初始化方法
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):

        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        adaptive_filter = True # True
        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x

class TimeAttnLayer(nn.Module):
    def __init__(self, d_model, layer_num):
        """
        :param d_model:
        :param layer_num: 堆叠的块的数量
        """
        super().__init__()
        self.time_attns = nn.ModuleList([
            TimeAttention_Block(d_model) for _ in range(layer_num)])

    def forward(self, src, base_shape):

        B, channel, K, L = base_shape
        src = src.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)  # 这一部分进行维度变换
        src = src.permute(2, 0, 1)

        output = src
        for mod in self.time_attns:
            output = mod(output)
            # print(output.shape)

        output = output.permute(1, 2, 0) # [12, 828, 64] → 828, 64, 12,
        output = output.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        # [828, 64, 12]  →  [4, 207, 64, 12 ]  →  [4, 207, 64, 12 ]

        return output

class TimeAttention_Block(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src):

        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # print("src.shape", src.shape)  # [12, 828, 64]
        return src





if __name__ == '__main__':
    ta = TimeAttention_Block(64)
    tl = TL(64, 128, 64)
    x = torch.randn(4, 64, 2484)
    res1 = ta(x, [4, 64, 207, 12])
    # print(res1.shape)
    res = tl(x)
    print(res.shape)  # torch.Size([12, 828, 64])