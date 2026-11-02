import torch.nn as nn
import torch


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else: x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.dropout = nn.Dropout(p=p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling, p=0.3):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim, p=p) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim, p=0.3):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim, p=p)

    def forward(self, x):
        return self.channel_mixing(x)



class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim=512, channels_hidden_dim=48, fac_T=False, fac_C=True, sampling=2, norm_flag=True, p=0.3):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling, p=p) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim, p=p)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim, p=p) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y