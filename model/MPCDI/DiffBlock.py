import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.MPCDI.Layers import AdaptiveGCN, SpatialLearning, TemporalLearning
from model.MPCDI.TemporalLearning import TL
# 扩散，用于数据生成
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlock(nn.Module):

    def __init__(self, num_node, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        #
        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.sl = SpatialLearning(num_node, channels=channels, nheads=nheads)
        # self.tl = TemporalLearning(channels)

        self.tl = TL(channels, channels, channels)

        # self.GCN = AdaptiveGCN(channels)  # 直接获取channels
        self.norm1_local = nn.GroupNorm(4, channels)

        self.gate_block = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()  # 输出[0,1]范围的权重
        )

    def forward(self, x, cond_info, diffusion_emb, support, time_mask=None):
        B, channel, K, L = x.shape # [4, 64, 207, 12] 64是hidden_dim

        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb  # (B,channel,K,L)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)

        # print(cond_info.shape)
        cond_info = self.cond_projection(cond_info)  # (B,channel,K*L)

        cond_info = cond_info

        y_t = y + cond_info
        y_s = y_t
        y_t = self.tl(y_t, base_shape)

        y_s = self.sl(y_s, base_shape, support)
        # 门控

        combined = torch.cat([y_s, y_t], dim=1)

        # 生成动态权重图
        fusion_gate = self.gate_block(combined)

        y = fusion_gate * y_s + (1 - fusion_gate) * y_t
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)  # (B,channel,K,L)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)  # (B,channel,K,L)
        return (x + residual) / math.sqrt(2.0), skip


class Encoder(nn.Module):
    def __init__(self,
                 num_node,
                 layers,
                 side_dim,
                 channels,
                 diffusion_embedding_dim,
                 nheads):
        super(Encoder, self).__init__()

        self.res_layers = nn.ModuleList([ResidualBlock(num_node,
                                                       side_dim=side_dim,
                                                   channels=channels,
                                                   diffusion_embedding_dim=diffusion_embedding_dim,
                                                   nheads=nheads) for _ in range(layers)])

    def forward(self, x, cond_info, diffusion_emb, support, time_mask=None):
        skips = []
        for i, layer in enumerate(self.res_layers):
            x, skip = layer(x, cond_info, diffusion_emb, support, time_mask)
            skips.append(skip)
        skip_concat = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.res_layers))
        B, _, K, L = skip_concat.shape
        res = skip_concat.reshape(B, -1, K * L)

        return res

from model.MPCDI.MixerBlock import MixerBlock
# 修改以下编解码器网络：编码器使用时空Transformer，解码器使用独立通道
class Decoder(nn.Module):
    def __init__(self, num_node, channels):
        super(Decoder, self).__init__()
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)
        self.output_projection2 = Conv1d_with_init(channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.decoder_layers = nn.ModuleList([MixerBlock(tokens_dim=12, channels_dim=num_node, p=0.02) for _ in range(2)])

    # 直接使用Channel_Mixer作为解码器
    def forward(self, x_hidden, B, K, L):  # (B, channel, K*L) => (B, K, L)
        x = self.output_projection1(x_hidden)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L).permute(0, 2, 1)

        x_skip = x
        for block in self.decoder_layers:
            x = block(x)
        # self.decoder_layers
        x = x_skip + x
        return x.permute(0, 2, 1)



class DenoisingNetwork(nn.Module):

    def __init__(self,
                 num_node,
                 channels,
                 num_steps,
                 diffusion_embedding_dim,
                 seqlen,
                 side_dim,
                 nheads,
                 layers,
                 inputdim=2):
        super().__init__()
        self.channels = channels

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=diffusion_embedding_dim,
        )
        self.seqlen = seqlen

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.hour_forward_projection = Conv1d_with_init(self.seqlen, self.seqlen, 1)
        self.day_forward_projection = Conv1d_with_init(self.seqlen, self.seqlen, 1)
        self.week_forward_projection = Conv1d_with_init(self.seqlen, self.seqlen, 1)

        # 条件映射器
        self.hour_cond_projection = Conv1d_with_init(self.channels + 1, self.channels, 1)
        self.day_cond_projection = Conv1d_with_init(self.channels + 1, self.channels, 1)
        self.week_cond_projection = Conv1d_with_init(self.channels + 1, self.channels, 1)


        self.encoder = Encoder(num_node,
                             layers,
                             side_dim,
                             channels,
                             diffusion_embedding_dim,
                             nheads)
        self.hour_decoder = Decoder(num_node, self.channels)
        self.day_decoder = Decoder(num_node, self.channels)
        self.week_decoder = Decoder(num_node, self.channels)


        self.weight_hour = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_day = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_week = nn.Parameter(torch.ones(1) * 0.5)




    # total_input,
    # side_info,
    # reverse_total_input,
    # reverse_side_info,
    # noisy_data,  # 噪声
    # noisy_cond_info,  # 全0矩阵
    # X_pred,  # 未来
    # t,  # 随机噪声程度矩阵
    # 去噪网络
    def get_cond(self, B, K, L, x, input_X, projection, forward_noise_hidden, cond_projection):

        random_mask = torch.rand(B, 1, K, L).to(input_X.device)
        cond = (
            projection(input_X.permute(0, 2, 1))
            .permute(0, 2, 1)
            .unsqueeze(1)
        )  # B,1,K,L

        cond = cond * random_mask + x * (1 - random_mask)
        cond = cond.reshape(B, 1, K * L)

        forward_noise_hidden = torch.cat([forward_noise_hidden, cond], dim=1)
        forward_noise_hidden = cond_projection(forward_noise_hidden)

        return forward_noise_hidden




    def forward(
        self,
        x, # total_input噪声替换的X_Tilde
        hour_X_tensor,
        hour_side_info,
        day_X_tensor,
        day_side_info,
        week_X_tensor,
        week_side_info,
        cond_info, # side_info其余信息
        diffusion_step, # 扩散步长
        support,
        time_mask=None,
        is_train=True
    ):

        B, inputdim, K, L = x.shape # [4, 1, 207, 12]
        # 编码方法
        x_hidden = self.__embedding(x, B, K, L, inputdim)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # 这里可以嵌入更多信息，比如说周期行
        forward_noise_hidden = self.encoder(x_hidden, cond_info, diffusion_emb, support, time_mask)

        forward_noise_hour_hidden = self.get_cond(B, K, L, x, hour_X_tensor, self.hour_forward_projection,
                                                  forward_noise_hidden, self.hour_cond_projection)
        forward_noise_day_hidden = self.get_cond(B, K, L, x, day_X_tensor, self.day_forward_projection,
                                                  forward_noise_hidden, self.day_cond_projection)
        forward_noise_week_hidden = self.get_cond(B, K, L, x, week_X_tensor, self.week_forward_projection,
                                                  forward_noise_hidden, self.week_cond_projection)

        forward_hour_noise = self.hour_decoder(forward_noise_hour_hidden, B, K, L)
        forward_day_noise = self.day_decoder(forward_noise_day_hidden, B, K, L)
        forward_week_noise = self.week_decoder(forward_noise_week_hidden, B, K, L)

        return (forward_hour_noise * self.weight_hour
                + forward_day_noise * self.weight_day
                + forward_week_noise * self.weight_week)

    def impute(self,
               x,
               hour_X_tensor,
               hour_side_info,
               day_X_tensor,
               day_side_info,
               week_X_tensor,
               week_side_info,
               cond_info,
               diffusion_step,
               support):

        x_noise = self.forward(x,
                    hour_X_tensor,
                    hour_side_info,
                    day_X_tensor,
                    day_side_info,
                    week_X_tensor,
                    week_side_info,
                    cond_info,
                    diffusion_step,
                    support,
                    is_train=False)
        return x_noise

    # Private helper functions
    def __embedding(self, x, B, K, L, input_dim):
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        return x
