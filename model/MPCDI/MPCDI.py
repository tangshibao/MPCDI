import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.MPCDI.DiffBlock import DenoisingNetwork
from model.MPCDI.AdjUtils import compute_support_gwn
## 主框架，实现对比学习
class MPCDI_Base(nn.Module):
    def __init__(self,
                 target_dim,
                 time_emb,
                 feature_emb,
                 num_steps,
                 schedule,
                 beta_start,
                 beta_end,
                 channels,
                 diffusion_embedding_dim,
                 seq_len,
                 nheads,
                 layers,
                 adj,
                 device):
        super().__init__()
        self.adj = adj
        self.device = device
        self.seq_len = seq_len
        self.num_node = target_dim
        self.target_dim = target_dim  # target_dim = number of features

        self.emb_time_dim = time_emb
        self.emb_feature_dim = feature_emb

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        side_dim = self.emb_total_dim

        input_dim = 1
        self.diffmodel = DenoisingNetwork(self.num_node,
                                          channels,
                                          num_steps,
                                          diffusion_embedding_dim,
                                          seq_len,
                                          side_dim,
                                          nheads,
                                          layers,
                                          input_dim)

        # parameters for diffusion models
        self.num_steps = num_steps
        if schedule == "quad":
            self.beta = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif schedule == "linear":
            self.beta = np.linspace(
                beta_start, beta_end, self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )
        # ===== 动态图生成部分 ===============#

        self.support = compute_support_gwn(self.adj.cpu(), device=self.device)
        # node_num = 207
        self.nodevec1 = nn.Parameter(torch.randn(self.num_node, 10).to(self.device), requires_grad=True).to(self.device) # 出度
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_node).to(self.device), requires_grad=True).to(self.device) # 入度
        self.support.append([self.nodevec1, self.nodevec2])

    def caculate_loss(self, forward_noise, noise, target_mask):

        residual = (noise - forward_noise) * target_mask
        num_eval = target_mask.sum()
        noise_loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        torch.cuda.empty_cache()
        return noise_loss


    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def get_side_info(self, observed_tp, cond_mask): # 添加一些其他额外数据，包括：
        B, N, L = cond_mask.shape

        time_embed = self.time_embedding(
            observed_tp, self.emb_time_dim
        )  # (B, L, emb_time)
        time_embed = time_embed.unsqueeze(2).expand(
            -1, -1, N, -1
        )  # (B, L, K, emb_time)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb_feature)
        feature_embed = (
            feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        )  # (B, L, K, emb_feature)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B, L, K, emb_total)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb_total, K, L)

        side_info = torch.cat(
            [side_info, cond_mask.unsqueeze(1)], dim=1
        )  # (B, emb_total + 1, K, L)

        return side_info # (B, emb_total + 1, N, L): Batch_size, emb_dim + 1, Node_num, Length


    def set_input_to_diffmodel(
        self,
        X_Tilde,
        noisy_data,
        X_mask,
        reverse_X_mask,
    ):

        def get_noisy_total_input(X_Tilde, X_noisy, mask):
            # mask: 1为可见。0为不可见
            return (mask * X_Tilde).unsqueeze(1) + ((1 - mask) * X_noisy).unsqueeze(1) # 可见的地方保持不变，不可见的地方替换成随机噪声
        # 对掩码掩盖的地方进行加噪，其他地方不加
        total_input = get_noisy_total_input(X_Tilde, noisy_data, X_mask) # 不可见的地方是人工掩码和Nan
        reverse_total_input = get_noisy_total_input(X_Tilde, noisy_data, reverse_X_mask) # 不可见的地方仅仅是人工掩码
        return (
            total_input,
            reverse_total_input,
            X_Tilde.unsqueeze(1),
            noisy_data.unsqueeze(1),
        )


    def impute(self,
               batch,
               n_samples): # 插补的时候才是逐渐去噪，需要将这里改进为多粒度去噪

        (X_Tilde,
         X_mask,
         X_Tilde_mask,
         indicating_mask,
         hour_X_tensor,
         hour_side_info,
         day_X_tensor,
         day_side_info,
         week_X_tensor,
         week_side_info,
         observed_tp,
         side_info) = self.process_data(batch)
        cond_mask = X_mask # 这里直接赋值

        B, K, L = X_Tilde.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            current_sample = torch.randn_like(X_Tilde)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * X_Tilde).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = cond_obs + noisy_target
                predicted = self.diffmodel(
                    diff_input,
                    hour_X_tensor,
                    hour_side_info,
                    day_X_tensor,
                    day_side_info,
                    week_X_tensor,
                    week_side_info,
                    side_info,
                    torch.tensor([t]).to(self.device),
                    self.support,
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        eval_mask = X_Tilde_mask - X_mask

        samples = imputed_samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
        X_Tilde = X_Tilde.permute(0, 2, 1)  # (B,L,K)
        eval_mask = eval_mask.permute(0, 2, 1)  # (B,L,K)
        samples = samples.median(dim=1).values

        return samples, X_Tilde, eval_mask

    def forward(self,
                batch,
                is_train=True,
                set_t=-1):
        (X_Tilde,
        X_mask,
        X_Tilde_mask,
        indicating_mask,
        hour_X_tensor,
        hour_side_info,
        day_X_tensor,
        day_side_info,
        week_X_tensor,
        week_side_info,
        observed_tp,
        side_info) = self.process_data(batch)

        B, K, L = X_Tilde.shape
        if is_train:
            # 如果是训练阶段，则是随机t，范围 [0, self.num_steps]
            t = torch.randint(0, self.num_steps, [B]).to(self.device)  # 生成一个Batch_size 大小，值为[0, num_steps) 的一维张量
        else:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)

        # add noise to observed data
        current_alpha = self.alpha_torch[t]  # (B, 1, 1)
        noise = torch.randn_like(X_Tilde)

        # 这个就是前向加噪过程 Noise Sampling 过程
        noisy_data = (current_alpha ** 0.5) * X_Tilde + (1.0 - current_alpha) ** 0.5 * noise
        reverse_X_mask = (X_Tilde_mask * indicating_mask)
        (total_input,
         reverse_total_input,
         X_Tilde,
         noisy_data
        ) = self.set_input_to_diffmodel(
            X_Tilde=X_Tilde,
            noisy_data=noisy_data,
            X_mask=X_mask,
            reverse_X_mask=reverse_X_mask)

        forward_pred_noise = self.diffmodel(
            total_input,
            hour_X_tensor,
            hour_side_info,
            day_X_tensor,
            day_side_info,
            week_X_tensor,
            week_side_info,
            side_info,
            t,
            self.support,
            time_mask=X_mask,
            is_train=is_train
        )
        target_mask = X_Tilde_mask - X_mask
        return forward_pred_noise, noise, target_mask

class MPCDI(MPCDI_Base):
    def __init__(self, args):
        target_dim = args.node_num
        super(MPCDI, self).__init__(
            target_dim,
            args.time_emb,
            args.feature_emb,
            args.num_steps,
            args.schedule,
            args.beta_start,
            args.beta_end,
            args.channels,
            args.diffusion_embedding_dim,
            args.seq_len,
            args.nheads,
            args.layers,
            args.A,
            args.device
        )

    def process_data(self, batch, istrain=True):
        day_X_tensor, week_X_tensor = None, None # 初始化二者为None
        (X_tensor,
         mask_tensor,
         X_Tilde_tensor,
         X_Tilde_mask_tensor,
         indicating_mask_tensor,
         hour_X_tensor,
         hour_mask_tensor,
         day_X_tensor,
         day_mask_tensor,
         week_X_tensor,
         week_mask_tensor) = batch

        X_Tilde = X_Tilde_tensor.to(self.device)  # B,L,F, with original missing values
        X_mask = mask_tensor.to(self.device)  # original missing + artifically missing (0: missing, 1: observed)
        indicating_mask = indicating_mask_tensor.to(self.device)  # indicating mask
        X_Tilde_mask = X_Tilde_mask_tensor.to(self.device)

        if day_X_tensor is not None and week_X_tensor is not None:
            hour_X_tensor = hour_X_tensor.to(self.device)
            hour_mask_tensor = hour_mask_tensor.to(self.device)

            day_X_tensor = day_X_tensor.to(self.device)
            day_mask_tensor = day_mask_tensor.to(self.device)

            week_X_tensor = week_X_tensor.to(self.device)
            week_mask_tensor = week_mask_tensor.to(self.device)

        batch_size = X_Tilde.shape[0] # 获取Batch_size

        observed_tp = torch.from_numpy(np.tile(np.arange(self.seq_len), batch_size).reshape(batch_size, self.seq_len)).to(self.device)

        X_Tilde = X_Tilde.permute(0, 2, 1)  # B,F,L
        X_Tilde_mask = X_Tilde_mask.permute(0, 2, 1)
        X_mask = X_mask.permute(0, 2, 1)
        indicating_mask = indicating_mask.permute(0, 2, 1)

        if day_X_tensor is not None and week_X_tensor is not None:
            hour_X_tensor = hour_X_tensor.permute(0, 2, 1)
            hour_mask_tensor = hour_mask_tensor.permute(0, 2, 1)

            day_X_tensor = day_X_tensor.permute(0, 2, 1)
            day_mask_tensor = day_mask_tensor.permute(0, 2, 1)

            week_X_tensor = week_X_tensor.permute(0, 2, 1)
            week_mask_tensor = week_mask_tensor.permute(0, 2, 1)
        side_info = self.get_side_info(
            observed_tp, X_mask
        )

        hour_side_info = self.get_side_info(
            observed_tp, hour_mask_tensor
        )

        day_side_info = self.get_side_info(
            observed_tp, day_mask_tensor
        )

        week_side_info = self.get_side_info(
            observed_tp, week_mask_tensor
        )


        return (X_Tilde,
                X_mask,
                X_Tilde_mask,
                indicating_mask,
                hour_X_tensor,
                hour_side_info,
                day_X_tensor,
                day_side_info,
                week_X_tensor,
                week_side_info,
                observed_tp,
                side_info)






