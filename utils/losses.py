from info_nce import InfoNCE, info_nce
#               forward_pred_noise,
#             forward_hidden,
#             reverse_hidden,
#             negative_hidden
import torch.nn as nn
import torch
# target_mask = X_Tilde_mask - cond_mask
# residual = (noise - forward_pred_noise) * target_mask
# num_eval = target_mask.sum()
# loss_noise = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
#
# # predict_hidden => first_hidden(missing) + reverse_hidden(not missing)
# predict_hidden = forward_hidden.reshape(B, -1, K, L) * (
#         1 - cond_mask.unsqueeze(1)
# ) + reverse_hidden.reshape(B, -1, K, L) * (1 - reverse_cond_mask.unsqueeze(1))
# loss_cons = self.calc_intra_cons_loss(
#     negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1)
# )

class MixLoss(nn.Module):
    def __init__(self,
                 alpha=1,
                 beta=10):
        super().__init__()
        self.intra_cons_loss = InfoNCE() # 通过KL散度来约束对比学习效果
        self.alpha = alpha
        self.beta = beta
        # self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        # self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    # X_Tilde_mask, X_mask, noise, reverse_X_mask, forward_pred_noise, forward_hidden, reverse_hidden, negative_hidden

    #
    # forward_noise_hidden,
    # forward_reverse_noise_hidden,
    # forward_negative_noise_hidden,
    # forward_noise
    def forward(self, X_Tilde_mask, X_mask, noise, forward_noise):

        B, F, L = X_mask.shape
        # print(B, F, L)
        target_mask = X_Tilde_mask - X_mask
        residual = (noise - forward_noise) * target_mask
        num_eval = target_mask.sum()
        noise_loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        # predict_hidden => first_hidden(missing) + reverse_hidden(not missing)
        # predict_hidden = (forward_hidden.reshape(B, -1, F, L) * (1 - X_mask.unsqueeze(1)))
        # predict_hidden = (forward_hidden.reshape(B, -1, F, L) * (1 - X_mask.unsqueeze(1))
        #                   + reverse_hidden.reshape(B, -1, F, L)
        #                   * (1 - reverse_X_mask.unsqueeze(1)))
        # cons_loss = self.intra_cons_loss(negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1))
        # mix_loss = self.alpha * noise_loss + self.beta * cons_loss
        # cons_loss.detach()
        torch.cuda.empty_cache()
        return noise_loss


