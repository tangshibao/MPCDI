import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SpatialLearning(nn.Module):
    def __init__(self, num_node, channels, nheads=8, dim=207, proj_t=64, order=2, include_self=True, is_adp=True, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpaDependLearning(num_node,
                                               channels,
                                               nheads=nheads,
                                               order=order,
                                               dim=num_node,
                                               include_self=include_self,
                                               is_adp=is_adp,
                                               proj_t=proj_t,
                                               is_cross=is_cross)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(self,
                 num_node,
                 channels,
                 nheads,
                 dim,
                 order,
                 include_self,
                 is_adp,
                 proj_t,
                 is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCNWithAttention(num_node, channels, order=order, include_self=include_self, is_adp=is_adp)
        self.attn = Attn_spa(dim=channels, seq_len=dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)  # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross and itp_y is not None:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_local + y_attn
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


class TemporalLearning(nn.Module): # 用于提取时间特征
    def __init__(self, channels, nheads=8, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None, time_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        # 这一部分y是具有高位特性的，其将1映射为隐藏状态64，如何将mask也同步映射，这部分就算完成了
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)
        if self.is_cross and itp_y is not None:
            itp_y = itp_y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v,  src_key_padding_mask=time_mask).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v, src_key_padding_mask=time_mask).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True, is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[-1][0]
            nodevec2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            support = support + [adp]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out

def Attn_tem(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MaskAttentionLayer(d_model, nhead)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src,attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.attention_weights = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # 初始化权重
        nn.init.xavier_uniform_(self.attention_weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj_matrix):
        # print(x.shape, adj_matrix.shape)
        """
        :param x: 输入特征矩阵 (Cx B x N)，其中 B 是批量大小，N 是节点数，F 是每个节点的特征维度
        :param adj_matrix: 邻接矩阵 (N x N)，用于表示图的结构
        :return: 加权传播后的输出特征 (B x N x F')
        """
        # 计算注意力得分
        # [48, 64, 207, 1]
        x = x.squeeze()
        att_scores = torch.matmul(x, self.attention_weights)  # B x N x F
        # print(att_scores.shape)
        att_scores = torch.matmul(att_scores.transpose(1, 2), att_scores)  # B x N x N
        att_scores = att_scores @ adj_matrix  # 应用图结构的掩码

        # 使用softmax归一化
        attention = F.softmax(att_scores, dim=-1)

        # 应用dropout
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # 加权传播
        out = torch.matmul(x, attention)
        out = out.unsqueeze(-1)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, hidden_dim=192, dropout=0.6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.projection = nn.Linear(in_channels, hidden_dim)

        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)

        self.linear_out = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x, adj_matrix):
        """x: (B, N, F), adj_matrix: (B, N, N)"""
        B, N, F, _ = x.shape
        x = x.squeeze()

        # 线性变换

        x = self.projection(x)
        q = self.linear_q(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.linear_k(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.linear_v(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力得分
        attn_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale.to(x.device)

        # 应用邻接矩阵掩码
        adj_expanded = adj_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(adj_expanded == 0, -1e9)

        # 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        # 输出变换
        out = self.linear_out(out)
        return out
class AdaptiveGCNWithAttention(nn.Module):
    def __init__(self, num_node, channels, order=2, include_self=True, is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        # 引入图注意力层
        self.gat_layer = GraphAttentionLayer(num_node, num_node)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape # [B, C, N, L]
        if K == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[-1][0]
            nodevec2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            support = support + [adp]

        for a in support:
            x1 = self.gat_layer(x, a).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.gat_layer(x1, a).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out
