import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from math import sqrt
from masking import  ProbMask

class InfoEmb(nn.Module):
    def __init__(self, space_emb_dim, day_emb_dim, week_emb_dim, node_num, device, num_input):
        super(InfoEmb, self).__init__()
        self.spatial_emb = nn.Parameter(torch.empty(node_num, space_emb_dim))
        nn.init.xavier_uniform_(self.spatial_emb)

        self.daily_emb = nn.Parameter(torch.empty(288, day_emb_dim))
        nn.init.xavier_uniform_(self.daily_emb)

        self.weekly_emb = nn.Parameter(torch.empty(7, week_emb_dim))
        nn.init.xavier_uniform_(self.weekly_emb)

        self.node_num = node_num
        self.num_input = num_input
        self.space_emb_dim = space_emb_dim
        self.day_emb_dim = day_emb_dim
        self.week_emb_dim = week_emb_dim
        self.device = device

    def forward(self, x):
        day_indices = x[..., 1]
        week_indices = x[..., 2]

        spatial_expanded = self.spatial_emb.unsqueeze(0).unsqueeze(2).expand(
            x.shape[0], self.node_num, self.num_input, self.space_emb_dim).to(self.device)
        x = torch.cat((x[..., :1], spatial_expanded), dim=-1)

        daily_values = self.daily_emb[day_indices.long()]
        x = torch.cat((x[..., :(1 + self.space_emb_dim)], daily_values), dim=-1)

        weekly_values = self.weekly_emb[week_indices.long()]
        x = torch.cat((x[..., :(1 + self.space_emb_dim + self.day_emb_dim)], weekly_values), dim=-1)

        return x


class DSFIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DSFIN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, out_channels * 2, (1, kernel_size), padding=(0, padding))
        self.path1_conv_main = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.path1_conv_gate = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.path2_conv_main = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.path2_conv_gate = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.fusion_fc = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x):
        x_initial = self.initial_conv(x)
        channels_half = round(x_initial.shape[1] / 2)

        x_path1 = x_initial[:, :channels_half, :, :]
        x_path2 = x_initial[:, channels_half:, :, :]

        x_path1_out = self.path1_conv_main(x_path1) * torch.sigmoid(self.path1_conv_gate(x_path1))
        x_path2_out = self.path2_conv_main(x_path2) * torch.sigmoid(self.path2_conv_gate(x_path2))

        x_fused = torch.cat((x_path1_out, x_path2_out), dim=1)
        x_out = F.relu(x_fused + x_initial)

        x_out = x_out.permute(0, 2, 3, 1)
        x_out = self.fusion_fc(x_out)

        return x_out


class TASTGC(nn.Module):
    def __init__(self, in_channels, out_channels, node_num, device):
        super(TASTGC, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.weightMatrix = nn.Parameter(torch.zeros(node_num, node_num))
        self.weightMatrix2 = nn.Parameter(torch.zeros(node_num, node_num))
        self.device = device
        self.node_num = node_num
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, X, A, S, V):
        mask = torch.zeros(A.shape, dtype=torch.float32, device=self.device)
        mask[A != 0] = 1
        delta_t = 300
        V_mps = (V * 1000 / 3600).to(self.device)

        T = S / V_mps

        alpha = torch.clamp(1 - T / delta_t, min=0, max=1).to(self.device)
        alpha = alpha
        beta = 1 - alpha

        X = X.permute(1, 0, 2, 3)
        X_forward = torch.cat([X[:, :, 0:1, :], X[:, :, :-1, :]], dim=2)

        X.to(dtype=torch.float32, device=self.device)
        X_forward.to(dtype=torch.float32, device=self.device)

        M = torch.zeros((self.node_num, self.node_num * 2)).to(device=self.device)
        M[:self.node_num, :self.node_num] = (alpha + self.weightMatrix) * mask
        M[:self.node_num, self.node_num:] = (beta + self.weightMatrix2) * mask
        a1 = (alpha + self.weightMatrix) * mask
        b1 = (beta + self.weightMatrix2) * mask

        A_expand = torch.zeros((self.node_num, self.node_num * 2)).to(device=self.device)
        A_expand[:self.node_num, :self.node_num] = A
        A_expand[:self.node_num, self.node_num:] = A

        A = A_expand * M
        X_two = torch.cat((X, X_forward), dim=0)
        result = torch.einsum('ij,jkmn->ikmn', A, X_two)
        lfs = torch.einsum("ijlm->jilm", result)
        output = F.relu(torch.matmul(lfs, self.weight))
        output = output.permute(0, 3, 1, 2)
        return output


class TemporalPreprocess(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, padding=0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if (queries.dim() == 2):
            queries = queries.unsqueeze(0)
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        index_sample = torch.randint(L_K, (L_Q, sample_k))

        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)

        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.linear = nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        lhs, rhs = torch.chunk(x, chunks=2, dim=1)
        return lhs * self.sigmoid(rhs)


class GFS(nn.Module):
    def __init__(self, inchannel, outchannel, device):
        super(GFS, self).__init__()
        self.fc = nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1)).to(device)
        self.glu = GLU(outchannel).to(device)

    def forward(self, x):
        x = self.fc(x)
        x = self.glu(x)
        return x


class CongestionLearn(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.time_fusion = nn.Sequential(
            nn.Linear(num_input, 1),
            nn.ReLU()
        )
        self.sig = nn.Sigmoid()
    def forward(self, flow):
        flow = flow.reshape(flow.shape[0], flow.shape[1], -1)
        flow = self.time_fusion(flow)
        flow_mean = flow.mean(dim=0)
        flow_mean_exchange = flow_mean.permute(1, 0)
        flow_fuse = (flow_mean + flow_mean_exchange) / 2
        flow_fuse = self.sig(flow_fuse)
        return flow_fuse


class TASTGCN(nn.Module):
    def __init__(self, args, node_num):
        super(TASTGCN, self).__init__()
        self.device = args.device
        self.infoEmb = InfoEmb(args.space_emb_dim, args.day_emb_dim, args.week_emb_dim, node_num, args.device
                               , args.num_input)
        self.dsfin1 = DSFIN(in_channels=(1 + args.space_emb_dim + args.day_emb_dim + args.week_emb_dim), out_channels=64
                            , kernel_size=5, padding=2)
        self.dsfin2 = DSFIN(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(node_num)
        self.tastgc1 = TASTGC(in_channels=65, out_channels=32, node_num=node_num, device=args.device)
        self.dsfin3 = DSFIN(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.dsfin4 = DSFIN(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(node_num)
        self.tastgc2 = TASTGC(in_channels=65, out_channels=32, node_num=node_num, device=args.device)
        self.gfs = GFS(64, 64, args.device)
        self.lconv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 3), padding=(0, 1))
        self.cl = CongestionLearn(args.num_input)

        self.preAddress = TemporalPreprocess(12, 3)
        self.encoder1 = EncoderLayer(
            AttentionLayer(
                ProbAttention(False, attention_dropout=0.1, output_attention=False),
                args.d_model, n_heads=4), args.d_model, args.d_ff, 0.1, activation="relu")

        self.encoder2 = EncoderLayer(
            AttentionLayer(
                ProbAttention(False, attention_dropout=0.1, output_attention=False),
                args.d_model, n_heads=4), args.d_model, args.d_ff, 0.1, activation="relu")

        self.fully = nn.Linear(in_features=args.num_output * 64, out_features=args.num_output)

    def forward(self, X, A, S, V):
        X = X.to(self.device)
        flow_raw = X[..., :1]

        x_flow_norm, means, stdev = utils.get_normalized_flow(X)
        x_flow_norm = x_flow_norm.to(self.device)
        x_flow_processed = self.preAddress(x_flow_norm)
        x_flow_processed = x_flow_processed.unsqueeze(1)
        x_flow_conv = self.lconv(x_flow_processed)

        x_flow_part1 = x_flow_conv[:, :1, :, :].squeeze()
        x_flow_part2 = x_flow_conv[:, 1:, :, :].squeeze()

        x_long_feat1, attn1 = self.encoder1(x_flow_part1, attn_mask=None, tau=None,
                                            delta=None)
        x_long_feat2, attn2 = self.encoder2(x_flow_part2, attn_mask=None, tau=None,
                                            delta=None)

        x_flow_restored1 = x_long_feat1.permute(0, 2, 1) * stdev + means
        x_flow_restored2 = x_long_feat2.permute(0, 2, 1) * stdev + means

        X_emb = self.infoEmb(X)
        congestion_rate = self.cl(flow_raw)
        V_adj = V * (1 - torch.log(1 + congestion_rate))

        X_emb = X_emb.permute(0, 3, 1, 2)
        X_dsfin1 = self.dsfin1(X_emb)
        X_concat1 = utils.concatenate_feature(x_flow_restored1, X_dsfin1)
        X_tastgc1 = self.tastgc1(X_concat1, A, S, V_adj)
        X_dsfin2 = self.dsfin2(X_tastgc1)
        X_bn1 = self.batch_norm1(X_dsfin2)

        X_bn1 = X_bn1.permute(0, 3, 1, 2)
        X_dsfin3 = self.dsfin3(X_bn1)
        X_concat2 = utils.concatenate_feature(x_flow_restored2, X_dsfin3)
        X_tastgc2 = self.tastgc2(X_concat2, A, S, V_adj)
        X_dsfin4 = self.dsfin4(X_tastgc2)
        X_bn2 = self.batch_norm2(X_dsfin4)

        X_bn2 = X_bn2.permute(0, 3, 1, 2)
        X_gfs = self.gfs(X_bn2)
        X_gfs = X_gfs.permute(0, 2, 3, 1)
        output = self.fully(X_gfs.reshape((X_gfs.shape[0], X_gfs.shape[1], -1)))
        return output
