from torch import nn
import torch
import math
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np


def softmax(src, index, dim, dim_size, margin: float = 0., num_stab=False):
    src_max = torch.clamp(scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0], min=0.)
    if num_stab:
        src = (src - src_max.index_select(dim=dim, index=index)).exp()
        out = src / (denom + (margin - src_max).exp()).index_select(dim, index)
    else:
        src = src.exp()
        denom = scatter_add(src, index, dim=dim, dim_size=dim_size)
    out = src / (denom).index_select(dim, index)

    return out


class SimpleAttention(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    """

    def __init__(self, q_dim, k_dim, v_dim, nhead=1, dropout=0.1):
        super(SimpleAttention, self).__init__()
        print("SimpleAttention")
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.nhead = 1
        self.aggr = lambda out, row, dim, x_size: scatter_add(out, row, dim=dim, dim_size=x_size)

    def forward(
            self,
            q: torch.tensor,
            k: torch.tensor,
            v: torch.tensor,
            self_indices: torch.tensor,
            neighbor_indices: torch.tensor):

        bs = q.size(0)
        # perform multi-head attention
        v = self._attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), self_indices, neighbor_indices, bs)
        # concatenate heads and put through final linear layer
        v = v.transpose(0, 1).contiguous().view(
            bs, self.v_dim)

        return v #, edge_index, edge_attr

    def _attention(self, q, k, v, self_indices, neighbor_indices, bs=None):
        scores = torch.matmul(
            q.index_select(1, self_indices).unsqueeze(dim=-2),
            k.index_select(1, neighbor_indices).unsqueeze(dim=-1))
        scores = scores.view(1, self_indices.shape[0], 1) / math.sqrt(self.k_dim)
        scores = softmax(scores, self_indices, 1, bs)
        out = scores * v.index_select(1, self_indices)  # H x e x hdim
        out = self.aggr(out, self_indices, 1, bs)  # H x bs x hdim
        if type(out) == tuple:
            out = out[0]
        return out


class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    """

    def __init__(self, q_dim, k_dim, v_dim, nhead, dropout=0.1):
        super(MultiHeadDotProduct, self).__init__()
        print("MultiHeadDotProduct")
        self.hdim_k = k_dim // nhead
        self.hdim_v = v_dim // nhead
        self.hdim_q = q_dim // nhead
        self.nhead = nhead
        self.aggr = lambda out, row, dim, x_size: scatter_add(out, row, dim=dim, dim_size=x_size)

        # FC Layers for input
        self.q_linear = nn.Linear(q_dim, q_dim)
        self.v_linear = nn.Linear(v_dim, v_dim)
        self.k_linear = nn.Linear(k_dim, k_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(v_dim, v_dim)

        self.reset_parameters()

    def forward(
            self,
            q: torch.tensor,
            k: torch.tensor,
            v: torch.tensor,
            self_indices: torch.tensor,
            neighbor_indices: torch.tensor):

        bs = q.size(0)

        # FC layer and split into heads --> h * bs * embed_dim
        k = self.k_linear(k).view(bs, self.nhead, self.hdim_k).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.nhead, self.hdim_q).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.nhead, self.hdim_v).transpose(0, 1)
        
        # perform multi-head attention
        v = self._attention(q, k, v, self_indices, neighbor_indices, bs)

        # concatenate heads and put through final linear layer
        v = v.transpose(0, 1).contiguous().view(
            bs, self.nhead * self.hdim_v)
        v = self.out(v)

        return v #, edge_index, edge_attr

    def _attention(self, q, k, v, self_indices, neighbor_indices, bs=None):
        scores = torch.matmul(
            q.index_select(1, self_indices).unsqueeze(dim=-2),
            k.index_select(1, neighbor_indices).unsqueeze(dim=-1))
        scores = scores.view(self.nhead, self_indices.shape[0], 1) / math.sqrt(self.hdim_k)
        scores = softmax(scores, self_indices, 1, bs)
        scores = self.dropout(scores)

        out = scores * v.index_select(1, self_indices)  # H x e x hdim
        out = self.aggr(out, self_indices, 1, bs)  # H x bs x hdim
        if type(out) == tuple:
            out = out[0]
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)

        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)
        
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)


class DotAttentionLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, attention='multihead', d_hid=None, num_heads=1, res1=True, res2=True, mlp=True, norm1=True, norm2=True, num_layers=1):
        super(DotAttentionLayer, self).__init__()
        self.attention = attention
        if attention=='multihead':
            self.att = MultiHeadDotProduct(q_dim, k_dim, v_dim, num_heads).cuda()
        else:
            self.att = SimpleAttention(q_dim, k_dim, v_dim).cuda()
            res1, res2, mlp, norm1, norm = False, False, False, False, False

        self.res1 = res1
        self.res2 = res2
        
        d_hid = 1 * v_dim if d_hid is None else d_hid
        self.mlp = mlp

        self.linear1 = nn.Linear(v_dim, d_hid) if self.mlp else None
        self.linear2 = nn.Linear(d_hid, v_dim) if self.mlp else None

        self.norm1 = LayerNorm(v_dim) if norm1 else None
        self.norm2 = LayerNorm(v_dim) if norm1 else None

        self.act = F.relu
        self.num_layers = num_layers
    
    def forward(self, q, k, v, self_indices, neighbor_indices):
        for i in range(self.num_layers):
            v2  = self.att(q, k, v, self_indices, neighbor_indices)
            v = v + v2 if self.res1 else v2
            v = self.norm1(v) if self.norm1 is not None else v
            
            if self.mlp:
                v2 = self.linear2(self.act(self.linear1(v)))
            else:
                v2 = v
            v = v + v2 if self.res2 else v2
            v = self.norm2(v) if self.norm2 is not None else v
        
        return v

    def reset_parameters(self):
        if self.mlp:
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.constant_(self.linear1.bias, 0.)

            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.constant_(self.linear2.bias, 0.)
        
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()

        if self.attention == 'multihead':
            self.att.reset_parameters()


class LayerNorm(nn.Module):
    def __init__(self, norm_shape=None, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.affine = affine

        if isinstance(norm_shape, int):
            norm_shape = (norm_shape,)
        self.norm_shape = norm_shape

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*self.norm_shape))
            self.bias = nn.Parameter(torch.Tensor(*self.norm_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        init_shape = [x.shape[i] for i in range(len(x.shape))]

        dims = len(self.norm_shape)
        shape = [x.shape[i] for i in range(len(x.shape) - dims)] + [
            int(np.prod(list(self.norm_shape)))]
        x = x.view(shape)

        x = (x - x.mean(dim=-1, keepdim=True)) / (
                    torch.sqrt(x.var(unbiased=False, dim=-1, keepdim=True) + self.eps))

        x = x.view(init_shape)

        if self.affine:
            x *= self.weight
            x += self.bias

        return x

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
