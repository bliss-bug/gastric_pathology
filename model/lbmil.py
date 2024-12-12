import math
from .rotary import apply_rotary_position_embeddings, Rotary2D
try:
    from xformers.ops import memory_efficient_attention
except:
    print('please install xformer')

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None



class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = q.view(1, N, 1, -1), k.view(1, N, 1, -1)
        if exists(freqs_cis):
            q, k = apply_rotary_position_embeddings(freqs_cis, q, k)
        q, k = q.view(1, N, self.num_heads, -1), k.view(1, N, self.num_heads, -1)
        attn_drop = self.attn_drop if self.training else 0
        if exists(bias):
            x = memory_efficient_attention(q, k, v, bias, p=attn_drop).reshape(B, N, C)
        else:
            x = memory_efficient_attention(q, k, v, p=attn_drop).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        attention = (q.permute(0,2,1,3) @ k.permute(0,2,3,1) / q.shape[-1]**0.5 + bias).softmax(dim=-1) if not self.training else None
        return x, attention



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class TransBlock(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=256, num_heads=4, attn_drop=0.05, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads,attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope=None, bias=None):
        temp = self.attn(self.norm(x), rope, bias)   # attn for spatial mixture
        # return x+temp[0], temp[1]
        x = x + self.drop_path(temp[0])
        x = x + self.drop_path(self.mlp(self.norm2(x))) # mlp for token mixture
        return x, temp[1]



class LearnableBiasMIL(nn.Module):
    def __init__(self, input_size, n_classes=2, feat_size=256, n_heads=4, n_blocks=1, table_size=[4,6,8,10]):
        super(LearnableBiasMIL, self).__init__()
        self.input_size = input_size
        self.feat_size = feat_size
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.table_size = table_size
        self._fc1 = nn.Sequential(nn.Linear(input_size, feat_size), nn.ReLU())

        self.layers = nn.ModuleList([TransBlock(dim=feat_size, num_heads=self.n_heads) for _ in range(n_blocks)])
        #self.global_layer = TransBlock(dim=feat_size, num_heads=self.n_heads)

        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)
        self.rotary = Rotary2D(dim=feat_size)

        assert n_heads == len(table_size)
        self.bias_table = nn.Parameter(torch.zeros(n_blocks, n_heads, max(table_size), max(table_size), dtype=torch.float))
        #nn.init.trunc_normal_(self.bias_table, std=0.02)


    def forward(self, x):
        '''
        :param x:  shape = N * (feat_size+2), 2 for x-y 2d-position
        :return:
        '''
        if self.n_blocks > 1 and x.shape[0] > 20000:
            x = x[:20000, :]
        h = x[:, :self.input_size].unsqueeze(0) # shape 1*N*input_size

        h = self._fc1(h)

        freqs_cis, bias = self.positional_embedding(x, use_bias=True)

        for i in range(self.n_blocks):
            h, attn = self.layers[i](h, freqs_cis, bias[i].unsqueeze(0) if exists(bias) else None)
        
        #h, attn = self.global_layer(h, freqs_cis, None)  # global attention
        self.saved_h = h
        if self.saved_h.requires_grad:
            self.saved_h.retain_grad()

        h = self.norm(h.mean(1))

        # ---->predict
        logits = self._fc2(h)  # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_hat, Y_prob, attn

    def positional_embedding(self, x, use_bias=True, use_rope=False):
        freqs_cis = None
        bias = None
        if use_rope or use_bias:
            abs_pos = x[::, -2:]
            x_pos, y_pos = abs_pos[:, 0], abs_pos[:, 1]
            x_pos, y_pos = torch.round(x_pos - x_pos.min()), torch.round(y_pos - y_pos.min())
            H, W = 500, 500
            selected_idx = (x_pos * W + y_pos).to(torch.int)

            if use_rope:
                pos_cached = self.rotary.forward(torch.tensor([H, W]).to(x.device))
                freqs_cis = pos_cached[selected_idx].to(x.device)
            if use_bias:
                x_dis, y_dis = torch.abs(x_pos.unsqueeze(1) - x_pos.unsqueeze(0)).int(), torch.abs(y_pos.unsqueeze(1) - y_pos.unsqueeze(0)).int()
                bias = torch.full((self.n_blocks, self.n_heads, x_dis.shape[0], x_dis.shape[1]), -100, dtype=torch.float).to(x.device)

                for i in range(self.n_heads):
                    valid_mask = (x_dis < self.table_size[i]) & (y_dis < self.table_size[i])
                    bias[:, i, valid_mask] = self.bias_table[:, i, x_dis[valid_mask], y_dis[valid_mask]]

                N = bias.shape[3]
                pad_num = (8 - N % 8) % 8
                padding_bias = torch.zeros(self.n_blocks, bias.shape[1], bias.shape[2], pad_num).to(x.device)
                bias = torch.cat([bias, padding_bias], dim=-1)
                bias = bias.contiguous()[:, :, :, :N]

        return freqs_cis, bias


if __name__ == "__main__":
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    data = torch.randn((8, 512+2)).cuda(3) # 512 is feature, 2 is coordinates
    model = LearnableBiasMIL(input_size=512, n_classes=2).cuda(3)
    results_dict = model(data)
    print(results_dict)
