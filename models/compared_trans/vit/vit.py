# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from UDL.Basis.module import PatchMergeModule

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.):
        self.__class__.__name__ = 'MSA_BNC'
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# from UDLada_module import NewSwinBlock
class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        # self.layers = nn.ModuleList([
        #     NewSwinBlock(dim, 256, dim, num_heads, 8, qkv_bias=False, hybrid_ffn=False) for _ in range(depth)
        # ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # for blk in self.layers:
        #     x = blk(x)
        return x

class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size=3,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size,
                               padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ViT(PatchMergeModule):
    def __init__(self, in_channels, image_size, patch_size, dim, depth, heads,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = dim * patch_size ** 2
        hidden_dim = 4 * dim
        # hidden_dim = 1024
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
                ResBlock(dim, kernel_size=5),
                ResBlock(dim, kernel_size=5)
            ) for _ in range(1)
        ])

        self.to_patch_embedding = nn.Sequential(
            #1 3 (8 32) (8 32) -> 1 (8 8) (32 32 3)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, hidden_dim),
            # nn.Linear(hidden_dim, patch_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))#+1
        # self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, 2 * hidden_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, patch_dim)
        self.tokens2img = nn.Sequential(Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_size // patch_size, w=image_size // patch_size, p1=patch_size, p2=patch_size))
        self.tail = OutConv(dim, in_channels)

    def forward(self, img):

        x = img
        for blk in self.head:
            x = blk(x)
        # print(x.shape)
        x = self.to_patch_embedding(x)
        # print(x.shape)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding#[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.mlp_head(self.norm(x))

        x = self.tokens2img(x)


        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return img + self.tail(x)#self.mlp_head(x)

    def init_eval_obj(self, args):
        self.args = args


def test_net():
    from torchstat import stat
    x = torch.randn(1, 3, 64, 64).cuda()
    net = ViT(
        in_channels=3,
        image_size=64,
        patch_size=16,
        dim=256,
        depth=8,
        heads=4,
        dropout=0,
        emb_dropout=0
    ).cuda()

    # net = ViT(
    #     in_channels=3,
    #     image_size=64,
    #     patch_size=16,
    #     dim=256,
    #     depth=6,
    #     heads=4,
    #     dropout=0,
    #     emb_dropout=0
    # ).cuda()

    out = net(x)
    print(net)
    stat(net, [x.size()])
    print([x.shape for x in out])

if __name__ == "__main__":
    # v = ViT(
    #     in_channels=3,
    #     image_size=48,
    #     patch_size=16,
    #     dim=256,
    #     depth=6,
    #     heads=8,
    #     dropout=0,
    #     emb_dropout=0
    # ).cuda()
    #
    # # img = torch.randn(1, 3, 48, 48).cuda()
    #
    # # preds = v(img)  # (1, 1000)
    # # print(preds.shape)
    # print(v)

    test_net()