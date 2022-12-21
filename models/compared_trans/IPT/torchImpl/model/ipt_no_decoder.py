# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import common
# import common
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from einops import rearrange
import copy


def make_model(args, parent=False):
    return ipt(args)


class ipt(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ipt, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        self.body = VisionTransformer(img_dim=args.patch_size, patch_dim=args.patch_dim, num_channels=n_feats,
                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim, num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                      num_queries=args.num_queries, dropout_rate=args.dropout_rate, mlp=args.no_mlp,
                                      pos_every=args.pos_every, no_pos=args.no_pos, no_norm=args.no_norm)

        self.tail = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ) for s in args.scale
        ])
        # self.apply(self._reset_parameters)

    def forward(self, x):
        #197,201->82.5560,86.5560
        x = self.sub_mean(x)#7.5560, 5.5560
        x = self.head[self.scale_idx](x)#-1.1079, 12.102
        #0.41843 0.19516
        # print("head_out:", x[0, 0, :2])
        res = self.body(x, self.scale_idx)
        res += x
        # print("body:", x[0, 0, :2])
        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)
        # print("out:", x[0, 0, 0])
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


    # def _reset_parameters(self, m):
    #
    #     if isinstance(m, nn.Conv2d):
    #         variance_scaling_initializer(m.weight)
    #         if hasattr(m, 'bias'):
    #             nn.init.constant_(m.bias, 0.)


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        # self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x,
            self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()
        if self.mlp == False:
            # -1.1079 12.1018
            # -0.4119, 0.0249 bias:0.25419 0.19826
            x = self.dropout_layer1(self.linear_encoding(x)) + x #transpose(0, 1)导致是576,bs,9C @ 9C,D

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            # x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            # x = self.decoder(x, x, query_pos=query_embed)
        else:
            # print(x.shape, pos.shape)
            #-0.5038, 7.4777, pos: -1.8190 -0.3637
            x = self.encoder(x + pos)
            # x = self.decoder(x, x, query_pos=query_embed)
        # print("enc_out:", x[:2, 0])
        # print(x[0, 0, 0])
        if self.mlp == False:
            x = self.mlp_head(x) + x
        # print("mlp_head:", x[:2, 0])
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x
        #256,8,576->8,64,48,48
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)
            # print("enc: ", output[:2, 0, :40])

        return output


class HAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
        #                                              nn.Linear(dim, dim * 2, bias=qkv_bias),
        #                                              nn.Dropout(attn_drop)) for i in range(num_patch)])

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))


        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)


        # self.apply(self._reset_parameters)
        # self._reset_parameters()

    def forward(self, x, H, W):
        # 256, 1, 576
        x = x.unsqueeze(1)
        # out = torch.zeros_like(x) #grad?
        # x = x.transpose(1, 2)
        # x = x.transpose(1, 0)
        B, p, N, C = x.shape # 256, 1, bs, 576
        # print(x.shape)
        # for i, (blk, sub_x) in enumerate(zip(self.H_module, x.chunk(p, dim=1))):
        #     q = blk[0]
        #     kv = blk[1]
        #     attn_drop = blk[2]
            # print(sub_x.shape, p)
            # 256, 1, bs, 576 -> 256, 1, bs, 12, 48 -> 256, 1, bsb * 12, 48
        # q = self.q_proj(x).reshape(B, p * N * self.num_heads, C // self.num_heads)  # .permute(0, 2, 1, 3)
        #
        # # 256, 1, bs, 576 -> 256, 1, bs, 576*2 -> 256, 1, bs*12, 576, ...
        # k, v = self.kv_proj(x).chunk(2, dim=-1)
        # k = k.reshape(B, p * N * self.num_heads, C // self.num_heads)
        # v = v.reshape(B, p * N * self.num_heads, C // self.num_heads)
        # .reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]
        #in_proj_weight -0.0647
        q, k, v = F.linear(x, self.in_proj_weight).chunk(3, dim=-1)
        # print(q.shape)
        # print("q", q[:2, 0, 0, :40])
        # print("in:", self.in_proj_weight[:2, :40])
        # print("x", x[:2, 0, 0, :40])

        # print("k", k[:2, 0, 0, :40])
        # print("v", v[:2, 0, 0, :40])
        #x:-0.0858 0.0217 0.2660
        #q:-0.4194 -1.0952 k:0.9805. -0.1406 v:0.65461, 0.54565
        q = q * self.scale#-0.0060531 -0.15808
        q = q.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        k = k.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        v = v.contiguous().view(B, p * N * self.num_heads, C // self.num_heads).transpose(0, 1)
        # 1 * bs*12， 256, 48 @ 1 * bs*12, 48, 256 -> 12*bs*1, 256, 256

        attn = torch.bmm(q, k.transpose(1, 2))#(q @ k.transpose(-2, -1))# * self.scale #？
        # 0.000050890 0.000024226
        # print("v:", v[:2, 0, :40])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print("attn_softmax:", attn[:2, 0, :40])
        # 12*bs, 256, 256@ bs*12, 256, 48-> bs*12, 256, 48
        #  bs*12, 256, 48-> 256, bs*12, 48 -> 256, bs, 576
        attn = torch.bmm(attn, v).transpose(0, 1).contiguous()
        attn = attn.view(B, p*N, C)
        #(attn @ v).transpose(0, 1).contiguous().view(B, p * N, C)
        # print("out_proj_weight", self.out_proj.weight)
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        # print("input_out_proj:", attn[:2, 0, :40])
        # print(sub_x.shape, out.shape)
        # out[:, i, ...] = sub_x

        return attn#out

    # def _reset_parameters(self):
    #     # if isinstance(m, nn.Linear):
    #     #     nn.init.xavier_uniform_(m.weight)
    #     #     if isinstance(m, nn.Linear) and m.bias is not None:
    #     #         nn.init.constant_(m.bias, 0.)
    #     # for blk in self.H_module:
    #     #     for m in blk:
    #     #         if hasattr(m, 'weight'):
    #     #         # nn.init.xavier_normal_(m.weight)
    #     #             nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #     #         if hasattr(m, 'bias') and m.bias is not None:
    #     #             nn.init.constant_(m.bias, 0.)
    #     nn.init.xavier_normal_(self.q_proj.weight)
    #     nn.init.xavier_normal_(self.kv_proj.weight)
    #     # nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
    #     # nn.init.kaiming_uniform_(self.kv_proj.weight, a=math.sqrt(5))
    #     if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
    #             nn.init.constant_(self.q_proj.bias, 0.)
    #     if hasattr(self.kv_proj, 'bias') and self.kv_proj.bias is not None:
    #         nn.init.constant_(self.kv_proj.bias, 0.)
    #     nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
    #     if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
    #             nn.init.constant_(self.out_proj.bias, 0.)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        # self.self_attn = HAttention(d_model, nhead, qkv_bias=False)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        # -2.3228 7.1141
        # print("src:", src[:2, 0, :40])
        # 256, bs, 576
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        # print(q.shape)
        # -0.0858 0.0217
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])#9.6205,  7.8124
        src2 = self.norm2(src)
        #src2_before: 0.1397 0.5305
        #after:-8.2498, -8.5746
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # print("linear2", self.linear2.weight[:2, :40])
        # print("linear1", self.linear1.weight[:2, :40])
        # print("src2_out:", src2[:2, 0, :40])
        src = src + self.dropout2(src2)
        return src

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        # self.self_attn = HAttention(d_model, nhead, qkv_bias=False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # self.multihead_attn = HAttention(d_model, nhead, qkv_bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        # tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
        #                            self.with_pos_embed(memory, pos),
        #                            memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
'''
decoder
- Epoch: [0]  [  0/900]  eta: 1:59:26  lr: 0.000100  grad_norm: 71.390587  Loss: 1.3943 (1.3943366)  psnr: 39.6943 (39.6942970)  time: 7.9627  data: 6.5101  max mem: 2626MB
- Epoch: [0]  [ 10/900]  eta: 0:13:45  lr: 0.000100  grad_norm: 57.637024  Loss: 2.1414 (1.3916275)  psnr: 36.1785 (40.1565912)  time: 0.9277  data: 0.5920  max mem: 4031MB
- Epoch: [0]  [ 20/900]  eta: 0:08:37  lr: 0.000100  grad_norm: 30.089693  Loss: 1.2767 (1.3634551)  psnr: 40.4983 (40.2040166)  time: 0.5879  data: 0.3102  max mem: 4031MB
- Epoch: [0]  [ 30/900]  eta: 0:06:48  lr: 0.000100  grad_norm: 31.871622  Loss: 1.7879 (1.3371066)  psnr: 36.9069 (40.2159932)  time: 0.4691  data: 0.2102  max mem: 4031MB
- Epoch: [0]  [ 40/900]  eta: 0:05:50  lr: 0.000100  grad_norm: 59.493645  Loss: 1.2511 (1.3636968)  psnr: 41.6513 (40.2932983)  time: 0.4078  data: 0.1591  max mem: 4031MB

std
- Epoch: [0]  [  0/900]  eta: 2:00:22  lr: 0.000100  grad_norm: 132.161758  Loss: 1.0172 (1.0171942)  psnr: 41.6087 (41.6087415)  time: 8.0252  data: 6.5462  max mem: 2545MB
- Epoch: [0]  [ 10/900]  eta: 0:14:11  lr: 0.000100  grad_norm: 54.195774  Loss: 1.2454 (1.2461083)  psnr: 40.4004 (41.3153618)  time: 0.9569  data: 0.5954  max mem: 3983MB
- Epoch: [0]  [ 20/900]  eta: 0:09:02  lr: 0.000100  grad_norm: 53.838734  Loss: 1.3910 (1.2832957)  psnr: 38.4496 (40.8889955)  time: 0.6167  data: 0.3121  max mem: 3983MB
- Epoch: [0]  [ 30/900]  eta: 0:07:11  lr: 0.000100  grad_norm: 49.673897  Loss: 0.9370 (1.2943243)  psnr: 44.3230 (40.5941960)  time: 0.4960  data: 0.2115  max mem: 3983MB
- Epoch: [0]  [ 40/900]  eta: 0:06:13  lr: 0.000100  grad_norm: 77.204575  Loss: 1.0302 (1.2662361)  psnr: 40.8896 (40.7342796)  time: 0.4343  data: 0.1600  max mem: 3983MB
'''

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
