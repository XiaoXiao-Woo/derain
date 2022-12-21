## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import os

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from UDL.Basis.dist_utils import reduce_mean
from UDL.Basis.module import PatchMergeModule
from models.base_model import DerainModel
from torch import optim
from UDL.Basis.criterion_metrics import SetCriterion
from .common.loss import BCMSLoss

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class CheapLP(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(CheapLP, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * ratio

        self.vit_mlp = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        )

        self.dw_mlp = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.vit_mlp(x)
        x2 = self.dw_mlp(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, patch_size, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.project_in = CheapLP(dim, hidden_features * 2)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.project_out = CheapLP(hidden_features, dim)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, patch_size):
        super(Attention, self).__init__()
        self.__class__.__name__ = 'XCTEB'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.qkv = CheapLP(dim, dim*3, dw_size=3)
        # self.qkv_cheap = CheapLP(dim*3, dim*3, kernel_size=3)
        # self.project_out = CheapLP(dim, dim, dw_size=3)

    def forward(self, x):
        b, c, h, w = x.shape
        # q, k, v = self.qkv(x, chunk=3)
        # qkv = self.qkv_cheap(torch.cat([q, k, v], dim=1))
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, patch_size, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, patch_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(patch_size, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(CheapLP(n_feat, n_feat * 2, stride=2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Upsample, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.body = nn.Sequential(
                                  nn.Conv2d(n_feat, out_feat, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.body(x)


##########################################################################
##---------- Arch -----------------------
class DFTLX(PatchMergeModule):
    def __init__(self,
                 args,
                 patch_size = [64, 32, 16, 8],
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(DFTLX, self).__init__(bs_axis_merge=False)
        self.args = args
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.ModuleList([
            TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            TransformerBlock(patch_size=patch_size[3], dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3) + int(dim * 2 ** 2), int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.up3_2 = Upsample(int(dim * 2 ** 2) + int(dim * 2 ** 1), int(dim * 2 ** 1))  ## From Level 3 to Level 2
        self.up2_1 = Upsample(int(dim * 2 ** 1) + dim, int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.refinement = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[0], dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        for blk in self.encoder_level1:
            inp_enc_level1 = blk(inp_enc_level1)
        # out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(inp_enc_level1)
        for blk in self.encoder_level2:
            inp_enc_level2 = blk(inp_enc_level2)
        # out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(inp_enc_level2)
        for blk in self.encoder_level3:
            inp_enc_level3 = blk(inp_enc_level3)
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(inp_enc_level3)
        for blk in self.latent:
            inp_enc_level4 = blk(inp_enc_level4)
        # latent = self.latent(inp_enc_level4)

        out_dec_level3 = self.up4_3(inp_enc_level4, inp_enc_level3)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        out_dec_level2 = self.up3_2(out_dec_level3, inp_enc_level2)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)

        out_dec_level1 = self.up2_1(out_dec_level2, inp_enc_level1)
        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    # def train_step(self, data, *args, **kwargs):
    #     samples, gt = data['O'].cuda(), data['B'].cuda()
    #     samples = sub_mean(samples)
    #     gt_y = sub_mean(gt)
    #
    #     outputs = self(samples)
    #     loss = self.criterion(outputs, gt_y, *args, **kwargs)
    #
    #     pred = add_mean(outputs)
    #
    #     loss.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, gt))))
    #     loss.update(psnr=reduce_mean(self.psnr(pred, gt * 255.0, 4, 255.0)))
    #
    #     return outputs, loss

    # def eval_step(self, batch, saved_path):
    #
    #     metrics = {}
    #
    #     O, B = batch['O'].cuda(), batch['B'].cuda()
    #     samples = sub_mean(O)
    #     derain = self.forward_chop(samples)
    #     pred = quantize(add_mean(derain), 255)
    #     normalized = pred[0]
    #     tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    #
    #     imageio.imwrite(os.path.join(saved_path, ''.join([batch['filename'][0], '.png'])),
    #                     tensor_cpu.numpy())
    #
    #     with torch.no_grad():
    #         metrics.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, B))))
    #         metrics.update(psnr=reduce_mean(self.psnr(pred, B * 255.0, 4, 255.0)))
    #
    #     return metrics

    def train_step(self, *args, **kwargs):

        return self(*args, **kwargs)

    def val_step(self, *args, **kwargs):

        return self.forward_chop(*args, **kwargs)

class build_DFTLX(DerainModel, name='DFTLX'):
    def __call__(self, args):
        scheduler = None
        loss = nn.L1Loss(size_average=True).cuda()
        # loss = BCMSLoss().cuda()
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = DFTLX(args).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  ## optimizer 1: Adam
        # model.set_metrics(criterion)

        return model, criterion, optimizer, scheduler
