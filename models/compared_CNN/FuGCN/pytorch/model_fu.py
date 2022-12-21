# This is a pytorch version code of this paper:
# X. Fu, Q. Qi, Z.-J. Zha, Y. Zhu, X. Ding. “Rain Streak Removal via Dual Graph Convolutional Network”, AAAI, 2021.
import os

import imageio
import torch
import torch.nn as nn
from torch import optim
from UDL.Basis.criterion_metrics import SetCriterion
from UDL.Basis.metrics import add_mean, sub_mean, PSNR_ycbcr, quantize
from common.cal_ssim import SSIM
from UDL.Basis.dist_utils import reduce_mean
from UDL.Basis.module import PatchMergeModule
from models.base_model import DerainModel
import torch.nn.functional as F
from einops import rearrange


class spatialGCN(nn.Module):

    def __init__(self, in_ch):
        super().__init__()

        self.__class__.__name__ = "sGCN"

        self.theta = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.nu = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.xi = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.F_sGCN = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1)

    def forward(self, input_tensor):
        inputs_shape = input_tensor.shape

        channels = inputs_shape[1] // 2

        theta = self.theta(input_tensor)
        # b, N, C
        # print(theta.shape)
        theta = theta.reshape([-1, channels, inputs_shape[2] * inputs_shape[3]])#.permute(0, 2, 1)
        # print(theta.shape)
        nu = self.nu(input_tensor)
        # b, N, C
        nu = nu.reshape([-1, channels, inputs_shape[2] * inputs_shape[3]])#.permute(0, 2, 1)
        nu_tmp = nu.reshape([-1, nu.shape[1] * nu.shape[2]])
        nu_tmp = F.softmax(nu_tmp, dim=-1)
        nu = nu_tmp.reshape([-1, nu.shape[1], nu.shape[2]])

        xi = self.xi(input_tensor)
        xi = xi.reshape([-1, channels, inputs_shape[2] * inputs_shape[3]])#.permute(0, 2, 1)
        xi_tmp = xi.reshape([-1, xi.shape[1] * xi.shape[2]])
        xi_tmp = F.softmax(xi_tmp, dim=-1)
        xi = xi_tmp.reshape([-1, xi.shape[1], xi.shape[2]])
        # (v @ k.T @ Q)W
        # 1, 36, 10000 @ 1, 10000, 36
        F_s = nu @ xi.transpose(-1, -2)
        # print(F_s.shape, theta.shape)
        # 1, 10000, 36 @ 1, 36, 36
        AF_s = theta.transpose(-1, -2) @ F_s
        # 1, N, C
        AF_s = AF_s.permute(0, 2, 1).reshape(shape=[-1, channels, inputs_shape[2], inputs_shape[3]])
        F_sGCN = self.F_sGCN(AF_s)

        return F_sGCN + input_tensor


class channelGCN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.__class__.__name__ = "cGCN"

        C = in_ch // 2
        N = in_ch // 4

        self.zeta = nn.Conv2d(in_ch, C, kernel_size=1)
        self.kappa = nn.Conv2d(in_ch, N, kernel_size=1)
        self.relu = nn.ReLU()

        self.F_c_mid = nn.Conv2d(C, C, kernel_size=1)
        self.F_c = nn.Conv2d(N, N, kernel_size=1)
        self.F_cGCN = nn.Conv2d(N, in_ch, kernel_size=1)

    def forward(self, input_tensor):
        inputs_shape = input_tensor.shape
        input_channel = inputs_shape[1]

        C = input_channel // 2
        N = input_channel // 4

        zeta = self.zeta(input_tensor)
        zeta = rearrange(zeta, 'b c h w -> b (h w) c')#zeta.reshape([inputs_shape[0], C, -1])
        # print(zeta.shape)
        # 1, C//2, N
        kappa = self.kappa(input_tensor)
        # print(kappa.shape)
        # 1, N, C//4
        # 得到通道间相似性
        kappa = rearrange(kappa, 'b c h w-> b c (h w)')#kappa.reshape(kappa, [inputs_shape[0], -1, N])
        # kappa = kappa.permute([0, 2, 1])
        # 1, C//2, N @ N,C//4
        # 1, 18, 10000 @ 1, 10000, 36
        F_c = kappa @ zeta

        F_c_tmp = F_c.reshape([-1, C * N])
        F_c_tmp = F.softmax(F_c_tmp, dim=-1)
        F_c = rearrange(F_c_tmp, 'b (c n) -> b c 1 n ', n=N, c=C)#F_c_tmp.reshape(-1, 1, N, C)

        F_c = self.relu(F_c + self.F_c_mid(F_c))
        F_c = rearrange(F_c, 'b c 1 n -> b n c 1')
        # b, N, C, 1
        F_c = self.F_c(F_c)
        F_c = rearrange(F_c, 'b n c 1 -> b c n')#F_c.reshape([inputs_shape[0], C, N])
        # 1, HW, C @ 1, C, N 通道加权
        # 1, 10000, 36 @ 1, 36, 18
        F_c = zeta @ F_c
        F_c = F_c.reshape([inputs_shape[0], inputs_shape[2], inputs_shape[3], N]).permute(0, 3, 1, 2)
        F_cGCN = self.F_cGCN(F_c)

        return F_cGCN + input_tensor


class basic(nn.Module):
    def __init__(self, in_ch=3, out_ch=72):
        super(basic, self).__init__()
        self.basic_fea0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.basic_fea1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        basic_fea0 = self.basic_fea0(input)
        basic_fea1 = self.basic_fea1(basic_fea0)
        return basic_fea1


class BasicUnit(nn.Module):
    def __init__(self, in_ch=72, out_ch=72):
        super(BasicUnit, self).__init__()

        self.F_sGCN = spatialGCN(in_ch)

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, dilation=3, padding=3)
        self.F_DCM = nn.Conv2d(in_channels=out_ch + out_ch + out_ch + out_ch, out_channels=out_ch, kernel_size=1,
                               stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.F_cGCN = channelGCN(in_ch=out_ch)

    # def forward(self, input):
    #     conv1 = self.relu(self.conv1(input))
    #     conv2 = self.relu(self.conv2(conv1))
    #     conv3 = self.relu(self.conv3(input)) #是input，1分2个，每个2次Conv
    #     conv4 = self.relu(self.conv4(conv3))
    #     tmp = torch.cat([conv1, conv2, conv3, conv4], dim=1)
    #     F_DCM = self.relu(self.F_DCM(tmp))
    #     return F_DCM + input

    def forward(self, input):

        F_sGCN = self.F_sGCN(input)
        conv1 = self.relu(self.conv1(F_sGCN))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(F_sGCN))
        conv4 = self.relu(self.conv4(conv3))
        tmp = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        F_DCM = self.relu(self.F_DCM(tmp))
        F_cGCN = self.F_cGCN(F_DCM)
        return F_cGCN + input



## backbone
class Net(PatchMergeModule):
    def __init__(self, args, channels=72):
        super(Net, self).__init__()

        self.args = args

        self.basic = basic()
        self.encode0 = BasicUnit()
        self.encode1 = BasicUnit()
        self.encode2 = BasicUnit()
        self.encode3 = BasicUnit()
        self.encode4 = BasicUnit()
        self.midle_layer = BasicUnit()
        self.deconv4 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder4 = BasicUnit()
        self.deconv3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder3 = BasicUnit()
        self.deconv2 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder2 = BasicUnit()
        self.deconv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder1 = BasicUnit()
        self.deconv0 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder0 = BasicUnit()
        self.decoding_end = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1,
                                      padding=1)
        self.res = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        basic = self.basic(input)
        encode0 = self.encode0(basic)
        encode1 = self.encode1(encode0)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        middle = self.midle_layer(encode4)
        decoder4 = self.deconv4(torch.cat([middle, encode4], dim=1))
        decoder4 = self.decoder4(decoder4)
        decoder3 = self.deconv3(torch.cat([decoder4, encode3], dim=1))
        decoder3 = self.decoder3(decoder3)
        decoder2 = self.deconv2(torch.cat([decoder3, encode2], dim=1))
        decoder2 = self.decoder2(decoder2)
        decoder1 = self.deconv1(torch.cat([decoder2, encode1], dim=1))
        decoder1 = self.decoder1(decoder1)
        decoder0 = self.deconv0(torch.cat([decoder1, encode0], dim=1))
        decoder0 = self.decoder0(decoder0)

        decoder_end = self.relu(self.decoding_end(torch.cat([decoder0, basic], dim=1)))
        res = self.res(decoder_end)

        return res + input

    def train_step(self, data, *args, **kwargs):
        samples, gt = data['O'].cuda(), data['B'].cuda()
        samples = sub_mean(samples)
        gt_y = sub_mean(gt)

        outputs = self(samples)
        loss = self.criterion(outputs, gt_y, *args, **kwargs)

        pred = add_mean(outputs)

        loss.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, gt))))
        loss.update(psnr=reduce_mean(self.psnr(pred, gt * 255.0, 4, 255.0)))

        return outputs, loss

    def eval_step(self, batch, saved_path):
        metrics = {}

        O, B = batch['O'].cuda(), batch['B'].cuda()
        samples = sub_mean(O)
        derain = self.forward(samples)
        pred = quantize(add_mean(derain), 255)
        normalized = pred[0]
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

        imageio.imwrite(os.path.join(saved_path, ''.join([batch['filename'][0], '.png'])),
                        tensor_cpu.numpy())

        with torch.no_grad():
            metrics.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, B))))
            metrics.update(psnr=reduce_mean(self.psnr(pred, B * 255.0, 4, 255.0)))

        return metrics

    def set_metrics(self, criterion, rgb_range=255.):
        self.criterion = criterion
        self.psnr = PSNR_ycbcr()
        self.ssim = SSIM(size_average=False, data_range=rgb_range)


class build_FuGCN(DerainModel, name='FuGCN'):

    def __call__(self, cfg):
        scheduler = None

        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'Loss': 1}
        losses = {'Loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = Net(cfg).cuda()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)  ## optimizer 1: Adam
        model.set_metrics(criterion)

        return model, criterion, optimizer, scheduler

