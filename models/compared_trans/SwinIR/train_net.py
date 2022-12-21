import argparse
import math

import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import torch.nn as nn

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from torchstat import stat


from dataset import derainSession

class PSNR_ycbcr(nn.Module):

    def __init__(self):
        super().__init__()
        self.gray_coeffs = torch.tensor([65.738, 129.057, 25.064],
                                        requires_grad=False).reshape((1, 3, 1, 1)) / 256
    def quantize(self, img, rgb_range):
        """metrics"""
        pixel_range = 255 / rgb_range
        img = torch.multiply(img, pixel_range)
        img = torch.clip(img, 0, 255)
        img = torch.round(img) / pixel_range
        return img

    @torch.no_grad()
    def forward(self, sr, hr, scale, rgb_range):
        """metrics"""
        sr = self.quantize(sr, rgb_range)
        gray_coeffs = self.gray_coeffs.to(sr.device)

        hr = hr.float()
        sr = sr.float()
        diff = (sr - hr) / rgb_range

        diff = torch.multiply(diff, gray_coeffs).sum(1)
        if hr.size == 1:
            return 0
        if scale != 1:
            shave = scale
        else:
            shave = scale + 6
        if scale == 1:
            valid = diff
        else:
            valid = diff[..., shave:-shave, shave:-shave]
        mse = torch.mean(torch.pow(valid, 2))
        return -10 * torch.log10(mse)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def quantize(img, rgb_range):
    pixel_range = 255.0 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    if diff == 0:
        return 100
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
    mse = np.mean(pow(valid, 2))
    return -10 * math.log10(mse)


def rgb2ycbcr(img, y_only=True):
    """metrics"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt

def sub_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x / 255.0

def add_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

psnr_v2 = PSNR_ycbcr().cuda()



def main():
    model_path = './results/114.pth.tar'
    epochs = 300
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='real_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str, default=model_path)
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--lr', default=1e-4, type=float)#1e-4 2e-4 8
    # parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('-b', '--batch-size', default=6, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--patch_size', type=int, default=48,
                        help='image2patch, set to model and dataset')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    print(f'loading model from {args.model_path}')
    model = define_model(args).to(device)
    # stat(model, [(3, 48, 48)])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)

    criterion = nn.L1Loss().cuda()

    sess = derainSession(args)

    train_loader, _ = sess.get_dataloader("train", False)
    nums = len(train_loader)

    model.train()
    best_psnr = 0
    start_epoch = model_path.split('/')[-1].split('.')[0]
    print(start_epoch)
    for epoch in range(int(start_epoch)+1, epochs+1):
        psnr_list = []
        for idx, batch in enumerate(train_loader):
            samples = batch['O'].to(device)
            gt = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]
            # samples = batch[0].to(device)
            # gt = batch[1].to(device)
            #
            # samples = sub_mean(samples)
            # gt_y = sub_mean(gt)
            outputs = model(samples)
            loss = criterion(outputs, gt)


            psnr = psnr_v2(outputs, gt * 255.0, 4, 255.0).item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            psnr_list.append(psnr)
            psnr_m = np.mean(psnr_list)
            print(f"[{epoch}]/[{epochs}] {idx}/{nums} Loss: {loss}, PSNR: {psnr:.4f} ({psnr_m:.4f})")

        psnr_m = np.mean(psnr_list)
        is_best = psnr_m > best_psnr
        best_psnr = max(psnr_m, best_psnr)
        print(f"PSNR: {psnr_m:.4f}, best PSNR {best_psnr:.4f}")

        if epoch % 10 == 0 or is_best:
            torch.save({"state_dict": model.state_dict(),
                        "psnr": psnr_m,
                        "best_psnr": best_psnr,
                        'epoch': epoch
                        },
                       f"./results/{epoch}.pth.tar")





def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')


    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    # 003 real-world image sr
    elif args.task == 'real_sr':
        # if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts

        model = net(upscale=1, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path)
        print("psnr: best_psnr: ", ckpt['psnr'], ckpt['best_psnr'])
        model.load_state_dict(ckpt['state_dict'])

    return model



if __name__ == '__main__':
    main()#7500MB


