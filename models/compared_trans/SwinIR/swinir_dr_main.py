"""
Backbone modules.
"""
import copy
import math
import os
import shutil
import time
from collections import OrderedDict
import imageio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import nn
from typing import Dict, List

from utils.utils import MetricLogger, SmoothedValue, set_random_seed

# from pytorch_msssim.pytorch_msssim import SSIM
from cal_ssim import SSIM
# from dataset import PSNR_t
from utils.logger import create_logger, log_string
import datetime
from SwinIR.models.network_swinir import SwinIR as net
from dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
import torch.multiprocessing as mp
from dataset import derainSession
from framework import model_amp, get_grad_norm, set_weight_decay
# from dataset import derainSession
from torch.utils.tensorboard import SummaryWriter

def partial_load_checkpoint(state_dict, amp, dismatch_list = []):
    pretrained_dict = {}
    # dismatch_list = ['dwconv']
    if amp is not None:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            k = '.'.join(['amp', k])
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})
    else:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            # if args.eval is not None:
            #     k = k.split('.')
            #     k = '.'.join(k[1:])
            # k = '.'.join(['model', k])
            #
            # k.__delitem__(0)
            k = k.split('.')
            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'ddp'])
            k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'model' and k_item != 'ddp'])  #
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict

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

# def quantize(img, rgb_range):
#     """metrics"""
#     pixel_range = 255 / rgb_range
#     img = np.multiply(img, pixel_range)
#     img = np.clip(img, 0, 255)
#     img = np.round(img) / pixel_range
#     return img
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

# psnr_t = PSNR_t()
psnr_v2 = PSNR_ycbcr().cuda()
g_ssim = SSIM(size_average=False, data_range=1)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    #.reshape((1, 1, 3)) / 256#
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)#(1)
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
        # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.mean(pow(valid, 2), axis=[1, 2, 3]), axis=0)
    mse = np.mean(pow(valid, 2))
    if mse == 0:
        return 100
    try:
        psnr = -10 * math.log10(mse)
    except Exception:
        print(mse)

    return psnr

class BCMSLoss(torch.nn.Module):

    def __init__(self, reduction='none'):
        super(BCMSLoss, self).__init__()

        self.reduction = reduction
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)
        self.l2_loss = torch.nn.MSELoss(reduction=reduction)
        self._ssim = SSIM(size_average=False, data_range=1)
        self.ind = 1

    def forward(self, x, gt):

        ind = self.ind
        # 好1%
        bce = self.bce_mse(x, gt)
        l1_loss = self.l1_loss(x, gt)
        l2_loss = self.l2_loss(x, gt)
        _ssim = self._ssim(x, gt)
        _ssim_loss = 1 - _ssim

        with torch.no_grad():
            w_1 = torch.mean(l1_loss)

            pred = torch.clamp(x, 0, 1)
            l2 = self.l2_loss(pred, gt)
            w_2 = torch.mean(l2)

            w_s = torch.mean(_ssim_loss)
            ssim_m = torch.mean(_ssim)
            w_bce = torch.mean(bce)
        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * \
        #        ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** (1 / ind)  # 119
        # loss = l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + 1e-8) ** ind + \
        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind  # 让l2朝更多方向解析，却增强l1 82

        loss = _ssim_loss.reshape(-1, 1, 1, 1) * (
                (l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind + \
               l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + (
                bce / w_bce) ** ind + 1e-8) ** ind
        # x = torch.sigmoid(x)
        # gt = torch.sigmoid(gt)
        # loss = self.l1(x, gt)
        loss = torch.mean(loss)
        return {'Loss': loss, 'l1_loss': w_1, 'mse_loss': w_2,
                'ssim_loss': w_s, 'bce_loss': w_bce, 'ssim': ssim_m}

    def bce_mse(self, x, gt):
        a = torch.exp(gt)
        b = 1
        loss = a * x - (a + b) * torch.log(torch.exp(x) + 1)
        if self.reduction == 'none':
            return -loss
        if self.reduction == 'mean':
            return -torch.mean(loss)
        else:
            assert False, "there have no self.reduction choices"

#############################################################################
# Backbone
#############################################################################


# if float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size
#
#
# def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
#     # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
#     """
#     Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
#     This will eventually be supported natively by PyTorch, and this
#     class can go away.
#     """
#     if float(torchvision.__version__[:3]) < 0.7:
#         if input.numel() > 0:
#             return torch.nn.functional.interpolate(
#                 input, size, scale_factor, mode, align_corners
#             )
#
#         output_shape = _output_size(2, input, size, scale_factor)
#         output_shape = list(input.shape[:-2]) + list(output_shape)
#         return _new_empty_tensor(input, output_shape)
#     else:
#         return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        # loss_dicts = {}

        for k in self.losses.keys():#.items():
            # k, loss = loss_dict
            if k == 'Loss':
                loss = self.losses[k]

                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts


##########################################################################################
# arch
##########################################################################################

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')


    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    # 003 real-world image sr
    elif args.task == 'real_sr':
        # if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts

        model = net(upscale=1, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume)
        if hasattr(ckpt, 'best_metric'):
            print("best_psnr: ", ckpt['best_metric'])
        # model.load_state_dict(ckpt['state_dict'])
        # partial_load_checkpoint(ckpt['state_dict'], args.amp)

    return model

def build(args):


    device = torch.device(args.device)

    # model = net(
    #         patch_size=128, in_channels=3, out_channels=3, patch_stride=[2, 2, 2, 2],
    #         hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #         depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])

    # model = net(patch_size=[torch.tensor([2, 2]), torch.tensor([2, 2])], image_size=128, in_channels=3, out_channels=3,
    #     patch_stride=[2, 2, 2, 2],
    #     hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #     depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])
    scale = 2
    model = define_model(args)
    #6666: 7675  head： 多卡 6728
    #3463: 7451-4789
    #2222: 7297-4789   head: 6566  多卡 6150

    # model = net(down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]], image_size=48, in_channels=3, out_channels=3,
    #     patch_stride=[2, 2, 2, 2],
    #     hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #     depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])
    #造成日志无法使用
    if args.global_rank == 0:
        log_string(model)

    # weight_dict = {'L1Loss': 1}
    # losses = {'L1Loss': torch.nn.L1Loss(reduction='mean').cuda()}

    weight_dict = {'Loss': 1}
    losses = {'Loss': nn.L1Loss().cuda()}
    # losses = {'Loss': BCMSLoss().cuda()}
    # weight_dict = {'Loss': 1, 'contrast': 1}
    # losses = {'Loss': BCMSLoss().cuda(), 'contrast': ErrMemory(K=4, feature_dims=3*128*128, T=1, momentum=0.9).cuda()}


    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses)
    criterion.to(device)

    return model, criterion

# import matplotlib.pyplot as plt
# plt.ion()
# f, axarr = plt.subplots(1, 3)
# fig, axes = plt.subplots(ncols=2, nrows=2)
def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, scaler):

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", dist_print=args.global_rank)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # psnr_list = []
    for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
        # index = batch['index']
        samples = batch['O'].to(device)
        gt = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]
        # samples = batch[0].to(device)
        # gt = batch[1].to(device)
        samples = sub_mean(samples)
        gt_y = sub_mean(gt)
        # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
        outputs, loss_dicts = model(samples, gt_y)
        # loss_dicts = model.model.ddp_step(loss_dicts)
        losses = loss_dicts['Loss']


        # weight_dict = criterion.weight_dict
        # losses = loss_dicts['contrast']
        # show_maps(axes, samples, gt, outputs)
        # plt.pause(0.4)
        # if epoch <= 946:
        #     losses = loss_dicts['Loss']
        # else:
        #     losses = loss_dicts['contrast']#sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
        losses = losses / args.accumulated_step

        model.backward(optimizer, losses, scaler)
        if args.clip_max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            # print(get_grad_norm(model.parameters()))
        else:
            grad_norm = get_grad_norm(model.parameters())

        if (idx + 1) % args.accumulated_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        # torch.cuda.synchronize()
        loss_dicts['Loss'] = losses
        metric_logger.update(**loss_dicts)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #reduce_mean(psnr_t(outputs, gt)))#
        # p = reduce_mean(psnr_t(outputs, gt))
        # p = psnr_t(outputs, gt)
        # p_1 = PSNR(outputs.cpu().detach().numpy(), gt.cpu().numpy())
        # if args.global_rank == 0:
        # psnr = reduce_mean(psnr_v2(add_mean(outputs), gt, 4, 255.0))
        # pred_np = add_mean(outputs.cpu().detach().numpy())
        # pred_np = quantize(pred_np, 255)
        # gt = gt.cpu().numpy() * 255
        # # psnr = calc_psnr(pred_np, gt, 4, 255.0)
        # p_1 = PSNR(pred_np, gt)
        # print("current psnr: ", psnr, p_1)
        # pred_np = quantize(add_mean(outputs), 255)
        # psnr = calc_psnr(pred_np.cpu().detach().numpy(), gt.cpu().numpy() * 255, 4, 255.0)
        #PSNR(outputs.cpu().detach().numpy(), gt.cpu().numpy()))
        # pred = quantize(add_mean(outputs), 255.0)
        metric_logger.update(ssim=reduce_mean(g_ssim(add_mean(outputs) / 255.0, gt)))
        metric_logger.update(psnr=reduce_mean(psnr_v2(add_mean(outputs), gt * 255.0, 4, 255.0))) #reduce_mean(psnr_v2(add_mean(outputs), gt, 4, 255.0))
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    metrics = {k: meter.avg for k, meter in metric_logger.meters.items()}


    if args.global_rank == 0:
        log_string("Averaged stats: {}".format(metric_logger))
        if args.use_tb:
            tfb_metrics = metrics.copy()
            tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
            args.train_writer.add_scalar(tfb_metrics, epoch)
            args.train_writer.flush()
    # plt.ioff()
    # 解耦，用于在main中调整日志
    return metrics#{k: meter.avg for k, meter in metric_logger.meters.items()}


################################################################################
# framework
################################################################################
def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))

def load_checkpoint(args, model, optimizer, ignore_params=[]):
    global_rank = args.global_rank
    checkpoint = {}
    if args.resume:
        if os.path.isfile(args.resume):
            if args.distributed:
                dist.barrier()
            init_checkpoint = torch.load(args.resume, map_location=f"cuda:{args.local_rank}")
            if init_checkpoint.get('state_dict') is None:
                checkpoint['state_dict'] = init_checkpoint
                del init_checkpoint
                torch.cuda.empty_cache()
            else:
                checkpoint = init_checkpoint
                print(checkpoint.keys())
            args.start_epoch = args.best_epoch = checkpoint.setdefault('epoch', 0)
            args.best_epoch = checkpoint.setdefault('best_epoch', 0)
            args.best_prec1 = checkpoint.setdefault('best_loss', 0)
            if args.amp is not None:
                print(checkpoint.keys())
                try:
                    amp.load_state_dict(checkpoint['amp'])
                except:
                    Warning("no loading amp_state_dict")
            # if ignore_params is not None:
            # if checkpoint.get('state_dict') is not None:
            #     ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
            # else:
            #     print(checkpoint.keys())
            #     ckpt = partial_load_checkpoint(checkpoint, args.amp, ignore_params)
            print(checkpoint['state_dict'].keys())
            ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
            if args.distributed:
                model.module.load_state_dict(ckpt)  # , strict=False
            else:
                model.load_state_dict(ckpt)#, strict=False
            if global_rank == 0:
                log_string(f"=> loading checkpoint '{args.resume}'\n ignored_params: \n{ignore_params}")
            # else:
            #     if global_rank == 0:
            #         log_string(f"=> loading checkpoint '{args.resume}'")
            #     if checkpoint.get('state_dict') is None:
            #         model.load_state_dict(checkpoint)
            #     else:
            #         model.load_state_dict(checkpoint['state_dict'])
                # print(checkpoint['state_dict'].keys())
                # print(model.state_dict().keys())
            if optimizer is not None:
                if hasattr(checkpoint, 'optimizer'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            if global_rank == 0:
                log_string("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']))

            del checkpoint
            torch.cuda.empty_cache()

        else:
            if global_rank == 0:
                log_string("=> no checkpoint found at '{}'".format(args.resume))

        return model, optimizer


class EpochRunner():

    def __init__(self, args, sess):
        self.args = args
        if args.use_log:
            out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
            self.args.out_dir = out_dir
            self.args.model_save_dir = model_save_dir
            self.args.tfb_dir = tfb_dir
        # self.tester = Tester(args)
        # self.std_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.sess = sess

        # self.ssim = SSIM().cuda()
        # self.bcmsl = BCMSLoss().cuda()  # torch.nn.L1Loss().cuda()

    def eval(self, eval_loader, model, criterion, eval_sampler):
        if self.args.distributed:
            eval_sampler.set_epoch(0)
        print(self.args.distributed)
        model.init_eval_obj(args)
        model = dist_train_v1(self.args, model)
        model, _ = load_checkpoint(self.args, model, None)
        val_loss = self.eval_framework(eval_loader, model, criterion)

    def run(self, train_loader, model, criterion, optimizer, val_loader, scheduler, **kwargs):
        args = self.args
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        model = model_amp(args, model, criterion)
        optimizer, scaler = model.apex_initialize(optimizer)
        model.dist_train()
        model, optimizer = load_checkpoint(args, model, optimizer)

        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if self.args.distributed:
                kwargs.get('train_sampler').set_epoch(epoch)
            # try:
            #     epoch_time = datetime.datetime.now()
            #     train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
            #                                  epoch, scaler)
            # except:
            #     # train_loader = self.sess.get_dataloader('train', self.args.)
            #     train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
            #                                  epoch, scaler)
            epoch_time = datetime.datetime.now()
            train_stats = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
                                         epoch, scaler)
            # val_stats = self.validate_framework(val_loader, model, criterion, epoch)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            val_metric = train_stats['psnr']
            val_loss = train_stats['Loss']
            is_best = val_metric > self.args.best_prec1
            self.args.best_prec1 = max(val_metric, args.best_prec1)

            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self.args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': self.args.best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, self.args.model_save_dir, is_best)

            if epoch % args.print_freq * 10 == 0 or is_best:
                if is_best:
                    self.args.best_epoch = epoch
                if args.global_rank == 0: #dist.get_rank() == 0
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric': self.args.best_prec1,
                        'loss': val_loss,
                        'best_epoch': self.args.best_epoch,
                        'amp': amp.state_dict() if args.amp_opt_level != 'O0' else None,
                        'optimizer': optimizer.state_dict()
                    }, args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")
            if args.global_rank == 0:
                log_string(' * Best validation psnr so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                    loss=args.best_prec1, best_epoch=args.best_epoch))

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                # log_string("one epoch time: {}".format(
                #     datetime.datetime.now() - epoch_time))
                log_string('Training time {}'.format(total_time_str))

    @torch.no_grad()
    def validate_framework(self, val_loader, model, criterion, epoch=0):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, args.epochs)
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(val_loader, 1, header):
            index = batch['index']
            samples = batch['O'].to(args.device, non_blocking=True)
            gt = batch['B'].to(args.device, non_blocking=True)
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])

            outputs, loss_dicts = model(sub_mean(samples), sub_mean(gt), index)
            # loss_dicts = criterion(outputs, gt)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)

            loss_dicts['Loss'] = losses
            # pred_np = add_mean(outputs.cpu().detach().numpy())
            # pred_np = quantize(pred_np, 255)
            # gt = gt.cpu().numpy() * 255
            # psnr = calc_psnr(pred_np, gt, 4, 255.0)
            psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255.0, 4, 255.0))
            metric_logger.update(**loss_dicts)
            metric_logger.update(
                psnr=psnr)  # reduce_mean(psnr_t(outputs, gt)))#PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        # metric_logger.synchronize_between_processes()
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}


        if args.global_rank == 0:
            log_string("[{}] Averaged stats: {}".format(epoch, metric_logger))
            if args.use_tb:
                tfb_metrics = stats.copy()
                tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
                args.test_writer.add_scalar(tfb_metrics, epoch)
                args.test_writer.flush()

        # stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        return stats#stats

    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        # switch to evaluate mode
        model.eval()
        psnr_list = []
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            # index = batch['index']
            # print("pass: ", batch['file_name'])
            # continue
            samples = batch['O'].to(args.device, non_blocking=True)
            # gt = batch['B'].to(args.device, non_blocking=True)
            filename = batch['file_name']
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            if args.distributed:
                outputs = model.module.forward_chop(sub_mean(samples))
            else:
                outputs = model.forward_chop(sub_mean(samples))
            pred = quantize(add_mean(outputs), 255)
            normalized = pred[0].mul(255 / args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            imageio.imwrite(os.path.join(f"./my_model_results/{args.eval}", ''.join([filename[0], '.png'])),
                            tensor_cpu.numpy())

            # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
            # psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255, 4, 255.0))
            # ssim = g_ssim(add_mean(outputs) / 255.0, gt)
            # psnr = calc_psnr(outputs.cpu().numpy(), gt.cpu().numpy() * 255, 4, 255.0)  # [0].permute(1, 2, 0)
            # metric_logger.update(**loss_dicts)
            # psnr_list.append(psnr.item())
            # print(args.local_rank, filename)
            # metric_logger.update(ssim=ssim)
            # metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        log_string("Averaged stats: {} ({})".format(metric_logger, np.mean(psnr_list)))

        return stats  # stats


def main(args):
    sess = derainSession(args)
    # torch.autograd.set_detect_anomaly(True)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        args.global_rank = args.local_rank
        runner = EpochRunner(args, sess)
    else:
        args.distributed = True
        init_dist(args.launcher, args)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        args.global_rank = rank
        # 多机多卡
        args.local_rank = local_rank
        runner = EpochRunner(args, sess)
        print(f"[init] == args local rank: {args.local_rank}, local rank: {local_rank}, global rank: {rank} ==")


        # _, world_size = get_dist_info()
    ##################################################
    if args.use_tb:
        if args.tfb_dir is None:
            args = runner.args
        args.train_writer = SummaryWriter(args.tfb_dir + '/train')
        args.test_writer = SummaryWriter(args.tfb_dir + '/test')




    # device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model, criterion = build(args)
    # model.to(device)
    model.cuda(args.local_rank)

    # SyncBN (https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)
    #net = nn.SyncBatchNorm.convert_sync_batchnorm(net) if cfg.MODEL.SYNCBN else net
    # if rank == 0:
    #     # from torch.utils.collect_env import get_pretty_env_info
    #     # logger.debug(get_pretty_env_info())
    #     # logger.debug(net)
    #     logger.info("\n\n\n            =======  TRAINING  ======= \n\n")
    #     logger.info(utils.count_parameters(net))
    # if rank == 0:
    #     logger.info(
    #         f"ACCURACY: TOP1 {acc1:.3f}(BEST {best_acc1:.3f}) | TOP{cfg.TRAIN.TOPK} {acck:.3f} | SAVED {checkpoint_file}"
    #     )


    # model_without_ddp = model
    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]

    ##################################################
    if args.eval is not None:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.eval, args.distributed)
        runner.eval(eval_loader, model, criterion, eval_sampler)
    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)

        # model = dist_train_v1(args, model)

        # skip = {}
        # skip_keywords = {}
        # if hasattr(model, 'no_weight_decay'):
        #     skip = model.no_weight_decay()
        # if hasattr(model, 'no_weight_decay_keywords'):
        #     skip_keywords = model.no_weight_decay_keywords()
        # parameters = set_weight_decay(model, skip, skip_keywords)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        # if args.lr_scheduler:
        #     from optim import lr_scheduler
        #     scheduler = lr_scheduler(optimizer.param_groups[0]['lr'], 600)
        #     scheduler.set_optimizer(optimizer, torch.optim.lr_scheduler.MultiStepLR)
        #     # scheduler.set_optimizer(optimizer, None)
        #     scheduler.get_lr_map("step_lr_100",
        #                          out_file=os.path.join(args.out_dir, f"./step_lr_100_{args.experimental_desc}.png"))
        # else:
        #     scheduler = None
        # train_loader = iter(list(train_loader))

        # for epoch in range(args.start_epoch, args.epochs):
        #     train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=None, train_sampler=train_sampler)

        if args.use_tb:
            args.train_writer.close()
            args.test_writer.close()


#-  * Best validation psnr so far@1 25.8624957 in epoch 0

if __name__ == "__main__":
    # from patch_aug_dataset import derainSession

    import numpy as np
    import argparse
    import random
    from torch.backends import cudnn

    torch.cuda.empty_cache()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # os.environ["RANK"] = "0"
    # model_path = './results/300.pth.tar'
    model_path = './results/Fu100H/HPT_new/swin/model_2021-09-03-23-34/495.pth.tar'
    #./results/100L/PVT/layernorm_100L/model_2021-06-10-22-19/181.pth.tar  1159
    #'./results/100h/PVT/amp_test_100H/model_2021-05-29-16-24/476.pth.tar1'
    # model_path = './results/100H/PVT/amp_test/model_2021-05-28-22-16/amp_model_best.pth.tar1'
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # * Logger
    parser.add_argument('--use-log', default=True
                        , type=bool)
    parser.add_argument('--out_dir', metavar='DIR', default='./results',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='HPT_new')
    parser.add_argument('--use-tb', default=False, type=bool)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'),
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    ## Train
    parser.add_argument('--patch_size', type=int, default=64,
                        help='image2patch, set to model and dataset')
    parser.add_argument('--lr', default=1e-4, type=float)#1e-4 2e-4 8
    # parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('-b', '--batch-size', default=8, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--resume',
                        default=model_path,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--accumulated-step', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')
    ## DDP
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', default=0, type=int,
                        help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
    # parser.add_argument('--world-size', default=2, type=int,
    #                     help='number of distributed processes, = gpus * nnodes')
    parser.add_argument('--backend', default='nccl', type=str,  # gloo
                        help='distributed backend')
    parser.add_argument('--dist-url', default='env://',
                        type=str,  # 'tcp://224.66.41.62:23456'
                        help='url used to set up distributed training')

    ## AMP
    parser.add_argument('--amp', default=None, type=bool,
                        help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    ##
    parser.add_argument('--eval', default='real', type=str,
                        choices=[None, 'rain200L', 'rain100L', 'rain200H', 'rain100H',
                                 'test12', 'real', 'DID', 'SPA'],
                        help="performing evalution for patch2entire")
    parser.add_argument('--crop_batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')

    #SRData
    parser.add_argument('--model', default='swin',
                        help='model name')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                        help='train dataset name')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--task', type=str, default='real_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    args = parser.parse_args()


    # log_string(args)#引发日志错误

    # assert args.opt_level != 'O0' and args.amp != None, print("you must have apex or torch.cuda.amp")
    args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
    # args.launcher = 'none' if dist.is_available() else args.launcher
    print(args.launcher)
    assert args.accumulated_step > 0
    # args.opt_level = 'O1'
    args.scale = [1]
    args.dir_data = '../IPT/torchImpl/data'
    args.experimental_desc = "swin"
    args.dataset = "Fu100H"

    set_random_seed(args.seed)
    ##################################################
    args.best_prec1 = 0
    args.best_prec5 = 0
    args.best_epoch = 0
    args.nprocs = torch.cuda.device_count()

    # print(f"deviceCount: {args.nprocs}")
    # mp.spawn(main, nprocs=2, args=(args, ))
    main(args)
    #40.30398910522461
    #m:PSNR: 40.110727 SSIM: 0.986556
    # PSNR: 40.223933 SSIM: 0.987300
    #8-28 modified HAttention split bug
    #- Averaged stats: psnr: 40.6850 (38.7817267) (38.781726684570316)