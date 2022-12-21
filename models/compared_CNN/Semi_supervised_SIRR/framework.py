import datetime
import time
import torch
import torch.distributed as dist
import shutil
import os
from utils import *
from utils import AverageMeter, accuracy
# from data_gen.prefetcher import data_prefetcher
from torch.backends import cudnn
# from logger import create_logger, log_string
# from main_train import Tester
import torch.nn as nn
from utils.utils import MetricLogger, SmoothedValue
from pytorch_msssim import SSIM
# from cal_ssim import SSIM
# from dataset import PSNR
import matplotlib.pyplot as plt
import numpy as np
from dist_utils import dist_train_v1
# from torch.autograd import Variable


try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    # from utils.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print("Currently using torch.cuda.amp")
    try:
        from torch.cuda import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex or use pytorch1.6+.")

class model_amp(nn.Module):
    def __init__(self, args, model, criterion):
        super(model_amp, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        # self.model = dist_train_v1(args, self.model)

    def dist_train(self):
        self.model = dist_train_v1(self.args, self.model)

    def __call__(self, x, gt, *args, **kwargs):
        if not self.args.amp or self.args.amp is None:
            output = self.model(x)
            loss = self.criterion(output, gt, *args, **kwargs)
        else:
            # torch.amp optimization
            with amp.autocast():
                output = self.model(x)
                loss = self.criterion(output, gt)
        if hasattr(self.model, 'ddp_step'):
            self.model.ddp_step(loss)
        return output, loss

    def backward(self, optimizer, loss, scaler=None):
        if self.args.amp is not None:
            if not self.args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # optimizer.step()
                if self.args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.clip_max_norm)
            else:
                # torch.amp optimization
                scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
        else:
            loss.backward()
            # optimizer.step()


    def apex_initialize(self, optimizer):

        scaler = None
        if self.args.amp is not None:
            cudnn.deterministic = False
            cudnn.benchmark = True
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


            if not self.args.amp:
                log_string("apex optimization")
                self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.amp_opt_level)
                # opt_level=args.opt_level,
                # keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                # loss_scale=args.loss_scale
                # )
            else:
                log_string("torch.amp optimization")
                scaler = amp.GradScaler()

        return optimizer, scaler

# def get_checkpoint_dir(_DIR_NAME = "checkpoints"):
#     """Retrieves the location for storing checkpoints."""
#     return os.path.join(cfg.OUT_DIR, _DIR_NAME)
#
# from iopath.common.file_io import g_pathmgr
#
# def get_last_checkpoint(_NAME_PREFIX = "ckpt_ep_"):
#     """Retrieves the most recent checkpoint (highest epoch number)."""
#     checkpoint_dir = get_checkpoint_dir()
#     checkpoints = [f for f in g_pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
#     last_checkpoint_name = sorted(checkpoints)[-1]
#     return os.path.join(checkpoint_dir, last_checkpoint_name)

# def reduce_mean(tensor, nprocs):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= nprocs
#     return rt

def load_checkpoint(args, model, optimizer, ignore_params=None):
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = args.best_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            try:
                args.best_epoch = checkpoint['best_epoch']
            except:
                args.best_epoch = 0
            args.best_prec1 = checkpoint['best_loss']
            # try:
            if args.amp is not None:
                print(checkpoint.keys())
                amp.load_state_dict(checkpoint['amp'])
            if ignore_params is not None:
                ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
                model.load_state_dict(ckpt, strict=False)
                print("partial_load_checkpoint")
            else:
                log_string(f"=> loading checkpoint '{args.resume}'")
                model.load_state_dict(checkpoint['state_dict'])

            if hasattr(checkpoint, 'optimizer'):
                optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            log_string("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']))

            del checkpoint
            torch.cuda.empty_cache()

        else:
            log_string("=> no checkpoint found at '{}'".format(args.resume))


        return model, optimizer

def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))

def show_maps(axes, O, B, outputs):
    pred = outputs[0, ...].cpu().detach().numpy().transpose(1, 2, 0)
    gt = B[0, ...].cpu().numpy().transpose(1, 2, 0)
    axes[0, 0].imshow(O[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[0, 1].imshow(pred)
    axes[1, 0].imshow(B[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[1, 1].imshow(np.abs(pred - gt))

    # pred = outputs[1, ...].cpu().detach().numpy().transpose(1, 2, 0)
    # gt = B[1, ...].cpu().numpy().transpose(1, 2, 0)
    # axes[1, 0].imshow(O[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 1].imshow(pred)
    # axes[1, 2].imshow(B[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 3].imshow(np.abs(pred - gt))

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

# import matplotlib.pyplot as plt
# plt.ion()
# f, axarr = plt.subplots(1, 3)
# fig, axes = plt.subplots(ncols=2, nrows=2)
def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, scaler):



    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
        index = batch['index']
        samples = batch['O'].to(device)
        gt = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]
        # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
        outputs, loss_dicts = model(samples, gt, index)
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

        if (idx + 1) % args.accumulated_step == 0:
            optimizer.step()
            optimizer.zero_grad()


        loss_dicts['Loss'] = losses
        metric_logger.update(**loss_dicts)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(psnr=PSNR(outputs.cpu().detach().numpy(), gt.cpu().numpy()))
    log_string("Averaged stats: {}".format(metric_logger))
    # plt.ioff()
    # 解耦，用于在main中调整日志
    return {k: meter.avg for k, meter in metric_logger.meters.items()}


class EpochRunner():

    def __init__(self, args, sess):
        self.args = args
        out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc)
        self.args.out_dir = out_dir
        self.args.model_save_dir = model_save_dir
        self.args.tfb_dir = tfb_dir
        # self.tester = Tester(args)
        # self.std_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.sess = sess

        # self.ssim = SSIM().cuda()
        # self.bcmsl = BCMSLoss().cuda()  # torch.nn.L1Loss().cuda()

    def run(self, train_loader, model, criterion, optimizer, val_loader, scheduler, **kwargs):
        args = self.args
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        model = model_amp(args, model, criterion)
        optimizer, scaler = model.apex_initialize(optimizer)
        model, optimizer = load_checkpoint(args, model, optimizer)

        start_time = time.time()
        for epoch in range(self.args.start_epoch, args.epochs):
            if self.args.distributed:
                kwargs.get('train_sampler').set_epoch(epoch)
            try:
                epoch_time = datetime.datetime.now()
                train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, self.args.device,
                                             epoch, scaler)
            except:
                # train_loader = self.sess.get_dataloader('train', self.args.)
                train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, self.args.device,
                                             epoch, scaler)

            val_loss = self.validate_framework(val_loader, model, criterion, epoch)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            val_loss = val_loss['Loss']
            is_best = val_loss < args.best_prec1
            self.args.best_prec1 = min(val_loss, args.best_prec1)

            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self.args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': self.args.best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, self.args.model_save_dir, is_best)

            if epoch % args.print_freq * 10 == 0 or is_best:
                if is_best:
                    args.best_epoch = epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    #'model': model,
                    'state_dict': model.state_dict(),
                    'best_loss': args.best_prec1,
                    'loss': val_loss,
                    'best_epoch': args.best_epoch,
                    'amp': amp.state_dict() if args.amp_opt_level != 'O0' else None
                    # 'optimizer': optimizer.state_dict(),
                }, args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")

            log_string(' * Best validation Loss so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                loss=args.best_prec1, best_epoch=args.best_epoch))

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # log_string("one epoch time: {}".format(
            #     datetime.datetime.now() - epoch_time))
            log_string('Training time {}'.format(total_time_str))

    @torch.no_grad()
    def validate_framework(self, val_loader, model, criterion, epoch=0):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, self.args.epochs)
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(val_loader, 1, header):
            index = batch['index']
            samples = batch['O'].to(self.args.device, non_blocking=True)
            gt = batch['B'].to(self.args.device, non_blocking=True)
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])

            outputs, loss_dicts = model(samples, gt, index)
            # loss_dicts = criterion(outputs, gt)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)

            loss_dicts['Loss'] = losses
            loss_dicts = model.ddp_step(loss_dicts)
            metric_logger.update(**loss_dicts)
            metric_logger.update(psnr=PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))
        log_string("Averaged stats: {}".format(metric_logger))
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        return stats

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin