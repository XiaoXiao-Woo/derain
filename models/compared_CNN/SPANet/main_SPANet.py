"""
Backbone modules.
"""
import copy
import math
import os
import shutil
import time
import imageio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import nn
from UDL.derain.common.cal_ssim import SSIM
from UDL.Basis.auxiliary import create_logger, log_string, MetricLogger, SmoothedValue, set_random_seed
import datetime
from UDL.Basis.framework_bak import model_amp, get_grad_norm, set_weight_decay
from UDL.Basis.dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
from torch.utils.tensorboard import SummaryWriter
from UDL.derain.common.derain_dataset import derainSession, calc_psnr, PSNR_ycbcr, add_mean, sub_mean, quantize
try:
    from apex import amp
except:
    print("don't have apex")
'''


inner randperm to generate idx:
 [3127, 3307, 1430, 138, 1634, 2466, 1435, 373, 2868, 57, 3031, 3249, 1913, 3560, 1951, 3266, 3121, 1078, 1907, 3350, 3554, 358, 1447, 496, 2331, 60, 716, 1528, 2177, 3073, 47, 1589, 2595, 3457, 1619, 749, 1587, 544, 1843, 1250, 1348, 1997, 3380, 304, 1602, 1425, 440, 183, 2672, 293, 2139, 789, 2483, 2340, 2468, 3430, 3191, 1486, 2944, 467, 1461, 2310, 14, 365, 581, 821, 2058, 579, 28, 667, 2892, 580, 3004, 18, 1953, 1647, 2708, 853, 3197, 1784, 2262, 2239, 2972, 188, 1822, 2860, 114, 2539, 602, 2780, 1839, 992, 1836, 3080, 1214, 246, 1959, 133, 940, 1273, 1192, 2759, 1306, 416, 802, 1867, 3265, 2906, 2658, 3199, 2307, 2329, 2313, 1948, 1677, 906, 3320, 
'''

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
            if not args.eval:
                k = k.split('.')
                k = '.'.join(k[1:])
            # k = '.'.join(['model', k])
            #
            # k.__delitem__(0)
            #
            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'ddp'])
            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'model' and k_item != 'ddp'])  #
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict


# psnr_t = PSNR_t()
psnr_v2 = PSNR_ycbcr().cuda()
g_ssim = SSIM(size_average=False, data_range=1)

#############################################################################
# Backbone
#############################################################################

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
        self.l1 = nn.L1Loss().to(args.device)
        self.l2 = nn.MSELoss().to(args.device)
        self.ssim = g_ssim.to(args.device)#SSIM().to(args.device)

    def forward(self, outputs, B, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        mask, out = outputs
        # print(out[0, :, 0, 0])
        # print(mask[0, :, 0, 0])

        O = kwargs.get('O')
        M = torch.clip((O - B).sum(dim=1), 0, 1).float()

        # loss
        l1_loss = self.l1(out, B)
        mask_loss = self.l2(mask[:, 0, :, :], M)
        ssim_loss = self.ssim(out, B)

        loss = l1_loss + (1 - ssim_loss) + mask_loss

        # log
        losses = {
            'l1_loss': l1_loss.item()
        }
        l2 = {
            'mask_loss': mask_loss.item()
        }
        losses.update(l2)
        ssimes = {
            'ssim_loss': ssim_loss.item()
        }
        losses.update(ssimes)
        allloss = {
            'Loss': loss#.item()
        }
        losses.update(allloss)

        return losses


##########################################################################################
# arch
##########################################################################################

from SPANet import SPANet

def build(args):


    device = torch.device(args.device)


    model = SPANet().to(args.device)

    #
    #造成日志无法使用
    if args.global_rank == 0:
        log_string(model)

    weight_dict = {'Loss': 1}
    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=None)
    criterion.to(device)

    return model, criterion


# fig, axes = plt.subplots(ncols=2, nrows=1)

def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, scaler):

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", dist_print=args.global_rank)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch, idx in metric_logger.log_every(data_loader, args.print_freq, header):
        # index = batch['index']
        samples = batch['O'].to(device)
        gt = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]

        samples = sub_mean(samples)
        gt_y = sub_mean(gt)
        # axes[0].imshow(samples[0].permute(1, 2, 0).cpu().numpy())
        # axes[1].imshow(gt[0].permute(1, 2, 0).cpu().numpy())
        # plt.savefig(f"{idx}_{batch['filename'][0]}_m.png")

        outputs, loss_dicts = model(samples, gt_y, O=samples)

        losses = loss_dicts['Loss']


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
        # PSNR(outputs.cpu().detach().numpy(), gt.cpu().numpy()))
        # pred = quantize(add_mean(outputs), 255.0)
        pred = add_mean(outputs[-1])
        # metric_logger.update(ssim=reduce_mean(torch.mean(g_ssim(pred / 255.0, gt))))
        metric_logger.update(psnr=reduce_mean(psnr_v2(pred, gt * 255.0, 4, 255.0))) #reduce_mean(psnr_v2(add_mean(outputs), gt, 4, 255.0))
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
            args.start_epoch = args.best_epoch = checkpoint.setdefault('epoch', 1)
            args.best_epoch = checkpoint.setdefault('best_epoch', 1)
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
                if checkpoint.get('optimizer') is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
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
        for epoch in range(args.start_epoch, args.epochs+1):
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

            if self.args.lr_scheduler and scheduler is not None:
                scheduler.step(epoch)

            # remember best prec@1 and save checkpoint
            val_metric = train_stats['psnr']
            val_loss = train_stats['Loss']
            is_best = val_metric > self.args.best_prec1
            self.args.best_prec1 = max(val_metric, args.best_prec1)

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
        saved_path = f"./my_model_results/{args.dataset}"

        if args.local_rank == 0:
            if os.path.exists(saved_path) is False:
                os.mkdir(saved_path)

        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            # index = batch['index']
            samples = batch['O'].to(args.device, non_blocking=True)
            gt = batch['B'].to(args.device, non_blocking=True)
            filename = batch['file_name']
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            if args.distributed:
                outputs = model.module.forward(sub_mean(samples))#forward_chop(sub_mean(samples))
            else:
                outputs = model.forward(sub_mean(samples))#forward_chop(sub_mean(samples))
            pred = quantize(add_mean(outputs), 255)
            normalized = pred[0].mul(255 / args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

            imageio.imwrite(os.path.join(saved_path, ''.join([filename[0], '.png'])),
                            tensor_cpu.numpy())

            # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
            psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255, 4, 255.0))
            ssim = g_ssim(add_mean(outputs) / 255.0, gt)
            # psnr = calc_psnr(outputs.cpu().numpy(), gt.cpu().numpy() * 255, 4, 255.0)  # [0].permute(1, 2, 0)
            # metric_logger.update(**loss_dicts)
            psnr_list.append(psnr.item())
            print(args.local_rank, filename)
            metric_logger.update(ssim=ssim)
            metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

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

    ##################################################
    if args.eval:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        runner.eval(eval_loader, model, criterion, eval_sampler)
    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.lr_scheduler:
            from UDL.Basis.optim import lr_scheduler
            scheduler = lr_scheduler(optimizer.param_groups[0]['lr'], 600)
            scheduler.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
        else:
            scheduler = None
        if args.once_epoch:
            train_loader = iter(list(train_loader))

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=scheduler, train_sampler=train_sampler)

        if args.use_tb:
            args.train_writer.close()
            args.test_writer.close()




if __name__ == "__main__":
    import numpy as np
    from options import args
    torch.cuda.empty_cache()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # os.environ["RANK"] = "0"
    set_random_seed(args.seed)
    '''
    938, 1177, 1986, 2351, 1753, 13, 484, 1178, 1013, 273, 69, 1143, 1230, 2051, 1309, 277, 250, 2214, 1340, 1277, 186, 346, 2329, 1741, 2056, 658, 322, 2333, 1144, 2082, 1248, 2213, 1586, 2189, 2208, 729, 603, 1617, 673, 
    '''
    ##################################################
    args.best_prec1 = 0
    args.best_prec5 = 0
    args.best_epoch = 0
    args.nprocs = torch.cuda.device_count()
    # print(f"deviceCount: {args.nprocs}")
    # mp.spawn(main, nprocs=2, args=(args, ))
    main(args)



