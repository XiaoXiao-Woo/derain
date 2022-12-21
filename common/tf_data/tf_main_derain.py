"""
Backbone modules.
"""
import copy
import math
import os
import shutil
import time
import warnings
from collections import OrderedDict
import datetime
import tensorflow as tf
import torch.distributed as dist
import torchvision
from torch import nn
from typing import Dict, List
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, set_random_seed, create_logger
from UDL.derain.common.tf_data.data_loader import derainSession as DataSession
from logging import info as log_string
import numpy as np


##########################################################################################
# arch
##########################################################################################

def train_one_epoch(args, model, criterion, data_loader, optimizer, saver, device, epoch):
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    metric_logger = MetricLogger(delimiter="  ", dist_print=args.global_rank)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) if args.print_freq <= 0 else args.print_freq
    with tf.Session(config=config) as sess:

        with tf.device('/gpu:0'):
            sess.run(init)
            tf.get_default_graph().finalize()



            for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
                outputs, loss_dicts = model(batch)
                weight_dict = criterion.weight_dict

                losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)




        metric_logger.update(**loss_dicts)


    metrics = {k: meter.avg for k, meter in metric_logger.meters.items()}

    if args.global_rank == 0:
        log_string("Averaged stats: {}".format(metric_logger))
        if args.use_tb:
            tfb_metrics = metrics.copy()
            tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'time'])
            args.train_writer.add_scalar(tfb_metrics, epoch)
            args.train_writer.flush()

    # plt.ioff()
    # 解耦，用于在main中调整日志
    return metrics


################################################################################
# framework
################################################################################
class EpochRunner():

    def __init__(self, args, sess):
        self.args = args
        if args.use_log:
            out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
            self.args.out_dir = out_dir
            self.args.model_save_dir = model_save_dir
            self.args.tfb_dir = tfb_dir
        self.sess = sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


    def load_checkpoint(self):
        save_model_path = self.args.save_model_path
        if tf.train.get_checkpoint_state(save_model_path):  # load previous trained model
            ckpt = tf.train.latest_checkpoint(save_model_path)
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r'(\w*[0-9]+)\w*', ckpt)
            start_point = int(ckpt_num[-1]) + 1
            print("Load success")

        else:  # re-training when no models found
            start_point = 0
            print("re-training")

    def best_record(self, train_stats, metrics):
        args = self.args
        if metrics == 'max':
            val_metric = train_stats['psnr']
            val_loss = train_stats['Loss']
            is_best = val_metric > args.best_prec1
            args.best_prec1 = max(val_metric, args.best_prec1)
        else:
            val_loss = train_stats['Loss']
            is_best = val_loss < args.best_prec1
            args.best_prec1 = min(val_loss, args.best_prec1)

        return is_best, val_loss

    def eval(self, eval_loader, model, criterion, eval_sampler):
        if self.args.distributed:
            eval_sampler.set_epoch(0)
        print(self.args.distributed)
        val_loss = self.eval_framework(eval_loader, model, criterion)

    def run(self, train_loader, model, criterion, optimizer, val_loader, scheduler, **kwargs):

        args = self.args
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        # model = model_amp(args, model, criterion, args.reg)
        # optimizer, scaler = model.apex_initialize(optimizer)
        # model.dist_train()
        # model, optimizer = load_checkpoint(args, model, optimizer)
        if args.start_epoch >= 1:
            args.epochs += 1
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            train_stats = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
                                          epoch)
            # val_stats = self.validate_framework(val_loader, model, criterion, epoch)

            if self.args.lr_scheduler and scheduler is not None:
                scheduler.step(epoch)

            # remember best prec@1 and save checkpoint
            is_best, val_loss = self.best_record(train_stats, args.metrics)

            if epoch % args.print_freq == 0 or is_best:
                if is_best:
                    args.best_epoch = epoch
                if args.global_rank == 0:  # dist.get_rank() == 0
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric': args.best_prec1,
                        'loss': val_loss,
                        'best_epoch': args.best_epoch,
                        'amp': amp.state_dict() if args.amp_opt_level != 'O0' else None,
                        'optimizer': optimizer.state_dict()
                    }, args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")
            if args.global_rank == 0:
                log_string(' * Best training metrics so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                    loss=args.best_prec1, best_epoch=args.best_epoch))

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                # log_string("one epoch time: {}".format(
                #     datetime.datetime.now() - epoch_time))
                log_string('Training time {}'.format(total_time_str))
    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        # switch to evaluate mode
        model.eval()
        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            if args.distributed:
                metrics = model.module.eval_step(batch)
            else:
                metrics = model.eval_step(batch)

            metric_logger.update(metrics)

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}

        return stats  # stats


def main(args):
    sess = DataSession(args)
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
        if not args.eval:
            args.train_writer = SummaryWriter(args.tfb_dir + '/train')
            args.test_writer = SummaryWriter(args.tfb_dir + '/test')

    # device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model, criterion, optimizer, scheduler = args.builder(args)
    # model.to(device)
    model.cuda(args.local_rank)

    ##################################################
    if args.eval:
        # eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        # runner.eval(eval_loader, model, criterion, eval_sampler)
        ...
    else:
        train_loader = sess.get_dataloader(mode="Slicer", dataset_name="Rain200L", gen="tensor_slices")()
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)

        # if args.once_epoch:
        #     train_loader = iter(list(train_loader))

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=scheduler, train_sampler=train_sampler)

        if args.use_tb and not args.eval:
            args.train_writer.close()
            args.test_writer.close()


if __name__ == "__main__":
    # FIXME 引发日志错误
    ...
