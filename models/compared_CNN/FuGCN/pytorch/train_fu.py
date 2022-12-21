import os
import re
import sys
import cv2
import argparse
import math

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import setting_fu
from dataset import TrainValDataset, TestDataset
from model_fu import Net
from cal_ssim import SSIM

logger = setting_fu.logger
os.environ['CUDA_VISIBLE_DEVICES'] = setting_fu.device_id
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
import numpy as np


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):
        self.log_dir = setting_fu.log_dir
        self.model_dir = setting_fu.model_dir
        self.ssim_loss = setting_fu.ssim_loss
        ensure_dir(setting_fu.log_dir)
        ensure_dir(setting_fu.model_dir)
        ensure_dir('../log_test')
        logger.info('set log dir as %s' % setting_fu.log_dir)
        logger.info('set model dir as %s' % setting_fu.model_dir)
        if len(setting_fu.device_id) > 1:
            self.net = nn.DataParallel(Net()).cuda()
        else:
            self.net = Net().cuda()

        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.step = 0
        self.save_steps = setting_fu.save_steps
        self.num_workers = setting_fu.num_workers
        self.batch_size = setting_fu.batch_size
        self.writers = {}
        self.dataloaders = {}
        self.opt_net = Adam(self.net.parameters(), lr=setting_fu.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[setting_fu.l1],
                                    gamma=0.2)#学习率不变

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        out['lr'] = self.opt_net.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def save_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_net.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.sche_net.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()
        if self.step == 0:
            self.print_network(self.net)

        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        derain = self.net(O)
        l1_loss = self.l1(derain, B)
        ssim = self.ssim(derain, B)
        loss = l1_loss
        if name == 'train':
            loss.backward()
            self.opt_net.step()
        losses = {'L1loss': l1_loss}
        ssimes = {'ssim': ssim}
        losses.update(ssimes)
        self.write(name, losses)

        return derain

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)
        h, w = pred.shape[-2:]
        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp
        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def inf_batch_test(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)

        with torch.no_grad():
            # print(O.shape)
            # input()
            derain = self.net(O)

        l1_loss = self.l1(derain, B)
        ssim = self.ssim(derain, B)
        psnr = PSNR(derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = {'L1 loss': l1_loss}
        ssimes = {'ssim': ssim}
        losses.update(ssimes)
        del O, B, derain
        torch.cuda.empty_cache()
        return l1_loss.data.cpu().numpy(), ssim.data.cpu().numpy(), psnr


def run_train_val(ckp_name_net):
    sess = Session()
    #sess.load_checkpoints_net(ckp_name_net)
    sess.tensorboard('train')
    dt_train = sess.get_dataloader('train')
    while sess.step < setting_fu.total_step:
        sess.sche_net.step()
        sess.net.train()
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)
        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints_net('latest_net')
        if sess.step % sess.save_steps == 0:
            sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])

        # observe tendency of ssim, psnr and loss
        ssim_all = 0
        psnr_all = 0
        loss_all = 0
        num_all = 0
        if sess.step % (setting_fu.one_epoch * 16) == 0:
            dt_val = sess.get_test_dataloader('test')
            sess.net.eval()
            for i, batch_v in enumerate(dt_val):
                loss, ssim, psnr = sess.inf_batch_test('test', batch_v)
                print(i)
                ssim_all = ssim_all + ssim
                psnr_all = psnr_all + psnr
                loss_all = loss_all + loss
                num_all = num_all + 1
            print('num_all:', num_all)
            loss_avg = loss_all / num_all
            ssim_avg = ssim_all / num_all
            psnr_avg = psnr_all / num_all
            logfile = open('../log_test/' + 'val_fu' + '.txt', 'a+')
            epoch = int(sess.step / setting_fu.one_epoch)
            logfile.write(
                'step  = ' + str(sess.step) + '\t'
                                              'epoch = ' + str(epoch) + '\t'
                                                                        'loss  = ' + str(loss_avg) + '\t'
                                                                                                     'ssim  = ' + str(
                    ssim_avg) + '\t'
                                'pnsr  = ' + str(psnr_avg) + '\t'
                                                             '\n\n'
            )
            logfile.close()
        if sess.step % (setting_fu.one_epoch * 10) == 0:
            sess.save_checkpoints_net('net_%d_fu__epoch' % int(sess.step / setting_fu.one_epoch))
            logger.info('save model as net_%d_epoch' % int(sess.step / setting_fu.one_epoch))
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='net_240_fu__epoch')
    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model_1)

