import os
import sys
import cv2
import argparse
import numpy as np
import logging

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
import skimage.measure as ms
# import progressbar

from derain_dataset import derainSession
from cal_ssim import SSIM
from SPANet import SPANet
import matplotlib.pyplot as plt
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

'''
tensor([ 0.0746, -0.0634,  0.0484], device='cuda:0', grad_fn=<SelectBackward>)
tensor([0.4881], device='cuda:0', grad_fn=<SelectBackward>)
2021-10-10 20:16:42,863 - INFO - train_derain--l1_loss:0.4806 mask_loss:0.1817 ssim_loss:-0.0396 all_loss:1.702 lr:0.005 step:0
tensor([0.0835], device='cuda:0', grad_fn=<SelectBackward>)
2021-10-10 20:16:43,670 - INFO - train_derain--l1_loss:0.1838 mask_loss:0.02242 ssim_loss:0.4589 all_loss:0.7473 lr:0.005 step:1

tensor([ 0.0746, -0.0634,  0.0484], device='cuda:0', grad_fn=<SelectBackward>)
tensor([0.4881], device='cuda:0', grad_fn=<SelectBackward>)
- Epoch: [0]  [  0/157]  eta: 0:31:00  lr: 0.005000  grad_norm: 4.791532  l1_loss: 0.4806 (0.4806307)  mask_loss: 0.1817 (0.1817380)  ssim_loss: -0.0396 (-0.0396044)  Loss: 1.7020 (1.7019731)  psnr: 7.2304 (7.2304273)  time: 11.8533  data: 9.6836  max mem: 5092MB
tensor([0.0518, 0.0560, 0.0519], device='cuda:0', grad_fn=<SelectBackward>)
tensor([0.0835], device='cuda:0', grad_fn=<SelectBackward>)
- Epoch: [0]  [  1/157]  eta: 0:16:23  lr: 0.005000  grad_norm: 4.352412  l1_loss: 0.1838 (0.3322037)  mask_loss: 0.0224 (0.1020805)  ssim_loss: 0.4589 (0.2096621)  Loss: 0.7473 (1.2246220)  psnr: 13.8770 (10.5537045)  time: 6.3064  data: 4.8418  max mem: 5100MB
'''

# torch.cuda.manual_seed_all(2019)
# torch.manual_seed(2019)


fig, axes = plt.subplots(ncols=2, nrows=1)

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class Session(derainSession):
    def __init__(self, args, net):
        super().__init__(args)
        self.device = torch.device("cuda")

        self.log_dir = './logdir'
        self.model_dir = './model'
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        self.log_name = 'train_derain'
        self.val_log_name = 'val_derain'
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        # self.test_data_path = 'testing/real_test_1000.txt'  # test dataset txt file path
        # self.train_data_path = 'training/real_world.txt'  # train dataset txt file path

        self.train_data_path = 'D:/Datasets/derain/Rain100L/train'
        self.test_data_path = 'D:/Datasets/derain/Rain100L/test'

        self.multi_gpu = True

        self.l1 = nn.L1Loss().to(self.device)
        self.l2 = nn.MSELoss().to(self.device)
        self.ssim = SSIM().to(self.device)
        self.net = SPANet().to(args.device)
        self.step = 0
        self.save_steps = 400
        # self.num_workers = 16
        # self.batch_size = 16
        self.writers = {}
        self.dataloaders = {}
        self.shuffle = True
        self.opt = Adam(self.net.parameters(), lr=5e-3)
        self.sche = MultiStepLR(self.opt, milestones=[30000], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    # def get_dataloader(self, dataset_name, train_mode=True):
    #     dataset = {
    #         True: TrainValDataset,
    #         False: TestDataset,
    #     }[train_mode](dataset_name)
    #     self.dataloaders[dataset_name] = \
    #         DataLoader(dataset, batch_size=self.batch_size,
    #                    shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
    #     if train_mode:
    #         return iter(self.dataloaders[dataset_name])
    #     else:
    #         return self.dataloaders[dataset_name]

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name, mode='train'):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            self.net.load_state_dict({k.replace('module.', ''): v for k, v in obj['net'].items()})
        except FileNotFoundError:
            return

        if mode == 'train':
            self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        if name == 'test':
            torch.set_grad_enabled(False)
        O, B = batch['O'], batch['B']  # , batch['M']
        O, B = O.to(self.device), B.to(self.device)  # ,M.to(self.device)
        M = torch.clip((O - B).sum(dim=1), 0, 1).float()#.astype(np.float32)

        axes[0].imshow(O[0].permute(1, 2, 0).cpu().numpy())
        axes[1].imshow(B[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"{self.step}_{batch['filename'][0]}.png")

        mask, out = self.net(O)
        print(out[0, :, 0, 0])
        print(mask[0, :, 0, 0])

        if name == 'test':
            return out.cpu().data, batch['B'], O, mask

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
            'all_loss': loss.item()
        }
        losses.update(allloss)
        return out, mask, M, loss, losses

    def heatmap(self, img):
        if len(img.shape) == 3:
            b, h, w = img.shape
            heat = np.zeros((b, 3, h, w)).astype('uint8')
            for i in range(b):
                heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, :, :], cv2.COLORMAP_JET), (2, 0, 1))
        else:
            b, c, h, w = img.shape
            heat = np.zeros((b, 3, h, w)).astype('uint8')
            for i in range(b):
                heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, 0, :, :], cv2.COLORMAP_JET), (2, 0, 1))
        return heat

    def save_mask(self, name, img_lists, m=0):
        data, pred, label, mask, mask_label = img_lists
        pred = pred.cpu().data

        mask = mask.cpu().data
        mask_label = mask_label.cpu().data
        data, label, pred, mask, mask_label = data * 255, label * 255, pred * 255, mask * 255, mask_label * 255
        pred = np.clip(pred, 0, 255)

        mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
        mask_label = np.clip(mask_label.numpy(), 0, 255).astype('uint8')
        h, w = pred.shape[-2:]
        mask = self.heatmap(mask)
        mask_label = self.heatmap(mask_label)
        gen_num = (1, 1)

        img = np.zeros((gen_num[0] * h, gen_num[1] * 5 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx], mask[idx], mask_label[idx]]
                    for k in range(5):
                        col = (j * 5 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp

        img_file = os.path.join(self.log_dir, '%d_%s.png' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(args, ckp_name='latest', net=None):
    sess = Session(args, net=net)
    # sess.load_checkpoints(ckp_name)
    if sess.multi_gpu:
        sess.net = nn.DataParallel(sess.net)
    sess.tensorboard(sess.log_name)
    sess.tensorboard(sess.val_log_name)

    dt_train, _ = sess.get_dataloader(args.dataset, None)#sess.train_data_path, None)
    # dt_val, _ = sess.get_dataloader(args.dataset, None)#sess.train_data_path, None)
    dt_train = iter(dt_train)

    while sess.step < 40001:
        sess.sche.step()
        sess.net.train()
        sess.net.zero_grad()

        batch_t = next(dt_train)
        pred_t, mask_t, M_t, loss_t, losses_t = sess.inf_batch(sess.log_name, batch_t)
        sess.write(sess.log_name, losses_t)
        loss_t.backward()
        sess.opt.step()

        # if sess.step % 4 == 0:
        #     sess.net.eval()
        #     batch_v = next(dt_val)
        #     pred_v, mask_v, M_v, loss_v, losses_v = sess.inf_batch(sess.val_log_name, batch_v)
        #     sess.write(sess.val_log_name, losses_v)
        # if sess.step % int(sess.save_steps / 16) == 0:
        #     sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_mask(sess.log_name, [batch_t['O'], pred_t, batch_t['B'], mask_t, M_t])
            # if sess.step % 4 == 0:
            #     sess.save_mask('valderain5', [batch_v['O'], pred_v, batch_v['B'], mask_v, M_v])
            # logger.info('save image as step_%d' % sess.step)
        # if sess.step % sess.save_steps == 0:
        #     sess.save_checkpoints('step_%d' % sess.step)
        #     logger.info('save model as step_%d' % sess.step)
        sess.step += 1


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name, 'test')
    if sess.multi_gpu:
        sess.net = nn.DataParallel(sess.net)
    sess.batch_size = 1
    sess.shuffle = False
    sess.outs = -1
    dt = sess.get_dataloader(sess.test_data_path, train_mode=False)

    ssim = []
    psnr = []

    # widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    # bar = progressbar.ProgressBar(widgets=widgets, maxval=len(dt)).start()
    for i, batch in enumerate(dt):
        pred, B, losses, mask = sess.inf_batch('test', batch)
        pred, B = pred[0], B[0]
        mask = mask.cpu().data
        mask = mask * 255
        mask = np.clip(mask.numpy(), 0, 255).astype('uint8')
        mask = sess.heatmap(mask)
        mask = np.transpose(mask[0], (1, 2, 0))
        pred = np.transpose(pred.numpy(), (1, 2, 0))
        B = np.transpose(B.numpy(), (1, 2, 0))
        pred = np.clip(pred, 0, 1)
        B = np.clip(B, 0, 1)
        ssim.append(ms.compare_ssim(pred, B, multichannel=True))
        psnr.append(ms.compare_psnr(pred, B))
        pred = pred * 255
        ensure_dir('../realtest/derain5_real/')
        cv2.imwrite('../realtest/derain5_real/{}.png'.format(i + 1), pred)
        cv2.imwrite('../realtest/derain5_real/{}m.jpg'.format(i + 1), mask)
        # bar.update(i + 1)
    print(np.mean(ssim), np.mean(psnr))


if __name__ == '__main__':
    import platform

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='latest')
    parser.add_argument('--dataset', default='Rain100L', type=str,
                        choices=[None, 'Rain200H', 'Rain100L', 'Rain200H',
                                 'Rain100H', 'PReNetData', 'DID', 'SPA', 'DDN',
                                 'test12', 'real', ],
                        help="set dataset name for training"
                             "real/test12 is eval-only")
    parser.add_argument('--test_every', type=float, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--no_augment', action='store_true', default=True,
                        help='do not use data augmentation')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args(sys.argv[1:])
    args.batch_size = 16
    args.workers = 4
    args.scale = [1]
    args.patch_size = 128 #256
    args.ext = 'sep'
    device = torch.device(args.device)

    set_random_seed(1)
    # net = SPANet().to(args.device)
    '''

    [938, 1177, 1986, 2351, 1753, 13, 484, 1178, 1013, 273, 69, 1143, 1230, 2051, 1309, 277, 250, 2214, 1340, 1277, 186, 346, 2329, 1741, 2056, 658, 322, 2333, 1144, 2082, 1248, 2213, 1586, 2189, 2208, 729, 603, 1617, 673, 
    '''
    if platform.system() == 'Linux':
        args.data_dir = '/home/office-409/Datasets/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'

    if args.action == 'train':
        run_train_val(args, args.model)
    elif args.action == 'test':
        run_test(args.model)
