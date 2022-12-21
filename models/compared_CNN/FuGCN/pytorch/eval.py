import os
import sys
import cv2
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

#from tensorboardX import SummaryWriter

import settings
from dataset import TestDataset,TrainValDataset
from model_fu import Net
from cal_ssim import SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#Torch.cuda.set_device(settings.device_id)
test_dir = '../testdir'
gt_dir = '../gtdir'

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def save_image( name, img):
    img = img.data.cpu().numpy() * 255
    img = np.squeeze(img)
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img_file = os.path.join(test_dir, '%d.png' % (name))
    cv2.imwrite(img_file, img)
def gt_image( name, img):
    img = img.data.cpu().numpy() * 255
    img = np.squeeze(img)
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img_file = os.path.join(gt_dir, '%d.png' % (name))
    cv2.imwrite(img_file, img)

class Session:
    def __init__(self):
        self.test_dir = settings.test_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if len(settings.device_id) >1:
            self.net = nn.DataParallel(Net()).cuda()
        else:
            self.net = Net().cuda()
        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        #dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
    def loss_vgg(self,input,groundtruth):
        vgg_gt = self.vgg.forward(groundtruth)
        eval = self.vgg.forward(input)
        loss_vgg = [self.l1(eval[m], vgg_gt[m]) for m in range(len(vgg_gt))]
        loss = sum(loss_vgg)
        return loss



    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)

       # B = B[:, :, :-1, :-1]
        gt_image(name,B)
        #O = O[:, :, :-1, :-1]
        with torch.no_grad():
            coarse, derain = self.net(O)
            save_image(name, coarse)
        l1_loss = self.l1(coarse, B)
        ssim = self.ssim(coarse, B)
        psnr = PSNR(coarse.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = { 'L1 loss' : l1_loss }
        ssimes = { 'ssim' : ssim }
        losses.update(ssimes)

        return losses, psnr




def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)
    dt = sess.get_dataloader('test')
    psnr_all = 0
    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses,psnr= sess.inf_batch( i,batch)
        psnr_all=psnr_all+psnr
        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))
    #psnr=sum(psnr_all)
    #print(psnr)
    print('psnr_ll:%8f'%(psnr_all/all_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='net_460_epoch')
    #parser.add_argument('-m', '--model', default='latest_net')

    args = parser.parse_args(sys.argv[1:])
    run_test(args.model)

