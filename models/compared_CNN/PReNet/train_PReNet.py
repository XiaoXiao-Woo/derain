import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DerainDataset import *
from util import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *
from common.derain_dataset import derainSession
import matplotlib.pyplot as plt
from UDL.Basis.auxiliary import set_random_seed

# 168963
parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--samples_per_gpu", type=int, default=18, help="Training batch size")
parser.add_argument('--workers_per_gpu', default=0, type=int)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_dir", type=str, default="D:/Datasets/derain",
                    help='path to training data')  # datasets/train/Rain12600
parser.add_argument('--dataset', default='Rain200L', type=str,
                    choices=[None, 'Rain200H', 'Rain100L', 'Rain200H',
                             'Rain100H', 'PReNetDataL', 'PReNetDataH', 'DID', 'SPA', 'DDN',
                             'test12', 'real', ],
                    help="set dataset name for training"
                         "real/test12 is eval-only")

parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--model', default='PReNet',
                    help='model name')

#Data
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=100,
                    help='image2patch, set to model and dataset')
parser.add_argument('--test_every', type=int, default=22,
                    help='do test per every N batches')
parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                    help='train dataset name')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
opt = parser.parse_args()
opt.scale = [1]

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

fig, axes = plt.subplots(ncols=2, nrows=1)


def main():
    print('Loading dataset ...\n')
    sess = derainSession(opt)
    loader_train, _ = sess.get_dataloader(opt.dataset, False)
    set_random_seed(1)
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    # initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    # if initial_epoch > 0:
    #     print('resuming by loading epoch %d' % initial_epoch)
    #     model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(0, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, batch in enumerate(loader_train, 0):
            input_train = batch['O']
            target_train = batch['B']
            # axes[0].imshow(input_train[0].permute(1, 2, 0).cpu().numpy())
            # axes[1].imshow(target_train[0].permute(1, 2, 0).cpu().numpy())
            # plt.savefig(f"{i}_{batch['filename'][0]}.png")

            # (input_train, target_train)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # # training curve
            # model.eval()
            # out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                # writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end

        # # log the images
        # model.eval()
        # out_train, _ = model(input_train)
        # out_train = torch.clamp(out_train, 0., 1.)
        # im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        # im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        # im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', im_target, epoch+1)
        # writer.add_image('rainy image', im_input, epoch+1)
        # writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    # if opt.preprocess:
    #     if opt.data_path.find('RainTrainH') != -1:
    #         print(opt.data_path.find('RainTrainH'))
    #         prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
    #     elif opt.data_path.find('RainTrainL') != -1:
    #
    #         prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
    #     elif opt.data_path.find('Rain12600') != -1:
    #         prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
    #     else:
    #         print('unkown datasets: please define prepare data function in DerainDataset.py')

    main()
