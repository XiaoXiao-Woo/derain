# -*- coding: utf-8 -*-
# !/usr/bin/env python2


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, X. Ding, Y. Liao and J. Paisley. “Clearing the Skies: A deep network architecture for single-image rain removal”, 
# IEEE Transactions on Image Processing, vol. 26, no. 6, pp. 2944-2956, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import time

import skimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from training_clear import inference
from skimage.io import imsave
##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

model_path = './model_DDN/'
pre_trained_model_path = './model_200H'
# pre_trained_model_path = 'D:/ProjectSets/NDA/Attention/UDL/UDL/results/derain/DID/Clear/Test/model_2022-01-06-16-16/170'
# img_path = 'D:/Datasets/derain/DDN/Rain1400/rainy_image/'  # the path of testing images
results_path = './TestData/results/real/'  # the path of de-rained images

os.makedirs(results_path, exist_ok=True)


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0
    return rainy


# class PatchMergeModule:
#
#     def __init__(self, args, inferece):
#         super().__init__()
#         self.args = args
#         self.forward = inference
#
#     def forward_chop(self, x, shave=12):
#         args = self.args
#         # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         x.cpu()
#         batchsize = self.args.crop_batch_size
#         h, w = x.size()[-2:]
#         padsize = int(args.patch_size)
#         shave = int(args.patch_size / 2)
#         # print(self.scale, self.idx_scale)
#         scale = 1  # self.scale[self.idx_scale]
#
#         h_cut = (h - padsize) % (int(shave / 2))
#         w_cut = (w - padsize) % (int(shave / 2))
#
#         x_unfold = tf.image.extract_image_patches(x, ksizes=[1, padsize, padsize, 1],
#                     strides=[1, int(shave / 2), int(shave / 2), 1], rates=[1, 1, 1, 1], padding='SAME')
#         #F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()
#
#         ################################################
#         # 最后一块patch单独计算
#         ################################################
#
#         x_hw_cut = x[..., (h - padsize):, (w - padsize):]
#         y_hw_cut = self.forward(x_hw_cut.cuda()).cpu()
#
#         x_h_cut = x[..., (h - padsize):, :]
#         x_w_cut = x[..., :, (w - padsize):]
#         y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#         y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#         # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # 左上patch单独计算，不是平均而是覆盖
#         ################################################
#
#         x_h_top = x[..., :padsize, :]
#         x_w_top = x[..., :, :padsize]
#         y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#         y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
#
#         # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # img->patch，最大计算crop_s个patch，防止bs*p*p太大
#         ################################################
#
#         x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
#         y_unfold = []
#
#         x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
#         x_unfold.cuda()
#         for i in range(x_range):
#             y_unfold.append(self.forward(
#                 x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
#             # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#
#         y_unfold = tf.concat(y_unfold, dim=0)
#
#         # y = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#         #            ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
#         #            stride=int(shave / 2 * scale))
#         y = tf.image.extract_image_patches(y_unfold, ksizes=[1, (h - h_cut) * scale, (w - w_cut) * scale, 1],
#                     strides=[1, int(shave / 2 * scale), int(shave / 2 * scale), 1], rates=[1, 1, 1, 1], padding='SAME')
#
#         # 312， 480
#         # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # 第一块patch->y
#         ################################################
#         y[..., :padsize * scale, :] = y_h_top
#         y[..., :, :padsize * scale] = y_w_top
#         # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         y_unfold = y_unfold[...,
#                    int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
#                    int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
#         # 1，3，24，24
#         # y_inter = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#         #                  ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
#         #                  padsize * scale - shave * scale,
#         #                  stride=int(shave / 2 * scale))
#         y_inter = tf.reshape(y_unfold, [1, (h - h_cut - shave) * scale, (w - w_cut - shave) * scale, 3])
#         y_inter = tf.space_to_depth(y_inter, p)
#         y_inter = tf.reshape(y_inter, [(h - h_cut - shave) * scale, (h - h_cut - shave) * scale, 3])
#
#
#         y_ones = tf.ones(y_inter.shape, dtype=y_inter.dtype)
#         # divisor = F.fold(F.unfold(y_ones, padsize * scale - shave * scale,
#         #                           stride=int(shave / 2 * scale)),
#         #                  ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
#         #                  padsize * scale - shave * scale,
#         #                  stride=int(shave / 2 * scale))
#
#
#         y_inter = y_inter / divisor
#         # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         ################################################
#         # 第一个半patch
#         ################################################
#         y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
#         int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_inter
#         # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
#                        y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
#         # 图分为前半和后半
#         # x->y_w_cut
#         # model->y_hw_cut
#         y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
#                              y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
#         y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
#                        y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
#         # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
#         # plt.show()
#
#         return y.cuda()
#
#     def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
#
#         x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0,
#                                                                                      2).contiguous()
#
#         x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
#         x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
#         y_h_cut_unfold = []
#         x_h_cut_unfold.cuda()
#         for i in range(x_range):
#             y_h_cut_unfold.append(self.forward(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
#                                                ...]).cpu())  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#         y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)
#
#         y_h_cut = F.fold(
#             y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#             (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
#         y_h_cut_unfold = y_h_cut_unfold[..., :,
#                          int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
#         y_h_cut_inter = F.fold(
#             y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#             (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
#             stride=int(shave / 2 * scale))
#
#         y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
#         divisor = F.fold(
#             F.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
#                      stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
#             (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
#         y_h_cut_inter = y_h_cut_inter / divisor
#
#         y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter
#         return y_h_cut
#
#     def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
#
#         x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
#                                                                                                        2).contiguous()
#
#         x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
#         x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
#         y_w_cut_unfold = []
#         x_w_cut_unfold.cuda()
#         for i in range(x_range):
#             y_w_cut_unfold.append(self.forward(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
#                                                ...]).cpu())  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
#         y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)
#
#         y_w_cut = torch.nn.functional.fold(
#             y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#             ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
#         y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
#                          :].contiguous()
#         y_w_cut_inter = torch.nn.functional.fold(
#             y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
#             ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
#             stride=int(shave / 2 * scale))
#
#         y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
#         divisor = torch.nn.functional.fold(
#             torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
#                                        stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
#             (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
#         y_w_cut_inter = y_w_cut_inter / divisor
#
#         y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter
#         return y_w_cut

if __name__ == '__main__':
    from UDL.derain.common.tf_data.data_loader import DataLoader, derainSession
    import argparse
    import platform

    parser = argparse.ArgumentParser(description='TF Testting')
    # parser.add_argument('--patch_size', type=int, default=128,
    #                     help='image2patch, set to model and dataset')

    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('-samples_per_gpu', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--test_every', type=int, default=-1,
                        help='do test per every N batches')
    args = parser.parse_args()
    if platform.system() == 'Linux':
        args.data_dir = '/Data/DataSet/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'

    args.eval = True

    # imgName = os.listdir(img_path)
    # num_img = len(imgName)
    #
    # whole_path = []
    # for i in range(num_img):
    #     whole_path.append(img_path + imgName[i])

    sess_t = derainSession(args)
    if args.eval:
        print(args.data_dir)
        loader = sess_t.get_eval_loader("torchGen", "real", "gen")

        # filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)
    # dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    # dataset = dataset.map(_parse_function)
    # dataset = dataset.prefetch(buffer_size=10)
    # dataset = dataset.batch(batch_size=1).repeat()
    # iterator = dataset.make_one_shot_iterator()
    t0 = time.time()
    # rain = iterator.get_next()
    rain = tf.placeholder(dtype=tf.float32, shape=[args.samples_per_gpu, None, None, 3])
    rain_pad = tf.pad(rain, [[0, 0], [10, 10], [10, 10], [0, 0]], "SYMMETRIC")

    detail, base = inference(rain_pad)

    detail = detail[:, 6:tf.shape(detail)[1] - 6, 6:tf.shape(detail)[2] - 6, :]
    base = base[:, 10:tf.shape(base)[1] - 10, 10:tf.shape(base)[2] - 10, :]

    output = tf.clip_by_value(base + detail, 0., 1.)
    output = output[0, :, :, :]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # GPU setting
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            if tf.train.get_checkpoint_state(pre_trained_model_path):
                ckpt = tf.train.latest_checkpoint(pre_trained_model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading model")
            else:
                saver.restore(sess, pre_trained_model_path)  # try a pre-trained model
                print("Loading pre-trained model")

            # for i in range(num_img):
            for i, batch in enumerate(loader):
                O = batch['O']
                filename = batch['filename']

                derained, ori = sess.run([output, rain], feed_dict={rain: O / 255.0})

                derained = np.uint8(derained * 255.)
                # index = imgName[i].rfind('.')
                # name = imgName[i][:index]
                imsave(results_path + filename[0] + '.png', derained)
                print('%d / %d images processed' % (i + 1, len(loader)), filename[0])
        print(time.time() - t0, len(loader))
        print('All done')
    sess.close()

    plt.subplot(1, 2, 1)
    plt.imshow(ori[0, :, :, :])
    plt.title('rainy')
    plt.subplot(1, 2, 2)
    plt.imshow(derained)
    plt.title('derained')
    plt.show()
