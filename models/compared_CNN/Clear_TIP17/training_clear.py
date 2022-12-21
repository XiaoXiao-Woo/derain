# -*- coding: utf-8 -*-
# !/usr/bin/env python2


# This is a re-implementation of training code of this paper:
# X. Fu, J. Huang, X. Ding, Y. Liao and J. Paisley. “Clearing the Skies: A deep network architecture for single-image rain removal”, 
# IEEE Transactions on Image Processing, vol. 26, no. 6, pp. 2944-2956, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from GuidedFilter import guided_filter

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

##################### Network parameters ###################################
num_feature = 512  # number of feature maps
num_channels = 3  # number of input's channels
patch_size = 64  # patch size
learning_rate = 1e-3  # learning rate
iterations = int(2e5)  # iterations
batch_size = 10  # batch size
save_model_path = "./model_DDN/"  # saved model's path
model_name = 'model-iter'  # saved model's name
############################################################################

# input_path = "./TrainData/input/"  # the path of training data
# gt_path = "./TrainData/label/"  # the path of training label
# input_path = "D:/Datasets/derain/Rain200L/train/rain/"
# gt_path = "D:/Datasets/derain/Rain200L/train/norain/"
#
# # randomly select image patches
# def _parse_function(filename, label):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#     rainy = tf.cast(image_decoded, tf.float32) / 255.0
#
#     image_string = tf.read_file(label)
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#     label = tf.cast(image_decoded, tf.float32) / 255.0
#
#     t = time.time()
#     rainy = tf.random_crop(rainy, [patch_size, patch_size, 3], seed=t)  # randomly select patch
#     label = tf.random_crop(label, [patch_size, patch_size, 3], seed=t)
#     return rainy, label


# DerainNet
def inference(images):
    with tf.variable_scope('DerainNet', reuse=tf.AUTO_REUSE):
        base = guided_filter(images, images, 15, 1, nhwc=True)  # using guided filter for obtaining base layer
        detail = images - base  # detail layer

        conv1 = tf.layers.conv2d(detail, num_feature, 16, padding="valid", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, num_feature, 1, padding="valid", activation=tf.nn.relu)
        output = tf.layers.conv2d_transpose(conv2, num_channels, 8, strides=1, padding="valid")

    return output, base


if __name__ == '__main__':
    import platform
    import argparse
    from UDL.UDL.derain.common.tf_data.data_loader import derainSession
    # RainName = os.listdir(input_path)
    # for i in range(len(RainName)):
    #     RainName[i] = input_path + RainName[i]
    #
    # LabelName = os.listdir(gt_path)
    # for i in range(len(LabelName)):
    #     LabelName[i] = gt_path + LabelName[i]

    # filename_tensor = tf.convert_to_tensor(RainName, dtype=tf.string)
    # labels_tensor = tf.convert_to_tensor(LabelName, dtype=tf.string)
    # dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
    # dataset = dataset.map(_parse_function)
    # dataset = dataset.prefetch(buffer_size=batch_size * 10)
    # dataset = dataset.batch(batch_size).repeat()
    # iterator = dataset.make_one_shot_iterator()

    # rainy, labels = iterator.get_next()
    parser = argparse.ArgumentParser(description='TF Training')
    # parser.add_argument('--patch_size', type=int, default=128,
    #                     help='image2patch, set to model and dataset')

    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('-samples_per_gpu', default=8, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--test_every', type=int, default=-1,
                        help='do test per every N batches')

    args = parser.parse_args()

    if platform.system() == 'Linux':
        args.data_dir = '/home/office-409/Datasets/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'
    args.patch_size = patch_size
    sess_t = derainSession(args)
    train_loader = sess_t.get_dataloader(mode="Slicer", dataset_name="DDN", gen="tensor_slices")()
    rainy, labels = train_loader.get_next()
    details_label = labels - guided_filter(labels, labels, 15, 1, nhwc=True)
    details_label = details_label[:, 4:patch_size - 4, 4:patch_size - 4, :]  # output size 56

    details_output, _ = inference(rainy)

    loss = tf.reduce_mean(tf.square(details_label - details_output))  # MSE loss

    all_vars = tf.trainable_variables()
    g_optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=all_vars)  # optimizer
    print("Total parameters' number: %d" % (np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))
    saver = tf.train.Saver(var_list=all_vars, max_to_keep=5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    with tf.Session(config=config) as sess:

        with tf.device('/gpu:0'):
            sess.run(init)
            tf.get_default_graph().finalize()

            if tf.train.get_checkpoint_state(save_model_path):  # load previous trained model
                ckpt = tf.train.latest_checkpoint(save_model_path)
                saver.restore(sess, ckpt)
                ckpt_num = re.findall(r'(\w*[0-9]+)\w*', ckpt)
                start_point = int(ckpt_num[-1]) + 1
                print("Load success")

            else:  # re-training when no models found
                start_point = 0
                print("re-training")

            check_data, check_label = sess.run([rainy, labels])
            # print("Check patch pair:")
            # plt.subplot(1,2,1)
            # plt.imshow(check_data[0,:,:,:])
            # plt.title('input')
            # plt.subplot(1,2,2)
            # plt.imshow(check_label[0,:,:,:])
            # plt.title('ground truth')
            # plt.show()

            start = time.time()

            for j in range(start_point, iterations):  # iterations

                _, Training_Loss = sess.run([g_optim, loss])  # training

                if np.mod(j + 1, 100) == 0 and j != 0:  # save the model every 100 iterations
                    end = time.time()
                    print('%d / %d iteraions, Training Loss  = %.4f, runtime = %.1f s' % (
                    j + 1, iterations, Training_Loss, (end - start)))
                    save_path_full = os.path.join(save_model_path, model_name)
                    saver.save(sess, save_path_full, global_step=j + 1, write_meta_graph=False)
                    start = time.time()

            print('Training is finished.')
    sess.close()
