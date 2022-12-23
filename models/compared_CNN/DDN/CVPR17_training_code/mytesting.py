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
from training_DDN import inference
from skimage.io import imsave
##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
############################################################################

tf.reset_default_graph()

model_path = './model_DDN/'
# img_path = 'D:/Datasets/derain/DDN/Rain1400/rainy_image/'  # the path of testing images
results_path = './TestData/results/real/'  # the path of de-rained images

os.makedirs(results_path, exist_ok=True)


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0
    return rainy


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
    # rain_pad = tf.pad(rain, [[0, 0], [10, 10], [10, 10], [0, 0]], "SYMMETRIC")

    output = inference(rain, is_training=False)

    output = tf.clip_by_value(output, 0., 1.)
    output = output[0, :, :, :]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # GPU setting
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            if tf.train.get_checkpoint_state(model_path):
                ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading model")
            else:
                saver.restore(sess, model_path)  # try a pre-trained model
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
