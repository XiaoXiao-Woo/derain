import datetime
import os
import re

import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
# from UDL.Basis.auxiliary import log_string, create_logger
from GuidedFilter import guided_filter
##################### Network parameters ###################################
num_feature = 512  # number of feature maps
num_channels = 3  # number of input's channels
patch_size = 64  # patch size
learning_rate = 1e-3  # learning rate
iterations = int(2e5)  # iterations
batch_size = 10  # batch size
save_model_path = "./model_DID/"  # saved model's path
model_name = 'model-iter'  # saved model's name
############################################################################

# DerainNet
def inference(images):
    with tf.variable_scope('DerainNet', reuse=tf.AUTO_REUSE):
        base = guided_filter(images, images, 15, 1, nhwc=True)  # using guided filter for obtaining base layer
        detail = images - base  # detail layer

        conv1 = tf.layers.conv2d(detail, num_feature, 16, padding="valid", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, num_feature, 1, padding="valid", activation=tf.nn.relu)
        output = tf.layers.conv2d_transpose(conv2, num_channels, 8, strides=1, padding="valid")

    return output, base


def train_one_epoch():
    ...

if __name__ == '__main__':
    import argparse
    import platform
    from UDL.UDL.derain.common.tf_data.data_loader import derainSession

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    script_path = os.path.dirname(os.path.dirname(__file__))
    root_dir = script_path.split("derain")[0]

    model_path = 'D:/ProjectSets/NDA/Attention/UDL/UDL/results/derain/DID/Clear/Test/model_2022-01-06-16-16/160'
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    parser = argparse.ArgumentParser(description='TF derain Training')
    parser.add_argument('--use-log', default=True
                        , type=bool)
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--use-tb', default=False, type=bool)
    parser.add_argument('--out_dir', metavar='DIR', default='{}/results/{}'.format(root_dir, "derain"),
                        help='path to save model')

    parser.add_argument('--lr', default=learning_rate, type=float)  # 1e-4 2e-4 8
    parser.add_argument('--lr_scheduler', default=True, type=bool)
    parser.add_argument('--samples_per_gpu', default=batch_size, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--epochs', default=170, type=int)
    parser.add_argument('--workers_per_gpu', default=0, type=int)
    parser.add_argument('--resume',
                        default=model_path,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    ##
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Clear', type=str)
    parser.add_argument('--dataset', default='Rain200L', type=str,
                        choices=[None, 'Rain100L', 'Rain100H',
                                 'Rain200L', 'Rain200H', 'DDN',
                                 'DID'],
                        help="performing evalution for patch2entire")
    parser.add_argument('--eval', default=False, type=bool,
                        help="performing evalution for patch2entire")

    parser.add_argument('--ext', type=str, default='sep',
                        help='dataset file extension')
    parser.add_argument('--crop_batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--test_every', type=int, default=-2,
                        help='do test per every N batches')
    parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                        help='train dataset name')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')

    args = parser.parse_args()
    args.start_epoch = args.best_epoch = 1
    args.eval = False
    args.experimental_desc = 'Test'
    args.patch_size = 64
    args.global_rank = 0
    args.workers = 4
    if platform.system() == 'Linux':
        args.data_dir = '/Data/DataSet/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'
    # out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
    model_save_dir = ' D:/ProjectSets/NDA/Attention/UDL/UDL/results/derain/DID/Clear/Test/model_2022-01-06-16-16/160'
    sess_t = derainSession(args)
    if args.eval:
        loader = sess_t.get_eval_loader("torchGen", args.dataset, "gen")
    else:
        loader = sess_t.get_dataloader(mode="torchGen", dataset_name=args.dataset, gen="gen")#()
    # log_string(args.dataset)
    # rainy, labels = train_loader.get_next()

    ############## placeholder for training
    labels = tf.placeholder(dtype=tf.float32, shape=[args.samples_per_gpu, args.patch_size, args.patch_size, 3])
    rainy = tf.placeholder(dtype=tf.float32, shape=[args.samples_per_gpu, args.patch_size, args.patch_size, 3])

    details_label = labels - guided_filter(labels, labels, 15, 1, nhwc=True)
    details_label = details_label[:, 4:patch_size - 4, 4:patch_size - 4, :]  # output size 56
    ######## network architecture
    details_output, _ = inference(rainy)
    ######## loss function
    loss = tf.reduce_mean(tf.square(details_label - details_output))  # MSE loss

    all_vars = tf.trainable_variables()
    g_optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=all_vars)  # optimizer
    print("Total parameters' number: %d" % (np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Run the above

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    saver = tf.train.Saver(var_list=all_vars, max_to_keep=5)
    with tf.Session() as sess:
        sess.run(init)

        resume = args.resume
        # if os.path.isfile(resume):
        #     print('Loading Model...')
        #     ckpt = tf.train.get_checkpoint_state(resume)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     start_epoch = int(resume.split('/')[-1])
        if tf.train.get_checkpoint_state(resume):  # load previous trained model
            ckpt = tf.train.latest_checkpoint(resume)
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r'(\w*[0-9]+)\w*', ckpt)
            start_epoch = int(ckpt_num[-1]) + 1
            print("Load success")
        else:
            start = datetime.datetime.now()
            start_epoch = 1# if model_path == '' else int(model_path.split('/')[-1]) + 1
            print("re-training")
        loss_train = []
        # mse_valid = []
        epochs = args.epochs
        # file = open('train_mse.txt', 'w')  # write the training error into train_mse.txt
        for epoch in range(start_epoch, args.epochs+1):
            # for i in range(1, iterations+1):
            for i, (O, B) in enumerate(loader):
                ###################################################################
                #### training phase! ###########################
                i += 1
                # print("Check patch pair:")
                # plt.subplot(1,2,1)
                # plt.imshow(O[0,:,:,:])
                # plt.title('input')
                # plt.subplot(1,2,2)
                # plt.imshow(B[0,:,:,:])
                # plt.title('ground truth')
                # plt.show()

                # image_batch, image_shape = sess.run([tf_image_batch, tf_shape])
                # image_batch_b = image_batch / 255.
                # O, B = crop(image_batch_b, image_shape)

                _, l1_loss = sess.run([g_optim, loss], feed_dict={labels: B / 255.0, rainy: O / 255.0}) #, all_sum
                if i % 30 == 0:
                    print("Epoch: " + str(epoch) + " Iter: " + str(i) + " l1: " + str(l1_loss))
            if epoch == epochs or epoch % 20 == 0 and epoch != 0:
                saver.save(sess, 'D:/ProjectSets/NDA/Attention/UDL/UDL/results/derain/DID/Clear/Test/model_2022-01-06-16-16' + '/' + str(epoch) + '/model-' + str(epoch) + '.ckpt')
                print("Save Model")

        end = datetime.datetime.now()
        print('time cost of Clear = {}s'.format (str(end - start)))
        # Total parameters' number: 754691
        # time cost of Clear = 9:25:41.977406s