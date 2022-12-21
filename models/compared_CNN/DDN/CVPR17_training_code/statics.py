import datetime
import os
import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import glob
import cv2
import numpy as np
import random
from UDL.UDL.derain.compared_CNN.DDN.CVPR17_training_code.training_DDN import inference
import matplotlib.pyplot as plt
from UDL.UDL.Basis.auxiliary import log_string, create_logger
patch_size = 48
image_size = 1024
# _, ax1 = plt.subplots(1, 1)
# fig, ax = plt.subplots(ncols=2, nrows=1)
# plt.ion()


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops/1000000000.0, params.total_parameters))


def crop(img_pair, shape):
    # patch_size = .patch_size
    O_list = []
    B_list = []
    for img, (h, ww, _) in zip(img_pair, shape):
    # h, ww, _ = shape
        h_pad = image_size - h
        w_pad = image_size - ww
        # print(h, ww, h_pad, w_pad)
        # ax1.imshow(img)

        img = img[h_pad:, w_pad:, :]

        h, ww, c = img.shape

        w = ww // 2
        p_h = p_w = patch_size

        # ax[0].imshow(img[:, :w, :])
        # ax[1].imshow(img[:, w+1:ww, :])
        # plt.pause(1000)
        # plt.show()

        # if aug:
        #     mini = - 1 / 4 * patch_size
        #     maxi = 1 / 4 * patch_size + 1
        #     p_h = patch_size + self.rand_state.randint(mini, maxi)
        #     p_w = patch_size + self.rand_state.randint(mini, maxi)
        # else:
        #     p_h, p_w = patch_size, patch_size
        #
        # r = self.rand_state.randint(0, h - p_h)
        # c = self.rand_state.randint(0, w - p_w)
        r = random.randrange(0, h - p_h + 1)
        c = random.randrange(0, w - p_w + 1)

        # O = img_pair[:, w:]
        # B = img_pair[:, :w]
        O_list.append(img[r: r + p_h, c + w: c + p_w + w, :])  # rain 右边
        B_list.append(img[r: r + p_h, c: c + p_w, :])  # norain 左边
        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)
        # ax[0].imshow(B_list[-1])
        # ax[1].imshow(O_list[-1])
        # print(O_list[-1].shape, B_list[-1].shape)
        # plt.pause(1)
        # plt.show()
    # if aug:
    #     O = cv2.resize(O, (patch_size, patch_size))
    #     B = cv2.resize(B, (patch_size, patch_size))

    return np.stack(O_list, axis=0),  np.stack(B_list, axis=0)

def decode(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
          'orishape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
      })

  #
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image = tf.reshape(image, features['shape'])
  # height =
  # image.set_shape((mnist.IMAGE_PIXELS))
  # label = tf.cast(features['label'], tf.int32)
  return image, features['orishape']



def inputs(tfrecords_path, patch_size, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    with tf.name_scope("input"):
        #读取tfrecords文件
        dataset = tf.data.TFRecordDataset(tfrecords_path)
        #tfrecords数据解码
        dataset = dataset.map(decode)
        # dataset = dataset.map(crop)
        # dataset = dataset.map(normalize)
        #打乱数据的顺序
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()



# model_path = './TrainedModel/model-Rain200L' # trained model



def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=3)
  rain = tf.cast(image_decoded, tf.float32)/255.
  return rain

# tf_image_batch, tf_shape = inputs("./rain100L", patch_size=100, batch_size=10, num_epochs=300)
# with tf.Session() as sess:
#     image_batch, shape = sess.run([tf_image_batch, tf_shape])
#     print(image_batch.shape, shape)

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

if __name__ == '__main__':
    import argparse
    import platform
    from UDL.UDL.derain.common.tf_data.data_loader import derainSession

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    script_path = os.path.dirname(os.path.dirname(__file__))
    root_dir = script_path.split("derain")[0]

    model_path = '{}/results/derain/FuGCN/Test/model_2021-11-22-23-26/.pth.tar'.format(root_dir)
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

    parser.add_argument('--lr', default=0.1, type=float)  # 1e-4 2e-4 8
    parser.add_argument('--lr_scheduler', default=True, type=bool)
    parser.add_argument('--samples_per_gpu', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--epochs', default=350, type=int) #iterations = int(2.1 * 1e5)  # iterations
    parser.add_argument('--workers_per_gpu', default=0, type=int)
    parser.add_argument('--resume',
                        default=model_path,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    ##
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DDN', type=str)
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
    args.experimental_desc = 'Test'
    args.patch_size = 64
    args.global_rank = 0
    args.workers = 4
    if platform.system() == 'Linux':
        args.data_dir = '/works/dataset/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'
    out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
    sess_t = derainSession(args)
    train_loader = sess_t.get_dataloader(mode="torchGen", dataset_name=args.dataset, gen="gen")#()
    log_string(args.dataset)
    # rainy, labels = train_loader.get_next()
    with tf.Graph().as_default() as graph:
        ############## placeholder for training
        gt = tf.placeholder(dtype=tf.float32, shape=[args.samples_per_gpu, args.patch_size, args.patch_size, 3])
        inputs = tf.placeholder(dtype=tf.float32, shape=[args.samples_per_gpu, args.patch_size, args.patch_size, 3])


        ######## network architecture
        outputs = inference(inputs, is_training=True)
        ######## loss function
        loss = tf.reduce_mean(tf.square(gt - outputs))  # MSE loss

        ##### Loss summary
        # l1_loss_sum = tf.summary.scalar("l1_loss", l1)
        # all_sum = tf.summary.merge([l1_loss_sum])

        lr_ = args.lr
        lr = tf.placeholder(tf.float32, shape=[])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_optim = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        #### Run the above




        stats_graph(graph)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            count_flops(graph)