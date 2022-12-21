import logging

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
import random
from torch.backends import cudnn
import imageio


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


fig, axes = plt.subplots(ncols=2, nrows=2)
# data_dir = "D:/Datasets/derain"

def make_dataset(dir):

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot, ', dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                item = path
                images.append(item)
    return images

class prepareDID_Data:
    def __init__(self, root):
        self.root = root

    def pix2pix(self, root):
        ...

    def pix2pix_class(self, index):
        index_sub = np.random.randint(0, 3)
        label = index_sub
        if index_sub == 0:
            index = np.random.randint(0, 4000)
            path = self.root + '/DID-MDN-training/Rain_Heavy/train2018new' + '/' + str(
                index) + '.jpg'

        if index_sub == 1:
            index = np.random.randint(0, 4000)
            path = self.root + '/DID-MDN-training/Rain_Medium/train2018new' + '/' + str(
                index) + '.jpg'

        if index_sub == 2:
            index = np.random.randint(0, 4000)
            path = self.root + '/DID-MDN-training/Rain_Light/train2018new/' + str(
                index) + '.jpg'


        img_pair = imageio.imread(path)

        return img_pair#, label


    def __call__(self, index):
        return self.pix2pix_class(index)


def prepareDDN_Data():
    input_path = "D:/Datasets/derain/DDN/Rain12600/rainy_image/"
    gt_path = "D:/Datasets/derain/DDN/Rain12600/ground_truth/"

    RainName = os.listdir(input_path)
    LabelName = []
    for i in range(len(RainName)):
        LabelName.append(gt_path + RainName[i].split('_')[0] + '.jpg')
        RainName[i] = input_path + RainName[i]

        # print(RainName[i], LabelName[i], len(RainName), len(LabelName))

    return RainName, LabelName



class SlicerDataLoader:
    def __init__(self, args, dataset_name, batch_size, patch_size, gen=2):

        self.args = args
        data_dir = args.data_dir
        self.gen = gen
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.imread = self.default_imread

        if dataset_name in ["Rain100L", "Rain100H", "Rain200L", "Rain200H"]:
            input_path = '/'.join([data_dir, dataset_name, 'train', 'rain/'])
            gt_path = '/'.join([data_dir, dataset_name, 'train', 'norain/'])

            self.RainName = os.listdir(input_path)
            for i in range(len(self.RainName)):
                self.RainName[i] = input_path + self.RainName[i]

            self.LabelName = os.listdir(gt_path)
            for i in range(len(self.LabelName)):
                self.LabelName[i] = gt_path + self.LabelName[i]

            self.file_num = len(self.RainName)

        elif dataset_name in ["Rain12600", "DDN"]:
            self.RainName, self.LabelName = prepareDDN_Data()
            self.file_num = len(self.RainName)

        elif dataset_name == "DID":
            data_dir = '/'.join([data_dir, 'DID-MDN-datasets/'])
            self.file_num = len(make_dataset(data_dir))
            self.imread = prepareDID_Data(data_dir)


        self.index_list = np.random.permutation(self.file_num)
        if dataset_name != "DID":
            self._repeat()
            assert self.file_num == len(self.LabelName), print("Error: data is not same with label")

        self.loader = {
            0: self.default_loader,
            1: self.stitch_loader,
            2: self.dataset_generator
        }[gen]()


    def default_imread(self, image_index):
        sample = imageio.imread(self.RainName[image_index])
        gt = imageio.imread(self.LabelName[image_index])

        return np.concatenate([sample, gt], axis=1)

    def default_loader(self):
        filename_tensor = tf.convert_to_tensor(self.RainName, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.LabelName, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 10)
        dataset = dataset.batch(self.batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()

        return iterator#.get_next()

    def stitch_loader(self):

        filename_tensor = tf.convert_to_tensor(self.RainName, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.LabelName, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 10)
        dataset = dataset.batch(self.batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()

        return iterator#.get_next()

    def dataset_generator(self):
        b = 0
        index = -1
        file_num = self.file_num
        index_list = self.index_list
        error_count = 1
        while True:
            try:
                image_index = index_list[(index + 1) % file_num]
                # img_pair = np.concatenate([sample, gt], axis=1)
                sample, gt = self.crop(self.imread(image_index), False)
                sample = np.expand_dims(sample, axis=0)
                gt = np.expand_dims(gt, axis=0)

                if b == 0:
                    batch_sample_s = sample
                    batch_gt_s = gt
                if b != 0:
                    batch_sample_s = np.append(batch_sample_s, sample, axis=0)
                    batch_gt_s = np.append(batch_gt_s, gt, axis=0)
                b += 1
                index += 1

                if b >= self.batch_size:
                    inputs = [batch_sample_s, batch_gt_s]

                    yield inputs

                    b = 0



            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}, {}".format(
                    self.RainName[image_index], self.LabelName[image_index]))
                error_count += 1
                if error_count > 5:
                    raise

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h = p_w = patch_size
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
        O = img_pair[r: r + p_h, c + w: c + p_w + w]  # rain
        B = img_pair[r: r + p_h, c: c + p_w]  # norain
        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)

        # print("coord:", [r, r + p_h, c + w, c + p_w + w, r, r + p_h, c, c + p_w])

        return O, B

    def __call__(self, *args, **kwargs):

        return self.loader#self.sess.run(self.loader)

    def __next__(self):

        return next(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def _repeat(self):
        batch_size = self.args.batch_size
        test_every = self.args.test_every
        file_num = self.file_num
        """srdata"""
        if test_every > 0:
            n_patches = batch_size * test_every#8000 // 2000
            # n_images = len(self.images_hr) #len(args.data_train) is list
            if file_num == 0:
                repeat = 0
            else:
                repeat = int(max(n_patches / file_num, 1))

            self.file_num = file_num * repeat
            self.RainName = self.RainName * repeat
            self.LabelName = self.LabelName * repeat
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(repeat, batch_size, self.file_num, len(self.RainName)))

        else:
            test_every = -test_every
            self.file_num = file_num * test_every
            self.RainName = self.RainName * test_every
            self.LabelName = self.LabelName * test_every
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(test_every, batch_size, self.file_num, len(self.RainName)))


    # randomly select image patches
    def _parse_function(self, filename, label):
        patch_size = self.patch_size
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        rainy = tf.cast(image_decoded, tf.float32) / 255.0

        image_string = tf.read_file(label)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        label = tf.cast(image_decoded, tf.float32) / 255.0

        ################################################################
        t = time.time()
        rainy = tf.random_crop(rainy, [patch_size, patch_size, 3], seed=t)  # randomly select patch
        label = tf.random_crop(label, [patch_size, patch_size, 3], seed=t)
        return rainy, label

    def default_parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        img = tf.cast(image_decoded, tf.float32) / 255.
        return img


class TFRecordDataloader:
    def __init__(self, dataset_name, patch_size, batch_size, num_epochs):

        if dataset_name == "Rain100L":
            self.tfrecords_path = "./rain100L"
        else:
            raise NotImplementedError

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def decode(self, serialized_example):
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

    def __call__(self):
        if not self.num_epochs:
            self.num_epochs = None
        with tf.name_scope("input"):
            # 读取tfrecords文件
            dataset = tf.data.TFRecordDataset(self.tfrecords_path)
            # tfrecords数据解码
            dataset = dataset.map(self.decode)
            # dataset = dataset.map(crop)
            # dataset = dataset.map(normalize)
            # 打乱数据的顺序
            dataset = dataset.shuffle(1000 + 3 * self.batch_size)
            dataset = dataset.repeat(self.num_epochs)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


class derainSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.patch_size = args.patch_size
        self.args = args

    def get_dataloader(self, mode, dataset_name, gen=2):

        if mode == "TFRecord":
            return TFRecordDataloader(dataset_name, patch_size=100, batch_size=10, num_epochs=300)

        elif mode == "Slicer":
            return SlicerDataLoader(self.args, dataset_name, patch_size=100, batch_size=10, gen=gen)


import argparse
import platform

def main():
    plt.ion()
    parser = argparse.ArgumentParser(description='TF Training')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='image2patch, set to model and dataset')

    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=8, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--test_every', type=int, default=-1,
                        help='do test per every N batches')

    args = parser.parse_args()

    if platform.system() == 'Linux':
        args.data_dir = '/home/office-409/Datasets/derain'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/derain'

    ##############################
    # tf.Graph

    ##############################
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    ##############################
    sess_t = derainSession(args)
    train_loader = sess_t.get_dataloader(mode="Slicer", dataset_name="Rain200L", gen=1)()
    for j in range(1, 100):
        # for samples, gt in train_loader:
        tf_samples, tf_gt = train_loader.get_next()
        samples, gt = sess.run([tf_samples, tf_gt])
        axes[0][0].imshow(samples[0])
        axes[0][1].imshow(gt[0])
        # plt.show()
        # plt.pause(0.5)
        axes[1][0].imshow(samples[1])
        axes[1][1].imshow(gt[1])
        plt.show()
        plt.pause(0.5)
        print(samples.shape, gt.shape)


if __name__ == '__main__':
    # plt.ion()
    # tf_image_batch, tf_shape = TFRecordDataloader("./rain100L", patch_size=100, batch_size=10, num_epochs=300)()
    # with tf.Session() as sess:
    #     image_batch, shape = sess.run([tf_image_batch, tf_shape])
    #     print(image_batch.shape, shape)

    # tf_rainy, tf_labels = SlicerDataLoader("Rain100L", patch_size=100, batch_size=10)()
    # tf_rainy, tf_labels = SlicerDataLoader("DDN", patch_size=100, batch_size=10, sess=sess)()

    # for j in range(1, 100):

    # axes[0].imshow(samples[0])
    # axes[1].imshow(gt[0])
    # plt.show()
    # plt.pause(0.5)
    # print(samples.shape, gt.shape)

    main()
