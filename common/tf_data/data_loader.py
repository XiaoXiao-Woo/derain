import logging

try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import numpy as np
# import torch
import random
# from torch.backends import cudnn
import imageio
import cv2
import collections.abc

container_abcs = collections.abc
from typing import Iterator, Iterable


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
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


import matplotlib.pyplot as plt

def default_loader(path):
    try:
        img_pair = cv2.imread(path).astype(np.float32)
        img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
    except Exception:
        print(path)
    return img_pair  # Image.open(path).convert('RGB')

class prepareDID_Data():
    def __init__(self, root, seed=None):
        self.root = root
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        self.root = root
        self.imgs = imgs
        self.loader = default_loader

        if seed is not None:
            np.random.seed(seed)

    def pix2pix_val(self, index):
        index = index % len(self.imgs)
        index = index + 1
        path = self.root + '/' + str(index) + '.jpg'

        # index_folder = np.random.randint(0, 4)
        # label = index_folder

        # path='/home/openset/Desktop/derain2018/facades/DB_Rain_test/Rain_Heavy/test2018'+'/'+str(index)+'.jpg'
        # print(path)
        img_pair = self.loader(path)

        # img = img.resize((w, h), Image.BILINEAR)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        # NOTE: split a sample into imgA and imgB
        imgA = img_pair[:, :w ]#img.crop((0, 0, w / 2, h))
        # imgC = img.crop((2*w/3, 0, w, h))

        imgB = img_pair[:, w:, :]#img.crop((w / 2, 0, w, h))
        # ax1[0].imshow(imgA)
        # ax1[1].imshow(imgB)
        # imgA = np.transpose(imgA, (2, 0, 1))
        # imgB = np.transpose(imgB, (2, 0, 1))

        return imgA, imgB, str(index)

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

        img_pair = imageio.imread(path)  # imageio.imread(path)

        return img_pair#, path  # , label

    # def __call__(self, index):
    #     return self.pix2pix_class(index)


def prepareDDN_Data(root_dir, dataset):
    input_path = "{}/DDN/{}/rainy_image/".format(root_dir, dataset)
    gt_path = "{}/DDN/{}/ground_truth/".format(root_dir, dataset)

    RainName = os.listdir(input_path)
    LabelName = []
    for i in range(len(RainName)):
        LabelName.append(gt_path + RainName[i].split('_')[0] + '.jpg')
        RainName[i] = input_path + RainName[i]

        # print(RainName[i], LabelName[i], len(RainName), len(LabelName))

    return RainName, LabelName


class Dataset(object):
    # 只要实现了 __getitem__ 方法就可以变成迭代器
    def __getitem__(self, index):
        raise NotImplementedError

    # 用于获取数据集长度
    def __len__(self):
        raise NotImplementedError


class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        # super().__init__(data_source)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        # 调用 sampler 内部的迭代器对象
        for idx in self.sampler:
            batch.append(idx)
            # 如果已经得到了 batch 个 索引，则可以通过 yield
            # 关键字生成生成器返回，得到迭代器对象
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            # 如果最后的索引数不够一个 batch，则抛弃
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class SequentialSampler(Sampler):

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        # 返回迭代器，不然无法 for .. in ..
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._drop_last = loader.drop_last
        self._index_sampler = loader.batch_sampler  # BatchSampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._persistent_workers = loader.persistent_workers
        self._sampler_iter = iter(self._index_sampler)
        # self._base_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        self._num_yielded += 1
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)


class _MapDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.auto_collation = auto_collation
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        # data = [self.dataset[idx] for idx in possibly_batched_index]
        # return self.collate_fn(data)
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0
        self._dataset_fetcher = _MapDatasetFetcher(loader.dataset, loader.auto_collation, loader.collate_fn,
                                                   loader.drop_last)

    # 迭代核心函数，返回的已经是 batch data 了
    def _next_data(self):
        # 输出 batch 个 index
        index = self._next_index()  # may raise StopIteration
        # 迭代 dataset+collate_fn
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        # 数据集
        self.data_source = data_source
        # 是否有放回抽象
        self.replacement = replacement
        # 采样长度，一般等于 data_source 长度
        self._num_samples = num_samples

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        # 通过 yield 关键字返回迭代器对象
        if self.replacement:
            # 有放回抽样
            # 可以直接写 yield from torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
            # 之所以按照每次生成32个，可能是因为想减少重复抽样概率 ?
            for _ in range(self.num_samples // 32):
                yield from np.random.randint(low=0, high=n, size=(32,))  # .tolist()
            yield from np.random.randint(low=0, high=n, size=(self.num_samples % 32,))  # , dtype=torch.int64).tolist()
        else:
            # 无放回抽样
            yield from np.random.permutation(n)  # torch.randperm(n).tolist()

    def __len__(self):
        return self.num_samples

string_classes = (str, bytes)
def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, np.ndarray):
        return np.vstack(batch)
    elif elem_type.__module__ == 'numpy':
        return default_collate([np.array(b) for b in batch])  # [torch.as_tensor(b) for b in batch])
    elif isinstance(elem, float):
        return batch
    elif isinstance(elem, int):
        return batch
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        raise NotImplementedError


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, collate_fn=None, drop_last=False,
                 num_workers=0, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False):
        self.dataset = dataset

        # 因为这两个功能是冲突的，假设shuffle=True,但是sampler里面是SequentialSampler，那么就违背设计思想了
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # 一旦设置了batch_sampler，那么batch_size、shuffle、sampler和drop_last四个参数就不能传入
            # 因为这4个参数功能和batch_sampler功能冲突了
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        # 也就是说batch_sampler必须要存在，你如果没有设置，那么采用默认类
        if batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

    def auto_collation(self):
        return self.batch_sampler is not None

    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self.auto_collation():
            return self.batch_sampler
        else:
            return self.sampler

    # 换一种迭代器写法
    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        # else:
        #     return _MultiProcessingDataLoaderIter(self)

    # 返回迭代器对象
    def __iter__(self):
        # return self._get_iterator()
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    def __len__(self):
        length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore
        if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
            from math import ceil
            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length


class SlicerDataset:
    def __init__(self, args, dataset_name, batch_size, patch_size):

        self.dataset_name = dataset_name
        self.args = args
        self.data_dir = args.data_dir
        # self.gen = gen
        self.batch_size = batch_size
        self.imread = self.default_imread

        if args.eval:
            self.get_eval_loader()
        else:
            self.patch_size = patch_size
            self.get_train_loader()

    def get_eval_loader(self):
        dataset_name = self.dataset_name
        data_dir = self.data_dir
        if dataset_name in ["Rain100L", "Rain100H", "Rain200L", "Rain200H"]:
            input_path = '/'.join([data_dir, dataset_name, 'test', 'rain/'])
            gt_path = '/'.join([data_dir, dataset_name, 'test', 'norain/'])

            self.RainName = os.listdir(input_path)
            for i in range(len(self.RainName)):
                self.RainName[i] = input_path + self.RainName[i]

            self.LabelName = os.listdir(gt_path)
            for i in range(len(self.LabelName)):
                self.LabelName[i] = gt_path + self.LabelName[i]

            self.file_num = len(self.RainName)

        elif dataset_name == "test12":
            input_path = '/'.join([data_dir, dataset_name, 'rainy/'])
            gt_path = '/'.join([data_dir, dataset_name, 'groundtruth/'])

            self.RainName = os.listdir(input_path)
            for i in range(len(self.RainName)):
                self.RainName[i] = input_path + self.RainName[i]

            self.LabelName = os.listdir(gt_path)
            for i in range(len(self.LabelName)):
                self.LabelName[i] = gt_path + self.LabelName[i]
            self.file_num = len(self.RainName)

        elif dataset_name == "real":
            input_path = '/'.join([data_dir, dataset_name, 'small/'])
            self.RainName = os.listdir(input_path)
            for i in range(len(self.RainName)):
                self.RainName[i] = input_path + self.RainName[i]
            self.file_num = len(self.RainName)

        elif dataset_name in ["Rain1400", "DDN"]:
            self.RainName, self.LabelName = prepareDDN_Data(data_dir, "Rain1400")
            self.file_num = len(self.RainName)

        elif dataset_name == "DID":
            data_dir = '/'.join([data_dir, 'DID-MDN-datasets/', 'DID-MDN-test'])
            self.file_num = len(make_dataset(data_dir))
            self.imread = prepareDID_Data(data_dir).pix2pix_val
            self.augment = lambda *x: x

        # if dataset_name not in ["DID", "DDN", "te"]:
        #     self._repeat()
        #     assert self.file_num == len(self.LabelName), print("Error: data is not same with label")
        # self.index_list = np.random.permutation(self.file_num)
        self.index_list = range(self.file_num)
        print("PyTorch SlicerDataset")

    def get_train_loader(self):
        dataset_name = self.dataset_name
        data_dir = self.data_dir
        if dataset_name in ["Rain100L", "Rain100H", "Rain200L", "Rain200H", "mini_rain_test"]:
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
            self.RainName, self.LabelName = prepareDDN_Data(data_dir, "Rain12600")
            self.file_num = len(self.RainName)

        elif dataset_name == "DID":
            data_dir = '/'.join([data_dir, 'DID-MDN-datasets/'])
            self.file_num = len(make_dataset(data_dir))
            self.imread = prepareDID_Data(data_dir).pix2pix_class

        if dataset_name not in ["DID", "DDN"]:
            self._repeat()
            assert self.file_num == len(self.LabelName), print("Error: data is not same with label")
        self.index_list = np.random.permutation(self.file_num)
        print("PyTorch SlicerDataset")
        # self.loader = {
        #     # "tensor_slices": self.default_loader,
        #     "tensor_slices": self.tensor_slices_loader,
        #     "gen": self.dataset_generator
        # }

        # if gen in self.loader.keys():
        #     self.loader = self.loader[gen]()
        # else:
        #     print("{} is not supported in {}".format(gen, self.loader.keys()))
        #     raise NotImplementedError

    def default_imread(self, image_index):
        sample = imageio.imread(self.RainName[image_index])
        # gt = imageio.imread(self.LabelName[image_index])
        filename = self.RainName[image_index].split('/')[-1]
        return sample, filename[:-4]#np.concatenate([sample, gt], axis=1) v

    # torchGen
    def __getitem__(self, idx):
        file_num = self.file_num
        index_list = self.index_list
        image_index = index_list[(idx + 1) % file_num]
        sample, filename = self.imread(image_index)
        # img_pair = np.concatenate([sample, gt], axis=1)
        # sample, gt = self.crop(img_pair, None)
        # print(sample.shape, gt.shape, filename)
        # sample, gt = self.crop(self.imread(image_index), False)
        # sample, gt = self.augment(sample, gt)
        sample = np.expand_dims(sample, axis=0)
        # gt = np.expand_dims(gt, axis=0)

        return {'O': sample, 'B': sample, 'filename': filename}


    def augment(self, *args, hflip=True, rot=True):
        """common"""
        hflip = hflip and random.random() < 0.5

        # vflip = rot and random.random() < 0.5
        # rot90 = rot and random.random() < 0.5

        def _augment(img):
            """common"""
            if hflip:
                img = img[:, ::-1, :]
                # img = np.flip(img, axis=1)
            # if vflip:
            #     img = img[::-1, :, :]
            #     # img = np.flip(img, axis=0)
            # if rot90:
            #     img = img.transpose(1, 0, 2)
            return img

        return _augment(args[0]), _augment(args[1])  # [_augment(a) for a in args]

    def crop(self, img_pair, aug):
        filename = img_pair[1]
        img_pair = img_pair[0]

        # patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        # p_h = p_w = patch_size
        # r = random.randrange(0, h - p_h + 1)
        # c = random.randrange(0, w - p_w + 1)

        O = img_pair[:, :w]
        B = img_pair[:, w:]
        # B = img_pair[r: r + p_h, c + w: c + p_w + w]  # norain
        # O = img_pair[r: r + p_h, c: c + p_w]  # rain

        # print("coord:", [r, r + p_h, c + w, c + p_w + w, r, r + p_h, c, c + p_w])

        return O, B, filename

    def __len__(self):
        return self.file_num

    def _repeat(self):
        batch_size = self.args.samples_per_gpu
        test_every = self.args.test_every
        file_num = self.file_num
        """srdata"""
        if test_every > 0:
            n_patches = batch_size * test_every  # 8000 // 2000
            # n_images = len(self.images_hr) #len(args.data_train) is list
            if file_num == 0:
                repeat = 0
            else:
                repeat = int(max(n_patches / file_num, 1))

            self.file_num = file_num * repeat
            self.RainName = self.RainName * repeat
            self.LabelName = self.LabelName * repeat
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(repeat, batch_size, self.file_num,
                                                                                 len(self.RainName)))

        else:
            test_every = -test_every
            self.file_num = file_num * test_every
            self.RainName = self.RainName * test_every
            self.LabelName = self.LabelName * test_every
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(test_every, batch_size, self.file_num,
                                                                                 len(self.RainName)))


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
            self.RainName, self.LabelName = prepareDDN_Data(data_dir, "Rain12600")
            self.file_num = len(self.RainName)

        elif dataset_name == "DID":
            data_dir = '/'.join([data_dir, 'DID-MDN-datasets/'])
            self.file_num = len(make_dataset(data_dir))
            self.imread = prepareDID_Data(data_dir)
            # self.RainName, self.LabelName = prepareDID_Data(data_dir)

        if dataset_name not in ["DID", 'DDN']:
            self._repeat()
            assert self.file_num == len(self.LabelName), print("Error: data is not same with label")
        self.index_list = np.random.permutation(self.file_num)

        self.loader = {
            # "tensor_slices": self.default_loader,
            "tensor_slices": self.tensor_slices_loader,
            "gen": self.dataset_generator
        }

        if gen in self.loader.keys():
            self.loader = self.loader[gen]()
        else:
            print("{} is not supported in {}".format(gen, self.loader.keys()))
            raise NotImplementedError

    def default_imread(self, image_index):
        sample = imageio.imread(self.RainName[image_index])
        gt = imageio.imread(self.LabelName[image_index])

        return np.concatenate([sample, gt], axis=1)

    # def default_loader(self):
    #     filename_tensor = tf.convert_to_tensor(self.RainName, dtype=tf.string)
    #     labels_tensor = tf.convert_to_tensor(self.LabelName, dtype=tf.string)
    #     dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
    #     dataset = dataset.map(self._parse_function)
    #     dataset = dataset.prefetch(buffer_size=self.batch_size * 10)
    #     dataset = dataset.batch(self.batch_size).repeat()
    #     iterator = dataset.make_one_shot_iterator()
    #
    #     return iterator#.get_next()

    def tensor_slices_loader(self):

        filename_tensor = tf.convert_to_tensor(self.RainName, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.LabelName, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 10)
        dataset = dataset.batch(self.batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()

        return iterator  # .get_next()

    def dataset_generator(self):
        b = 0
        index = -1
        file_num = self.file_num
        index_list = self.index_list
        error_count = 1
        while True:
            try:
                if index + 1 < file_num:
                    image_index = index_list[(index + 1) % file_num]
                    # img_pair = np.concatenate([sample, gt], axis=1)
                    sample, gt, filename = self.crop(self.imread(image_index), False)
                    sample = np.expand_dims(sample, axis=0)
                    gt = np.expand_dims(gt, axis=0)

                    if b == 0:
                        batch_sample_s = sample
                        batch_gt_s = gt
                        batch_filename = [filename]
                    if b != 0:
                        batch_sample_s = np.append(batch_sample_s, sample, axis=0)
                        batch_gt_s = np.append(batch_gt_s, gt, axis=0)
                        batch_filename.append(filename)
                    b += 1
                    index += 1

                    if b >= self.batch_size:
                        inputs = [batch_sample_s, batch_gt_s, np.array(batch_filename)]

                        yield inputs

                        b = 0
                else:
                    raise StopIteration

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

        return O, B, aug

    def __call__(self, *args, **kwargs):

        return self.loader  # self.sess.run(self.loader)

    def __next__(self):

        return next(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def _repeat(self):
        batch_size = self.args.samples_per_gpu
        test_every = self.args.test_every
        file_num = self.file_num
        """srdata"""
        if test_every > 0:
            n_patches = batch_size * test_every  # 8000 // 2000
            # n_images = len(self.images_hr) #len(args.data_train) is list
            if file_num == 0:
                repeat = 0
            else:
                repeat = int(max(n_patches / file_num, 1))

            self.file_num = file_num * repeat
            self.RainName = self.RainName * repeat
            self.LabelName = self.LabelName * repeat
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(repeat, batch_size, self.file_num,
                                                                                 len(self.RainName)))

        else:
            test_every = -test_every
            self.file_num = file_num * test_every
            self.RainName = self.RainName * test_every
            self.LabelName = self.LabelName * test_every
            print("trainData_repeat: {}, batch_size: {}, n_images: {}/{}".format(test_every, batch_size, self.file_num,
                                                                                 len(self.RainName)))

    # randomly crop image to obtain patches
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
        self.samples_per_gpu = args.samples_per_gpu
        self.num_workers = args.workers
        if not args.eval:
            self.patch_size = args.patch_size
        self.args = args

    def get_dataloader(self, mode, dataset_name, gen):

        if mode == "TFRecord":
            return TFRecordDataloader(dataset_name, patch_size=self.patch_size, batch_size=self.samples_per_gpu,
                                      num_epochs=300)

        elif mode == "Slicer":
            return SlicerDataLoader(self.args, dataset_name, patch_size=self.patch_size, batch_size=self.samples_per_gpu,
                                    gen=gen)

        elif mode == "torchGen":
            dataset = SlicerDataset(self.args, dataset_name, batch_size=self.samples_per_gpu, patch_size=self.patch_size)
            # print(dataset.augment)
            dataloader = DataLoader(dataset, batch_size=self.samples_per_gpu, shuffle=True, num_workers=0, drop_last=True)
            return dataloader

    def get_eval_loader(self, mode, dataset_name, gen):

        if mode == "torchGen":
            dataset = SlicerDataset(self.args, dataset_name, batch_size=self.samples_per_gpu,
                                    patch_size=None)
            # print(dataset.augment)
            dataloader = DataLoader(dataset, batch_size=self.samples_per_gpu, shuffle=True, num_workers=0,
                                    drop_last=True)
            return dataloader
        else:
            raise NotImplementedError

def guided_filter(data):
    r = 15
    eps = 1.0
    num_patches, height, width, channel = data.shape
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :, j] = q
    return batch_q


import argparse
import platform


def main():
    plt.ion()
    parser = argparse.ArgumentParser(description='TF Training')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='image2patch, set to model and dataset')

    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int,  # 8
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
    train_loader = sess_t.get_dataloader(mode="torchGen", dataset_name="DID", gen="gen")  # ()

    for j in range(1):
        for idx, (samples, gt, filename) in enumerate(train_loader):
            print(samples.shape, gt.shape, len(filename))

            # # if idx == 179:
            # #     print(len(train_loader))
            #     # continue
            # # tf_samples, tf_gt = train_loader.get_next()
            # # samples, gt = sess.run([tf_samples, tf_gt])
            # # detail_data = samples - guided_filter(samples)  # detail layer
            # #
            # axes[0][0].imshow(samples[0])
            # axes[0][1].imshow(gt[0])
            # # plt.show()
            # # plt.pause(0.5)
            # axes[1][0].imshow(samples[1])
            # axes[1][1].imshow(gt[1])
            #
            # # axes[0][2].imshow(detail_data[0] / 255)
            # # axes[1][2].imshow(detail_data[1] / 255)
            # #
            # plt.show()
            # plt.pause(0.5)
            # # print(samples.shape, gt.shape)
            # print(idx)
            # filename = filename[0].split('/')[-1]
            # plt.imsave('D:\Datasets\derain\DID-MDN-datasets\DID-MDN-training1/rain/'+filename, samples[0])
            # plt.imsave('D:\Datasets\derain\DID-MDN-datasets\DID-MDN-training1/norain/'+filename, gt[0])

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
