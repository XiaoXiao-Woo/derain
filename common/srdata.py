# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""srdata"""
import os
import glob
import random
import pickle

import cv2
import numpy as np
import imageio
from common import common
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SRData:
    """srdata"""

    def __init__(self, args, name='', dataset_name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.args.derain = True
        self.args.ext = 'sep'

        self.load_PReNet_data(dataset_name)
        self._set_filesystem(args.dir_data)
        self._set_img(args)
        if self.args.derain and self.train:
            self.images_hr, self.images_lr = self.clear_train, self.rain_train
            print("images_hr:\n", self.images_hr)
            print("images_lr:\n", self.images_lr)
        if train:
            self._repeat(args)


    def load_PReNet_data(self, dataset_name):
        if self.args.dsderain:
            if self.train:
                self.derain_dataroot = os.path.join(self.args.dir_data, dataset_name)
                self.clear_train = common.search(self.derain_dataroot, "norain")
                self.rain_train = []
                for path in self.clear_train:
                    change_path = path.split('/')
                    # change_path[-2] = change_path[-2][2:]
                    change_path[-1] = change_path[-1][2:]
                    # change_path[-1] = ''.join([change_path[-1][:-4], "x2", '.png'])
                    # change_path[-2] = change_path[-2][2:]
                    self.rain_train.append('/'.join(change_path))
                    # print(path, self.rain_train[-1])
                # self.rain_train.extend(self.clear_train)
                # self.clear_train.extend(self.clear_train)
                if dataset_name == "RainTrainL":
                    self.derain_test = os.path.join(self.args.dir_data, "test")
                elif dataset_name == "RainTrainH":
                    self.derain_test = os.path.join(self.args.dir_data, "test")
                self.deblur_lr_test = common.search(self.derain_test, "rain")
                self.deblur_hr_test = [path.replace("rainy/", "no") for path in self.deblur_lr_test]
            else:
                ...
                # if dataset_name == "RainTrainL":
                #     self.derain_test = os.path.join(args.dir_data, "Rain100L")
                # elif dataset_name == "RainTrainH":
                #     self.derain_test = os.path.join(args.dir_data, "Rain100H")
                # self.deblur_lr_test = common.search(self.derain_test, "rain")
                # self.derain_lr_test = common.search(self.derain_test, "rain")
                # self.derain_hr_test = [path.replace("rainy/", "no") for path in self.derain_lr_test]


    def _set_img(self, args):
        """srdata"""
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or self.benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            for s in self.scale:
                if s == 1:
                    os.makedirs(os.path.join(self.dir_hr), exist_ok=True)
                else:
                    os.makedirs(
                        os.path.join(self.dir_lr.replace(self.apath, path_bin), 'X{}'.format(s)), exist_ok=True)

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

    def _repeat(self, args):
        """srdata"""
        n_patches = args.samples_per_gpu * args.test_every#8000 // 2000
        n_images = len(self.images_hr) #len(args.data_train) is list
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = max(n_patches / n_images, 1)
        print("trainData_repeat:", n_patches, n_images, self.repeat)

    def _scan(self):
        """srdata"""
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    scale = s
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}x{}{}' \
                                                     .format(s, filename, scale, self.ext[1])))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si] = names_hr
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name[0])
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if self.args.derain and self.scale[self.idx_scale] == 1:
            if self.train:
                lr, hr, filename = self._load_file_deblur(idx)
                pair = self.get_patch(lr, hr)
                # img, shape = self.get_padding_image(lr, hr)
                # pair = [img, img]
                pair = common.set_channel(*pair, n_channels=self.args.n_colors)
                pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            else:
                norain, rain, filename = self._load_rain_test(idx)
                pair = common.set_channel(*[rain, norain], n_channels=self.args.n_colors)
                pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return {'O': pair_t[0], 'B': pair_t[1], "scale": [self.idx_scale], "filename": filename}#pair_t[0], pair_t[1], [self.idx_scale], [filename]#, shape

        if self.args.denoise and self.scale[self.idx_scale] == 1:
            hr, filename = self._load_file_hr(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            noise = np.random.randn(*pair_t[0].shape) * self.args.sigma
            lr = pair_t[0] + noise
            lr = np.float32(np.clip(lr, 0, 255))
            return lr, pair_t[0], [self.idx_scale], [filename]
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return {'O': pair_t[0], 'B': pair_t[1]}  # , '''[self.idx_scale], [filename]

    def __len__(self):
        if self.train:
            return int(len(self.images_hr) * self.repeat)

        if self.args.derain and not self.args.alltask:
            return int(len(self.derain_hr_test) / self.args.derain_test)
        return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        return idx

    def _load_file_deblur(self, idx, train=True):
        """srdata"""
        idx = self._get_index(idx)
        if train:
            f_hr = self.images_hr[idx]
            f_lr = self.images_lr[idx]

        else:
            f_hr = self.deblur_hr_test[idx]
            f_lr = self.deblur_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        # filename = f_hr[-27:-17] + filename
        # print(f_hr, f_lr)
        hr = imageio.imread(f_hr) / 255.0
        lr = imageio.imread(f_lr) / 255.0
        return lr, hr, filename

    def _load_file_hr(self, idx):
        """srdata"""
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
        return hr, filename

    def _load_rain_test(self, idx):
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename

    def _load_file(self, idx):
        """srdata"""
        idx = self._get_index(idx)

        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_padding_image(self, lr, hr):

        img = np.concatenate([hr, lr], axis=-2)

        h, w, c = img.shape
        h_pad = (1024 - h)
        w_pad = (1024 - w)
        img = cv2.copyMakeBorder(img, h_pad, 0, w_pad, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if img.shape != (1024, 1024, 3):
            raise StopIteration

        return img, [h, w, c]

    def get_patch_hr(self, hr):
        """srdata"""
        if self.train:
            hr = self.get_patch_img_hr(hr, patch_size=self.args.patch_size, scale=1)
        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        """srdata"""
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret

    def get_patch(self, lr, hr):
        """srdata"""
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size * scale,
                scale=scale)
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)


# class SRData:
#     """srdata"""
#     def __init__(self, args, name='', train=True, benchmark=False):
#         self.args = args
#         self.name = name
#         self.train = train
#         self.split = 'train' if train else 'test'
#         self.do_eval = True
#         self.benchmark = benchmark
#         self.input_large = (args.model == 'VDSR')
#         self.scale = args.scale
#         self.idx_scale = 0
#
#         if self.args.derain:
#             if self.train:
#                 self.derain_dataroot = os.path.join(args.dir_data, "RainTrainL")
#                 self.clear_train = common.search(self.derain_dataroot, "norain")
#                 self.rain_train = []
#                 for path in self.clear_train:
#                     change_path = path.split('/')
#                     change_path[-1] = change_path[-1][2:]
#                     change_path[-1] = ''.join([change_path[-1][:-4], "x2", '.png'])
#                     change_path[-2] = change_path[-2][2:]
#                     change_path[-3] = "Rain100L"
#                     self.rain_train.append('/'.join(change_path))
#                 self.derain_test = os.path.join(args.dir_data, "Rain100L")
#                 self.deblur_lr_test = common.search(self.derain_test, "rain")
#                 self.deblur_hr_test = [path.replace("rainy/", "no") for path in self.deblur_lr_test]
#                 self.derain_hr_test = self.deblur_hr_test
#             else:
#                 self.derain_test = os.path.join(args.dir_data, "Rain100L")
#                 self.derain_lr_test = common.search(self.derain_test, "rain")
#                 self.derain_hr_test = [path.replace("rainy/", "no") for path in self.derain_lr_test]
#         self._set_filesystem(args.dir_data)
#         self._set_img(args)
#         if self.args.derain and self.train:
#             self.images_hr, self.images_lr = self.clear_train, self.rain_train
#         # print("norain:\n", self.images_hr)
#         # print("rain:\n", self.images_lr)
#         if train:
#             self._repeat(args)
#
#     def _set_img(self, args):
#         """srdata"""
#         if args.ext.find('img') < 0:
#             path_bin = os.path.join(self.apath, 'bin')
#             os.makedirs(path_bin, exist_ok=True)
#
#         list_hr, list_lr = self._scan()
#         if args.ext == 'img' or self.benchmark:
#             self.images_hr, self.images_lr = list_hr, list_lr
#         elif args.ext == 'seq':
#             os.makedirs(self.dir_hr, exist_ok=True)#.replace(self.apath, path_bin)
#             for s in self.scale:
#                 if s == 1:
#                     os.makedirs(os.path.join(self.dir_hr), exist_ok=True)
#                 else:
#                     os.makedirs(
#                         os.path.join(self.dir_lr, 'X{}'.format(s)), exist_ok=True)#.replace(self.apath, path_bin)
#
#             self.images_hr, self.images_lr = [], [[] for _ in self.scale]
#             for h in list_hr:
#                 # b = h.replace(self.apath, path_bin)
#                 b = b.replace(self.ext[0], '.pt')
#                 self.images_hr.append(b)
#                 self._check_and_load(args.ext, h, b, verbose=True)
#             for i, ll in enumerate(list_lr):
#                 for l in ll:
#                     # b = l.replace(self.apath, path_bin)
#                     b = b.replace(self.ext[1], '.pt')
#                     self.images_lr[i].append(b)
#                     self._check_and_load(args.ext, l, b, verbose=True)
#
#     def _repeat(self, args):
#         """srdata"""
#         n_patches = args.batch_size * args.test_every # 64 * 1000
#         n_images = len(args.data_train) * len(self.images_hr) # "DIV2K", 5 * 3600
#         if n_images == 0:
#             self.repeat = 0
#         else:
#             self.repeat = max(n_patches // n_images, 1)
#
#     def _scan(self):
#         """srdata"""
#         names_hr = sorted(
#             glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
#         names_lr = [[] for _ in self.scale]
#         # print(self.dir_hr)
#         # print(names_hr)
#         for f in names_hr:
#             filename, _ = os.path.splitext(os.path.basename(f))
#             for si, s in enumerate(self.scale):
#                 if s != 1:
#                     scale = s
#                     names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}{}' \
#                         .format(s, filename, self.ext[1])))#x{} , scale
#         for si, s in enumerate(self.scale):
#             if s == 1:
#                 names_lr[si] = names_hr
#         return names_hr, names_lr
#
#     def _set_filesystem(self, dir_data):
#         self.apath = os.path.join(dir_data, self.name[0])
#         self.dir_hr = os.path.join(self.apath, 'HR')
#         self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
#         self.ext = ('.png', '.png')
#
#     def _check_and_load(self, ext, img, f, verbose=True):
#         if not os.path.isfile(f) or ext.find('reset') >= 0:
#             if verbose:
#                 print('Making a binary: {}'.format(f))
#             with open(f, 'wb') as _f:
#                 pickle.dump(imageio.imread(img), _f)
#
#     def __getitem__(self, idx):
#         if self.args.derain and self.scale[self.idx_scale] == 1:
#             if self.train:
#                 lr, hr, filename = self._load_file_deblur(idx)
#                 pair = self.get_patch(lr, hr)
#                 pair = common.set_channel(*pair, n_channels=self.args.n_colors)
#                 pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
#             else:
#                 norain, rain, filename = self._load_rain_test(idx)
#                 pair = common.set_channel(*[rain, norain], n_channels=self.args.n_colors)
#                 pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
#             return pair_t[0], pair_t[1], [self.idx_scale], [filename]
#
#         if self.args.denoise and self.scale[self.idx_scale] == 1:
#             hr, filename = self._load_file_hr(idx)
#             pair = self.get_patch_hr(hr)
#             pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
#             pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
#             noise = np.random.randn(*pair_t[0].shape) * self.args.sigma
#             lr = pair_t[0] + noise
#             lr = np.float32(np.clip(lr, 0, 255))
#             return lr, pair_t[0], [self.idx_scale], [filename]
#         lr, hr, filename = self._load_file(idx)
#         pair = self.get_patch(lr, hr)
#         pair = common.set_channel(*pair, n_channels=self.args.n_colors)
#         pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
#
#         return pair_t[0], pair_t[1], [self.idx_scale], [filename]
#
#     def __len__(self):
#         if self.train:
#             return len(self.images_hr) * self.repeat
#
#         if self.args.derain and not self.args.alltask:
#             return int(len(self.derain_hr_test) / self.args.derain_test)
#         return len(self.images_hr)
#
#     def _get_index(self, idx):
#         if self.train:
#             return idx % len(self.images_hr)
#         return idx
#
#     def _load_file_deblur(self, idx, train=True):
#         """srdata"""
#         idx = self._get_index(idx)
#         if train:
#             f_hr = self.images_hr[idx]
#             f_lr = self.images_lr[idx]
#         else:
#             f_hr = self.deblur_hr_test[idx]
#             f_lr = self.deblur_lr_test[idx]
#         filename, _ = os.path.splitext(os.path.basename(f_hr))
#         filename = f_hr[-27:-17] + filename
#         hr = imageio.imread(f_hr)
#         lr = imageio.imread(f_lr)
#         return lr, hr, filename
#
#     def _load_file_hr(self, idx):
#         """srdata"""
#         idx = self._get_index(idx)
#         f_hr = self.images_hr[idx]
#
#         filename, _ = os.path.splitext(os.path.basename(f_hr))
#         if self.args.ext == 'img' or self.benchmark:
#             hr = imageio.imread(f_hr)
#         elif self.args.ext.find('sep') >= 0:
#             with open(f_hr, 'rb') as _f:
#                 hr = pickle.load(_f)
#         return hr, filename
#
#     def _load_rain_test(self, idx):
#         f_hr = self.derain_hr_test[idx]
#         f_lr = self.derain_lr_test[idx]
#         filename, _ = os.path.splitext(os.path.basename(f_lr))
#         norain = imageio.imread(f_hr)
#         rain = imageio.imread(f_lr)
#         return norain, rain, filename
#
#     def _load_file(self, idx):
#         """srdata"""
#         idx = self._get_index(idx)
#         # print(self.images_lr, len(self.images_lr[self.idx_scale]), idx)
#         f_hr = self.images_hr[idx]
#         f_lr = self.images_lr[self.idx_scale][idx]
#
#         filename, _ = os.path.splitext(os.path.basename(f_hr))
#         # print(f_hr, f_lr)
#         if self.args.ext == 'img' or self.benchmark:
#             hr = imageio.imread(f_hr)
#             lr = imageio.imread(f_lr)
#             print("read filename:", f_hr, f_lr)
#             # print("read:", hr.shape, lr.shape)
#         elif self.args.ext == 'seq':  # .find('sep') >= 0
#             print(f_hr, f_lr)
#             with open(f_hr, 'rb') as _f:
#                 hr = pickle.load(_f)
#             with open(f_lr, 'rb') as _f:
#                 lr = pickle.load(_f)
#
#         return lr, hr, filename
#
#     def get_patch_hr(self, hr):
#         """srdata"""
#         if self.train:
#             hr = self.get_patch_img_hr(hr, patch_size=self.args.patch_size, scale=1)
#         return hr
#
#     def get_patch_img_hr(self, img, patch_size=96, scale=2):
#         """srdata"""
#         ih, iw = img.shape[:2]
#
#         tp = patch_size
#         ip = tp // scale
#
#         ix = random.randrange(0, iw - ip + 1)
#         iy = random.randrange(0, ih - ip + 1)
#
#         ret = img[iy:iy + ip, ix:ix + ip, :]
#
#         return ret
#
#     def get_patch(self, lr, hr):
#         """srdata"""
#         scale = self.scale[self.idx_scale]
#         if self.train:
#             # print("get_patch_b:", lr.shape, hr.shape, "scale:", scale)
#             lr, hr = common.get_patch(
#                 lr, hr,
#                 patch_size=self.args.patch_size * scale,
#                 scale=scale)
#             # print("get_patch:", lr.shape, hr.shape)
#             if not self.args.no_augment:
#                 lr, hr = common.augment(lr, hr)
#         else:
#             ih, iw = lr.shape[:2]
#             hr = hr[0:ih * scale, 0:iw * scale]
#
#         return lr, hr
#
#     def set_scale(self, idx_scale):
#         if not self.input_large:
#             self.idx_scale = idx_scale
#         else:
#             self.idx_scale = random.randint(0, len(self.scale) - 1)


class TrainValDataset():
    def __init__(self, args, name, train):
        super().__init__()

        self.args = args
        self.root_dir = os.path.join("/IPT/MindSporeImpl/rain100L", name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = args.patch_size
        self.file_num = len(self.mat_files)
        print(self.file_num)
        self.idx_scale = 0
        self.train = train
        print(os.path.join(self.root_dir, self.mat_files[0]))

    def __len__(self):
        return self.file_num

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.mat_files)
        return idx

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)

        img_pair = imageio.imread(img_file).astype(np.float32)  # / 255.
        # print("img_pair.shape:", img_pair.shape)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        lr = img_pair[:, w:]
        hr = img_pair[:, :w]

        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        # print(pair_t[0].shape)

        return pair_t[0], pair_t[1], [self.idx_scale], [file_name]

    def get_patch(self, lr, hr):
        """srdata"""
        scale = self.args.scale[self.idx_scale]
        if self.train:
            # print("get_patch_before:", lr.shape, hr.shape, scale)
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size * scale,
                scale=scale)
            # print("get_patch:", lr.shape, hr.shape)
            if not self.args.no_augment:
                # print("data augment") done
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

# import os
# import imageio
# from multiprocessing import Pool
#
# file_list = os.listdir('./RainTrainL/rain')
# print(file_list[0])
#
#
# def replace_name(file):
#     img = imageio.imread(os.path.join('./RainTrainL/rain', file))
#     new_file = ''.join([file.split('.')[0][2:-2], '.png'])
#     print(file, new_file)
#     imageio.imwrite(os.path.join('./RainTrainL/tmp', new_file), img)
# pool = Pool(4)
# pool.map(replace_name, file_list)
# pool.close()
# pool.join()

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    import torch
    import matplotlib.pyplot as plt
    import tensorflow as tf

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', default='ipt',
                        help='model name')
    parser.add_argument('-b', '--batch-size', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--patch_size', type=int, default=100,
                        help='image2patch, set to model and dataset')
    parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                        help='train dataset name')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    args = parser.parse_args()
    args.scale = [1]
    args.dir_data = '../IPT/torchImpl/data'


    def encode_utf8_string(text, length, dic, null_char_id=5462):
        char_ids_padded = [null_char_id] * length
        char_ids_unpadded = [null_char_id] * len(text)
        for i in range(len(text)):
            hash_id = dic[text[i]]
            char_ids_padded[i] = hash_id
            char_ids_unpadded[i] = hash_id
        return char_ids_padded, char_ids_unpadded


    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    dataset = SRData(args, 'train', train=True)  # TrainValDataset(dataset_name, self.patch_size)
    dataloader = DataLoader(dataset, batch_size=1,
                shuffle=False, num_workers=1, drop_last=False, sampler=None)
    tfrecord_writer = tf.python_io.TFRecordWriter("K:rain100L")
    for idx, batch in enumerate(dataloader):
        imgA = batch[0]
        imgB = batch[1]
        shape = batch[-1]
        h, w = int(shape[0]), int(shape[1])
        print('Train data: {}/{}'.format(idx, len(dataloader)))
        # img = torch.cat([imgB, imgA], dim=-1)
        np_data = imgA.permute(0, 2, 3, 1).numpy()
        print(np_data.shape, h, w)

        # plt.imshow(np_data[0])
        # plt.show()
        image_data = np_data[0].astype(np.uint8).tobytes()
        # for text in open(addrs_label[j], encoding="utf"):
        #     char_ids_padded, char_ids_unpadded = encode_utf8_string(
        #         text=text,
        #         dic=dict,
        #         length=37,
        #         null_char_id=5462)

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(image_data),
                'image/format': _bytes_feature(b"raw"),
                'orishape': tf.train.Feature(int64_list=tf.train.Int64List(value=(h, w, 3))),
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=np_data[0].shape)),
                # 'image/width': _int64_feature([np_data.shape[1]]),
                # 'image/height': _int64_feature([np_data.shape[0]]),
                # 'image/orig_width': _int64_feature([np_data.shape[1]]),
                # 'image/class': _int64_feature(char_ids_padded),
                # 'image/unpadded_class': _int64_feature(char_ids_unpadded),
                # 'image/text': _bytes_feature(bytes(text, 'utf-8')),
                # 'height': _int64_feature([crop_data.shape[0]]),
            }
        ))
        tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()