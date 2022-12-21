import os

import imageio
import torch
from data import srdata
from torch.utils.data import Dataset
import numpy as np

class RainHeavyTest(srdata.SRData):
    def __init__(self, args, name='RainHeavyTest', train=True, benchmark=False):
        super(RainHeavyTest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.args = args

    def _scan(self):
        names_hr, names_lr = super(RainHeavyTest, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavyTest, self)._set_filesystem(dir_data)
        # self.apath = '../data/test/small/'
        # self.apath = '../data/rain100H/test/small/'
        # self.apath = '/home/office-409/Datasets/derain/Rain100L/test'
        self.apath = self.args.dir_data + '/Rain100L/test'
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')

class TestRealDataset(Dataset):
    def __init__(self, rgb_dir):
        super(TestRealDataset, self).__init__()

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir).replace('\\', '/')))
        self.noisy_filenames = [os.path.join(rgb_dir, x) for x in noisy_files]
        print(self.noisy_filenames)
        # self.root_dir = os.path.join(rgb_dir, "train")
        # self.mat_files = os.listdir(self.root_dir)

        # self.img_options = img_options

        self.tar_size = len(self.noisy_filenames)  # get the size of target clean_filenames

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        noisy = torch.from_numpy(np.float32(imageio.imread(self.noisy_filenames[tar_index]))).permute(2, 0, 1)
        # file_name = self.mat_files[tar_index]
        # img_file = os.path.join(self.root_dir, file_name)

        # img_pair = torch.from_numpy(np.float32(load_img(img_file)))
        # h, ww, c = img_pair.shape
        # w = int(ww / 2)
        # clean = img_pair[:, :w, :]
        # noisy = img_pair[:, w:, :]
        # clean = clean.permute(2, 0, 1)
        # noisy = noisy.permute(2, 0, 1)

        noisy_filenames = os.path.split(self.noisy_filenames[tar_index])[-1]


        return noisy, noisy_filenames#{'O': noisy, 'file_name': noisy_filenames}  # clean_filename, noisy_filename