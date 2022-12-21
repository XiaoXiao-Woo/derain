import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import settings 

def image_jittor():
    from torch.utils.data import DataLoader
    class TrainValDataset(Dataset):
        def __init__(self, name):
            super().__init__()
            self.rand_state = RandomState(66)
            # self.rain_dir = os.path.join(settings.data_dir, "train/rain")
            # self.rain_dir_list = os.listdir(self.rain_dir)
            # self.derain_dir = os.path.join(settings.data_dir, "train/norain")
            # self.derain_dir_list = os.listdir(self.derain_dir)
            self.dir = os.path.join(settings.data_dir, "train_bak")
            self.dir_list = os.listdir(self.dir)
            print(self.dir_list, len(self.dir_list))


            self.patch_size = settings.patch_size
            self.file_num = len(self.dir_list)


        def __len__(self):
            return self.file_num

        def __getitem__(self, idx):
            # file_name = self.mat_files[idx % self.file_num]
            file_name = self.dir_list[idx]
            im_file = os.path.join(self.dir, file_name)

            o_file_name, suffix = im_file[4:].split('.')
            suffix = "." + suffix
            if 'x2' in im_file:

                rain_im_file = im_file[:-6] + suffix
                derain_im_file = im_file
            else:
                # file_name, suffix = im_file[2:].split('.')
                rain_im_file = im_file
                derain_im_file = im_file[:-4] + 'x2' + suffix




            # derain_name = self.derain_dir_list[idx]
            # rain_im_file = os.path.join(self.dir, rain_im_file)
            # derain_im_file = os.path.join(self.dir, derain_im_file)
            img_left = cv2.imread(rain_im_file).astype(np.float32) / 255
            img_right = cv2.imread(derain_im_file).astype(np.float32) / 255
            h,w,c = img_left.shape

            img_left = np.tile(img_left, [1, 2, 1])
            img_left[:, w:2*w, :] = img_right
            #
            # cv2.imshow("left", img_left)
            # cv2.imshow("right", img_right)
            # cv2.waitKey(1000)
            # cv2.imwrite(settings.data_dir+"/train/"+file_name, img_left)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            plt.imsave(settings.data_dir+"/train/"+file_name, img_left)


            return img_left, img_right

    dataset = TrainValDataset("train/rain")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in loader:
        ...




class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)
        if name == "test":
            print(f"test aug state: {settings.aug_data}, will be changed to False")
            settings.aug_data = False

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        if settings.aug_data:
            O, B = self.crop(img_pair, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img_pair, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]


        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        #h_8=h%8
        #w_8=w%8
        O = np.transpose(img_pair[:, w:], (2, 0, 1))
        O = O[:, :-1, :-1]
        B = np.transpose(img_pair[:, :w], (2, 0, 1))
        B = B[:, :-1, :-1]
        sample = {'O': O, 'B': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        #h_8 = h % 8
        #w_8 = w % 8
        if settings.pic_is_pair:
            O = np.transpose(img_pair[:, w:], (2, 0, 1))
            B = np.transpose(img_pair[:, :w], (2, 0, 1))
        else:
            O = np.transpose(img_pair[:, :], (2, 0, 1))
            B = np.transpose(img_pair[:, :], (2, 0, 1))
        sample = {'O': O, 'B': B,'file_name':file_name}

        return sample


if __name__ == '__main__':

    image_jittor()

    # dt = TrainValDataset('val')
    # print('TrainValDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # dt = TestDataset('test')
    # print('TestDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # print('ShowDataset')
    # dt = ShowDataset('test')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
