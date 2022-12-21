# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:
import torch.utils.data as data
import random
# from PIL import Image
import os
import os.path
import numpy as np
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
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


def default_loader(path):
    try:
        img_pair = cv2.imread(path).astype(np.float32) / 255.0
        img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
    except Exception:
        print(path)
    return img_pair  # Image.open(path).convert('RGB')


def crop(img_pair, patch_size):
    # patch_size = .patch_size
    h, ww, c = img_pair.shape
    w = int(ww / 2)
    p_h = p_w = patch_size[0]
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
    B = img_pair[r: r + p_h, c + w: c + p_w + w]  # norain 右边
    O = img_pair[r: r + p_h, c: c + p_w]  # rain 左边
    # cv2.imshow("O", O)
    # cv2.imshow("B", B)
    # cv2.waitKey(1000)

    # if aug:
    #     O = cv2.resize(O, (patch_size, patch_size))
    #     B = cv2.resize(B, (patch_size, patch_size))

    return O, B


class pix2pix(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader, seed=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        # index = np.random.randint(self.__len__(), size=1)[0]
        # index = np.random.randint(self.__len__(), size=1)[0]+1
        # index = np.random.randint(self.__len__(), size=1)[0]

        # index_folder = np.random.randint(1,4)
        index_folder = np.random.randint(0, 1)

        index_sub = np.random.randint(2, 5)

        label = index_folder

        if index_folder == 0:
            path = '/home/openset/Desktop/derain2018/facades/training2' + '/' + str(index) + '.jpg'

        if index_folder == 1:
            if index_sub < 4:
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Heavy/train2018new' + '/' + str(
                    index) + '.jpg'
            if index_sub == 4:
                index = np.random.randint(0, 400)
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Heavy/trainnew' + '/' + str(
                    index) + '.jpg'

        if index_folder == 2:
            if index_sub < 4:
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Medium/train2018new' + '/' + str(
                    index) + '.jpg'
            if index_sub == 4:
                index = np.random.randint(0, 400)
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Medium/trainnew' + '/' + str(
                    index) + '.jpg'

        if index_folder == 3:
            if index_sub < 4:
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain_new/Rain_Light/train2018new' + '/' + str(
                    index) + '.jpg'
            if index_sub == 4:
                index = np.random.randint(0, 400)
                path = '/home/openset/Desktop/derain2018/facades/DB_Rain/Rain_Light/trainnew' + '/' + str(
                    index) + '.jpg'

        # img = self.loader(path)

        img = self.loader(path)

        # NOTE: img -> PIL Image
        # w, h = img.size
        # w, h = 1024, 512
        # img = img.resize((w, h), Image.BILINEAR)
        # pix = np.array(I)
        #
        # r = 16
        # eps = 1
        #
        # I = img.crop((0, 0, w/2, h))
        # pix = np.array(I)
        # base=guidedfilter(pix, pix, r, eps)
        # base = PIL.Image.fromarray(numpy.uint8(base))
        #
        #
        #
        # imgA=base
        # imgB=I-base
        # imgC = img.crop((w/2, 0, w, h))

        if self.transform is None:
            imgA, imgB = crop(img, patch_size=self.patch_size)

        elif self.transform is not None:
            w, h = img.size
            # img = img.resize((w, h), Image.BILINEAR)

            # NOTE: split a sample into imgA and imgB
            imgA = img.crop((0, 0, w / 2, h))
            # imgC = img.crop((2*w/3, 0, w, h))

            imgB = img.crop((w / 2, 0, w, h))
            # NOTE preprocessing for each pair of images
            imgA, imgB = self.transform(imgA, imgB)

        return {'O': imgA, 'B': imgB, 'label': label}

    def __len__(self):
        # return 679
        print(len(self.imgs))
        return len(self.imgs)


class pix2pix_val(data.Dataset):
    def __init__(self, root, patch_size, transform=None, loader=default_loader, seed=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        # index = np.random.randint(self.__len__(), size=1)[0]
        # index = np.random.randint(self.__len__(), size=1)[0]

        # path = self.imgs[index]
        index = index % len(self.imgs)
        index = index + 1
        path = self.root + '/' + str(index) + '.jpg'

        index_folder = np.random.randint(0, 4)
        label = index_folder

        # path='/home/openset/Desktop/derain2018/facades/DB_Rain_test/Rain_Heavy/test2018'+'/'+str(index)+'.jpg'
        # print(path)
        img_pair = self.loader(path)

        # # NOTE: img -> PIL Image
        # w, h = img.size
        # w, h = 1024, 512
        # img = img.resize((w, h), Image.BILINEAR)
        # # NOTE: split a sample into imgA and imgB
        # imgA = img.crop((0, 0, w/2, h))
        # imgB = img.crop((w/2, 0, w, h))
        # if self.transform is not None:
        #   # NOTE preprocessing for each pair of images
        #   imgA, imgB = self.transform(imgA, imgB)
        # return imgA, imgB

        # w, h = 1536, 512
        # img = img.resize((w, h), Image.BILINEAR)
        #
        #
        # # NOTE: split a sample into imgA and imgB
        # imgA = img.crop((0, 0, w/3, h))
        # imgC = img.crop((2*w/3, 0, w, h))
        #
        # imgB = img.crop((w/3, 0, 2*w/3, h))

        # w, h = 1024, 512
        # img = img.resize((w, h), Image.BILINEAR)
        #
        # r = 16
        # eps = 1
        #
        # # I = img.crop((0, 0, w/2, h))
        # # pix = np.array(I)
        # # print
        # # base[idx,:,:,:]=guidedfilter(pix[], pix[], r, eps)
        # # base[]=guidedfilter(pix[], pix[], r, eps)
        # # base[]=guidedfilter(pix[], pix[], r, eps)
        #
        #
        # # base = PIL.Image.fromarray(numpy.uint8(base))
        #
        # # NOTE: split a sample into imgA and imgB
        # imgA = img.crop((0, 0, w/3, h))
        # imgC = img.crop((2*w/3, 0, w, h))
        #
        # imgB = img.crop((w/3, 0, 2*w/3, h))
        # imgA=base
        # imgB=I-base
        # imgC = img.crop((w/2, 0, w, h))
        # h, ww, c = img_pair.shape
        # w, h = 586*2, 586

        # img = img.resize((w, h), Image.BILINEAR)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        # NOTE: split a sample into imgA and imgB
        imgA = img_pair[:, :w ]#img.crop((0, 0, w / 2, h))
        # imgC = img.crop((2*w/3, 0, w, h))

        imgB = img_pair[:, w:, :]#img.crop((w / 2, 0, w, h))
        # ax1[0].imshow(imgA)
        # ax1[1].imshow(imgB)
        imgA = np.transpose(imgA, (2, 0, 1))
        imgB = np.transpose(imgB, (2, 0, 1))
        # plt.show()
        # plt.pause(1)
        # print(h, ww, imgA.shape, imgB.shape, path)
        if self.transform is not None:
            # NOTE preprocessing for each pair of images
            # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
            imgA, imgB = self.transform(imgA, imgB)

        return {'O': imgA, 'B': imgB, 'filename': str(index)}

    def __len__(self):
        return len(self.imgs)


class pix2pix_class(data.Dataset):
    def __init__(self, root, patch_size, transform=None, loader=default_loader, seed=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.patch_size = patch_size
        self.loader = loader

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):

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

        img = self.loader(path)
        # print(path)

        # NOTE: split a sample into imgA and imgB

        if self.transform is None:
            imgA, imgB = crop(img, patch_size=self.patch_size)
            imgA = np.transpose(imgA, (2, 0, 1))  # rain
            imgB = np.transpose(imgB, (2, 0, 1))

        elif self.transform is not None:
            w, h = img.size
            imgA = img.crop((0, 0, w / 2, h))
            imgB = img.crop((w / 2, 0, w, h))
            # NOTE preprocessing for each pair of images
            imgA, imgB = self.transform(imgA, imgB)

        return {'O': imgA, 'B': imgB, 'label': label}

    def __len__(self):
        # return 679
        # print(len(self.imgs))
        return len(self.imgs)
