import random

import numpy as np
# import skimage.color as sc
import cv2
import torch
from torchvision import transforms


def resize_image(*args, patch_size):
    ih, iw = args[0].shape[:2]
    ratio = ih / iw

    if ratio >= 1:
        short_side = int(patch_size / ratio)
        return [cv2.resize((a * 255).astype(np.uint8), (patch_size, short_side)) / 255.0
                for a in args]
    else:
        short_side = int(patch_size * ratio)
        return [cv2.resize((a * 255).astype(np.uint8), (patch_size, short_side)) / 255.0
                for a in args]

def get_patch(*args, patch_size=96, scale=1, multi_scale=False, resize=False):
    ih, iw = args[0].shape[:2]

    #p = scale if multi_scale else 1
    #tp = p * patch_size
    #ip = tp // scale

    tp = patch_size
    ip = patch_size


    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    #tx, ty = scale * ix, scale * iy
    tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    '''
    coord: [32, 80, 68, 116] [32, 80, 68, 116]
    coord: [194, 242, 333, 381] [194, 242, 333, 381]
    coord: [221, 269, 199, 247] [221, 269, 199, 247]
    coord: [117, 165, 136, 184] [117, 165, 136, 184]
    coord: [13, 61, 11, 59] [13, 61, 11, 59]
    coord: [110, 158, 351, 399] [110, 158, 351, 399]
    coord: [253, 301, 224, 272] [253, 301, 224, 272]
    coord: [235, 283, 389, 437] [235, 283, 389, 437]
    coord: [51, 99, 284, 332] [51, 99, 284, 332]
    coord: [170, 218, 61, 109] [170, 218, 61, 109]
    coord: [259, 307, 216, 264] [259, 307, 216, 264]
    coord: [433, 481, 255, 303] [433, 481, 255, 303]
    coord: [124, 172, 245, 293] [124, 172, 245, 293]
    coord: [191, 239, 187, 235] [191, 239, 187, 235]
    coord: [266, 314, 83, 131] [266, 314, 83, 131]
    coord: [22, 70, 240, 288] [22, 70, 240, 288]
    '''
    # print("coord:", [iy, iy + ip, ix, ix + ip], [ty, ty + tp, tx, tx + tp])

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

