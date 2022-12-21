# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import os
from importlib import import_module

import imageio
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.nn.functional as F
import torch.utils.model_zoo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def partial_load_checkpoint(state_dict, amp, dismatch_list = []):
    pretrained_dict = {}
    # dismatch_list = ['dwconv']

    for module in state_dict.items():
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
        # 2. overwrite entries in the existing state dict
        k, v = module
        k = '.'.join(['model', k])
        if all(m not in k for m in dismatch_list):
            # print(k)
            pretrained_dict.update({k: v})

    return pretrained_dict

class Model(nn.Module):
    def __init__(self, args, ckp, show=False):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.idx_scale = 0
        # self.input_large = (args.model == 'VDSR')
        self.self_ensemble = False  # args.self_ensemble
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        # module = import_module(args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        # self.load(
        #     ckp.get_path('model'),
        #     resume=args.resume,
        #     cpu=args.cpu
        # )
        # print(self.model, file=ckp.log_file)
        self.show = show
        if show:
            # plt.ion()
            # f, axarr = plt.subplots(1, 3)
            fig, axes = plt.subplots(ncols=4, nrows=2)
            self.axes = axes

    def show_patches(self, x, fig_num):
        # grid_size = x.shape[0]
        if len(x.shape) == 5:
            grid_x, grid_y, _, _, _ = x.shape

            g = plt.figure(fig_num, figsize=(grid_x, grid_y))
            gs1 = gridspec.GridSpec(grid_x, grid_y)
            gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

            # for i in range(grid_size * grid_size):
            #     n = i // grid_size
            #     m = i % grid_size
            for n in range(grid_x):
                for m in range(grid_y):
                    ax1 = plt.subplot(gs1[m + grid_y * n])
                    plt.axis('off')

                    patch = x[n][m].permute(1, 2, 0)
                    plt.imshow(patch)

            g.show()


    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale

        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            forward_function = self.forward_chop
            # forward_function = self.forward_once

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

    def forward_once(self, x):

        return self.model(x)

    def forward_patchify(self, x, shave=None):
        # batchsize = self.args.crop_batch_size
        # patch_size = self.patch_size
        # shave = patch_size // 2 if shave is None else shave
        # h, w = x.size()[-2:]
        # print(x.unfold(2, patch_size, shave).shape)
        # print(x.unfold(2, patch_size, shave)
        #     .unfold(3, patch_size, shave).shape)
        # #481,321->bs,c,12,19,48,48->bs*12*19,48,48
        # x_unfold = (
        #     x.unfold(2, patch_size, shave)
        #     .unfold(3, patch_size, shave)
        #     .permute(0, 2, 3, 1, 4, 5)
        #     .flatten(0, 2)
        #     .contiguous()
        # )
        # # x_unfold = F.unfold(x, patch_size, stride=shave).transpose(0, 2).contiguous()
        # # x_unfold = x_unfold.reshape(-1, 3, patch_size, patch_size).contiguous()
        # y_unfold = []
        #
        # # 大图切成小图，可能patch太多了爆显存，设置最大batchsize个，余数要单独计算
        # # 64+64+64+36 = 228
        # x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        # # x_unfold.cuda()
        # for i in range(x_range):
        #     y_unfold.append(x_unfold[i * batchsize:(i + 1) * batchsize, ...])
        #     # y_unfold.append(self.model(
        #     #     x_unfold[i * batchsize:(i + 1) * batchsize, ...]))
        #     # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        #
        # y_unfold = torch.cat(y_unfold, dim=0)
        #
        # y = F.fold(y_unfold.view(x_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
        #            (h, w),
        #            patch_size,
        #            stride=shave)
        batchsize, c, w, h = x.shape
        patch_size = 48
        overlap = 10
        deraintemp = torch.Tensor(np.zeros(x.shape)).cuda()
        ditemp = torch.Tensor(np.zeros(x.shape)).cuda()
        # print(ditemp.shape)
        # input()
        gap = patch_size - overlap
        # gridx = np.linspace(0, h - (h % gap) - gap, num=(h // gap))
        gridx = []
        for i in range(0, (w // gap) + 1): gridx.append(i * gap)
        gridy = []
        for i in range(0, (h // gap) + 1): gridy.append(i * gap)
        for i in range(0, len(gridx)):
            for j in range(0, len(gridy)):
                xx = gridx[i]
                yy = gridy[j]
                opatch = x[0, :, xx:xx + patch_size, yy:yy + patch_size].unsqueeze(0)

                # if i == len(gridx) - 1:
                #     opatch = O[0, :, w - patch_size:w+1, h - patch_size:h+1].unsqueeze(0)
                #     xx = w - patch_size
                #     yy = h - patch_size
                # if j == len(gridy) - 1:
                #     opatch = O[0, :, w - patch_size:w+1, h - patch_size:h+1].unsqueeze(0)
                #     xx = w - patch_size
                #     yy = h - patch_size

                if xx + patch_size > w:
                    opatch = x[0, :, w - patch_size:w + 1, yy:yy + patch_size].unsqueeze(0)
                    xx = w - patch_size
                    # yy = h - patch_size
                    gridx[i] = xx
                    # gridy[j] = yy
                if yy + patch_size > h:
                    opatch = x[0, :, xx:xx + patch_size, h - patch_size:h + 1].unsqueeze(0)
                    # xx = w - patch_size
                    yy = h - patch_size
                    # gridx[i] = xx
                    gridy[j] = yy

                derainp = self.model(opatch)
                deraintemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = derainp + deraintemp[0, :,
                                                                                     xx:xx + patch_size,
                                                                                     yy:yy + patch_size,
                                                                                     ]
                ditemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = 1 + ditemp[0, :,
                                                                           xx:xx + patch_size,
                                                                           yy:yy + patch_size]

        for i in range(0, len(gridx)):
            for j in range(0, len(gridy)):
                xx = gridx[i] + (w % gap)
                yy = gridy[j]
                opatch = x[0, :, xx:xx + patch_size, yy:yy + patch_size].unsqueeze(0)
                if xx + patch_size > w:
                    opatch = x[0, :, w - patch_size:w + 1, yy:yy + patch_size].unsqueeze(0)
                    xx = w - patch_size
                    # yy = h - patch_size
                    gridx[i] = xx
                    # gridy[j] = yy
                if yy + patch_size > h:
                    opatch = x[0, :, xx:xx + patch_size, h - patch_size:h + 1].unsqueeze(0)
                    # xx = w - patch_size
                    yy = h - patch_size
                    # gridx[i] = xx
                    gridy[j] = yy
                # print(opatch.shape)
                # input()
                derainp = self.model(opatch)
                deraintemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = derainp + deraintemp[0, :,
                                                                                     xx:xx + patch_size,
                                                                                     yy:yy + patch_size,
                                                                                     ]
                ditemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = 1 + ditemp[0, :,
                                                                           xx:xx + patch_size,
                                                                           yy:yy + patch_size]

        for i in range(0, len(gridx)):
            for j in range(0, len(gridy)):
                xx = gridx[i]
                yy = gridy[j] + (h % gap)
                opatch = x[0, :, xx:xx + patch_size, yy:yy + patch_size].unsqueeze(0)
                # print(opatch.shape)
                # input()
                if xx + patch_size > w:
                    opatch = x[0, :, w - patch_size:w + 1, yy:yy + patch_size].unsqueeze(0)
                    xx = w - patch_size
                    # yy = h - patch_size
                    gridx[i] = xx
                    # gridy[j] = yy
                if yy + patch_size > h:
                    opatch = x[0, :, xx:xx + patch_size, h - patch_size:h + 1].unsqueeze(0)
                    # xx = w - patch_size
                    yy = h - patch_size
                    # gridx[i] = xx
                    gridy[j] = yy
                derainp = self.model(opatch)
                deraintemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = derainp + deraintemp[0, :,
                                                                                     xx:xx + patch_size,
                                                                                     yy:yy + patch_size,
                                                                                     ]
                ditemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = 1 + ditemp[0, :,
                                                                           xx:xx + patch_size,
                                                                           yy:yy + patch_size]

        for i in range(0, len(gridx)):
            for j in range(0, len(gridy)):
                xx = gridx[i] + (w % gap)
                yy = gridy[j] + (h % gap)
                opatch = x[0, :, xx:xx + patch_size, yy:yy + patch_size].unsqueeze(0)
                # print(opatch.shape)
                # input
                if xx + patch_size > w:
                    opatch = x[0, :, w - patch_size:w + 1, yy:yy + patch_size].unsqueeze(0)
                    xx = w - patch_size
                    # yy = h - patch_size
                    gridx[i] = xx
                    # gridy[j] = yy
                if yy + patch_size > h:
                    opatch = x[0, :, xx:xx + patch_size, h - patch_size:h + 1].unsqueeze(0)
                    # xx = w - patch_size
                    yy = h - patch_size
                    # gridx[i] = xx
                    gridy[j] = yy
                derainp = self.model(opatch)
                deraintemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = derainp + deraintemp[0, :,
                                                                                     xx:xx + patch_size,
                                                                                     yy:yy + patch_size,
                                                                                     ]
                ditemp[0, :, xx:xx + patch_size, yy:yy + patch_size] = 1 + ditemp[0, :,
                                                                           xx:xx + patch_size,
                                                                           yy:yy + patch_size]
        derain = deraintemp / ditemp

        return derain

    #pytorch的fold、unfold一起使用要求x是方形，因为雨图不是，所以作者切成大小不一的方形，多次输入到模型中，耗费了大量的测试时间
    # # 大图切成小图，可能patch太多了爆显存，设置最大batchsize个，余数要单独计算
    # # 64+64+64+36 = 228
    def forward_chop(self, x, shave=12):
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size / 2)
        # print(self.scale, self.idx_scale)
        scale = self.scale[self.idx_scale]

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        ################################################
        # 最后一块patch单独计算
        ################################################

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 左上patch单独计算，不是平均而是覆盖
        ################################################

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # img->patch，最大计算crop_s个patch，防止bs*p*p太大
        ################################################

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(self.model(
            x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
            # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        y_unfold = torch.cat(y_unfold, dim=0)
        #bs*p*p, c, h, w -> bs*p*p, c*h*w, 1
        y = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
                   stride=int(shave / 2 * scale))
        # 312， 480
        # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 第一块patch->y
        ################################################
        y[..., :padsize * scale, :] = y_h_top
        y[..., :, :padsize * scale] = y_w_top
        # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        y_unfold = y_unfold[...,
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        #1，3，24，24
        y_inter = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                         ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                         padsize * scale - shave * scale,
                         stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = F.fold(F.unfold(y_ones, padsize * scale - shave * scale,
                                  stride=int(shave / 2 * scale)),
                         ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                         padsize * scale - shave * scale,
                         stride=int(shave / 2 * scale))

        y_inter = y_inter / divisor
        # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 第一个半patch
        ################################################
        y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
        int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_inter
        # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        #图分为前半和后半
        # x->y_w_cut
        # model->y_hw_cut
        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
        # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        # plt.show()

        return y.cuda()

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(self.model(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                             ...]).cpu())  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_h_cut_inter = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = F.fold(
            F.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
                                       stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter
        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(self.model(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                             ...]).cpu())  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                         :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
                                       stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter
        return y_w_cut

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    #.reshape((1, 1, 3)) / 256#
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)#(1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
        # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.mean(pow(valid, 2), axis=[1, 2, 3]), axis=0)
    mse = np.mean(pow(valid, 2))
    if mse == 0:
        return 100
    try:
        psnr = -10 * math.log10(mse)
    except Exception:
        print(mse)

    return psnr

if __name__ == "__main__":
    import numpy as np
    import cv2
    import argparse

    parser = argparse.ArgumentParser(description='IPT')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    # parser.add_argument('--batch_size', type=int, default=8,
    #                     help='input batch size for training')
    parser.add_argument('--crop_batch_size', type=int, default=64,
                        help='input batch size for training')

    parser.add_argument('--patch_dim', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)  # 4
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--freeze_norm', action='store_true')
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--pos_every', action='store_true')
    parser.add_argument('--no_pos', action='store_true')
    parser.add_argument('--num_queries', type=int, default=1)  # 6
    parser.add_argument('--patch_size', type=int, default=48,
                        help='output patch size')

    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='save output results')
    parser.add_argument('--save_gt', action='store_true',
                        help='save low-resolution and high-resolution images together')
    parser.add_argument('--model', default='ipt',
                        help='model name')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    args, unparsed = parser.parse_known_args()
    args.scale = [1]
    fig, axes = plt.subplots(ncols=3, nrows=1)
    img = cv2.imread("../../derain/dataset/rain100L/test/norain-1.png").astype(np.float32)
    h, ww, _ = img.shape #321, 962, 3
    w = ww // 2
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img.astype(int))
    x = img[:, w:ww, ...]
    # x = x[125:173, 164:212]
    axes[1].imshow(x.astype(int))
    x = np.transpose(x, (2, 0, 1))
    # x = cv2.resize(img[:, w:ww, ...], (48, 48))
    x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()

    gt = img[:, :w, ...]#cv2.resize(img[:, :w, ...], (48, 48))
    # gt = gt[125:173, 164:212]
    axes[2].imshow(gt.astype(int))
    gt = np.transpose(gt, (2, 0, 1))
    print(x.shape)
    gt = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0).cuda()

    net = Model(args, None).cuda()
    net.load_state_dict(partial_load_checkpoint(torch.load("./IPT_derain.pt"), None), strict=False)
    # net.train()
    net.eval()
    with torch.no_grad():
        # 197 201
        outputs = net(x, idx_scale=0)
        outputs = quantize(outputs, 255)
        normalized = outputs[0].mul(255 / args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    imageio.imwrite("./img.png", tensor_cpu.numpy())
    psnr = calc_psnr(outputs.cpu().detach().numpy(), gt.cpu().numpy(), 4, 255.0)
    print(psnr)
    # plt.imshow(y[0].byte().permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
'''
D:\anaconda\envs\mmlab\python.exe D:/ProjectSets/NDA/Attention/IPT/torchImpl/test_ipt.py
(321, 962, 3)
torch.Size([1, 3, 321, 481])
Making model...
46.80181219109498

Process finished with exit code 0

'''