# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import utility
import torch
# from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, args, loader_train, loader_test, loader_eval, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loader_eval = loader_eval
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self, epoch):
        # torch.set_grad_enabled(False)

        # epoch = self.optimizer.get_last_epoch()
        # self.ckp.write_log('\nTraining:')
        print('\nTraining')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_train), len(self.scale))
        # )
        self.model.train()
        timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        psnr_list = []
        for idx_data, d in enumerate(self.loader_train):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                # d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    # for norain, rain, filename in tqdm(d, ncols=80):
                    # norain, rain = self.prepare(norain, rain)
                    rain = d['O']
                    norain = d['B']
                    norain, rain = self.prepare(norain, rain)
                    sr = self.model(rain, idx_scale)
                    loss = self.loss(sr, norain)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                    #     sr, norain, scale, self.args.rgb_range
                    # )
                    psnr = utility.calc_psnr(
                        sr, norain, scale, self.args.rgb_range
                    )
                    psnr_list.append(psnr)
                    # if self.args.save_results:
                    #     self.ckp.save_results(d, filename[0], save_list, 1)
                    # self.ckp.log[-1, idx_data, idx_scale] /= self.args.batch_size#len(d)
                    # best = self.ckp.log.max(0)
                    avg = np.mean(psnr_list)
                    # avg = self.ckp.log.mean()
                    #(Best: {:.3f} @epoch {})
                    print(
                        '[{} x1] Epoch: {} [{}]/[{}] Loss: {} \tPSNR: {:.3f} (Avg: {:.7f})'.format(
                            'rain100L',
                            epoch,
                            idx_data,
                            len(self.loader_train),
                            loss,
                            psnr,
                            # best[0][idx_data, idx_scale],
                            # best[1][idx_data, idx_scale] + 1,
                            avg
                        )
                    )
                    isderain = 0

                elif self.args.denoise:
                    for hr, _, filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise+hr).clamp(0, 255)
                        sr = self.model(nois_hr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i = i+1
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                torch.cuda.empty_cache()
        print('Forward: {:.2f}s\n'.format(timer_test.toc()))
        print('Saving...')
        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        # self.ckp.write_log('Saving...')

        # if self.args.save_results:
        #     self.ckp.end_background()

        print(
            'Total: {:.2f}s\n'.format(timer_test.toc())
        )

        # torch.set_grad_enabled(True)
    def eval(self, epoch):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        print('\nEvaluation:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_test), len(self.scale))
        # )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        psnr_list = []
        for idx_data, d in enumerate(self.loader_eval):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                # d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    # for norain, rain, filename in tqdm(d, ncols=80):
                    # norain, rain = self.prepare(norain, rain)
                    rain = d['O']
                    # norain = d['B']
                    rain = self.prepare(rain)[0]
                    sr = self.model(rain, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                    #     sr, norain, scale, self.args.rgb_range
                    # )
                    # psnr = utility.calc_psnr(
                    #     sr, norain, scale, self.args.rgb_range)
                    # psnr_list.append(psnr)
                    # avg = np.mean(psnr_list)
                    if self.args.save_results:
                        self.ckp.save_results(d, d['file_name'][0], save_list, 1)
                    # self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    # best = self.ckp.log.max(0)
                    # print(
                    #     '[{} x1] Epoch: {} [{}]/[{}]\tPSNR: {:.3f} (Avg: {:.7f})'.format(
                    #         'rain100L',
                    #         epoch,
                    #         idx_data,
                    #         len(self.loader_test),
                    #         psnr,
                    #         # self.ckp.log[-1, idx_data, idx_scale],
                    #         # best[0][idx_data, idx_scale],
                    #         # best[1][idx_data, idx_scale] + 1,
                    #         avg
                    #     )
                    # )
                    # isderain = 0

                elif self.args.denoise:
                    for hr, _, filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise+hr).clamp(0, 255)
                        sr = self.model(nois_hr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i = i+1
                    # self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

        print('Forward: {:.2f}s\n'.format(timer_test.toc()))
        print('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        print(
            'Total: {:.2f}s\n'.format(timer_test.toc())
        )

        torch.set_grad_enabled(True)

    def test(self, epoch):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        print('\nEvaluation:')
        # self.ckp.add_log(
        #     torch.zeros(1, len(self.loader_test), len(self.scale))
        # )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        psnr_list = []
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                # d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    # for norain, rain, filename in tqdm(d, ncols=80):
                    # norain, rain = self.prepare(norain, rain)
                    rain = d['O']
                    norain = d['B']
                    norain, rain = self.prepare(norain, rain)
                    sr = self.model(rain, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                    #     sr, norain, scale, self.args.rgb_range
                    # )
                    psnr = utility.calc_psnr(
                        sr, norain, scale, self.args.rgb_range)
                    psnr_list.append(psnr)
                    avg = np.mean(psnr_list)
                    if self.args.save_results:
                        self.ckp.save_results(d, d['file_name'][0], save_list, 1)
                    # self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    # best = self.ckp.log.max(0)
                    print(
                        '[{} x1] Epoch: {} [{}]/[{}]\tPSNR: {:.3f} (Avg: {:.7f})'.format(
                            'rain100L',
                            epoch,
                            idx_data,
                            len(self.loader_test),
                            psnr,
                            # self.ckp.log[-1, idx_data, idx_scale],
                            # best[0][idx_data, idx_scale],
                            # best[1][idx_data, idx_scale] + 1,
                            avg
                        )
                    )
                    isderain = 0

                elif self.args.denoise:
                    for hr, _, filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise+hr).clamp(0, 255)
                        sr = self.model(nois_hr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i = i+1
                    # self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

        print('Forward: {:.2f}s\n'.format(timer_test.toc()))
        print('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        print(
            'Total: {:.2f}s\n'.format(timer_test.toc())
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs