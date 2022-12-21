import os
import math
from decimal import Decimal
import utility

import IPython
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        #self.scheduler.last_epoch = 10
        self.scheduler.step()
        self.loss.step()

        #print(torch.load(os.path.join(self.ckp.dir, 'model', 'model_93.pt')))
        #self.model.load_state_dict(
        #        torch.load(os.path.join(self.ckp.dir, 'model', 'model_93.pt'))
        #    )

        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        

        self.model.train()
        huber_loss = torch.nn.SmoothL1Loss(reduce=True, size_average=True)

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            import random
            up_or_down = random.randint(0, 1)

            factor = random.randint(0, 3)
            sf = factor+1
            #print(sf)

            [b, c, h, w] = lr.shape
            new_h = int(h/sf)
            new_w = int(w/sf)

            downer =torch.nn.Upsample(size=[new_h, new_w], mode='bilinear', align_corners=True)
            uper = torch.nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=True)

            hr_rs = uper(downer(hr)) 

            hr_res, hr_res_feat = self.model(hr_rs, idx_scale)

            area_weight = sf*sf

            if up_or_down==0:

                sr, sr_feat = self.model(lr, idx_scale)

                lr_up = uper(lr)
                hr_up = uper(hr)
                sr_up, sr_feat_up = self.model(lr_up, idx_scale)

                sr_edge_x = sr[:,:,0:h-1,:]-sr[:,:,1:h,:]
                sr_edge_y = sr[:,:,:,0:w-1]-sr[:,:,:,1:w]

                hr_edge_x = hr[:,:,0:h-1,:]-hr[:,:,1:h,:]
                hr_edge_y = hr[:,:,:,0:w-1]-hr[:,:,:,1:w]

                [b, c, h, w] = lr_up.shape
                new_h = int(h/sf)
                new_w = int(w/sf)

                downer =torch.nn.Upsample(size=[new_h, new_w], mode='bilinear', align_corners=True)

                sr_up_down = downer(sr_up)
                [b, c, h, w] = sr_up_down.shape
                sr_up_down_edge_x = sr_up_down[:,:,0:h-1,:]-sr_up_down[:,:,1:h,:]
                sr_up_down_edge_y = sr_up_down[:,:,:,0:w-1]-sr_up_down[:,:,:,1:w]

                loss = self.loss(downer(sr_up), hr) + self.loss(sr, hr) + 0.05*self.loss(downer(sr_feat_up), sr_feat.detach())
                loss += 0.05*self.loss(hr_res, hr)

            else:
                downer =torch.nn.Upsample(size=[new_h, new_w], mode='bilinear', align_corners=True)
                lr_down = downer(lr)
                hr_down = downer(hr)
                sr_down, sr_feat_down = self.model(lr_down, idx_scale)
                sr, sr_feat = self.model(lr, idx_scale)

                [b, c, h, w] = hr.shape
                sr_edge_x = sr[:,:,0:h-1,:]-sr[:,:,1:h,:]
                sr_edge_y = sr[:,:,:,0:w-1]-sr[:,:,:,1:w]

                hr_edge_x = hr[:,:,0:h-1,:]-hr[:,:,1:h,:]
                hr_edge_y = hr[:,:,:,0:w-1]-hr[:,:,:,1:w]


                [b, c, h, w] = hr_down.shape
                sr_down_edge_x = sr_down[:,:,0:h-1,:]-sr_down[:,:,1:h,:]
                sr_down_edge_y = sr_down[:,:,:,0:w-1]-sr_down[:,:,:,1:w]

                hr_down_edge_x = hr_down[:,:,0:h-1,:]-hr_down[:,:,1:h,:]
                hr_down_edge_y = hr_down[:,:,:,0:w-1]-hr_down[:,:,:,1:w]

                loss = self.loss(sr_down, hr_down.detach())*area_weight + self.loss(sr, hr) + 0.05*self.loss(sr_feat_down, downer(sr_feat.detach()))*area_weight

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    lr = lr[:,:3,:,:]
                    hr = hr[:,:3,:,:]

                    sr, sr_feat = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

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
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
