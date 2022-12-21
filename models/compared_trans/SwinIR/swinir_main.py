"""
Backbone modules.
"""
import os
import torch
from torch import nn
from UDL.Basis.criterion_metrics import SetCriterion
from models.compared_trans.SwinIR.network_swinir import SwinIR as net
from models.base_model import DerainModel

def define_model(args):
    # 001 classical image sr
    if args.task_head == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')


    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task_head == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    # 003 real-world image sr
    elif args.task_head == 'real_sr':
        # if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts

        model = net(upscale=1, in_chans=3, img_size=args.patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path)['params'], strict=True)

    if os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from)
        print("best_psnr: ", ckpt['best_metric'])
        # model.load_state_dict(ckpt['state_dict'])
        # partial_load_checkpoint(ckpt['state_dict'], args.amp)

    return model


class build_SwinIR(DerainModel, name='SwinIR'):

    def __call__(self, cfg):

        schedular = None

        device = torch.device(cfg.device)

        model = define_model(cfg)
        # 造成日志无法使用
        # if cfg.global_rank == 0:
        #     log_string(model)

        weight_dict = {'Loss': 1}
        losses = {'Loss': nn.L1Loss().cuda()}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=1e-4)

        return model, criterion, optimizer, schedular
