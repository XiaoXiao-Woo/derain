"""
Backbone modules.
"""
import os
import torch
from torch import nn
from UDL.Basis.criterion_metrics import SetCriterion
from UDL.mmcv.mmcv.utils import print_log
from models.base_model import DerainModel

class build_NLDEN(DerainModel, name='NLEDN'):
    def __call__(self, cfg):
        from .lib.NLEDN import NLEDN

        device = torch.device(cfg.device)

        model = NLEDN().cuda()

        weight_dict = {'Loss': 1}
        losses = {'Loss': nn.L1Loss().cuda()}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=1e-4)
        if cfg.lr_scheduler:
            from UDL.Basis.optim import lr_scheduler
            scheduler = lr_scheduler(optimizer.param_groups[0]['lr'], 600)
            scheduler.set_optimizer(optimizer, torch.optim.lr_scheduler.MultiStepLR)
            # scheduler.set_optimizer(optimizer, None)
            scheduler.get_lr_map("step_lr_100",
                                 out_file=os.path.join(cfg.out_dir, f"./step_lr_100_{cfg.experimental_desc}.png"))
        else:
            scheduler = None

        return model, criterion, optimizer, scheduler


