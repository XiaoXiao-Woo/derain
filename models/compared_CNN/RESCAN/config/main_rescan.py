"""
Backbone modules.
"""
import torch
from UDL.Basis.criterion_metrics import SetCriterion
from .model import RESCAN
from models.base_model import DerainModel


class build_RESCAN(DerainModel, name='RESCAN'):

    def __call__(self, cfg):
        device = torch.device(cfg.device)

        model = RESCAN(cfg).cuda()
        #造成日志无法使用
        # if args.global_rank == 0:
        #     log_string(model)

        weight_dict = {'Loss': 1}
        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=None)
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if cfg.lr_scheduler:
            from UDL.Basis.optim import lr_scheduler, optim
            scheduler = optim.lr_scheduler.StepLR
            scheduler.step_size = [150, 175]
            scheduler.gamma = 0.1
        else:
            scheduler = None

        # from torchstat import stat
        # stat(model.cuda(), [[1, 3, 64, 64]])

        return model, criterion, optimizer, scheduler

