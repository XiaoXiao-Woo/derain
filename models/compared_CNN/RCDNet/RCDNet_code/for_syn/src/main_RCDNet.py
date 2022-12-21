import torch
from .model import Model
from . import loss
from models.base_model import DerainModel
from UDL.Basis.criterion_metrics import SetCriterion

class build_RCDNet(DerainModel, name='RCDNet'):

    def __call__(self, cfg):
        device = torch.device(cfg.device)

        model = Model(cfg, cfg.resume_from)

        # 造成日志无法使用
        # if args.global_rank == 0:
        #     log_string(model)

        weight_dict = {'Loss': -1}
        losses = {'Loss':  loss.Loss(cfg, cfg.resume_from) if not cfg.test_only else None}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR
        else:
            scheduler = None

        model.set_metrics(criterion)

        # from torchstat import stat
        # stat(model.cuda(), [[1, 3, 64, 64]])

        return model, criterion, optimizer, scheduler
