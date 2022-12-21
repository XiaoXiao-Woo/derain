"""
Backbone modules.
"""
import torch
from torch import nn
from UDL.Basis.criterion_metrics import SetCriterion
from models.base_model import DerainModel


class build_Uformer(DerainModel, name='Uformer'):

    def __call__(self, cfg):
        from . import utils

        schedular = None

        device = torch.device(cfg.device)

        model = utils.get_arch(cfg)

        weight_dict = {'Loss': 1}
        losses = {'Loss': nn.L1Loss().cuda()}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=1e-4)

        return model, criterion, optimizer, schedular


