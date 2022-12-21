"""
Backbone modules.
"""
import torch
from torch import nn
from UDL.Basis.criterion_metrics import SetCriterion
from models.base_model import DerainModel

class build_IPT(DerainModel, name='IPT'):

    def __call__(self, cfg):
        from . import model

        device = torch.device(cfg.device)
        model = model.Model(cfg, cfg.resume_from)
        weight_dict = {'Loss': 1}
        losses = {'Loss': nn.L1Loss().cuda()}
        schedule = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=1e-4)

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        return model, criterion, optimizer, schedule

