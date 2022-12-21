# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:
"""
Backbone modules.
"""
import torch
from torch import nn
from models.compared_trans.vit.vit import ViT as net

from models.base_model import DerainModel
from UDL.Basis.criterion_metrics import SetCriterion
class build_ViT_derain(DerainModel, name='ViT'):

    def __call__(self, cfg):
        schedule = None
        device = torch.device(cfg.device)
        model = net(
            in_channels=3,
            image_size=cfg.patch_size[0],
            patch_size=16,
            dim=256,
            depth=24,
            heads=16,
            dropout=0,
            emb_dropout=0
        ).cuda()

        # 造成日志无法使用
        # if cfg.global_rank == 0:
        #     log_string(model)

        weight_dict = {'l1': 1}
        losses = {'l1': nn.L1Loss().cuda()}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=1e-4)

        return model, criterion, optimizer, schedule
