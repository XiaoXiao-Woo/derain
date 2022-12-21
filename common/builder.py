# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/9/16 22:10
# @Author  : Xiao Wu
# @reference: 
#
import UDL.Basis.option
from .derain_dataset import derainSession
import models

def build_model(arch, task, cfg=None):

    if task == "derain":
        from UDL.Basis.python_sub_class import ModelDispatcher

        return ModelDispatcher.build_model(cfg)
    else:
        raise NotImplementedError(f"It's not supported in {task}")