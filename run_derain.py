# GPL License
# Copyright (C) 2022, UESTC
# All Rights Reserved 
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference: 
from configs.configs import TaskDispatcher
from UDL.AutoDL.trainer import main
from common.builder import build_model, derainSession

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='derain', mode='entrypoint', arch='Restormer')
    print(TaskDispatcher._task.keys())
    main(cfg, build_model, derainSession)
