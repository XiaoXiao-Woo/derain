"""
Backbone modules.
"""
import torch
# from UDL.derain.builder import DERAIN_MODELS
# from UDL.Basis.auxiliary import create_logger, log_string, MetricLogger, SmoothedValue, set_random_seed
from .networks import *
from .SSIM import SSIM as SSIM_PReNet


from models.base_model import DerainModel
from UDL.Basis.criterion_metrics import SetCriterion

class build_PReNet(DerainModel, name='PReNet'):

    def __call__(self, cfg):
        device = torch.device(cfg.device)

        model = PReNet(recurrent_iter=6, use_GPU=True)

        # 造成日志无法使用
        # if args.global_rank == 0:
        #     log_string(model)

        weight_dict = {'Loss': -1}
        losses = {'Loss': SSIM_PReNet().cuda()}

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        if cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR
            scheduler.step_size = [30, 50, 80]
            scheduler.gamma = 0.2
        else:
            scheduler = None

        model.set_metrics(criterion)

        # from torchstat import stat
        # stat(model.cuda(), [[1, 3, 64, 64]])

        return model, criterion, optimizer, scheduler


