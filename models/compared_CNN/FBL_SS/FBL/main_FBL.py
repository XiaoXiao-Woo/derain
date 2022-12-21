import torch
from models.base_model import DerainModel
from . import utility
from . import model as pre_model
from . import loss as pre_loss
from UDL.Basis.criterion_metrics import SetCriterion

class build_FBL(DerainModel, name="FBL"):

    def __call__(self, cfg):
        cfg.decay_type = "step"
        cfg.optimizer = "Adam"
        cfg.model = "RFBLL"
        device = torch.device(cfg.device)

        model = pre_model.Model(cfg, None).cuda()
        # weight_dict = {'L1Loss': 1}
        # losses = {'L1Loss': torch.nn.L1Loss(reduction='mean').cuda()}

        loss = pre_loss.Loss(cfg, None)
        weight_dict = {'Loss': 1}
        losses = {'Loss': loss}

        optimizer = utility.make_optimizer(cfg, model)
        scheduler = utility.make_scheduler(cfg, optimizer)

        criterion = SetCriterion(weight_dict=weight_dict,
                                 losses=losses)
        criterion.to(device)

        return model, criterion, optimizer, scheduler



