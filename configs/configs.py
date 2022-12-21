import platform
from UDL.Basis.option import common_cfg
from UDL.Basis.python_sub_class import TaskDispatcher

class derain_cfg(TaskDispatcher, name='derain'):

    def __init__(self, cfg=None, arch=None):
        super(derain_cfg, self).__init__()

        if cfg is None:
            cfg = common_cfg()

        cfg.scale = [1]
        if platform.system() == 'Linux':
            cfg.data_dir = '/Data/Dataset/derain'
        if platform.system() == "Windows":
            cfg.data_dir = 'G:/woo/derain/data'
        cfg.best_prec1 = 10000
        cfg.best_prec5 = 10000
        cfg.metrics = 'min'
        cfg.task = "derain"
        cfg.save_fmt = "mat" # fmt is mat or not mat
        cfg.adjust_size_mode = "patch"

        if arch is not None:
            cfg = self.new(cfg=cfg, task=cfg.arch)
        self.merge_from_dict(cfg)
