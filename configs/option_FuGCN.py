import argparse
import os
from UDL.Basis.python_sub_class import TaskDispatcher

class parser_args(TaskDispatcher, name='FuGCN'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = ''


        parser = argparse.ArgumentParser(description='PyTorch derain Training')
        # * Logger
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default='{}/results/{}'.format(root_dir, cfg.task),
                            help='path to save model')
        parser.add_argument('--lr', default=1e-4, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=10, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        ##
        parser.add_argument('--arch', '-a', metavar='ARCH', default='FuGCN', type=str)
        parser.add_argument('--dataset', default={'train': 'Rain200L'}, type=str,
                            choices=[None, 'Rain100L', 'Rain100H',
                                     'Rain200L', 'Rain200H', 'DDN',
                                     'rain12600'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")

        parser.add_argument('--ext', type=str, default='sep',
                            help='dataset file extension')
        parser.add_argument('--crop_batch_size', type=int, default=32,
                            help='input batch size for training')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')
        parser.add_argument('--test_every', type=int, default=1000,
                            help='do test per every N batches')
        parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                            help='train dataset name')
        parser.add_argument('--no_augment', action='store_true',
                            help='do not use data augmentation')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')


        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'
        cfg.merge_args2cfg(args)
        cfg.patch_size = (100, 100)
        cfg.adjust_derain_mode = "patch"
        cfg.reg = False
        cfg.reset_lr = False
        cfg.mode = "none"
        cfg.workflow = [('train', 1)]
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)