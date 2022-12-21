import argparse
from UDL.Basis.option import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='DFTLW'):

    def __init__(self, cfg=None):
        super(parser_args, self).__init__()
        if cfg is None:
            from UDL.Basis.option import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))

        model_path = f''
        parser = argparse.ArgumentParser(description='PyTorch derain Training')
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('-samples_per_gpu', default=64, type=int,
                            metavar='N', help='mini-batch size (max: 180 10G GPU memory)') #64?32?
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--epochs', default=5000, type=int)
        parser.add_argument('--workers_per_gpu', default=4, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='DFTLW', type=str)
        parser.add_argument('--dataset', default={'train': 'Rain200L', 'val': 'Rain200L'}, type=str,
                            choices=['Rain200L', 'Rain100L', 'Rain200H', 'Rain100H',
                                     'test12', 'real', 'DID', 'SPA', 'DDN'],
                            help="Datasets")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")
        parser.add_argument('--crop_batch_size', type=int, default=128,
                            help='input batch size for training')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')


        # SRData.py dataset setting
        parser.add_argument('--model', default='ipt',
                            help='model name')
        parser.add_argument('--test_every', type=int, default=1000,
                            help='do test per every N batches')
        parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                            help='train dataset name')
        parser.add_argument('--no_augment', action='store_true',
                            help='do not use data augmentation')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--ext', type=str, default='sep',
                            help='dataset file extension')

        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = "Test"
        cfg.reg = False
        cfg.mode = "none"
        cfg.scale = [1]
        cfg.save_top_k = 1
        cfg.adjust_size_mode = "patch"
        cfg.patch_size = (64, 64)
        cfg.merge_args2cfg(args)
        cfg.workflow = [('val', 10), ('train', 1)]
        cfg.reset_lr = True

        print(cfg.pretty_text)

        self.merge_from_dict(cfg)


























