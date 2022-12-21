import argparse
import os
from UDL.Basis.python_sub_class import TaskDispatcher

class parser_args(TaskDispatcher, name='SPANet'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        script_path = script_path.split(cfg.task)[0]

        model_path = f''

        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        # * Logger
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{cfg.task}',
                            help='path to save model')
        parser.add_argument('--log_dir', metavar='DIR', default='logs',
                            help='path to save log')
        parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                            help='useless in this script.')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='SPANet')
        parser.add_argument('--use-tb', default=False, type=bool)

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')

        ## Train
        parser.add_argument('--patch_size', type=int, default=128, #256
                            help='image2patch, set to model and dataset')
        parser.add_argument('--lr', default=5e-3, type=float)
        parser.add_argument('-samples_per_gpu', default=16, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=1, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--epochs', default=400, type=int)
        parser.add_argument('--workers_per_gpu', default=4, type=int)
        parser.add_argument('--resume',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--accumulated-step', default=1, type=int)
        parser.add_argument('--clip_max_norm', default=0, type=float,
                            help='gradient clipping max norm')

        parser.add_argument('--lr_scheduler', default=True, type=bool)

        ## Data
        parser.add_argument('--model', default='rescan',
                            help='model name')
        parser.add_argument('--test_every', type=float, default=1000,
                            # {r:1000, epoch:100}-RCDNet, 即RESCAN {r:156.25,epoch:400} or {r:625, epoch:100}
                            help='do test per every N batches')
        parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                            help='train dataset name')
        parser.add_argument('--no_augment', action='store_true', default=True,
                            help='do not use data augmentation')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--ext', type=str, default='sep',
                            help='dataset file extension')

        # Benchmark
        parser.add_argument('--eval', default=False, type=bool, help="performing evalution for patch2entire")
        parser.add_argument('--dataset', default={'train': 'Rain200H'}, type=str,
                            choices=[None, 'Rain200H', 'Rain100L', 'Rain200H',
                                     'Rain100H', 'PReNetData', 'DID', 'SPA', 'DDN',
                                     'test12', 'real', ],
                            help="set dataset name for training"
                                 "real/test12 is eval-only")
        parser.add_argument('--crop_batch_size', type=int, default=128,
                            help='input batch size for training')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')

        args = parser.parse_args()
        args.experimental_desc = "Test"
        args.adjust_size_mode = "patch"

        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)