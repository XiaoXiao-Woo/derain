import argparse
import os
from UDL.Basis.python_sub_class import TaskDispatcher

class parser_args(TaskDispatcher, name='ViT'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        script_path = script_path.split(cfg.task)[0]

        model_path = f''
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{cfg.task}',
                            help='path to save model')
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'),
                            help="Type of positional embedding to use on top of the image features")

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        ## Train
        parser.add_argument('--patch_size', type=int, default=(64, 64),
                            help='image2patch, set to model and dataset')
        parser.add_argument('--lr', default=1e-4, type=float)  # 1e-4 2e-4 8
        # parser.add_argument('--lr_backbone', default=1e-5, type=float)
        # parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('-samples_per_gpu', default=100, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--epochs', default=5000, type=int)
        parser.add_argument('--workers_per_gpu', default=4, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--accumulated-step', default=1, type=int)
        parser.add_argument('--clip_max_norm', default=0, type=float,
                            help='gradient clipping max norm')
        parser.add_argument('--dataset', default={'train': "Rain200L"}, type=str,
                            choices=[None, 'Rain200L', 'Rain100L', 'Rain200H', 'Rain100H',
                                     'test12', 'real', 'DID', 'SPA'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")
        # SRData
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

        parser.add_argument('--lr_scheduler', default=False, type=bool) # True
        args = parser.parse_args()

        args.experimental_desc = "Test"
        args.adjust_size_mode = "patch"
        cfg.workflow = [('train', 1)]


        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)