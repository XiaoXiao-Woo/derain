import argparse
import os
from UDL.Basis.python_sub_class import TaskDispatcher
from configs.template import set_template

class parser_args(TaskDispatcher, name='RCDNet'):

    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        script_path = script_path.split(cfg.task)[0]

        model_path = f''

        parser = argparse.ArgumentParser(description='RCDNet')
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/derain',
                            help='path to save model')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='RCDNet')
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--accumulated-step', default=1, type=int)
        parser.add_argument('--clip_max_norm', default=0, type=float,
                            help='gradient clipping max norm')
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--print-freq', '-p', default=5, type=int,
                            metavar='N', help='print frequency (default: 10)')
        # Benchmark
        parser.add_argument('--eval', default=False, type=bool, help="performing evalution for patch2entire")
        parser.add_argument('--dataset', default={'train': 'Rain200L'}, type=str,
                            choices=['Rain200H', 'Rain100L', 'Rain200H',
                                     'Rain100H', 'PReNetData', 'DID', 'SPA', 'DDN',
                                     'test12', 'real'],
                            help="set dataset name for training"
                                 "real/test12 is eval-only")
        parser.add_argument('--crop_batch_size', type=int, default=256,
                            help='input batch size for training')

        ###
        parser.add_argument('--debug', action='store_true',
                            help='Enables debug mode')
        parser.add_argument('--template', default='.',
                            help='You can set various templates in option.py')

        # Hardware specifications
        parser.add_argument('--n_threads', type=int, default=0,
                            help='number of threads for data loading')
        parser.add_argument('--cpu', action='store_true',
                            help='use cpu only')
        parser.add_argument('--n_GPUs', type=int, default=1,
                            help='number of GPUs')
        parser.add_argument('--seed', type=int, default=1,
                            help='random seed')

        # Data specifications
        parser.add_argument('--dir_data', type=str, default='../data',
                            help='dataset directory')
        parser.add_argument('--dir_demo', type=str, default='../test',
                            help='demo image directory')
        parser.add_argument('--data_train', type=str, default='RainHeavy',  # 'DIV2K',
                            help='train dataset name')
        parser.add_argument('--data_test', type=str, default='RainHeavyTest',  # 'DIV2K',
                            help='test dataset name')
        parser.add_argument('--data_range', type=str, default='1-200/1-100',  # 1-200/1-100 1400
                            help='train/test data range')
        parser.add_argument('--ext', type=str, default='sep',
                            help='dataset file extension')
        parser.add_argument('--scale', type=str, default='2',
                            help='super resolution scale')
        parser.add_argument('--patch_size', type=int, default=64,
                            help='output patch size')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--chop', action='store_true',
                            help='enable memory-efficient forward')
        parser.add_argument('--no_augment', action='store_true', default=True,
                            help='do not use data augmentation')

        # Log specifications
        parser.add_argument('--save', type=str, default='RCDNet_syn',
                            help='file name to save')
        parser.add_argument('--load', type=str, default='RCDNet_syn',
                            help='file name to load')
        parser.add_argument('--resume', type=int, default=0,
                            help='resume from specific checkpoint')
        parser.add_argument('--save_models', action='store_true', default=True,
                            help='save all intermediate models')
        parser.add_argument('--print_every', type=int, default=100,
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save_results', action='store_true', default=True,
                            help='save output results')

        # Model specifications
        parser.add_argument('--model', default='RCDNet',
                            help='model name')
        parser.add_argument('--act', type=str, default='relu',
                            help='activation function')
        parser.add_argument('--pre_train', type=str, default='.',
                            # '../experiment/RCDNet_syn/model/model_latest.pt', ../../../Pretrained Model/rain1400/model_best.pt
                            help='pre-trained model directory')
        parser.add_argument('--extend', type=str, default='.',
                            help='pre-trained model directory')
        parser.add_argument('--n_resblocks', type=int, default=16,
                            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
        parser.add_argument('--res_scale', type=float, default=1,
                            help='residual scaling')
        parser.add_argument('--shift_mean', default=True,
                            help='subtract pixel mean from the input')
        parser.add_argument('--dilation', action='store_true',
                            help='use dilated convolution')
        parser.add_argument('--precision', type=str, default='single',
                            choices=('single', 'half'),
                            help='FP precision for test (single | half)')

        # Network settings
        parser.add_argument('--num_M', type=int, default=32,
                            help='the number of rain maps')
        parser.add_argument('--num_Z', type=int, default=32,
                            help='the number of dual channles')
        parser.add_argument('--T', type=int, default=4,
                            help='Resblocks number in each proxNet')
        parser.add_argument('--stage', type=int, default=17,
                            help='Stage number S')

        # Training specifications
        parser.add_argument('--test_every', type=int, default=1000,
                            help='do test per every N batches')
        parser.add_argument('--epochs', type=int, default=450,
                            help='number of epochs to train')
        parser.add_argument('--samples_per_gpu', type=int, default=16,
                            help='input batch size for training')
        parser.add_argument('--split_batch', type=int, default=1,
                            help='split the batch into smaller chunks')
        parser.add_argument('--self_ensemble', action='store_true',
                            help='use self-ensemble method for test')
        parser.add_argument('--test_only', action='store_true', default=False,
                            help='set this option to test the model')
        parser.add_argument('--reset', action='store_true',
                            help='reset the training')
        # Optimization specifications
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
        parser.add_argument('--lr_decay', type=int, default=110,
                            help='learning rate decay per N epochs')
        parser.add_argument('--decay_type', type=str, default='step',
                            help='learning rate decay type')
        parser.add_argument('--gamma', type=float, default=0.2,
                            help='learning rate decay factor for step decay')
        parser.add_argument('--optimizer', default='ADAM',
                            choices=('SGD', 'ADAM', 'RMSprop'),
                            help='optimizer to use (SGD | ADAM | RMSprop)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='SGD momentum')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='ADAM beta1')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='ADAM beta2')
        parser.add_argument('--epsilon', type=float, default=1e-8,
                            help='ADAM epsilon for numerical stability')
        parser.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay')

        # Loss specifications
        parser.add_argument('--loss', type=str, default='1*MSE',
                            help='loss function configuration')
        parser.add_argument('--skip_threshold', type=float, default='1e6',
                            help='skipping batch that has large error')

        args = parser.parse_args()
        set_template(args)

        args.experimental_desc = "Test"
        args.scale = list(map(lambda x: int(x), args.scale.split('+')))
        args.scale = [2]

        if args.epochs == 0:
            args.epochs = 1e8

        for arg in vars(args):
            if vars(args)[arg] == 'True':
                vars(args)[arg] = True
            elif vars(args)[arg] == 'False':
                vars(args)[arg] = False

        args.adjust_size_mode = "patch"
        # args.reset_lr = True
        # args.lr = 0.0002
        # args.last_epoch = 31

        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)