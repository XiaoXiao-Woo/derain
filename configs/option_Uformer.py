import os
import torch
from UDL.Basis.python_sub_class import TaskDispatcher
import argparse
import platform
import os

class parser_args(TaskDispatcher, name='Uformer'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from configs.configs import derain_cfg
            cfg = derain_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        script_path = script_path.split(cfg.task)[0]

        model_path = f'{script_path}/results/Uformer/models/.pth'

        parser = argparse.ArgumentParser(description='PyTorch Training')

        parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{cfg.task}',
                            help='path to save model')
        parser.add_argument('--config', help='train config file path', default='')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='Uformer')

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
        # parser.add_argument('--train_ps', type=int, default=48,
        #                     help='image2patch, set to model and dataset')
        parser.add_argument('--lr', default=1e-4, type=float)  # 1e-4 2e-4 8
        # parser.add_argument('--lr_backbone', default=1e-5, type=float)
        # parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('-samples_per_gpu', default=8, type=int, #8
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

        parser.add_argument('--patch_size', type=int, default=128, #128
                            help='output patch size')
        parser.add_argument('--dataset', default={'train': 'Rain200L'}, type=str,
                            choices=[None, 'Rain200L', 'Rain100L', 'Rain200H', 'Rain100H',
                                     'test12', 'real', 'DID', 'SPA', 'DDN'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")
        parser.add_argument('--crop_batch_size', type=int, default=32,
                            help='input batch size for training')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')
        # SRData
        parser.add_argument('--model', default='uformer',
                            help='model name')
        parser.add_argument('--test_every', type=int, default=1000,
                            help='do test per every N batches')
        parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                            help='train dataset name')
        parser.add_argument('--no_augment', action='store_true',
                            help='do not use data augmentation')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--lr_scheduler', default=False, type=bool)
        parser.add_argument('--ext', type=str, default='sep',
                            help='dataset file extension')


        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_embed', type=str, default='linear', help='linear/conv token embedding')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        args = parser.parse_args()

        args.scale = [1]

        args.experimental_desc = "Test"
        args.mode = None
        args.adjust_size_mode = "patch"
        args.workflow = [('train', 1)]

        cfg.merge_args2cfg(args)

        print(cfg.pretty_text)

        self.merge_from_dict(cfg)


class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=40, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        # parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='Rain100L')#SIDD
        parser.add_argument('--pretrain_weights',type=str, default='./log_bak/Uformer_/models/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0005, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default ='Uformer',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='deraining',  help='image restoration mode')#denoising

        # args for saving
        # parser.add_argument('--save_dir', type=str, default ='',  help='save dir')#/home/ma-user/work/deNoTr/log
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_embed', type=str,default='linear', help='linear/conv token embedding')
        parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        # args for training
        parser.add_argument('--train_ps', type=int, default=64, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default ='../../derain/dataset/rain100L',  help='dir of train data')#../datasets/SIDD/train'
        parser.add_argument('--val_dir', type=str, default ='../../derain/datasets/SIDD/val',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')


        return parser
