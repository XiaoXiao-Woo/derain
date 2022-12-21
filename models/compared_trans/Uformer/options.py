import argparse
import platform
import os

task = 'derain'

script_path = os.path.dirname(os.path.dirname(__file__))
script_path = script_path.split(task)[0]

# model_path = './results/Fu100H/HPT_new/OneAttn/model_2021-08-30-14-47/747.pth.tar'
model_path = f'{script_path}/results/Uformer/models/.pth'
# ./results/100L/PVT/layernorm_100L/model_2021-06-10-22-19/181.pth.tar  1159
# './results/100h/PVT/amp_test_100H/model_2021-05-29-16-24/476.pth.tar1'
# model_path = './results/100H/PVT/amp_test/model_2021-05-28-22-16/amp_model_best.pth.tar1'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# * Logger
parser.add_argument('--use-log', default=True
                    , type=bool)
parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{task}',
                    help='path to save model')
parser.add_argument('--log_dir', metavar='DIR', default='logs',
                    help='path to save log')
parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                    help='useless in this script.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='Uformer')
parser.add_argument('--use-tb', default=False, type=bool)

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
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--accumulated-step', default=1, type=int)
parser.add_argument('--clip_max_norm', default=0, type=float,
                    help='gradient clipping max norm')
## DDP
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', default=0, type=int,
                    help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
# parser.add_argument('--world-size', default=2, type=int,
#                     help='number of distributed processes, = gpus * nnodes')
parser.add_argument('--backend', default='nccl', type=str,  # gloo
                    help='distributed backend')
parser.add_argument('--dist-url', default='env://',
                    type=str,  # 'tcp://224.66.41.62:23456'
                    help='url used to set up distributed training')

## AMP
parser.add_argument('--amp', default=None, type=bool,
                    help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
##
parser.add_argument('--patch_size', type=int, default=128, #128
                    help='output patch size')
parser.add_argument('--dataset', default='Rain200L', type=str,
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
args.once_epoch = False
# log_string(args)#引发日志错误
# assert args.opt_level != 'O0' and args.amp != None, print("you must have apex or torch.cuda.amp")
args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
print(args.launcher)
assert args.accumulated_step > 0
args.scale = [1]
if platform.system() == 'Linux':
    args.data_dir = '/home/office-409/Datasets/derain'
if platform.system() == "Windows":
    args.data_dir = 'D:/Datasets/derain'

args.experimental_desc = "Test"
args.mode = None
args.adjust_size_mode = "patch"