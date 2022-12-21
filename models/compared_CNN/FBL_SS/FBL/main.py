import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from utils.utils import set_random_seed


# torch.manual_seed(args.seed)
set_random_seed(args.seed)
checkpoint = utility.checkpoint(args)
'''
tensor(60.9937, device='cuda:0', grad_fn=<AddBackward0>)
tensor(41.1471, device='cuda:0', grad_fn=<AddBackward0>)
tensor(36.2177, device='cuda:0', grad_fn=<AddBackward0>)
tensor(24.5334, device='cuda:0', grad_fn=<AddBackward0>)
tensor(23.3344, device='cuda:0', grad_fn=<AddBackward0>)
'''
args.resume = 0
'''
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
'''

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

