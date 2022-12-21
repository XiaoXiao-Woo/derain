# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from option import args
import torch
import utility
import data
import loss
from trainer import Trainer
import warnings
# import argparse
from dataset import derainSession
warnings.filterwarnings('ignore')
import os
import model

# os.system('pip install einops')

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
# parser = argparse.ArgumentParser(description='PyTorch IPT Training')
# parser.add_argument('-b', '--batch-size', default=8, type=int,  # 8
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--workers', default=4, type=int)

# args = parser.parse_args()
args.workers = 4
args.distributed = False

sess = derainSession(args)
sess.scale = 1
train_loader, train_sampler = sess.get_dataloader('train', args.distributed)
val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)
eval_loader = sess.get_eval_dataloader('eval', args.distributed)



def main():
    global model
    if checkpoint.ok:
        # loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        # if args.pretrain != "":
        # if 's3' in args.pretrain:
        #     import moxing as mox
        #     mox.file.copy_parallel(args.pretrain, "/cache/models/ipt.pt")
        #     args.pretrain = "/cache/models/ipt.pt"
        state_dict = torch.load(args.pretrain)
        _model.model.load_state_dict(state_dict, strict=False)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args,  train_loader, val_loader, eval_loader, _model, _loss, checkpoint)#loader.loader_train, loader.loader_test,
        for epoch in range(args.epochs):
            if epoch == 0:
                t.eval(epoch)
            t.train(epoch)
            t.test(epoch)
        checkpoint.done()


if __name__ == '__main__':
    main()
