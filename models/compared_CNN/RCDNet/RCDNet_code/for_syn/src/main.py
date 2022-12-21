import torch
import os
from models.compared_CNN.RCDNet.RCDNet_code.for_syn.src import utility, data, loss
from models.compared_CNN.RCDNet.RCDNet_code.for_syn.src import model
from option import args
from models.compared_CNN.RCDNet.RCDNet_code.for_syn.src.trainer import Trainer
from UDL.Basis.auxiliary import set_random_seed
import multiprocessing
import time


args.resume = 0

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    # torch.manual_seed(args.seed)
    set_random_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        print_network(model)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            # t.test()
        # t.test_real()
        checkpoint.done()
    



