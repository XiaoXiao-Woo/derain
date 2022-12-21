"""train finetune"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import warnings
warnings.filterwarnings("ignore")
import os
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from src.args import args
from src.data.imagenet import ImgData
from src.data.srdata import SRData, TrainValDataset
from src.data.div2k import DIV2K
from src.data.bicubic import bicubic
from src.ipt_model import IPT, IPT_post
from src.utils import Trainer
import copy


def eval_net(iargs):

    args = copy.deepcopy(iargs)
    args.batch_size = 128
    args.decay = 70
    args.patch_size = 48
    args.num_queries = 6
    args.model = 'vtip'
    args.num_layers = 4


    if args.epochs == 0:
        args.epochs = 1e8

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    # train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    print("TrainValDataset")
    train_dataset = TrainValDataset(args, name="test", train=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR', "idx", "filename"], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

    num_imgs = train_de_dataset.get_dataset_size()

    return args, train_loader, num_imgs

def train_net(distribute, imagenet):
    """Train net with finetune"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))#GRAPH_MODE PYNATIVE_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=device_id)

    if imagenet == 1:
        train_dataset = ImgData(args)
    elif not args.derain:
        train_dataset = DIV2K(args, name=args.data_train, train=True, benchmark=False)
        train_dataset.set_scale(args.task_id)
    else:
        # train_dataset = SRData(args, name=args.data_train, train=True, benchmark=False)
        # train_dataset.idx_scale(args.task_id)
        print("TrainValDataset")
        train_dataset = TrainValDataset(args, name="train", train=True)

    if distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=rank_size, gradients_mean=True)
        print('Rank {}, group_size {}'.format(rank_id, rank_size))
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   num_shards=rank_size, shard_id=rank_id, shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR", "idx", "filename"],
                                                   num_shards=rank_size, shard_id=rank_id, shuffle=True)
    else:
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR", "idx", "filename"], shuffle=True)

    if args.imagenet == 1:
        resize_fuc = bicubic()
        train_de_dataset = train_de_dataset.batch(
            args.batch_size,
            input_columns=["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
            output_columns=["LR", "HR", "idx", "filename"], drop_remainder=True,
            per_batch_map=resize_fuc.forward)
    else:
        print("Task: derain")
        train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    test_args, test_loader, test_num_imgs = eval_net(args)




    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)
    steps = train_de_dataset.get_dataset_size()
    net_m = IPT(args)
    inference = IPT_post(net_m, args)
    for m in net_m.parameters_and_names():
        print(m)
    # net_m = None
    print("Init net weights successfully")
    args.pth_path = './cache/results/edsr_baseline_x2/model_40.ckpt'
    start_epoch = 0
    if os.path.isfile(args.pth_path):
        start_epoch = int(args.pth_path.split('/')[-1].split('_')[1].replace('.ckpt', '')) + 1
        print("start_epoch:", start_epoch)
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")

    train_func = Trainer(args, train_loader, net_m)
    # train_func.eval_net(test_args, inference, test_loader, test_num_imgs)
    for epoch in range(start_epoch, args.epochs):
        # if epoch == 0:
        #     train_func.eval_net(test_args, inference, test_loader, test_num_imgs)
        train_func.update_learning_rate(epoch)
        train_func.train(epoch, steps)
        if (epoch >= 100 and epoch % 10 == 0):
            train_func.eval_net(test_args, inference, test_loader, test_num_imgs)

if __name__ == "__main__":
    train_net(distribute=args.distribute, imagenet=args.imagenet)
