import os
import logging


#######################  Settings of Network　　############################################################
channel = 20
Num_encoder = 8
ssim_loss = False
############################################
uint = "GRU"   #'RNN','GRU','LSTM'
########################################################################################
aug_data = True # Set as False for fair comparison

patch_size = 100
pic_is_pair = True  #input picture is pair or single

lr = 0.0001

data_dir = '../dataset/rain100H'
if pic_is_pair is False:
    data_dir = '/dataset/real-world-images'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'
test_dir = '../testdir'
gt_dir = '../gtdir'
log_level = 'info'
model_path = os.path.join(model_dir, 'latest_net')
save_steps = 1500

num_workers = 0

num_GPU = 1

device_id = '0'

epoch = 300
batch_size = 6


if pic_is_pair:
    root_dir = os.path.join(data_dir, 'train')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(1 * epoch * num_datasets / batch_size)
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


