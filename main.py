from global_utils.torch_utils.cuda import find_gpus
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import Logger, easymail
from global_utils.torch_utils.torch_utils import random_seed
from global_utils.torch_utils.Args import TorchArgs
from local_utils.data_utils import Ecg_Dataset
import random
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import warnings
warnings.filterwarnings('ignore')
#time seed is the time of this second
import time
# time_seed = int(time.time())
time_seed = 2024
random_seed(time_seed)


from torch.utils.tensorboard import SummaryWriter

summarywriter = SummaryWriter(log_dir='log/summary')

noise_intensities = [-4, -2, 0, 2, 4]
noise_type = ['bw', 'ma', 'em', 'emb']
models = ['unet', 'DANet', "ralenet_nra", "ralenet_mlp", "ralenet", "ACDAE", "mpmtransformer", "pathformer"]

args = TorchArgs()
args.add_argument("--intensity_index", type=int, default=0)
args.add_argument("--noise_type_index", type=int, default=0)
args.add_argument("--model_index", type=int, default=0)


args_dict = vars(args.parse_args())
for k,v in args_dict.items():
    print(k, v)

batch_size = args_dict['batch_size']    
noise_name = noise_type[args_dict['noise_type_index']]
noise_intensity = noise_intensities[args_dict['intensity_index']]

data_path = '/local/storage/ding/ecg_data/dict_data/'

ecg_data = Ecg_Dataset(noise_name=noise_name, noise_intensity=noise_intensity, path=data_path)

def custom_collate_fn(batch):
    # change the data type into FloatTensor
    inputs, targets = zip(*batch)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)

# since random seed is defined, we can use random.sample to select samples
total_samples = len(ecg_data)
select_samples = random.sample(range(total_samples), 10000)
select_dataset = Subset(ecg_data, select_samples)
train_ratio = 0.8
test_ratio = 0.2 
train_size = int(train_ratio * len(select_dataset))
test_size = len(select_dataset) - train_size
train_dataset, test_dataset = random_split(select_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn)

from denoise_train import train 
if args_dict['model_index'] == 0:
    from model.UNet import UNet
    model = UNet()
if args_dict['model_index'] == 1:
    from model.DAM import Seq2Seq2
    model = Seq2Seq2()
if args_dict['model_index'] == 2:
    from model.transformer import ralenet
    model = ralenet()
if args_dict['model_index'] == 3:
    from model.ralenet import ralenet
    model = ralenet(low_level_enhence=False)
if args_dict['model_index'] == 4:
    from model.ralenet import ralenet
    model = ralenet(high_level_enhence=True)
if args_dict['model_index'] == 5:
    from model.ACDAE import ACDAE
    model = ACDAE()
if args_dict['model_index'] == 6:
    from model.mpmtransformer import mpmtransformer
    model = mpmtransformer()
if args_dict['model_index'] == 7:
    # config = {
    #     'layer_nums': 3,
    #     'num_nodes' : 2,
    #     'pred_len' : 256,
    #     'seq_len' : 256,
    #     'k': 2,
    #     'num_experts_list': [3, 3, 3],
    #     'patch_size_list': [[16, 8, 4], [16, 8, 4], [8, 4, 2]],
    #     'd_model': 16,
    #     'd_ff': 64,
    #     'residual_connection': 0,
    #     'revin': 1,
    #     'gpu': '0',
    # }
    # config = type('Config', (object,), config)()
    # from model.PathFormer import Model
    from model.PathUformer import PathUformer
    model = PathUformer(device='cuda:0')

model_name = models[args_dict['model_index']]


epochs = args_dict['epochs']
batch_size = args_dict['batch_size']
train(epochs=epochs, batch_size=batch_size, model=model, train_loader=train_loader, test_loader=test_loader, model_name=model_name, use_gpu=True, noise_name=noise_name, noise_intensity=noise_intensity, model_index=args_dict['model_index'], writer=summarywriter)


summarywriter.close()
# if __name__ == "__main__":
#     x = torch.randn(32, 2, 256)
#     for _, (x, y) in enumerate(train_loader):
#         print(x.shape)
#         z = model(x)
#         print(z.shape)

