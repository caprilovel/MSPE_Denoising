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
from local_utils.data_utils import Ecg_Dataset, EasyDataset
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


noise_type = ['bw', 'ma', 'em', 'emb']
models = ['unet', 'DANet', "ralenet_nra", "ralenet_mlp", "ralenet", "ACDAE", 
          "mpmtransformer", "pathformer",'denoisegan', 'datransformer',
          'newmpm']

args = TorchArgs()
args.add_argument("--intensity_index", type=int, default=0)
args.add_argument("--noise_type_index", type=int, default=0)
args.add_argument("--model_index", type=int, default=0)
args.add_argument("--in_channels", type=int, default=15)
args.add_argument("--dataset", type=str, default='ptbdb')


if args.parse_args().dataset == 'mitdb':
    noise_intensities = [-4, -2, 0, 2, 4]
elif args.parse_args().dataset == 'ptbdb':
    noise_intensities = [-8, -4, 0, 4, 8]


args_dict = vars(args.parse_args())
for k,v in args_dict.items():
    print(k, v)

batch_size = args_dict['batch_size']    
noise_name = noise_type[args_dict['noise_type_index']]
noise_intensity = noise_intensities[args_dict['intensity_index']]

if args_dict['dataset'] == 'ptbdb':
    data_path = '/local/storage/ding/processed_data/'
    try:
        os.path.exists(data_path + 'clean_snr{}.npy'.format(noise_intensity))
    except:
        print("data path not exist")
        exit(0)
    ecg_data = EasyDataset(data_path + 'clean_snr{}.npy'.format(noise_intensity), data_path + 'corrupt_snr{}.npy'.format(noise_intensity))
elif args_dict['dataset'] == 'mitdb':
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


model_args = {}
model_args['in_channels'] = args_dict['in_channels']
if args_dict['dataset'] == 'mitdb':
    model_args['in_channels'] = 2
elif args_dict['dataset'] == 'ptbdb':
    model_args['in_channels'] = 15
from denoise_train import train, train_gan
if args_dict['model_index'] == 0:
    from model.UNet import UNet
    model = UNet()
if args_dict['model_index'] == 1:
    from model.DAM import Seq2Seq2
    model = Seq2Seq2(**model_args)
if args_dict['model_index'] == 2:
    from model.transformer import ralenet
    model = ralenet()
if args_dict['model_index'] == 3:
    from model.ralenet import ralenet
    model = ralenet(low_level_enhence=False)
if args_dict['model_index'] == 4:
    from model.ralenet import ralenet
    model = ralenet(high_level_enhence=True, **model_args)
if args_dict['model_index'] == 5:
    from model.ACDAE import ACDAE
    model = ACDAE(**model_args)
if args_dict['model_index'] == 6:
    from model.mpmtransformer import mpmtransformer
    # deliver the model args to the model
    model = mpmtransformer(**model_args)
    
if args_dict['model_index'] == 7:

    from model.PathUformer import PathUformer
    model = PathUformer(device='cuda:0')
if args_dict['model_index'] == 8:

    from model.DCLSGAN import DCLSGAN
    model = DCLSGAN(**model_args)
if args_dict['model_index'] == 9:
    from model.DATransformer import TransformerDAE
    model = TransformerDAE(**model_args)
if args_dict['model_index'] == 10:
    from model.newmpm import EnhancedDAE
    model = EnhancedDAE(**model_args)
model_name = models[args_dict['model_index']]


epochs = args_dict['epochs']
batch_size = args_dict['batch_size']
def calculate_snr_for_dataloader(dataloader, use_gpu=True):
    snr = 0
    for _, (x, y) in enumerate(dataloader):
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        snr += 10 * torch.log10(torch.mean(y**2) / torch.mean((y - x)**2))
    return snr / len(dataloader)
train_ground_snr = calculate_snr_for_dataloader(train_loader)
test_ground_snr = calculate_snr_for_dataloader(test_loader)
print("train_ground_snr: ", train_ground_snr)
print("test_ground_snr: ", test_ground_snr)
summarywriter.add_scalar('train_ground_snr', train_ground_snr, 0)
summarywriter.add_scalar('test_ground_snr', test_ground_snr, 0)



if args_dict['model_index'] == 8:
    train_gan(epochs=epochs, 
          batch_size=batch_size, 
          model=model,
          train_loader=train_loader, 
          test_loader=test_loader, 
          model_name=model_name, 
          use_gpu=True, 
          noise_name=noise_name, 
          noise_intensity=noise_intensity, 
          writer=summarywriter)
else:
    train(epochs=epochs, batch_size=batch_size, model=model, train_loader=train_loader, test_loader=test_loader, model_name=model_name, use_gpu=True, noise_name=noise_name, noise_intensity=noise_intensity, model_index=args_dict['model_index'], writer=summarywriter)
summarywriter.close()

