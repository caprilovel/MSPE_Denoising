import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os  # 新增os模块
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import Logger, easymail
from global_utils.torch_utils.torch_utils import random_seed
from global_utils.torch_utils.Args import TorchArgs
from local_utils.data_utils import Ecg_Dataset
import random
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import warnings
warnings.filterwarnings('ignore')

random_seed(2023)

# 新增数据集参数
noise_intensities = [-8, -4, 0, 4, 8]
noise_type = ['bw', 'ma', 'em', 'emb']
models = ['unet', 'DANet', "ralenet", "testmodel"]
datasets = ['ptbdb', 'mitdb']  # 新增数据集选项

args = TorchArgs()
args.add_argument("--intensity_index", type=int, default=0)
args.add_argument("--noise_type_index", type=int, default=0)
args.add_argument("--model_index", type=int, default=0)
args.add_argument("--dataset", type=str, default='ptbdb', choices=datasets)  # 新增dataset参数

args_dict = vars(args.parse_args())
for k,v in args_dict.items():
    print(k, v)

batch_size = args_dict['batch_size']
dataset_type = args_dict['dataset']  # 获取数据集类型

# 修改后的数据加载部分
if dataset_type == 'ptbdb':
    data_path = '/local/storage/ding/processed_data/'
    noise_intensity = noise_intensities[args_dict['intensity_index']]
    
    # 检查文件是否存在
    clean_path = os.path.join(data_path, f'clean_snr{noise_intensity}.npy')
    corrupt_path = os.path.join(data_path, f'corrupt_snr{noise_intensity}.npy')
    
    if not os.path.exists(clean_path) or not os.path.exists(corrupt_path):
        raise FileNotFoundError(f"Required data files not found in {data_path}")
    
    # 假设EasyDataset的实现
    class EasyDataset(Dataset):
        def __init__(self, clean_path, corrupt_path):
            self.clean_data = np.load(clean_path)
            self.corrupt_data = np.load(corrupt_path)
            
            # 确保数据对齐
            assert len(self.clean_data) == len(self.corrupt_data), "Data mismatch!"
            
        def __len__(self):
            return len(self.clean_data)
        
        def __getitem__(self, idx):
            return self.corrupt_data[idx], self.clean_data[idx]
    
    ecg_data = EasyDataset(clean_path, corrupt_path)

elif dataset_type == 'mitdb':
    data_path = '/local/storage/ding/ecg_data/dict_data/'
    noise_name = noise_type[args_dict['noise_type_index']]
    noise_intensity = noise_intensities[args_dict['intensity_index']]
    ecg_data = Ecg_Dataset(
        noise_name=noise_name, 
        noise_intensity=noise_intensity, 
        path=data_path
    )

# 公共数据处理部分
def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)

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

# 后续评估代码保持不变
from local_utils.evaluate import RMSE, SNR
from local_utils.denoisefunc import wavelet_denoise

rmse_array = []
snr_array = []
for _, (inputs, targets) in enumerate(train_loader):
    input_np = inputs.numpy()
    out = wavelet_denoise(input_np)
    out = torch.FloatTensor(out)
    rmse_array.append(RMSE(out, targets))
    snr_array.append(SNR(out, targets))
    
rmse_array = torch.concat(rmse_array, dim=0)
snr_array = torch.concat(snr_array, dim=0)

# 添加数据集类型到输出信息
print(f"Dataset: {dataset_type}")
print(f"Noise intensity: {noise_intensity}")
print(f"RMSE: {rmse_array.mean().item():.4f}")
print(f"SNR: {snr_array.mean().item():.4f}")

with open("dwt_dn_result.txt", "a") as f:
    f.write(
        f"Dataset: {dataset_type}\n"
        f"Noise intensity: {noise_intensity}\n"
        f"RMSE: {rmse_array.mean().item():.4f}\n"
        f"SNR: {snr_array.mean().item():.4f}\n\n"
    )