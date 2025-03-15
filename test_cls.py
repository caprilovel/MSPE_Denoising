import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from torch.utils.data import Dataset, DataLoader

from model.ResNet_cls import ResNet_cls


def acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred == label).item() / len(label)


def precision(pred, label):
    pred = torch.argmax(pred, dim=1)
    TP = torch.sum(pred * label).item()
    FP = torch.sum(pred * (1 - label)).item()
    return TP / (TP + FP)

def f1_score(pred, label):
    pred = torch.argmax(pred, dim=1)
    TP = torch.sum(pred * label).item()
    FP = torch.sum(pred * (1 - label)).item()
    FN = torch.sum((1 - pred) * label).item()
    return TP / (TP + 0.5 * (FP + FN))

class ClsDataset(Dataset):
    def __init__(self, train=True, noised=False, path='/local/storage/ding/ecg_data/') -> None:
        super().__init__()
        if not noised:
            if train:
                self.N_data = np.load('/local/storage/ding/ecg_data/cls_data/N_train_data.npy')
                self.V_data = np.load('/local/storage/ding/ecg_data/cls_data/V_train_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
            else:
                self.N_data = np.load('/local/storage/ding/ecg_data/cls_data/N_test_data.npy')
                self.V_data = np.load('/local/storage/ding/ecg_data/cls_data/V_test_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
        else:
            if train:
                self.N_data = np.load('/local/storage/ding/ecg_data/cls_data/noised_N_train_data.npy')
                self.V_data = np.load('/local/storage/ding/ecg_data/cls_data/noised_V_train_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
            else:
                self.N_data = np.load('/local/storage/ding/ecg_data/cls_data/noised_N_test_data.npy')
                self.V_data = np.load('/local/storage/ding/ecg_data/cls_data/noised_V_test_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
        
            
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
    

test_dataset = ClsDataset(train=False)
noised_test_dataset = ClsDataset(train=False, noised=True)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
noised_test_dataloader = DataLoader(noised_test_dataset, batch_size=128, shuffle=False)

model = ResNet_cls()
model.load_state_dict(torch.load('./model_save/cls_model.pth'))
model = model.cuda()


# from model.mpmtransformer import mpmtransformer
# denoised_model = mpmtransformer()
# denoised_model.load_state_dict(torch.load('./model_save/mpmtransformer/mpmtransformer_199_emb_intensity-4.pth'))
# denoised_model = denoised_model.cuda()

# from model.UNet import UNet
# denoised_model1 = UNet()
# denoised_model1.load_state_dict(torch.load('./model_save/UNet/UNet_99_emb_intensity-4.pth'))
# denoised_model1 = denoised_model1.cuda()


# from model.UNet import UNet
# denoised_model1 = UNet()
# denoised_model1.load_state_dict(torch.load('./model_save/UNet/UNet_99_emb_intensity-4.pth'))
# denoised_model1 = denoised_model1.cuda()

# from model.DAM import Seq2Seq2
# denoised_model2 = Seq2Seq2()
# denoised_model2.load_state_dict(torch.load('./model_save/Seq2Seq2/Seq2Seq2_99_emb_intensity-4.pth'))
# denoised_model2 = denoised_model2.cuda()

from model.ralenet import ralenet
denoised_model3 = ralenet(high_level_enhence=True)
denoised_model3.load_state_dict(torch.load('./model_save/ralenet/ralenet_99_emb_intensity-4.pth'))
denoised_model3 = denoised_model3.cuda()

# from model.ACDAE import ACDAE
# denoised_model4 = ACDAE()
# denoised_model4.load_state_dict(torch.load('./model_save/ACDAE/ACDAE_99_emb_intensity-4.pth'))
# denoised_model4 = denoised_model4.cuda()

from model.transformer import ralenet
denoised_model5 = ralenet()
denoised_model5.load_state_dict(torch.load('./model_save/ralenet_nra/ralenet_nra_99_emb_intensity-4.pth'))
denoised_model5 = denoised_model5.cuda()

from model.ralenet import ralenet
denoised_model6 = ralenet(low_level_enhence=False)
denoised_model6.load_state_dict(torch.load('./model_save/ralenet_mlp/ralenet_mlp_99_emb_intensity-4.pth'))
denoised_model6 = denoised_model6.cuda()






# 选择可视化样本
sample_idx = 0

# 获取原始信号
clean_ecg = test_dataset[sample_idx][0].squeeze()

# 获取加噪信号 
noised_ecg = noised_test_dataset[sample_idx][0].squeeze()

# 存储各模型输出
denoised_results = {}

# 生成降噪信号
with torch.no_grad():
    input_tensor = torch.from_numpy(noised_ecg).unsqueeze(0).float().cuda()  # 添加batch维度
    # print(input_tensor)
    # 对每个降噪模型进行处理
    models = {

        'mpmtransformer': denoised_model3,
        # 'ralenet_nra': denoised_model5,
        # 'ralenet_mlp': denoised_model6,
    }
    
    for name, model in models.items():
        
        output = model(input_tensor)  
        denoised_results[name] = output.squeeze().cpu().numpy()
import matplotlib.pyplot as plt
# 创建对比图
plt.figure(figsize=(15, 12))
line_width = 5
font_size = 20
# 原始信号
plt.subplot(len(models)+2, 1, 1)
print(clean_ecg.shape)
plt.plot(clean_ecg[0], color='#1f77b4', label='Clean ECG', linewidth=line_width)
# plt.legend(loc='upper right')
plt.ylabel('Amplitude', fontsize=font_size)
plt.title('Original Clean Signal', fontsize=font_size)

# 加噪信号
plt.subplot(len(models)+2, 1, 2)
plt.plot(noised_ecg[0], color='#ff7f0e', label='Noised ECG', linewidth=line_width)
# plt.legend(loc='upper right')
plt.ylabel('Amplitude', fontsize=font_size)
plt.title('Noised Signal', fontsize=font_size)

# 各模型降噪结果
for idx, (name, ecg) in enumerate(denoised_results.items(), 3):
    plt.subplot(len(models)+2, 1, idx)
    plt.plot(ecg[0], color='#2ca02c', label=f'{name} Denoised', linewidth=line_width)
    # plt.legend(loc='upper right')
    plt.ylabel('Amplitude', fontsize=font_size)
    plt.title(f'Denoised Signal', fontsize=font_size)

plt.xlabel('Sample Points', fontsize=font_size)
plt.tight_layout()
plt.savefig('test.png')
plt.show()

exit()












# denoised_model = denoised_model5
with torch.no_grad():
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('test acc: ', acc(pred_array, label_array))
    print('test precision: ', precision(pred_array, label_array))
    print('test f1 score: ', f1_score(pred_array, label_array))
    
    
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('test acc: ', acc(pred_array, label_array))
    print('test precision: ', precision(pred_array, label_array))
    print('test f1 score: ', f1_score(pred_array, label_array))
    
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised test acc: ', acc(pred_array, label_array))
    print('noised test precision: ', precision(pred_array, label_array))
    print('noised test f1 score: ', f1_score(pred_array, label_array))
    
    exit()
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised test acc: ', acc(pred_array, label_array))
    print('noised test precision: ', precision(pred_array, label_array))
    print('noised test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model1(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised1 test acc: ', acc(pred_array, label_array))
    print('noised1 test precision: ', precision(pred_array, label_array))
    print('noised1 test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model2(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised2 test acc: ', acc(pred_array, label_array))
    print('noised2 test precision: ', precision(pred_array, label_array))
    print('noised2 test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model3(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised3 test acc: ', acc(pred_array, label_array))
    print('noised3 test precision: ', precision(pred_array, label_array))
    print('noised3 test f1 score: ', f1_score(pred_array, label_array))
    
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model4(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised4 test acc: ', acc(pred_array, label_array))
    print('noised4 test precision: ', precision(pred_array, label_array))
    print('noised4 test f1 score: ', f1_score(pred_array, label_array))    

    
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model5(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('ralenet_nra test acc: ', acc(pred_array, label_array))
    print('ralenet_nra test precision: ', precision(pred_array, label_array))
    print('ralenet_nra test f1 score: ', f1_score(pred_array, label_array))    

    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model6(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('ralenet_mlp test acc: ', acc(pred_array, label_array))
    print('ralenet_mlp test precision: ', precision(pred_array, label_array))
    print('ralenet_mlp test f1 score: ', f1_score(pred_array, label_array))    

    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.numpy()
        label = label.long().cuda()
        from local_utils.denoisefunc import wavelet_denoise
        data = wavelet_denoise(data)
        data = torch.FloatTensor(data).cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('dwt test acc: ', acc(pred_array, label_array))
    print('dwt test precision: ', precision(pred_array, label_array))
    print('dwt test f1 score: ', f1_score(pred_array, label_array))
    
    
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.numpy()
        label = label.long().cuda()
        from local_utils.denoisefunc import fft_denoise
        data = fft_denoise(data)
        data = torch.FloatTensor(data).cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('fft test acc: ', acc(pred_array, label_array))
    print('fft test precision: ', precision(pred_array, label_array))
    print('fft test f1 score: ', f1_score(pred_array, label_array))
    

