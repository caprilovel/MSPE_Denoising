import matplotlib.pyplot as plt

# 选择可视化样本
sample_idx = 0

# 获取原始信号
clean_ecg = test_dataset[sample_idx][0].squeeze().numpy()

# 获取加噪信号 
noised_ecg = noised_test_dataset[sample_idx][0].squeeze().numpy()

# 存储各模型输出
denoised_results = {}

# 生成降噪信号
with torch.no_grad():
    input_tensor = torch.from_numpy(noised_ecg).unsqueeze(0).float().cuda()  # 添加batch维度
    
    # 对每个降噪模型进行处理
    models = {
        'mpmtransformer': denoised_model,
        'ralenet': denoised_model3,
        'ralenet_nra': denoised_model5,
        'ralenet_mlp': denoised_model6,
    }
    
    for name, model in models.items():
        output = model(input_tensor.unsqueeze(1))  # 添加通道维度 (batch, channel, length)
        denoised_results[name] = output.squeeze().cpu().numpy()

# 创建对比图
plt.figure(figsize=(15, 12))

# 原始信号
plt.subplot(len(models)+2, 1, 1)
plt.plot(clean_ecg, color='#1f77b4', label='Clean ECG')
plt.legend(loc='upper right')
plt.ylabel('Amplitude')
plt.title('Original Clean Signal')

# 加噪信号
plt.subplot(len(models)+2, 1, 2)
plt.plot(noised_ecg, color='#ff7f0e', label='Noised ECG')
plt.legend(loc='upper right')
plt.ylabel('Amplitude')
plt.title('Noised Signal')

# 各模型降噪结果
for idx, (name, ecg) in enumerate(denoised_results.items(), 3):
    plt.subplot(len(models)+2, 1, idx)
    plt.plot(ecg, color='#2ca02c', label=f'{name} Denoised')
    plt.legend(loc='upper right')
    plt.ylabel('Amplitude')
    plt.title(f'{name} Denoised Signal')

plt.xlabel('Sample Points')
plt.tight_layout()
plt.show()