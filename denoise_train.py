import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import train_log
from datetime import datetime
from local_utils.evaluate import RMSE, SNR
from local_utils.loss import RateDiffLoss

def multiscale_loss(output, target):
    loss = 0
    for k in [1, 2, 4]:
        pool = nn.AvgPool1d(k)
        loss += F.mse_loss(pool(output), pool(target))
    return loss
@train_log()
def train(epochs, model, batch_size, train_loader, test_loader, use_gpu,model_path=None, writer=None, *args, **kwargs):
    model_name = kwargs["model_name"]
    for k,v in kwargs.items():
        print(k,":", v)
    if use_gpu:
        model = model.cuda()
    from tqdm import tqdm
    if model_path:
        model = model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_list = []
    eval_loss_list = []
    train_snr_list = []
    test_snr_list = []
    train_rmse_list = []
    test_rmse_list = []
    max_snr = -np.inf
    max_rmse = np.inf
    max_snr_epoch = 0
    max_rmse_epoch = 0
    
    
    train_nums = len(train_loader.dataset)
    test_nums = len(test_loader.dataset)
    
    init_time = datetime.now()
    for epoch in range(epochs):
        train_snr = []
        train_rmse = []
        test_snr = []
        test_rmse = [] 
        with tqdm(total=(train_nums-1)//batch_size + 1, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_init_time = datetime.now()
            
            model.train()
            epoch_loss_list = []
            
            for _, (data, target) in enumerate(train_loader):
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                if kwargs['model_index'] == 7:
                    pre, balance_loss = model(data)
                    loss = F.mse_loss(pre, target) 
                elif kwargs['model_index'] == 10:
                    pre= model(data)
                    loss = multiscale_loss(pre, target)
                else:
                    pre = model(data)
                    loss = F.mse_loss(pre, target)
                loss += RateDiffLoss(pre, target) * 0.01
                epoch_loss_list.append(loss.item())
                
                loss.backward()
                optimizer.step()
                if kwargs['model_index'] == 6:
                    optimizer.zero_grad()
                    target_feature = model(target)
                    pre = model(data)
                    c_loss = F.mse_loss(target_feature, pre)
                    c_loss.backward()
                    optimizer.step()
                
                target = target.cpu()
                pre = pre.cpu()
                train_snr.append(SNR(target, pre))
                train_rmse.append(RMSE(target, pre))
                total_time = datetime.now() - init_time
                epoch_time = datetime.now() - epoch_init_time
                t.update(1)
                t.set_description_str(f"\33[36m【Train Epoch {epoch + 1:03d}】") 
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 
                torch.cuda.empty_cache()
                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        with tqdm(total=(test_nums-1)//batch_size + 1, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            with torch.no_grad():
                model.eval()
                # change model to cpu
                
                epoch_eval_loss_list = []
                for _, (data, target) in enumerate(test_loader):
                    if use_gpu:
                        data, target = data.cuda(), target.cuda()
                    if kwargs['model_index'] == 7:
                        pre, balance_loss = model(data)
                        loss = F.mse_loss(pre, target) + balance_loss
                    else:
                        pre = model(data)
                        loss = F.mse_loss(pre, target)
                    epoch_eval_loss_list.append(loss.item())
                    total_time = datetime.now() - init_time
                    epoch_time = datetime.now() - epoch_init_time
                    target = target.cpu()
                    pre = pre.cpu()
                    test_snr.append(SNR(target, pre))
                    # print("test_snr:", SNR(target, pre))
                    test_rmse.append(RMSE(target, pre))
                    t.update(1)
                    t.set_description_str(f"\33[36m【Eval Epoch {epoch + 1:03d}】")
                    t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 
                    
        
        
        
        
        
        train_snr = torch.cat(train_snr, dim=0)
        train_rmse = torch.cat(train_rmse, dim=0)
        test_snr = torch.cat(test_snr, dim=0)
        test_rmse = torch.cat(test_rmse, dim=0)
        
        writer.add_scalar(f"{model_name}/train_snr", train_snr.mean().item(), epoch)
        writer.add_scalar(f"{model_name}/test_snr", test_snr.mean().item(), epoch)
        writer.add_scalar(f"{model_name}/train_rmse", train_rmse.mean().item(), epoch)
        writer.add_scalar(f"{model_name}/test_rmse", test_rmse.mean().item(), epoch)
        
        train_snr_list.append(train_snr.mean().item())
        test_snr_list.append(test_snr.mean().item())
        train_rmse_list.append(train_rmse.mean().item())
        test_rmse_list.append(test_rmse.mean().item())
        if test_snr.mean().item() > max_snr:
            max_snr = test_snr.mean().item()
            max_snr_epoch = epoch
        if test_rmse.mean().item() < max_rmse:
            max_rmse = test_rmse.mean().item()
            max_rmse_epoch = epoch
        
        if (epoch + 1) % 10 == 0:
            from global_utils.torch_utils.log_utils import mkdir
            mkdir(f"./model_save/{model_name}")
            torch.save(model.state_dict(), f"model_save/{model_name}/{model_name}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
            print(f"model_save/{model_name}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
            print("epoch:", epoch + 1)
            print("train snr:", train_snr.mean().item())
            print("test snr:", test_snr.mean().item())
            print("train rmse:", train_rmse.mean().item())
            print("test rmse:", test_rmse.mean().item())
    with open("./output.txt", "a") as output_file:
        output_file.write(f"{kwargs['model_name']}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}:snr:{test_snr.mean().item()}, rmse:{test_rmse.mean().item()}\n") 
        output_file.write(f"max_snr:{max_snr}, max_snr_epoch:{max_snr_epoch}, max_rmse:{max_rmse}, max_rmse_epoch:{max_rmse_epoch}\n")
        
    return train_snr_list, test_snr_list, train_rmse_list, test_rmse_list



@train_log()
def train_gan_bak(epochs, model, batch_size, train_loader, test_loader, use_gpu, writer=None, *args, **kwargs):
    """GAN训练函数"""
    model_name = kwargs["model_name"]
    if use_gpu:
        model = model.cuda()
    
    # 初始化优化器
    g_optimizer = optim.Adam(model.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # 使用带标签的损失函数
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.MSELoss()
    
    # 记录指标
    train_hist = {'D_loss': [], 'G_loss': [], 'Recon_loss': []}
    best_snr = -np.inf

    for epoch in range(epochs):
        # 训练阶段
        model.G.train()
        model.D.train()
        
        for batch_idx, (clean, noisy) in enumerate(train_loader):
            if use_gpu:
                clean, noisy = clean.cuda(), noisy.cuda()
            
            real_labels = torch.ones(clean.size(0), 1).cuda()
            fake_labels = torch.zeros(clean.size(0), 1).cuda()
            
            # ---------------------
            #  训练判别器
            # ---------------------
            d_optimizer.zero_grad()
            
            # 真实样本损失
            real_output = model.D(clean.unsqueeze(1), noisy.unsqueeze(1))  # 正确传入两个参数

            # 生成样本损失（将生成的信号和条件一起传入）
            fake_output = model.D(generated.detach(), noisy.unsqueeze(1))
            d_real_loss = adversarial_loss(real_output, real_labels)
            
            # 生成样本损失
            z = torch.randn(clean.size(0), 62).cuda()
            generated = model.G(z, noisy.unsqueeze(1))
            fake_output = model.D(generated.detach(), noisy.unsqueeze(1))
            d_fake_loss = adversarial_loss(fake_output, fake_labels)
            
            d_total_loss = (d_real_loss + d_fake_loss) * 0.5
            d_total_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            #  训练生成器
            # ---------------------
            g_optimizer.zero_grad()
            
            # 对抗损失
            validity = model.D(generated, noisy.unsqueeze(1))
            g_adv_loss = adversarial_loss(validity, real_labels)
            
            # 重建损失
            g_recon_loss = reconstruction_loss(generated, clean.unsqueeze(1))
            
            g_total_loss = g_adv_loss + 100 * g_recon_loss  # λ=100
            g_total_loss.backward()
            g_optimizer.step()
            
            # 记录损失
            train_hist['D_loss'].append(d_total_loss.item())
            train_hist['G_loss'].append(g_adv_loss.item())
            train_hist['Recon_loss'].append(g_recon_loss.item())
        
        # 评估阶段
        model.G.eval()
        test_snr = []
        test_rmse = []
        with torch.no_grad():
            for clean, noisy in test_loader:
                if use_gpu:
                    clean, noisy = clean.cuda(), noisy.cuda()
                
                z = torch.randn(clean.size(0), 62).cuda()
                generated = model.G(z, noisy.unsqueeze(1))
                
                # 计算SNR
                test_snr.append(10 * torch.log10(
                    torch.mean(clean**2) / 
                    torch.mean((clean - generated.squeeze())**2)
                ).item())
                test_rmse.append(torch.sqrt(torch.mean((clean - generated.squeeze())**2)).item())
        
        avg_test_snr = np.mean(test_snr)
        avg_test_rmse = np.mean(test_rmse)
        if avg_test_snr > best_snr:
            best_snr = avg_test_snr
            torch.save(model.state_dict(), 
                      f"model_save/{model_name}_best_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
        
        # 记录到TensorBoard
        writer.add_scalars(f'{model_name}/Loss', {
            'D': np.mean(train_hist['D_loss'][-len(train_loader):]),
            'G': np.mean(train_hist['G_loss'][-len(train_loader):]),
            'Recon': np.mean(train_hist['Recon_loss'][-len(train_loader):])
        }, epoch)
        writer.add_scalar(f'{model_name}/Test_SNR', avg_test_snr, epoch)
        
        # 每10个epoch保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      f"model_save/{model_name}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
            
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {np.mean(train_hist['D_loss'][-len(train_loader):]):.4f} "
              f"| G Loss: {np.mean(train_hist['G_loss'][-len(train_loader):]):.4f} | SNR: {avg_test_snr:.2f} dB"
              f"| RMSE: {avg_test_rmse:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 
              f"model_save/{model_name}_final_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
    return train_hist

def train_gan(epochs, model, batch_size, train_loader, test_loader, use_gpu, writer=None, *args, **kwargs):
    model_name = kwargs["model_name"]
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)
    
    # 初始化优化器
    g_optimizer = optim.Adam(model.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # 损失函数
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()  # 使用L1损失保留细节
    
    # 训练记录
    train_hist = {'D_loss': [], 'G_loss': [], 'Recon_loss': []}
    best_snr = -np.inf

    for epoch in range(epochs):
        model.G.train()
        model.D.train()
        
        for clean, noisy in train_loader:
            # 数据维度处理
            B, C, L = clean.shape
            clean = clean.to(device)       # (B, C, L)
            noisy = noisy.to(device)       # (B, C, L)
            
            # 生成噪声
            z = torch.randn(B, model.latent_dim).to(device)  # (B, latent_dim)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            d_optimizer.zero_grad()
            
            # 真实样本损失
            real_pred = model.D(clean, noisy)  # (B, 1)
            real_loss = adversarial_loss(real_pred, torch.ones(B, 1).to(device))
            
            # 生成样本损失
            fake_signals = model.G(z, noisy).detach()  # (B, C, L)
            fake_pred = model.D(fake_signals, noisy)   # (B, 1)
            fake_loss = adversarial_loss(fake_pred, torch.zeros(B, 1).to(device))
            
            d_total_loss = (real_loss + fake_loss) * 0.5
            d_total_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            #  训练生成器
            # ---------------------
            g_optimizer.zero_grad()
            
            # 生成样本
            gen_signals = model.G(z, noisy)
            
            # 对抗损失
            validity = model.D(gen_signals, noisy)
            g_adv_loss = adversarial_loss(validity, torch.ones(B, 1).to(device))
            
            # 重建损失
            g_recon_loss = reconstruction_loss(gen_signals, clean)
            
            g_total_loss = g_adv_loss + 100 * g_recon_loss  # λ=100
            g_total_loss.backward()
            g_optimizer.step()
            
            # 记录损失
            train_hist['D_loss'].append(d_total_loss.item())
            train_hist['G_loss'].append(g_adv_loss.item())
            train_hist['Recon_loss'].append(g_recon_loss.item())
        
        # 评估阶段
        model.G.eval()
        test_snr = []
        test_rmse = []
        with torch.no_grad():
            for clean, noisy in test_loader:
                clean = clean.to(device)
                noisy = noisy.to(device)
                z = torch.randn(clean.size(0), model.latent_dim).to(device)
                
                generated = model.G(z, noisy)
                mse = F.mse_loss(generated, clean)
                snr = 10 * torch.log10(torch.mean(clean**2) / mse)
                test_snr.append(snr.item())
                rmse = torch.sqrt(mse)
                test_rmse.append(rmse.item())
        
        avg_test_snr = np.mean(test_snr)
        avg_test_rmse = np.mean(test_rmse)
        if avg_test_snr > best_snr:
            best_snr = avg_test_snr
            torch.save(model.state_dict(), 
                      f"model_save/{model_name}_best_snr.pth")
        
        # 记录到TensorBoard
        writer.add_scalars(f'{model_name}/Loss', {
            'D': np.mean(train_hist['D_loss'][-len(train_loader):]),
            'G': np.mean(train_hist['G_loss'][-len(train_loader):]),
            'Recon': np.mean(train_hist['Recon_loss'][-len(train_loader):])
        }, epoch)
        writer.add_scalar(f'{model_name}/Test_SNR', avg_test_snr, epoch)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"D Loss: {np.mean(train_hist['D_loss'][-len(train_loader):]):.4f} | "
              f"G Loss: {np.mean(train_hist['G_loss'][-len(train_loader):]):.4f} | "
              f"SNR: {avg_test_snr:.2f} dB | "
              f"RMSE: {avg_test_rmse:.4f}")
    
    torch.save(model.state_dict(), f"model_save/{model_name}_final.pth")
    return train_hist