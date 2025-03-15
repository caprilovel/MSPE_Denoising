import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AddGatedNoise(nn.Module):
    """带门控的乘性噪声层，仅在训练时添加噪声"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            # 生成[-1, 1)均匀分布的噪声
            noise = torch.rand_like(x) * 2 - 1  
            return x * (1 + noise)
        return x

class PositionalEncoding1D(nn.Module):
    """位置编码层，适配多通道输入"""
    def __init__(self, channels):
        super().__init__()
        self.channels = int(np.ceil(channels / 2)) * 2  # 确保channels为偶数
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        # 输入形状: (batch_size, channels, seq_len)
        batch_size, _, seq_len = x.size()
        
        # 生成位置编码
        pos_x = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        
        # 调整维度匹配 [batch, seq_len, features]
        emb = emb_x[None, :, :self.channels].repeat(batch_size, 1, 1)  
        return emb.permute(0, 2, 1)  # 输出形状: (batch, features, seq_len)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码层，适配通道优先格式"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Conv1d(embed_dim, ff_dim, 1),
            nn.ReLU(),
            nn.Conv1d(ff_dim, embed_dim, 1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # 输入形状: (seq_len, batch, embed_dim)
        
        # 自注意力
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = src.permute(1, 2, 0)  # (batch, embed_dim, seq_len)
        src2 = self.ffn(src2).permute(2, 0, 1)  # 恢复形状
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerDAE(nn.Module):
    """基于Transformer的去噪自动编码器，支持多通道输入"""
    def __init__(self, 
                 in_channels=15,       # 输入通道数
                 signal_size=512,      # 输入信号长度
                 head_size=64,         # 注意力头维度
                 num_heads=8,          # 注意力头数量
                 ff_dim=64,            # 前馈网络隐藏层维度
                 num_transformer_blocks=6,  # Transformer块数量
                 dropout=0.1,          # Dropout率
                 ks=13):               # 卷积核大小
        super().__init__()
        
        # ----------------- 编码器部分 -----------------
        # 第一卷积层 (输入通道: in_channels -> 16)
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, ks, stride=2, padding=ks//2),
            AddGatedNoise(),
            nn.Sigmoid()
        )
        self.enc1_res = nn.Conv1d(in_channels, 16, ks, stride=2, padding=ks//2)
        
        # 第二卷积层 (16 -> 32)
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, ks, stride=2, padding=ks//2),
            AddGatedNoise(),
            nn.Sigmoid()
        )
        self.enc2_res = nn.Conv1d(16, 32, ks, stride=2, padding=ks//2)
        
        # 第三卷积层 (32 -> 64)
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, ks, stride=2, padding=ks//2),
            AddGatedNoise(),
            nn.Sigmoid()
        )
        self.enc3_res = nn.Conv1d(32, 64, ks, stride=2, padding=ks//2)
        
        # ----------------- Transformer部分 -----------------
        self.pos_encoder = PositionalEncoding1D(64)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=64,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])
        
        # ----------------- 解码器部分 -----------------
        # 第一转置卷积层 (64 -> 64)
        self.dec1 = nn.ConvTranspose1d(64, 64, ks, 
                                     stride=1, 
                                     padding=ks//2,
                                     output_padding=0)
        
        # 第二转置卷积层 (64 -> 32)
        self.dec2 = nn.ConvTranspose1d(64, 32, ks, 
                                     stride=2, 
                                     padding=ks//2,
                                     output_padding=1)  # 补偿下采样的尺寸损失
        
        # 第三转置卷积层 (32 -> 16)
        self.dec3 = nn.ConvTranspose1d(32, 16, ks, 
                                     stride=2, 
                                     padding=ks//2,
                                     output_padding=1)
        
        # 最终输出层 (16 -> in_channels)
        self.final = nn.ConvTranspose1d(16, in_channels, ks, 
                                      stride=2, 
                                      padding=ks//2,
                                      output_padding=1)
        
        # ----------------- 公共层 -----------------
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入形状: (batch, in_channels, seq_len)
        
        # ----------------- 编码过程 -----------------
        # 第一层编码
        x0 = self.enc1(x) * self.enc1_res(x)  # 形状: (B,16,256)
        x0 = F.batch_norm(x0, running_mean=None, running_var=None, training=True)  # 简化BN
        
        # 第二层编码
        x1 = self.enc2(x0) * self.enc2_res(x0)  # (B,32,128)
        x1 = F.batch_norm(x1, running_mean=None, running_var=None, training=True)
        
        # 第三层编码
        x2 = self.enc3(x1) * self.enc3_res(x1)  # (B,64,64)
        x2 = F.batch_norm(x2, running_mean=None, running_var=None, training=True)
        
        # ----------------- Transformer处理 -----------------
        # 添加位置编码
        pos_emb = self.pos_encoder(x2)  # (B,64,64)
        x3 = x2 + pos_emb
        
        # 调整维度顺序供Transformer使用 (seq_len, batch, features)
        x3 = x3.permute(2, 0, 1)  # (64, B, 64)
        
        # 通过多个Transformer块
        for block in self.transformer_blocks:
            x3 = block(x3)  # 保持形状 (64, B, 64)
        
        # 恢复原始维度顺序 (B, 64, 64)
        x3 = x3.permute(1, 2, 0)
        
        # ----------------- 解码过程 -----------------
        # 第一层解码
        x4 = self.dec1(x3) + x2  # (B,64,64)
        x4 = F.batch_norm(x4, running_mean=None, running_var=None, training=True)
        
        # 第二层解码
        x5 = self.dec2(x4) + x1  # (B,32,128)
        x5 = F.batch_norm(x5, running_mean=None, running_var=None, training=True)
        
        # 第三层解码
        x6 = self.dec3(x5) + x0  # (B,16,256)
        x6 = F.batch_norm(x6, running_mean=None, running_var=None, training=True)
        
        # 最终输出
        out = self.final(x6)  # (B,in_channels,512)
        return out