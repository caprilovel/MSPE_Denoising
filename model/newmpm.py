import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 核心改进组件 ----------------------------------------------------------------

class AddGatedNoise(nn.Module):
    """门控噪声注入层（保持与原始TransformerDAE兼容）"""
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            gate = torch.sigmoid(x.mean(dim=1, keepdim=True))  # 通道感知门控
            return x + noise * gate
        return x

# class ChannelAwarePositionEncoding(nn.Module):
#     """改进的通道感知位置编码"""
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         self.channel_embed = nn.Parameter(torch.randn(1, d_model, 1))
#         position = torch.arange(max_len).unsqueeze(0).unsqueeze(0)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#         pe = torch.zeros(1, d_model, max_len)
#         pe[0, 0::2, :] = torch.sin(position * div_term)
#         pe[0, 1::2, :] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         # x shape: [B, C, T]
#         return x + self.pe[:, :, :x.size(2)] + self.channel_embed

class HybridConvBlock(nn.Module):
    """混合卷积特征提取块（包含门控噪声和残差）"""
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, stride, kernel_size//2),
            AddGatedNoise(),
            nn.BatchNorm1d(out_c),
            nn.GELU()
        )
        self.res = nn.Conv1d(in_c, out_c, 1, stride) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        return self.main(x) * 0.7 + self.res(x) * 0.3  # 加权残差连接

class EnhancedTransformerBlock(nn.Module):
    """增强型Transformer块（带门控注意力）"""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # 形状转换 [B,C,T] -> [T,B,C]
        residual = x.permute(2,0,1)
        x_norm = self.norm1(residual)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        gate = self.gate(residual)
        attn_out = attn_out * gate
        residual = residual + F.dropout(attn_out, p=0.1)
        
        # MLP部分
        mlp_in = self.norm2(residual)
        mlp_out = self.mlp(mlp_in)
        return (residual + mlp_out).permute(1,2,0)

# 完整模型实现 ----------------------------------------------------------------

# class EnhancedDAE(nn.Module):
#     """增强版去噪自动编码器"""
#     def __init__(self, 
#                 in_channels=15,
#                 base_dim=64,
#                 num_heads=8,
#                 num_blocks=6,
#                 kernel_size=13):
#         super().__init__()
        
#         # 编码器
#         self.encoder = nn.Sequential(
#             HybridConvBlock(in_channels, base_dim, kernel_size, 2),
#             HybridConvBlock(base_dim, base_dim*2, kernel_size, 2),
#             HybridConvBlock(base_dim*2, base_dim*4, kernel_size, 2)
#         )
        
#         # Transformer处理层
#         self.pos_encoder = ChannelAwarePositionEncoding(base_dim*4)
#         self.transformer_blocks = nn.ModuleList([
#             EnhancedTransformerBlock(base_dim*4, num_heads)
#             for _ in range(num_blocks)
#         ])
        
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(base_dim*4, base_dim*2, kernel_size, 2, kernel_size//2, output_padding=1),
#             nn.BatchNorm1d(base_dim*2),
#             nn.GELU(),
            
#             nn.ConvTranspose1d(base_dim*2, base_dim, kernel_size, 2, kernel_size//2, output_padding=1),
#             nn.BatchNorm1d(base_dim),
#             nn.GELU(),
            
#             nn.ConvTranspose1d(base_dim, in_channels, kernel_size, 2, kernel_size//2, output_padding=1),
#             nn.Tanh()
#         )
        
#         # 初始化
#         self._init_weights()
        
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
                
#     def forward(self, x):
#         # 编码
#         enc_out = self.encoder(x)
        
#         # 加入位置信息
#         enc_out = self.pos_encoder(enc_out)
        
#         # Transformer处理
#         trans_out = enc_out
#         for block in self.transformer_blocks:
#             trans_out = block(trans_out)
            
#         # 残差连接
#         trans_out = trans_out + enc_out
        
#         # 解码
#         return self.decoder(trans_out)

class ChannelAwarePositionEncoding(nn.Module):
    """改进的通道感知位置编码（修正版）"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 通道嵌入
        self.channel_embed = nn.Parameter(torch.randn(1, d_model, 1))
        
        # 位置编码
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [B, C, T]
        pos_emb = self.pe[:, :x.size(2), :]  # [1, T, C]
        pos_emb = pos_emb.permute(0, 2, 1)  # [1, C, T]
        return x + pos_emb.to(x.device) + self.channel_embed.to(x.device)

class EnhancedDAE(nn.Module):
    """修正后的增强版模型"""
    def __init__(self, 
                in_channels=15,
                base_dim=64,
                num_heads=8,
                num_blocks=6,
                kernel_size=13):
        super().__init__()
        
        # 编码器保持不变
        self.encoder = nn.Sequential(
            HybridConvBlock(in_channels, base_dim, kernel_size, 2),
            HybridConvBlock(base_dim, base_dim*2, kernel_size, 2),
            HybridConvBlock(base_dim*2, base_dim*4, kernel_size, 2)
        )
        
        # 使用修正后的位置编码
        self.pos_encoder = ChannelAwarePositionEncoding(d_model=base_dim*4)
        
        # 其他组件保持不变
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(base_dim*4, num_heads)
            for _ in range(num_blocks)
        ])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_dim*4, base_dim*2, kernel_size, 2, kernel_size//2, output_padding=1),
            nn.BatchNorm1d(base_dim*2),
            nn.GELU(),
            
            nn.ConvTranspose1d(base_dim*2, base_dim, kernel_size, 2, kernel_size//2, output_padding=1),
            nn.BatchNorm1d(base_dim),
            nn.GELU(),
            
            nn.ConvTranspose1d(base_dim, in_channels, kernel_size, 2, kernel_size//2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        enc_out = self.pos_encoder(enc_out)
        
        trans_out = enc_out
        for block in self.transformer_blocks:
            trans_out = block(trans_out)
            
        trans_out = trans_out + enc_out
        return self.decoder(trans_out)


