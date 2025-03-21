from typing import Iterator, Optional, Callable, List, Any

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torchvision.ops
from einops import rearrange
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx
from torch.nn.init import trunc_normal_
import math 


    
    

    
#%%

class PartialConv_1d(nn.Module):
    """Partial Convolution for 1d data
    
    from cvpr 2023 paper 'Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks' 

    """
    def __init__(self, dim, n_div, forward) -> None:
        """Partial Convolution for 1d data
    
        from cvpr 2023 paper 'Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks'
        
        Args:
            dim (int): the input channels of 1d data
            n_div (int): the number of division of the input channels
            forward (str, optional): slicing | split_cat. Defaults to "slicing".

        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv1d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        
        if forward == "slicing":
            self.forward = self.forward_slicing
            
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
        
    def forward_slicing(self, x):
        # only for inference 
        x = x.clone()
        x[:, :self.dim_conv3, :] = self.partial_conv3(x[:, :self.dim_conv3, :])
        
        return x 
        
        
    def forward_split_cat(self, x):
        # for training / inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    
    reading the code, it seems that the drop path is applied to drop each instance in a batch but not the whole batch.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class eca_layer_1d(nn.Module):
    def __init__(self, channels, k_size=3) -> None:
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channels = channels
        self.k_size = k_size
        
    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)


        

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., local_enhance=False, use_partial=True, use_eca=False):
        """Feed Forward Layer 

        Args:
            in_features (int): input dims 
            hidden_features (int, optional): hidden dims in the Linear Layer of the feed forward layer. if None, hidden dim is same as in_features. Defaults to None.
            out_features (int, optional): output dims. Defaults to None.
            act_layer (_type_, optional): activate function layer. Defaults to nn.GELU.
            drop (float, optional): dropout rate. Defaults to 0..
            local_enhance (bool, optional): whether to use local enhance. Defaults to False.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.local_enhance = local_enhance
        self.use_partial = use_partial
        self.eca = eca_layer_1d(out_features) if use_eca else nn.Identity()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if local_enhance:
            if use_partial:
                self.leconv = PartialConv_1d(hidden_features, hidden_features, forward="split_cat")
            else:
                self.leconv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=False)
                

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.local_enhance:
            x = rearrange(x, 'b c l -> b l c')
            x = self.leconv(x)
            x = self.act(x)
            x = rearrange(x, 'b l c -> b c l')
        x = self.fc2(x)
        x = self.eca(x)
        x = self.drop(x)
        return x




class AbsPositionalEncoding(nn.Module):
    """Position Encoding for Transformer"""
    def __init__(self, num_hiddens, dropout=0., max_len=1000):
        super(AbsPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class LinearProjection(nn.Module):
    """Linear Projection for generate Q K V vectors

    Args:
        dim (int):  the dim of the input 
        heads (int, optional): the head would be divided. Defaults to 8.
        dim_head (int, optional): each head dim. Defaults to 64.
        dropout (float, optional): dropout rate. Defaults to 0..
        bias (bool, optional): whether to use bias. Defaults to True.
        
    Input Tensor size should be as Batch x Length x Dim
    And output tensor size would be a tuple of (q, k, v):
        q: Batch x heads x length x head_dim
        k: Batch x heads x length_kv x head_dim
        v: Batch x heads x length_kv x head_dim
    
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias = True) -> None:
        """ Linear Projection for generate Q K V vectors

        Args:
            dim (int):  the dim of the input 
            heads (int, optional): the head would be divided. Defaults to 8.
            dim_head (int, optional): each head dim. Defaults to 64.
            dropout (float, optional): dropout rate. Defaults to 0..
            bias (bool, optional): whether to use bias. Defaults to True.
            
        Input Tensor size should be as Batch x Length x Dim
        And output tensor size would be a tuple of (q, k, v):
            q: Batch x heads x length x head_dim
            k: Batch x heads x length_kv x head_dim
            v: Batch x heads x length_kv x head_dim
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads 
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_kv=None):
        '''
        Args:
          x: Batch x Length x Dim
        
        Output:
          q: Batch x heads x length x head_dim
          k: Batch x heads x length_kv x head_dim
          v: Batch x heads x length_kv x head_dim
        
        '''
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k,v = kv[0], kv[1]
        return q, k, v
    
    
class MSAttention(nn.Module):
    """Multi-head Self Attention

    Args:
        dim (int): input dims
        num_heads (int): number of heads
        token_projection (str, optional): token projection method. Defaults to 'linear'.
        qkv_bias (bool, optional): whether to use qkv bias. Defaults to True.
        qk_scale (float, optional): qkv scale. Defaults to None.
        attn_drop (float, optional): attention dropout rate. Defaults to 0..
        proj_drop (float, optional): projection dropout rate. Defaults to 0..
    """
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ) -> None:
        """Multi-head Self Attention

        Args:
            dim (int): input dims
            num_heads (int): number of heads
            token_projection (str, optional): token projection method. Defaults to 'linear'.
            qkv_bias (bool, optional): whether to use qkv bias. Defaults to True.
            qk_scale (float, optional): qkv scale. Defaults to None.
            attn_drop (float, optional): attention dropout rate. Defaults to 0..
            proj_drop (float, optional): projection dropout rate. Defaults to 0..
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv_proj = LinearProjection(dim, num_heads, head_dim, dropout=0, bias=qkv_bias)

        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):
        """
        Args:
            x (tensor): input features with shape of (B, N, C)
            attn_kv (_type_, optional): _description_. Defaults to None.
            mask : (0/inf) mask with shape of (inner_dim, inner_dim) or None. Defaults to None.

        """
        B_, N, C = x.shape
        q, k, v = self.qkv_proj(x, attn_kv)
        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))
        
        long_attn_mask = torch.full((N, N), -10000.0, dtype=torch.float32, device=x.device)
        
        long_attn_mask = torch.triu(long_attn_mask, diagonal=5)
        # long_attn_mask = torch.triu(long_attn_mask, diagonal=5) + torch.tril(long_attn_mask, diagonal=-5)
        
        if mask is not None:
            if len(mask.shape) != len(attn.shape):
                mask = mask.unqueeze(0)
            
            attn = attn + mask 
            attn = self.softmax(attn)
        else:
            # print(attn.shape)
            # attn += long_attn_mask
            attn = self.softmax(attn)
        # 
        attn =self.attn_drop(attn)
        
        # v: batch x heads x length x head_dim
        # attn: batch x heads x length x length
        x = attn @ v
        # x: batch x heads x length x head_dim
        x = rearrange(x, 'b h l hd -> b l (h hd)')
        # x: batch x length x channels 
        x = self.proj(x)
        x = self.proj_drop(x)
        # x: batch * length * dims 
        return x 


class BlockSegmentMHSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv_proj = LinearProjection(dim, num_heads, head_dim, dropout=0, bias=qkv_bias)

        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, attn_kv=None, mask=None):
        """
        Args:
            x (tensor): input features with shape of (B, N, C)
            attn_kv (_type_, optional): _description_. Defaults to None.
            mask : (0/inf) mask with shape of (inner_dim, inner_dim) or None. Defaults to None.

        """
        B_, N, C = x.shape
        q, k, v = self.qkv_proj(x, attn_kv)
        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            if len(mask.shape) != len(attn.shape):
                mask = mask.unsqueeze(0)
            
            attn = attn + mask 
            attn = self.softmax(attn)
        else:
            # print(attn.shape)
            attn = self.softmax(attn)
    
        attn =self.attn_drop(attn)
        
        # v: batch x heads x length x head_dim
        # attn: batch x heads x length x length
        x = attn @ v
        # x: batch x heads x length x head_dim
        x = rearrange(x, 'b h l hd -> b l (h hd)')
        # x: batch x length x channels 
        x = self.proj(x)
        x = self.proj_drop(x)
        # x: batch * length * dims 
        return x
        
        


class TransformerBlock(nn.Module):
    """Transformer Block

    Args:
        dim (int): _description_
        num_heads (_type_): _description_
        mlp_ratio (_type_, optional): _description_. Defaults to 4..
        qkv_bias (bool, optional): _description_. Defaults to True.
        qk_scale (_type_, optional): _description_. Defaults to None.
        drop (_type_, optional): _description_. Defaults to 0..
        attn_drop (_type_, optional): _description_. Defaults to 0..
        drop_path (_type_, optional): the rate of the drop path. Defaults to 0..
        act_layer (_type_, optional): activate function layer. Defaults to nn.GELU.
        norm_layer (_type_, optional): norm function layer. Defaults to nn.LayerNorm.
        local_enhance (bool, optional): whether to use local enhance. Defaults to False.
        abs_emd (bool, optional): whether to use abs embedding. Defaults to True.
        use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False.
    """
    def __init__(self, dim, num_heads,  
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_enhance=False, use_partial=True, use_eca=False, pe='abs', use_checkpoint=False, *args, **kwargs) -> None:
        """Transformer Block

        Args:
            dim (int): _description_
            num_heads (_type_): _description_
            mlp_ratio (_type_, optional): _description_. Defaults to 4..
            qkv_bias (bool, optional): _description_. Defaults to True.
            qk_scale (_type_, optional): _description_. Defaults to None.
            drop (_type_, optional): _description_. Defaults to 0..
            attn_drop (_type_, optional): _description_. Defaults to 0..
            drop_path (_type_, optional): the rate of the drop path. Defaults to 0..
            act_layer (_type_, optional): activate function layer. Defaults to nn.GELU.
            norm_layer (_type_, optional): norm function layer. Defaults to nn.LayerNorm.
            local_enhance (bool, optional): whether to use local enhance. Defaults to False.
            abs_emd (bool, optional): whether to use abs embedding. Defaults to True.
            use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads 
        self.mlp_ratio = mlp_ratio
        self.pe = pe 
        self.use_checkpoint = use_checkpoint
        
        self.attn = MSAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, )
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, local_enhance=local_enhance, use_partial=use_partial, use_eca=use_eca)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if pe == 'abs':
            self.abs_pos_enc = AbsPositionalEncoding(dim)
        elif pe == 'rel':
            from global_utils.torch_utils.Embed import RelativePositionEmbedding
            
    def forward_part1(self, x, mask):
        B, L, C = x.shape
        if self.pe == 'abs':
            x = self.abs_pos_enc(x * math.sqrt(self.dim)) 
        x = self.norm1(x)
        msa_in = x
        attn = self.attn(msa_in, mask=mask)
        return attn
    
    def forward_part2(self, x):
        ffn_in = self.norm2(x)
        ffn_out = self.mlp(ffn_in)
        return self.drop_path(ffn_out)        
    
    
    def forward(self, x, mask=None):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask)
        else:
            x = self.forward_part1(x, mask)
            
        x = shortcut + self.drop_path(x)
        
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x
class PatchSeparate(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)
    def forward(self, x):
        B, L, C = x.shape 
        x = rearrange(x, 'b l (c1 c2) -> b (c1 l) c2', c1=2)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x 
        
class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape

        # padding
        pad_input = (L % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, L % 2))

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads,  
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_enhance=False, abs_emd=True, downsample=None, upsample=None, use_checkpoint=False) -> None:
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, 
                num_heads=num_heads,  
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop, 
                attn_drop=attn_drop, 
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path, 
                act_layer=act_layer,
                norm_layer=norm_layer, 
                local_enhance=local_enhance, 
                abs_emd=abs_emd, 
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            
    def forward(self, x, mask=None):
        B, L, C = x.shape
        # x = rearrange(x, 'b c l -> b l c')
        # add mask here
        if mask is None:
            for blk in self.blocks:
                x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x, mask)
            
        if self.downsample is not None:
            x = self.downsample(x)
        # x = rearrange(x, 'b l c -> b c l')
        return x 
        
class RelativePositionEmbedding(nn.Module):
    def __init__(self, Length, whole_length, num_heads) -> None:
        super().__init__()
        self.Length = Length
        self.whole_length = whole_length
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Length - 1), num_heads))
        
        coords_l = torch.arange(Length)
        coords = torch.stack(torch.meshgrid([coords_l], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wl, Wl, 2
        relative_coords[:, :, 0] += Length - \
            1  # shift to start from 0
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer("relative_position_index",
                             relative_position_index)
    def parameters_normalize(self, ):
        self.relative_position_bias_table.data = torch.randn_like(self.relative_position_bias_table.data) * 0.02
    
        
    def forward(self, R_pos=None):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.Length, self.Length, -1)  # Wl, Wl, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wl, Wl
        attn_length = self.whole_length
        if R_pos is None:
            relative_position_bias = mask_fill(relative_position_bias, (attn_length-self.Length)//2, attn_length)
        else:
            relative_position_bias = mask_fill(relative_position_bias, R_pos-self.Length//2, attn_length)
    
        return relative_position_bias.unsqueeze(0)    

def mask_fill(mask, init_len, length):
    """_summary_

    Args:
        mask (_type_): _description_
        length (_type_): _description_
    """
    num_head, window_size, _ = mask.shape
    assert mask.shape[1] == mask.shape[2]
    pad_total = length - window_size 
    
    return F.pad(mask, (init_len, pad_total-init_len, init_len, pad_total-init_len), value=0)

class MultiPatchInEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels_per, patch_sizes=[2, 4, 8, 16, 32]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.out_channels_per = out_channels_per
        self.in_channels = in_channels
        self.projs = nn.ModuleList([nn.Conv1d(in_channels, out_channels_per, kernel_size=patch_size, stride=patch_size) for patch_size in patch_sizes])
        # padding 16 zeros for 1d
        self.padding = nn.ZeroPad1d(20)
    def forward(self, x):
        B, C, L = x.shape
        patches = [proj(x) for proj in self.projs]
        return self.padding(torch.cat(patches, 2))


    
class MultiPatchOutEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels_per, patch_sizes=[1, 2, 4, 8, 16]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.out_channels_per = out_channels_per
        self.in_channels = in_channels
        self.projs = nn.ModuleList([nn.ConvTranspose1d(in_channels, out_channels_per, kernel_size=patch_size, stride=patch_size) for patch_size in patch_sizes])
        
    def forward(self, x):
        B, C, L = x.shape
        patches = [proj(x) for proj in self.projs]
        return torch.cat(patches, 2)
        

class MultiPatchChannelInEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels_per, patch_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.out_channels_per = out_channels_per
        self.in_channels = in_channels
        self.projs = nn.ModuleList([nn.Conv1d(in_channels, out_channels_per, kernel_size=patch_size, padding=(patch_size-1)//2) for patch_size in patch_sizes])
        
    def forward(self, x):
        B, C, L = x.shape
        patches = [proj(x) for proj in self.projs]
        return torch.cat(patches, 1)
    
class MultiPatchChannelOutEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels_per, patch_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.out_channels_per = out_channels_per
        self.in_channels = in_channels
        self.projs = nn.ModuleList([nn.ConvTranspose1d(in_channels, out_channels_per, kernel_size=patch_size, padding=(patch_size-1)//2) for patch_size in patch_sizes])
        
    def forward(self, x):
        B, C, L = x.shape
        out = 0
        for i, proj in enumerate(self.projs):
            out += proj(x)
        return out

def mask_x(tensor, p):
    # Create a tensor filled with the probability p
    prob_tensor = torch.full(tensor.shape, p)
    # Generate the mask using Bernoulli distribution
    mask = torch.bernoulli(prob_tensor).bool()
    mask = mask.to(tensor.device)
    return mask * tensor

class mpmtransformer(nn.Module):
    def __init__(
        self, in_channels=15, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,  use_partial=True, use_eca=False, pe='abs', use_checkpoint=False, low_level_enhance=True, high_level_enhance=False
        ) -> None:
        super().__init__()        
        
        channels = [2**(i+3) for i in range(5)]
        heads = [2**(i+1) for i in range(5)]
        length = [2**(-i+8) for i in range(5)]

        
        self.conv1 = nn.Sequential( 
            nn.Conv1d(2, channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels[0]),
        )
        # self.conv2 = MultiPatchInEmbedding(in_channels=in_channels, out_channels_per=channels[0])
        # self.conv2 = MultiPatchInEmbedding(2, channels[0])
        # self.conv3 = MultiPatchChannelInEmbedding(in_channels=channels[0], out_channels_per=channels[0])
        self.conv3 = MultiPatchChannelInEmbedding(in_channels=in_channels, out_channels_per=2)
        # self.conv3 = MultiPatchChannelInEmbedding(2, 2)
        
        self.transconv = nn.Linear(288, 256)

        self.transconv3 = MultiPatchChannelOutEmbedding(channels[0], in_channels)
        
        self.rwattn1 = RelativePositionEmbedding(length[0], length[0], heads[0])
        self.rwattn2 = RelativePositionEmbedding(length[1], length[1], heads[1])
        self.rwattn3 = RelativePositionEmbedding(length[2], length[2], heads[2])
        self.rwattn4 = RelativePositionEmbedding(length[3], length[3], heads[3])
        
        
        self.dtransformer1 = BasicLayer(channels[0], depth=2, num_heads=heads[0], local_enhance=high_level_enhance, ) 

        self.pm1 = PatchMerging(channels[0], norm_layer=norm_layer)
        
        self.dtransformer2 = BasicLayer(channels[1], depth=2, num_heads=heads[1], local_enhance=high_level_enhance, ) 
        
        self.pm2 = PatchMerging(channels[1], norm_layer=norm_layer)
        
        self.dtransformer3 = BasicLayer(channels[2], depth=2, num_heads=heads[2], local_enhance=high_level_enhance, )
        
        self.pm3 = PatchMerging(channels[2], norm_layer=norm_layer)
        
        self.dtransformer34 = BasicLayer(channels[3], depth=2, num_heads=heads[3], local_enhance=high_level_enhance, )
        
        self.pm4 = PatchMerging(channels[3], norm_layer=norm_layer)
        
        self.transformer = BasicLayer(channels[4], depth=2, num_heads=heads[4], local_enhance=high_level_enhance, ) 
        
        self.utransformer4 = BasicLayer(channels[4], depth=2, num_heads=heads[4], local_enhance=high_level_enhance, )
        
        self.ps4 = PatchSeparate(channels[4], norm_layer=norm_layer)
        
        self.utranformer3 = BasicLayer(channels[3], depth=2, num_heads=heads[3], local_enhance=high_level_enhance, )
        
        self.ps3 = PatchSeparate(channels[3], norm_layer=norm_layer)
        
        self.utransformer2 = BasicLayer(channels[2], depth=2, num_heads=heads[2], local_enhance=high_level_enhance, )
        
        self.ps2 = PatchSeparate(channels[2], norm_layer=norm_layer)
        
        self.utransformer1 = BasicLayer(channels[1], depth=2, num_heads=heads[1], local_enhance=high_level_enhance, )
        
        self.ps1 = PatchSeparate(channels[1], norm_layer=norm_layer)
        
        

        
    
    
    
    def forward(self, x, target=None):
        B, C, L = x.shape
        

        
        x = mask_x(x, 0.85)
        
        
        

        x = self.conv3(x) 
        
        
        
        attn1 = self.rwattn1()
        attn2 = self.rwattn2()
        attn3 = self.rwattn3()
        attn4 = self.rwattn4()
                
        x1 = rearrange(x, 'b c l -> b l c')
        
        x1 = self.dtransformer1(x1)
        x1 = self.pm1(x1)
        
        x2 = self.dtransformer2(x1)
        x2 = self.pm2(x2)
        
        x3 = self.dtransformer3(x2, attn3)
        x3 = self.pm3(x3)
        
        x4 = self.dtransformer34(x3, attn4)
        x4 = self.pm4(x4)

        x_mid = self.transformer(x4)
        


        x_mid += x4
        
        x_4 = self.utransformer4(x_mid)
        x_4 = self.ps4(x_4)
        x_4 = x_4 + x3
        
        x_3 = self.utranformer3(x_4, attn4)
        x_3 = self.ps3(x_3)
        x_3 = x_3 + x2
        
        x_2 = self.utransformer2(x_3, attn3)
        x_2 = self.ps2(x_2)
        x_2 = x_2 + x1
        
        x_1 = self.utransformer1(x_2)
        x_1 = self.ps1(x_1)
        x_1 = x_1
        
        x_1 = rearrange(x_1, 'b l c -> b c l')
        x_1 +=  x
        
        out =  self.transconv3(x_1)
        

                


        return out
    
    def get_middle_feature(self, x):
        B, C, L = x.shape
        
        x = mask_x(x, 0.85)
        
        
        x = self.conv3(x) 
        
        
        
        attn1 = self.rwattn1()
        attn2 = self.rwattn2()
        attn3 = self.rwattn3()
        attn4 = self.rwattn4()
                
        x1 = rearrange(x, 'b c l -> b l c')
        
        x1 = self.dtransformer1(x1)
        x1 = self.pm1(x1)
        
        x2 = self.dtransformer2(x1)
        x2 = self.pm2(x2)
        
        x3 = self.dtransformer3(x2, attn3)
        x3 = self.pm3(x3)
        
        x4 = self.dtransformer34(x3, attn4)
        x4 = self.pm4(x4)

        x_mid = self.transformer(x4)
        return x_mid
        








