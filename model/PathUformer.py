import math
import torch
import torch.nn as nn
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul

class PathUformer(nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__()
        self.enc1 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8], 
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=1, residual_connection=1)
        
        self.enc2 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[4,8,16], 
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=2, residual_connection=1)
        
        self.enc3 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8], 
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=3, residual_connection=1)
        
        self.enc4 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8],
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=4, residual_connection=1)
        
        self.dec1 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8], 
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=3, residual_connection=1)
        
        self.dec2 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8],
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=4, residual_connection=1)
        
        self.dec3 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[4,8,16],
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=3, residual_connection=1)
        
        self.dec4 = AMS(256, 256, 3, device, k=1, num_nodes=2, patch_size=[2,4,8],
                        noisy_gating=True, d_model=16, d_ff=32, layer_number=4, residual_connection=1)
        
        self.start_fc = nn.Linear(in_features=1, out_features=16)
        # set start_fc freeze
        self.start_fc = self.start_fc.eval()
        
        self.out_fc = nn.Linear(in_features=16, out_features=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.start_fc(x.unsqueeze(-1))
        # print(x.shape)
        balance = 0
        x1, balance1 = self.enc1(x)
        x2, balance2 = self.enc2(x1)
       
        
        
        y1, balance5 = self.dec1(x2)
        y2, balance6 = self.dec2(y1 + x1)

        
        y = self.out_fc(y2).squeeze(-1)
        y = y.permute(0, 2, 1)
        
        return y, balance
        
        