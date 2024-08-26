import torch 
import torch.nn as nn
import torch.nn.functional as F



def RateDiffLoss(a:torch.tensor, b:torch.tensor) -> torch.tensor:
    """_summary_

    Args:
        a (torch.tensor): _description_
        b (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """
    assert len(a.shape) == len(b.shape) == 3
    diff_a = torch.abs(a[:, :, 1:] - a[:, :, :-1])
    diff_b = torch.abs(b[:, :, 1:] - b[:, :, :-1])
    
    return torch.mean(torch.abs(diff_a - diff_b))
    
    
def SSIMLoss(a:torch.tensor, b:torch.tensor, c1=0.01, c2=0.01) -> torch.tensor:
    """_summary_

    Args:
        a (torch.tensor): first tensor. Usually be the input tensor. 
        b (torch.tensor): Second tensor. Usually be the target tensor.
        c1 ([float]): the first hyper parameter for SSIM loss.
        c2 ([float]): the second hyper parameter for SSIM loss.

    Returns:
        [type]: the SSIM loss between a and b.
    """
    C1 = c1
    C2 = c2

    mu_x = torch.mean(a)
    mu_y = torch.mean(b)
    sigma_x = torch.std(a)
    sigma_y = torch.std(b)
    sigma_xy = torch.cov(a, b)[0, 1]

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))

    return ssim