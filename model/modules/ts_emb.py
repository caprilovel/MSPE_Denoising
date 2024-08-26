import math

from typing import Optional

import torch
from torch import nn
from einops import einsum, rearrange

from jaxtyping import Bool, Float, Int

def size_to_mask(
    max_size: int,
    sizes: Int[torch.Tensor, "*batch"],
) -> Bool[torch.Tensor, "*batch max_size"]:
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))

class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls)), dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )
        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch max_feat"],
        in_feat_size: Int[torch.Tensor, "*batch"],
    ) -> Float[torch.Tensor, "*batch out_feat"]:
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            
            weight = self.weight[idx] * self.mask[idx]
            print(weight.shape)
            print(self.mask[idx])
            bias = self.bias[idx] if self.bias is not None else 0
            print(torch.eq(in_feat_size, feat_size).unsqueeze(-1))
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )

if __name__ == "__main__":
    model = MultiInSizeLinear(in_features_ls=(1, 2, 4), out_features=7)
    print(model)

    x = torch.randn(1, 3, 4)
    in_feat_size = torch.tensor([1, 2, 4])
    out = model(x, in_feat_size)
    print(out.shape)
    print(out)
