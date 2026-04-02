from typing import Tuple

import torch
from check_shapes import check_shapes
from torch import nn


class ModuleOnPreSpecifiedDomain(nn.Module):
    def __init__(
        self,
        *,
        module: type[nn.Module],
        x_range: Tuple[
            Tuple[float,],
            Tuple[float,],
        ],
        default_val: float = 0.0,
    ):
        super().__init__()

        self.module = module
        self.x_range = torch.as_tensor(x_range)
        self.default_val = default_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)

        in_domain = torch.all(x < self.x_range[1].to(out), dim=-1) & torch.all(
            x > self.x_range[0].to(out), dim=-1
        )
        out = torch.where(
            in_domain[..., None], out, torch.ones_like(out) * self.default_val
        )

        return out


class ModuleOnFourierExpandedInput(nn.Module):
    def __init__(
        self,
        *,
        module: type[nn.Module],
        x_range: Tuple[
            Tuple[float,],
            Tuple[float,],
        ],
        num_fourier: int,
    ):
        super().__init__()

        self.module = module
        x_range_transposed = [list(item) for item in zip(*x_range)]
        self.x_range = torch.as_tensor(x_range_transposed)
        self.num_fourier = num_fourier

        periods = [
            torch.stack(
                [
                    (2 * abs(self.x_range[1, dim] - self.x_range[0, dim])) // (n + 1)
                    for dim in range(len(self.x_range[0]))
                ],
                dim=-1,
            )
            for n in range(num_fourier)
        ]
        #print(periods)
        self.fourier_expansion = lambda x: torch.cat(
            [torch.sin(x / period.to(x.device)) for period in periods], dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded_x = self.fourier_expansion(x)

        out = self.module(expanded_x)
        in_domain = torch.all(x < self.x_range[1].to(out), dim=-1) & torch.all(
            x > self.x_range[0].to(out), dim=-1
        )

        out = torch.where(in_domain[..., None], out, torch.zeros_like(out))

        return out
