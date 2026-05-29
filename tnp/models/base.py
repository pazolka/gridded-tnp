from abc import ABC
from typing import Dict, Optional

import torch
from check_shapes import check_shapes
from torch import nn


class BaseNeuralProcess(nn.Module, ABC):
    """Represents a neural process base class"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class ConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))


class NeuralProcess(BaseNeuralProcess):
    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 1
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, num_samples), xt))


class OOTGConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xc_grid: [m, ..., dx]",
        "yc_grid: [m, ..., dy_grid]",
        "xt: [m, nt, dx]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xc_grid: torch.Tensor,
        yc_grid: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        return self.likelihood(
            self.decoder(self.encoder(xc, yc, xc_grid, yc_grid, xt), xt)
        )


class MultiModalConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc.values(): [m, nc, dx]",
        "yc.values(): [m, nc, dy]",
        "xt: [m, nt, dx]",
        "time_grid: [m, t_grid]",
    )
    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, time_grid), xt))
