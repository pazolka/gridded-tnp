from typing import Dict, List, Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.grid_decoders import GridDecoder
from ..networks.grid_encoders import OOTGSetConv, SetConv
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder


class ConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        grid_encoder: SetConv,
        grid_decoder: GridDecoder,
        z_encoder: nn.Module,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        # Add density.
        yc = torch.cat((yc, torch.ones(yc.shape[:-1] + (1,)).to(yc)), dim=-1)

        # Encode to grid.
        x_grid, z_grid = self.grid_encoder(xc, yc)

        # Encode to z.
        z_grid = self.z_encoder(z_grid)

        # Convolve.
        z_grid = self.conv_net(z_grid)

        # Decode.
        zt = self.grid_decoder(x_grid, z_grid, xt)
        return zt


class MultiModalConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        grid_encoder: SetConv,
        grid_decoder: GridDecoder,
        z_encoder: nn.Module,
        mode_names: List[str],
    ):
        super().__init__()

        self.conv_net = conv_net
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder
        self.mode_names = mode_names

    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Add density and dummy values for all modes.
        yc_list: List[torch.Tensor] = []
        for i, mode in enumerate(self.mode_names):
            modes_before = self.mode_names[:i]
            modes_after = self.mode_names[i + 1 :]
            dims_before = sum(yc[m].shape[-1] for m in modes_before)
            dims_after = sum(yc[m].shape[-1] for m in modes_after)
            yc_ = torch.cat(
                (
                    torch.zeros(yc[mode].shape[:-1] + (dims_before,)).to(yc[mode]),
                    yc[mode],
                    torch.zeros(yc[mode].shape[:-1] + (dims_after,)).to(yc[mode]),
                    torch.zeros(yc[mode].shape[:-1] + (len(modes_before),)).to(
                        yc[mode]
                    ),
                    torch.ones(yc[mode].shape[:-1] + (1,)).to(yc[mode]),
                    torch.zeros(yc[mode].shape[:-1] + (len(modes_after),)).to(yc[mode]),
                ),
                dim=-1,
            )
            yc_list.append(yc_)

        yc = torch.cat(yc_list, dim=1)
        xc = torch.cat(list(xc.values()), dim=1)

        # Encode to grid.
        if time_grid is not None:
            x_grid, z_grid = self.grid_encoder(xc, yc, time_grid)
        else:
            x_grid, z_grid = self.grid_encoder(xc, yc)

        # Encode to z.
        z_grid = self.z_encoder(z_grid)

        # Convolve.
        z_grid = self.conv_net(z_grid)

        # Decode.
        zt = self.grid_decoder(x_grid, z_grid, xt)
        return zt


class OOTGConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        grid_encoder: OOTGSetConv,
        grid_decoder: GridDecoder,
        z_encoder: nn.Module,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xc_grid: [m, ..., dx]",
        "yc_grid: [m, ..., dy_grid]",
        "xt: [m, nt, dx]",
        "return: [m, n, dz]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xc_grid: torch.Tensor,
        yc_grid: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        yc_dims, yc_grid_dims = yc.shape[-1], yc_grid.shape[-1]

        # Add density and padding for the yc_grid channels.
        yc = torch.cat(
            (
                yc,
                torch.zeros(yc.shape[:-1] + (yc_grid_dims,)).to(yc),
                torch.ones(yc.shape[:-1] + (1,)).to(yc),
                torch.zeros(yc.shape[:-1] + (1,)).to(yc),
            ),
            dim=-1,
        )
        yc_grid = torch.cat(
            (
                torch.zeros(yc_grid.shape[:-1] + (yc_dims,)).to(yc_grid),
                yc_grid,
                torch.zeros(yc_grid.shape[:-1] + (1,)).to(yc_grid),
                torch.ones(yc_grid.shape[:-1] + (1,)).to(yc_grid),
            ),
            dim=-1,
        )

        # Encode to grid.
        x_grid, z_grid = self.grid_encoder(xc, yc, xc_grid, yc_grid)

        # Encode to z.
        z_grid = self.z_encoder(z_grid)

        # Convolve.
        z_grid = self.conv_net(z_grid)

        # Decode.
        zt = self.grid_decoder(x_grid, z_grid, xt)
        return zt


class GriddedConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        z_encoder: nn.Module,
    ):
        super().__init__()
        self.conv_net = conv_net
        self.z_encoder = z_encoder

    @check_shapes(
        "mc: [m, ...]",
        "mt: [m, ...]",
        "y: [m, ..., dy]",
        "return: [m, dt, dz]",
    )
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.z_encoder(z_grid)
        z_grid = self.conv_net(z_grid)
        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        return zt


class ConvCNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalConvCNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGConvCNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class GriddedConvCNP(nn.Module):
    def __init__(
        self,
        encoder: GriddedConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    @check_shapes("mc: [m, ...]", "y: [m, ..., dy]", "mt: [m, ...]")
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(mc, y, mt)))
