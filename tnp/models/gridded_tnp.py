import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.grid_encoders import (
    BasePseudoTokenGridEncoder,
    MultiModalGridEncoder,
    OOTGPseudoTokenGridEncoder,
    OOTGSetConv,
    SetConv,
)
from ..networks.transformer import GriddedTransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder


class GriddedTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[BasePseudoTokenGridEncoder, SetConv],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.grid_encoder = grid_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, nt, dz]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        zt = self.transformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class MultiModalGriddedTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: GriddedTransformerEncoder,
        grid_encoder: MultiModalGridEncoder,
        xy_encoder: nn.Module,
        zt_encoder: nn.Module,
        mode_names: List[str],
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        x_grid_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.grid_encoder = grid_encoder
        self.mode_names = mode_names

        # Shared x_encoder.
        self.x_encoder = x_encoder

        self.zt_encoder = zt_encoder

        # Copy encoder for each mode.
        self.xy_encoder = nn.ModuleDict(
            {mode: copy.deepcopy(xy_encoder) for mode in mode_names}
        )
        self.y_encoder = nn.ModuleDict(
            {mode: copy.deepcopy(y_encoder) for mode in mode_names}
        )
        self.x_grid_dims = x_grid_dims

    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Use separate encoders for each modality.
        xc_list = list(xc.values())
        yc_list = list(yc.values())
        nc_list = [xcm.shape[-2] for xcm in xc_list]
        x = torch.cat(xc_list, dim=1)
        x = torch.cat((x, xt), dim=1)
        x_encoded = self.x_encoder(x)

        # Split x_encoded into xc_list_encoded and xt_encoded.
        xc_list_encoded = x_encoded[:, : sum(nc_list)].split(nc_list, dim=1)
        xt_encoded = x_encoded[:, sum(nc_list) :]

        # Encode xt.
        zt = self.zt_encoder(xt_encoded)

        # Encode y for each modality.
        yc_encoded_list = []

        if self.y_encoder is not None:
            for mod, yc_ in zip(yc, yc_list):
                yc_encoded_list.append(self.y_encoder[mod](yc_))
        else:
            yc_encoded_list = yc_list

        # Combine x and y encodings for each modality
        zc: Dict[str, torch.Tensor] = {}
        for mod, xc_encoded, yc_encoded in zip(
            yc.keys(), xc_list_encoded, yc_encoded_list
        ):
            zc[mod] = torch.cat((xc_encoded, yc_encoded), dim=-1)
            zc[mod] = self.xy_encoder[mod](zc[mod])

        if self.x_grid_dims is not None:
            xc = {mod: xc[mod][..., self.x_grid_dims] for mod in xc.keys()}
            xt = xt[..., self.x_grid_dims]

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        # Apply transformer encoder
        zt = self.transformer_encoder(xc_grid, zc_grid, xt, zt)
        return zt


class OOTGGriddedTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[OOTGPseudoTokenGridEncoder, OOTGSetConv],
        xy_encoder: nn.Module,
        xy_grid_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        y_grid_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.grid_encoder = grid_encoder
        self.xy_encoder = xy_encoder
        self.xy_grid_encoder = xy_grid_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.y_grid_encoder = y_grid_encoder

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
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        xc_grid_encoded = self.x_encoder(xc_grid)
        yc_grid_encoded = self.y_grid_encoder(yc_grid)

        # Tokenise non-gridded observations.
        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Tokenise gridded observations.
        zc_grid = torch.cat((xc_grid_encoded, yc_grid_encoded), dim=-1)
        zc_grid = self.xy_grid_encoder(zc_grid)

        # Encode to grid.
        xc_grid, zc_grid = self.grid_encoder(xc, zc, xc_grid, zc_grid)

        zt = self.transformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class GriddedTNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: GriddedTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGGriddedTNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGGriddedTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalGriddedTNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalGriddedTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
