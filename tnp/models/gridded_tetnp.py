import copy
from typing import Dict, List, Optional, Tuple, Union

import einops
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


class GriddedTETNPEncoder(nn.Module):
    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[BasePseudoTokenGridEncoder, SetConv],
        y_encoder: nn.Module,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
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

        y = torch.cat((yc, yt), dim=1)
        z = self.y_encoder(y)
        zc, zt = z.split((yc.shape[1], yt.shape[1]), dim=1)

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        zt = self.tetransformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class MultiModalGriddedTETNPEncoder(nn.Module):
    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: MultiModalGridEncoder,
        y_encoder: nn.Module,
        mode_names: List[str],
        x_grid_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
        self.mode_names = mode_names

        # Copy encoder for each mode.
        self.y_encoder = nn.ModuleDict(
            {mode: copy.deepcopy(y_encoder) for mode in mode_names}
        )
        self.x_grid_dims = x_grid_dims

        # Construct initial target token.
        self.zt = nn.Parameter(
            torch.randn(self.grid_encoder.embed_dim),
            requires_grad=True,
        )

    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Encode y for each modality.
        zc: Dict[str, torch.Tensor] = {}
        for mod, ycm in yc.items():
            zc[mod] = self.y_encoder[mod](ycm)

        if self.x_grid_dims is not None:
            # Only use the x_grid_dims dimensions for the gridded variables, e.g. for time.
            xc = {mod: xc[mod][..., self.x_grid_dims] for mod in xc.keys()}
            xt = xt[..., self.x_grid_dims]

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        # Duplicate initial target token to correct shape.
        zt = einops.repeat(self.zt, "d -> m n d", m=xt.shape[0], n=xt.shape[1])

        # Apply transformer encoder
        zt = self.tetransformer_encoder(xc_grid, zc_grid, xt, zt)
        return zt


class OOTGGriddedTETNPEncoder(nn.Module):
    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[OOTGPseudoTokenGridEncoder, OOTGSetConv],
        y_encoder: nn.Module,
        y_grid_encoder: nn.Module,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
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

        # Tokenise observations.
        y = torch.cat((yc, yt), dim=1)
        z = self.y_encoder(y)
        zc, zt = z.split((yc.shape[1], yt.shape[1]), dim=1)
        zc_grid = self.y_grid_encoder(yc_grid)

        # Encode to grid.
        xc_grid, zc_grid = self.grid_encoder(xc, zc, xc_grid, zc_grid)

        zt = self.tetransformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class GriddedTETNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: GriddedTETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGGriddedTETNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGGriddedTETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalGriddedTETNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalGriddedTETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
