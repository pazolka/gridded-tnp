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
from ..utils.dropout import dropout_all
from ..utils.helpers import preprocess_observations
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder


class GriddedATETNPEncoder(nn.Module):
    force_dropout = False

    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[BasePseudoTokenGridEncoder, SetConv],
        y_encoder: nn.Module,
        basis_fn: nn.Module,
        p_basis_dropout: float = 0.5,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
        self.y_encoder = y_encoder
        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout

    @check_shapes("xc: [m, ..., dx]", "zc: [m, ..., dz]", "return: [m, ..., dz]")
    def add_basis(self, xc, zc):
        # Obtain basis functions and concatenate.
        if self.basis_fn.num_fourier != 0:
            zc_basis = self.basis_fn(xc)

            # Dropout
            if self.force_dropout:
                zc_basis = dropout_all(zc_basis, 1.0, True)
            else:
                zc_basis = dropout_all(zc_basis, self.p_basis_dropout)

            # Sum grid values with basis functions.
            zc = zc + zc_basis
        return zc

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

        # Add basis functions.
        zc = self.add_basis(xc, zc)

        # Encode to grid using original xc.
        if time_grid is not None:
            xc_grid, zc_grid = self.grid_encoder(xc, zc, time_grid)
        else:
            xc_grid, zc_grid = self.grid_encoder(xc, zc)

        zt = self.tetransformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class MultiModalGriddedATETNPEncoder(nn.Module):
    force_dropout = False

    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: MultiModalGridEncoder,
        y_encoder: nn.Module,
        mode_names: List[str],
        basis_fn: nn.Module,
        p_basis_dropout: float = 0.5,
        x_grid_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
        self.mode_names = mode_names
        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout

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

    @check_shapes("xc: [m, ..., dx]", "zc: [m, ..., dz]", "return: [m, ..., dz]")
    def add_basis(self, xc, zc):
        # Obtain basis functions and concatenate.
        if self.basis_fn.num_fourier != 0:
            # Take only the spatial dimensions
            if self.grid_encoder.time_dim is not None:
                zc_basis = self.basis_fn(
                    xc[..., torch.arange(xc.shape[-1], device=xc.device) != self.grid_encoder.time_dim]
                )
            else:
                zc_basis = self.basis_fn(xc)

            # Dropout
            if self.force_dropout:
                zc_basis = dropout_all(zc_basis, 1.0, True)
            else:
                zc_basis = dropout_all(zc_basis, self.p_basis_dropout)

            # Sum grid values with basis functions.
            zc = zc + zc_basis
        return zc

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

        # Add grid encoder basis functions.
        for key in xc.keys():
            zc[key] = self.add_basis(xc[key], zc[key])

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


class OOTGGriddedATETNPEncoder(nn.Module):
    force_dropout = False

    def __init__(
        self,
        tetransformer_encoder: GriddedTransformerEncoder,
        grid_encoder: Union[OOTGPseudoTokenGridEncoder, OOTGSetConv],
        y_encoder: nn.Module,
        y_grid_encoder: nn.Module,
        basis_fn: nn.Module,
        p_basis_dropout: float = 0.5,
    ):
        super().__init__()

        self.tetransformer_encoder = tetransformer_encoder
        self.grid_encoder = grid_encoder
        self.y_encoder = y_encoder
        self.y_grid_encoder = y_grid_encoder
        self.basis_fn = basis_fn
        self.p_basis_dropout = p_basis_dropout

    @check_shapes("xc: [m, ..., dx]", "zc: [m, ..., dz]", "return: [m, ..., dz]")
    def add_basis(self, xc, zc):
        # Obtain basis functions and concatenate.
        if self.basis_fn.num_fourier != 0:
            zc_basis = self.basis_fn(xc)

            # Dropout
            if self.force_dropout:
                zc_basis = dropout_all(zc_basis, 1.0, True)
            else:
                zc_basis = dropout_all(zc_basis, self.p_basis_dropout)

            # Sum grid values with basis functions.
            zc = zc + zc_basis
        return zc

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

        # Add grid encoder basis functions.
        if self.basis_fn:
            zc = self.add_basis(xc, zc)
            zc_grid = self.add_basis(xc_grid, zc_grid)

        # Encode to grid.
        xc_grid, zc_grid = self.grid_encoder(xc, zc, xc_grid, zc_grid)

        zt = self.tetransformer_encoder(xc_grid, zc_grid, xt, zt)

        return zt


class GriddedATETNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: GriddedATETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGGriddedATETNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGGriddedATETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalGriddedATETNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalGriddedATETNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
