import copy
from typing import Dict, List, Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .pt_grid_encoders import BasePseudoTokenGridEncoder
from .pt_te_grid_encoders import BasePseudoTokenTEGridEncoder
from .setconv_grid_encoders import BaseSetConv


class MultiModalGridEncoder(nn.Module):
    def __init__(
        self,
        grid_encoder: Union[
            BasePseudoTokenGridEncoder, BasePseudoTokenTEGridEncoder, BaseSetConv
        ],
        mode_names: List[str],
        output_mixer: nn.Module,
    ):
        super().__init__()

        # Seperate grid encoders for each mode.
        self.grid_encoders = nn.ModuleDict(
            {mode: copy.deepcopy(grid_encoder) for mode in mode_names}
        )
        # Mixes the outputs of the pseudo-token grid encoders.
        self.output_mixer = output_mixer
        if hasattr(grid_encoder, "embed_dim"):
            self.embed_dim = grid_encoder.embed_dim
        if hasattr(grid_encoder, "time_dim"):
            self.time_dim = grid_encoder.time_dim
        else:
            self.time_dim = None

    @check_shapes(
        "x.values(): [m, n, dx]",
        "z.values(): [m, n, dz]",
        "time_grid: [m, nt]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        z: Dict[str, torch.Tensor],
        time_grid: Optional[torch.Tensor] = None,
    ):

        z_grids: Dict[str, torch.Tensor] = {}
        for mode, grid_encoder in self.grid_encoders.items():
            if time_grid is not None:
                x_grid, z_grid = grid_encoder(x[mode], z[mode], time_grid)
            else:
                x_grid, z_grid = grid_encoder(x[mode], z[mode])
            z_grids[mode] = z_grid

        z_grid = torch.cat(list(z_grids.values()), dim=-1)
        z_grid = self.output_mixer(z_grid)

        return x_grid, z_grid


class MultiModalSingleGridEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_encoder: Union[
            BasePseudoTokenGridEncoder, BasePseudoTokenTEGridEncoder, BaseSetConv
        ],
    ):
        super().__init__()

        self.grid_encoder = grid_encoder

    @check_shapes(
        "x.values(): [m, n, dx]",
        "z.values(): [m, n, dz]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        z: Dict[str, torch.Tensor],
        time_grid: Optional[torch.Tensor] = None,
    ):

        # Concatenate all modalities.
        x = torch.cat(list(x.values()), dim=1)
        z = torch.cat(list(z.values()), dim=1)

        if time_grid is not None:
            x_grid, z_grid = self.grid_encoder(x, z, time_grid)
        else:
            x_grid, z_grid = self.grid_encoder(x, z)

        return x_grid, z_grid
