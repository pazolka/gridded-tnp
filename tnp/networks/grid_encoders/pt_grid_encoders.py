from abc import ABC
from typing import Callable, Optional, Tuple, Union

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ...utils.grids import (
    coarsen_grid,
    compute_cumcount_idx,
    construct_grid,
    construct_nearest_neighbour_matrix,
    flatten_grid,
    nearest_gridded_neighbours,
)
from ..attention_layers import MultiHeadCrossAttention, MultiHeadCrossAttentionLayer


class BasePseudoTokenGridEncoder(nn.Module, ABC):
    def __init__(
        self,
        *,
        embed_dim: int,
        mhca_layer: Union[MultiHeadCrossAttentionLayer, MultiHeadCrossAttention],
        grid_range: Tuple[Tuple[float, float], ...],
        points_per_dim: Tuple[int, ...],
    ):
        super().__init__()

        # Construct grid of pseudo-tokens.
        self.register_buffer("grid", construct_grid(grid_range, points_per_dim))

        # Construct pseudo-tokens for each grid point.
        self.latents = nn.Parameter(
            torch.randn(*self.grid.shape[:-1], embed_dim),
            requires_grad=True,
        )

        self.mhca_layer = mhca_layer


class PseudoTokenGridEncoder(BasePseudoTokenGridEncoder):
    @check_shapes(
        "x: [m, n, dx]",
        "z: [m, n, dz]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        grid_shape = self.grid.shape[:-1]
        grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
        x_grid = einops.repeat(
            self.grid, grid_str + " d -> b " + grid_str + " d", b=x.shape[0]
        )
        z_grid = einops.repeat(
            self.latents, grid_str + " e -> b " + grid_str + " e", b=z.shape[0]
        )
        z_grid = mhca_to_grid(x, z, x_grid, z_grid, z_grid, self.mhca_layer)

        return x_grid, z_grid


class PseudoTokenGridEncoderThroughTime(BasePseudoTokenGridEncoder):
    def __init__(self, *, time_dim: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.time_dim = time_dim

    @check_shapes(
        "x: [m, n, dx]",
        "z: [m, n, dz]",
        "time_grid: [m, nt]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(self, x: torch.Tensor, z: torch.Tensor, time_grid: torch.Tensor):
        # Extend self.grid to include time dimension.
        grid_shape = self.grid.shape[:-1]

        n_strings = [f"n{i}" for i in range(len(grid_shape))]
        n_vars = dict(zip(n_strings, grid_shape))
        n_str = " ".join(n_strings)

        new_n_strings = n_strings[: self.time_dim] + ["t"] + n_strings[self.time_dim :]
        new_n_str = " ".join(new_n_strings)

        x_grid = einops.repeat(
            self.grid,
            n_str + " d -> m " + new_n_str + " d",
            t=time_grid.shape[1],
            m=x.shape[0],
        )
        time_grid_reshaped = einops.repeat(time_grid, "m t -> m " + new_n_str, **n_vars)

        # Concatenate time_grid to the grid in the correct dimension.
        x_grid = torch.cat(
            [
                x_grid[..., : self.time_dim],
                time_grid_reshaped.unsqueeze(-1),
                x_grid[..., self.time_dim :],
            ],
            dim=-1,
        )

        # Extend self.latents to include time dimension.
        z_grid = einops.repeat(
            self.latents,
            n_str + " e -> m " + new_n_str + " e",
            m=x.shape[0],
            t=time_grid.shape[1],
        )

        z_grid = mhca_to_grid(x, z, x_grid, z_grid, z_grid, self.mhca_layer)

        return x_grid, z_grid


class OOTGPseudoTokenGridEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        mhca_layer: Union[MultiHeadCrossAttentionLayer, MultiHeadCrossAttention],
        grid_shape: Optional[Tuple[int, ...]] = None,
        coarsen_fn: Callable = coarsen_grid,
    ):
        super().__init__()

        if grid_shape is None:
            # Construct pseudo-tokens shared for each grid point.
            self.latent = nn.Parameter(torch.randn(embed_dim), requires_grad=True)
        else:
            self.latent = nn.Parameter(
                torch.randn(*grid_shape, embed_dim), requires_grad=True
            )

        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.coarsen_fn = coarsen_fn
        self.mhca_layer = mhca_layer

    @check_shapes(
        "x: [m, n, dx]",
        "z: [m, n, dz]",
        "x_grid: [m, ..., dx]",
        "z_grid: [m, ..., dz]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_grid: torch.Tensor,
        z_grid: Optional[torch.Tensor] = None,
    ):

        # Check whether to coarsen grid.
        if self.grid_shape is not None:
            # Move grid data to off the grid.
            x = torch.cat((x, flatten_grid(x_grid)[0]), dim=1)
            z = torch.cat((z, flatten_grid(z_grid)[0]), dim=1)

            x_grid = self.coarsen_fn(
                x_grid,
                output_grid=self.grid_shape,
            )
            z_grid = None

        grid_shape = x_grid.shape[1:-1]
        grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
        grid_pattern = grid_str + " e"
        grid_vars = {f"n{i}": dim for i, dim in enumerate(grid_shape)}

        if self.grid_shape is not None:
            assert self.grid_shape == grid_shape, "Grid shape mismatch."

            z0_grid = einops.repeat(
                self.latent,
                grid_pattern + " -> b " + grid_pattern,
                b=x.shape[0],
            )
        else:
            z0_grid = einops.repeat(
                self.latent, "e -> b " + grid_pattern, b=x.shape[0], **grid_vars
            )

        if z_grid is None:
            z_grid = z0_grid

        z_grid = mhca_to_grid(x, z, x_grid, z_grid, z0_grid, self.mhca_layer)

        return x_grid, z_grid


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "z0_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def mhca_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
    z0_grid: torch.Tensor,
    mhca: Union[MultiHeadCrossAttention, MultiHeadCrossAttentionLayer],
) -> torch.Tensor:
    # Get number of grid points.
    x_grid_flat, _ = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, k)
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    # Cumulative count.
    cumcount_idx = compute_cumcount_idx(nearest_idx)

    joint_grid, att_mask = construct_nearest_neighbour_matrix(
        nearest_idx, cumcount_idx, z, z_grid, num_grid_points, concatenate_x_grid=True
    )

    # Rearrange latents.
    z0_grid_flat, flat_to_grid_fn = flatten_grid(z0_grid)
    z0_grid_flat = einops.rearrange(z0_grid_flat, "b m e -> (b m) 1 e")

    # Finally! Do the MHCA operation.
    z_grid = mhca(z0_grid_flat, joint_grid, mask=att_mask)

    # Reshape output.
    z_grid = einops.rearrange(z_grid, "(b s) 1 e -> b s e", b=x.shape[0])

    # Unflatten and return.
    z_grid = flat_to_grid_fn(z_grid)

    return z_grid
