from abc import ABC
from typing import Callable, Optional, Tuple

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
from ..teattention_layers import MultiHeadCrossTEAttentionLayer


class BasePseudoTokenTEGridEncoder(nn.Module, ABC):
    def __init__(
        self,
        *,
        embed_dim: int,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
        points_per_unit_dim: Tuple[int, ...],
        margin: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        # Construct shared pseudo-token for each grid point.
        self.latent = nn.Parameter(
            torch.randn(embed_dim),
            requires_grad=True,
        )

        self.embed_dim = embed_dim
        self.mhca_layer = mhca_layer

        self.points_per_unit_dim = tuple(points_per_unit_dim)

        if margin is None:
            self.margin = tuple(0.0 for _ in range(len(points_per_unit_dim)))
        else:
            assert len(margin) == len(points_per_unit_dim)
            self.margin = tuple(margin)


class PseudoTokenTEGridEncoder(BasePseudoTokenTEGridEncoder):
    @check_shapes(
        "x: [m, n, dx]",
        "z: [m, n, dz]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute the grid based on min(x), max(x) and points per dimension.
        grid_range = tuple(
            (x[:, :, i].min() - self.margin[i], x[:, :, i].max() + self.margin[i])
            for i in range(x.shape[-1])
        )
        points_per_dim = tuple(
            int((grid_range[i][1] - grid_range[i][0]) * self.points_per_unit_dim[i])
            for i in range(len(grid_range))
        )
        x_grid = construct_grid(grid_range, points_per_dim).to(x)

        grid_shape = x_grid.shape[:-1]
        grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
        grid_vars = dict(zip(grid_str.split(" "), grid_shape))
        x_grid = einops.repeat(
            x_grid, grid_str + " d -> b " + grid_str + " d", b=x.shape[0]
        )
        z_grid = einops.repeat(
            self.latent, "e -> b " + grid_str + " e", b=z.shape[0], **grid_vars
        )
        z_grid, x_grid = te_mhca_to_grid(x, z, x_grid, z_grid, z_grid, self.mhca_layer)

        return x_grid, z_grid


class PseudoTokenTEGridEncoderThroughTime(BasePseudoTokenTEGridEncoder):
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
    def forward(
        self, x: torch.Tensor, z: torch.Tensor, time_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute the grid based on min(x), max(x) and points per dimension.
        grid_range = tuple(
            (x[:, :, i].min() - self.margin[j], x[:, :, i].max() + self.margin[j])
            for j, i in enumerate(i for i in range(x.shape[-1]) if i != self.time_dim)
        )
        points_per_dim = tuple(
            int((grid_range[i][1] - grid_range[i][0]) * self.points_per_unit_dim[i])
            for i in range(len(grid_range))
        )
        x_grid = construct_grid(grid_range, points_per_dim).to(x)
        grid_shape = x_grid.shape[:-1]
        grid_strings = [f"n{i}" for i in range(len(grid_shape))]
        grid_str = " ".join(grid_strings)
        grid_vars = dict(zip(grid_str.split(" "), grid_shape))

        new_grid_strings = (
            grid_strings[: self.time_dim] + ["t"] + grid_strings[self.time_dim :]
        )
        new_grid_str = " ".join(new_grid_strings)

        x_grid = einops.repeat(
            x_grid,
            grid_str + " d -> b " + new_grid_str + " d",
            b=x.shape[0],
            t=time_grid.shape[1],
        )
        time_grid_reshaped = einops.repeat(
            time_grid,
            "b t -> b " + new_grid_str,
            **grid_vars,
        )
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
            self.latent,
            "e -> b " + new_grid_str + " e",
            b=z.shape[0],
            **grid_vars,
            t=time_grid.shape[1],
        )
        z_grid, x_grid = te_mhca_to_grid(x, z, x_grid, z_grid, z_grid, self.mhca_layer)

        return x_grid, z_grid


class OOTGPseudoTokenTEGridEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
        grid_shape: Optional[Tuple[int, ...]] = None,
        coarsen_fn: Callable = coarsen_grid,
    ):
        super().__init__()

        # Construct pseudo-tokens shared for each grid point.
        self.latent = nn.Parameter(torch.randn(embed_dim), requires_grad=True)

        self.mhca_layer = mhca_layer
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.coarsen_fn = coarsen_fn

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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

        z0_grid = einops.repeat(
            self.latent, "e -> b " + grid_pattern, b=x.shape[0], **grid_vars
        )

        if z_grid is None:
            z_grid = z0_grid

        z_grid, x_grid = te_mhca_to_grid(x, z, x_grid, z_grid, z0_grid, self.mhca_layer)

        return x_grid, z_grid


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "z0_grid: [m, ..., dz]",
    "return[0]: [m, ..., dz]",
    "return[1]: [m, ..., dx]",
)
def te_mhca_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
    z0_grid: torch.Tensor,
    mhca: MultiHeadCrossTEAttentionLayer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get number of grid points.
    x_grid_flat, _ = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, k)
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    # Cumulative count.
    cumcount_idx = compute_cumcount_idx(nearest_idx)

    joint_grid_x, _ = construct_nearest_neighbour_matrix(
        nearest_idx, cumcount_idx, x, x_grid, num_grid_points, concatenate_x_grid=True
    )
    joint_grid_z, att_mask = construct_nearest_neighbour_matrix(
        nearest_idx, cumcount_idx, z, z_grid, num_grid_points, concatenate_x_grid=True
    )

    # Rearrange inputs and latents.
    x_grid_flat, flat_to_grid_fn_x = flatten_grid(x_grid)
    x_grid_flat = einops.rearrange(x_grid_flat, "b m e -> (b m) 1 e")
    z0_grid_flat, flat_to_grid_fn_z = flatten_grid(z0_grid)
    z0_grid_flat = einops.rearrange(z0_grid_flat, "b m e -> (b m) 1 e")

    # Finally! Do the MHCA operation.
    z_grid, x_grid = mhca(
        z0_grid_flat, joint_grid_z, x_grid_flat, joint_grid_x, mask=att_mask
    )

    # Reshape output.
    z_grid = einops.rearrange(z_grid, "(b s) 1 e -> b s e", b=x.shape[0])
    x_grid = einops.rearrange(x_grid, "(b s) 1 e -> b s e", b=x.shape[0])

    # Unflatten and return.
    z_grid = flat_to_grid_fn_z(z_grid)
    x_grid = flat_to_grid_fn_x(x_grid)

    return z_grid, x_grid
