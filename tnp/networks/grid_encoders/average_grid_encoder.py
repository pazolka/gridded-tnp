from typing import Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ...utils.grids import (
    associative_scan,
    construct_grid,
    flatten_grid,
    nearest_gridded_neighbours,
)


class AverageGridEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        grid_range: Tuple[Tuple[float, float], ...],
        points_per_dim: Tuple[int, ...],
    ):
        super().__init__()

        # Construct grid.
        self.register_buffer("grid", construct_grid(grid_range, points_per_dim))

        # Construct pseudo-tokens for each grid point.
        self.latents = nn.Parameter(
            torch.randn(*self.grid.shape[:-1], embed_dim),
            requires_grad=True,
        )

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
            self.latents, grid_str + " e -> b " + grid_str + " e", b=x.shape[0]
        )

        # Average the the token values z across all x that appear in the same grid cell.
        z_grid = reduce_to_grid(x, z, x_grid, z_grid)

        return x_grid, z_grid


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
)
def reduce_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    num_batches, num_points = x.shape[:2]

    # Flatten grid.
    x_grid_flat, flat_to_grid_fn = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, n).
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    n_batch_idx = torch.arange(num_batches, device=z.device).unsqueeze(-1).repeat(1, num_points)
    n_range_idx = torch.arange(num_points, device=z.device).repeat(num_batches, 1)

    _, inverse_indices = torch.unique(nearest_idx, return_inverse=True)

    sorted_indices = nearest_idx.argsort(dim=1, stable=True)
    inverse_indices_sorted = inverse_indices.gather(1, sorted_indices).type(torch.long)
    unsorted_indices = sorted_indices.argsort(dim=1, stable=True)

    # Store changes in value.
    inverse_indices_diff = inverse_indices_sorted - inverse_indices_sorted.roll(
        1, dims=1
    )
    inverse_indices_diff = torch.where(
        inverse_indices_diff == 0,
        torch.ones_like(inverse_indices_diff),
        torch.zeros_like(inverse_indices_diff),
    )
    inverse_indices_diff[:, 0] = torch.zeros_like(inverse_indices_diff[:, 0])

    adjusted_cumsum = associative_scan(
        inverse_indices_diff, inverse_indices_diff, dim=1
    )
    adjusted_cumsum = adjusted_cumsum.round().int()
    cumcount_idx = adjusted_cumsum.gather(1, unsorted_indices)

    max_patch = cumcount_idx.amax() + 1

    # Create tensor with for each grid-token all nearest off-grid + itself in third axis.
    # (b * num_grid_points, max_patch, z.shape[-1])
    joint_grid = torch.full(
        (num_batches * num_grid_points, max_patch, z.shape[-1]),
        torch.nan,
        device=z.device,
    )

    # Add nearest off the grid points to each on_the_grid point.
    idx_shifter = torch.arange(
        0, num_batches * num_grid_points, num_grid_points, device=z.device
    ).repeat_interleave(num_points)
    joint_grid[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = z[
        n_batch_idx.flatten(), n_range_idx.flatten()
    ]

    # Now do the reduction.
    joint_grid = torch.nanmean(joint_grid, dim=1, keepdim=True)

    # Reshape output.
    joint_grid = einops.rearrange(joint_grid, "(b s) 1 e -> b s e", b=num_batches)

    # Unflatten and return.
    joint_grid = flat_to_grid_fn(joint_grid)

    # Replace nans with 0 in joint_grid.
    joint_grid = torch.nan_to_num(joint_grid, nan=0.0)
    z_grid = z_grid + joint_grid
    return z_grid
