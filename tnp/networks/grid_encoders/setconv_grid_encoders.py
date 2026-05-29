from abc import ABC
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ...utils.distances import sq_dist
from ...utils.grids import (
    coarsen_grid,
    compute_cumcount_idx,
    construct_grid,
    construct_nearest_neighbour_matrix,
    flatten_grid,
    nearest_gridded_neighbours,
)


class BaseSetConv(nn.Module, ABC):
    def __init__(
        self,
        *,
        dims: int,
        grid_range: Tuple[Tuple[float, float], ...],
        points_per_dim: Tuple[int, ...],
        init_lengthscale: float = 0.1,
        dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
        use_nn: bool = False,
    ):
        super().__init__()

        # Construct grid.
        self.register_buffer("grid", construct_grid(grid_range, points_per_dim))

        # Construct lengthscales.
        init_lengthscale = torch.as_tensor(dims * [init_lengthscale], dtype=torch.float)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=True,
        )

        self.dist_fn = dist_fn
        self.use_nn = use_nn

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )


class SetConv(BaseSetConv):
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

        if self.use_nn:
            z_grid = setconv_to_grid_nn(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )
        else:
            z_grid = setconv_to_grid(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )

        return x_grid, z_grid


class SetConvThroughTime(BaseSetConv):
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
        # Extend self.grid to include time dimension.
        grid_shape = self.grid.shape[:-1]

        n_strings = [f"n{i}" for i in range(len(grid_shape))]
        n_vars = dict(zip(n_strings, grid_shape))
        n_str = " ".join(n_strings)

        new_n_strings = n_strings[: self.time_dim] + ["t"] + n_strings[self.time_dim :]
        new_n_str = " ".join(new_n_strings)

        x_grid = einops.repeat(
            self.grid,
            n_str + " d -> b " + new_n_str + " d",
            t=time_grid.shape[1],
            b=x.shape[0],
        )
        time_grid_reshaped = einops.repeat(time_grid, "b t -> b " + new_n_str, **n_vars)

        # Concatenate time_grid to the grid in the correct dimension.
        x_grid = torch.cat(
            [
                x_grid[..., : self.time_dim],
                time_grid_reshaped.unsqueeze(-1),
                x_grid[..., self.time_dim :],
            ],
            dim=-1,
        )

        if self.use_nn:
            z_grid = setconv_to_grid_nn(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )
        else:
            z_grid = setconv_to_grid(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )

        return x_grid, z_grid


class OOTGSetConv(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        grid_shape: Optional[torch.Tensor] = None,
        init_lengthscale: float = 0.1,
        coarsen_fn: Callable = coarsen_grid,
        dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
        use_nn: bool = False,
    ):
        super().__init__()

        # Construct lengthscales.
        init_lengthscale = torch.as_tensor(dims * [init_lengthscale], dtype=torch.float)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=True,
        )

        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.coarsen_fn = coarsen_fn
        self.dist_fn = dist_fn
        self.use_nn = use_nn

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

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

            x_grid = self.coarsen_fn(x_grid, output_grid=self.grid_shape)
            z_grid = None

        if self.use_nn:
            z_grid = setconv_to_grid_nn(
                x, z, x_grid, self.lengthscale, z_grid, dist_fn=self.dist_fn
            )
        else:
            z_grid = setconv_to_grid(
                x, z, x_grid, self.lengthscale, z_grid, dist_fn=self.dist_fn
            )

        return x_grid, z_grid


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def setconv_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    lengthscale: torch.Tensor,
    z_grid: Optional[torch.Tensor] = None,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
):
    x_grid_flat, flat_to_grid_fn = flatten_grid(x_grid)

    dists2 = dist_fn(x_grid_flat, x)
    pre_exp = torch.sum(dists2 / lengthscale.pow(2), dim=-1)
    weights = torch.exp(-0.5 * pre_exp)

    # Multiply context outputs by weights.
    # (batch_size, num_grid_points, embed_dim).
    z_grid_flat = weights @ z

    # Reshape grid.
    if z_grid is None:
        z_grid = 0

    z_grid = z_grid + flat_to_grid_fn(z_grid_flat)

    return z_grid


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def setconv_to_grid_nn(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    lengthscale: torch.Tensor,
    z_grid: Optional[torch.Tensor] = None,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
):
    # Get number of grid points.
    x_grid_flat, flat_to_grid_fn = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, k)
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    # Cumulative count.
    cumcount_idx = compute_cumcount_idx(nearest_idx)

    joint_grid_x, att_mask = construct_nearest_neighbour_matrix(
        nearest_idx, cumcount_idx, x, x_grid, num_grid_points
    )
    joint_grid_z, _ = construct_nearest_neighbour_matrix(
        nearest_idx, cumcount_idx, z, z_grid, num_grid_points
    )

    # Rearrange grids.
    x_grid_flat = einops.rearrange(x_grid_flat, "b m e -> (b m) 1 e")

    # Now we can do the setconv.
    dists2 = dist_fn(x_grid_flat, joint_grid_x)
    pre_exp = torch.sum(dists2 / lengthscale.pow(2), dim=-1)
    weights = torch.exp(-0.5 * pre_exp)

    # Apply mask to weights.
    weights = weights * att_mask
    z_grid_flat = weights @ joint_grid_z

    # Reshape output.
    z_grid_flat = einops.rearrange(z_grid_flat, "(b m) 1 e -> b m e", b=x.shape[0])

    # Reshape grid.
    if z_grid is None:
        z_grid = 0

    z_grid = z_grid + flat_to_grid_fn(z_grid_flat)

    return z_grid
