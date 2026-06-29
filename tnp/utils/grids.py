import itertools
import math
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn


def flatten_grid(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    grid_shape = x.shape[1:-1]
    n_strings = [f"n{i}" for i in range(len(grid_shape))]
    grid_pattern = f"b {' '.join(n_strings)} e"
    flat_pattern = f"b ({' '.join(n_strings)}) e"
    grid_to_flat = grid_pattern + " -> " + flat_pattern
    flat_to_grid = flat_pattern + " -> " + grid_pattern
    reshape_vars = dict(zip(n_strings, grid_shape))

    def grid_to_flat_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, grid_to_flat)

    def flat_to_grid_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, flat_to_grid, **reshape_vars)

    return grid_to_flat_fn(x), flat_to_grid_fn


def coarsen_grid(
    grid: torch.Tensor,
    output_grid: Optional[Tuple[int, ...]] = None,
    coarsen_factors: Optional[Tuple[int, ...]] = None,
    ignore_dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:

    # dim = grid.ndim - 2
    assert (
        output_grid is not None or coarsen_factors is not None
    ), "Must specify either output_grid or coarsen_factors."

    grid_shape = torch.as_tensor(grid.shape[1:-1])
    if ignore_dims is not None:
        # Make ignore_dims positive indexes.
        ignore_dims = tuple(
            dim + len(grid_shape) if dim < 0 else dim for dim in ignore_dims
        )
        # Modify output_grid to account for ignored dimensions.
        if output_grid is not None:
            output_grid = tuple(
                output_grid[i] for i in range(len(output_grid)) if i not in ignore_dims
            )
        if coarsen_factors is not None:
            coarsen_factors = tuple(
                coarsen_factors[i]
                for i in range(len(coarsen_factors))
                if i not in ignore_dims
            )

        # Merge ignored dimension with batch dimension.
        ignore_dims_vars = dict(
            zip(
                [f"d{i}" for i in ignore_dims],
                [grid_shape[i] for i in ignore_dims],
            )
        )
        original_pattern = (
            "m " + " ".join([f"d{i}" for i in range(len(grid.shape[1:-1]))]) + " d"
        )
        merged_pattern = (
            "(m "
            + " ".join([f"d{i}" for i in ignore_dims])
            + ") "
            + " ".join(
                [f"d{i}" for i in range(len(grid.shape[1:-1])) if i not in ignore_dims]
            )
            + " d"
        )
        # So original grid is not modified.
        grid_ = einops.rearrange(grid, original_pattern + " -> " + merged_pattern)
    else:
        grid_ = grid

    # Coarsen inputs using interpolation.
    coarse_grid = torch.nn.functional.interpolate(
        grid_.movedim(-1, 1),
        size=output_grid,
        scale_factor=coarsen_factors,
        mode="bilinear",
        align_corners=False,
    )
    coarse_grid = coarse_grid.movedim(1, -1)

    # Unmerge ignored dimensions.
    if ignore_dims is not None:
        coarse_grid = einops.rearrange(
            coarse_grid, merged_pattern + " -> " + original_pattern, **ignore_dims_vars
        )

    return coarse_grid


def construct_grid(
    grid_range: Tuple[Tuple[float, float], ...],
    points_per_dim: Tuple[int, ...],
) -> torch.Tensor:
    grid_range_ = torch.as_tensor(grid_range)
    grid = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(
                    grid_range_[i, 0],
                    grid_range_[i, 1],
                    steps=points_per_dim[i],
                    dtype=torch.float,
                )
                for i in range(len(grid_range))
            ]
        ),
        dim=-1,
    )

    return grid


@check_shapes(
    "x: [m, n, dx]",
    "x_grid: [m, ..., dx]",
    "return[0]: [m, n, k]",
    "return[1]: [m, n, k]",
)
def nearest_gridded_neighbours(
    x: torch.Tensor,
    x_grid: torch.Tensor,
    k: int = 1,
    roll_dims: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_shape = torch.as_tensor(x_grid.shape[1:-1], device=x.device)
    x_grid_flat, _ = flatten_grid(x_grid)

    # Get number of neighbors along each dimension.
    dim_x = x.shape[-1]
    num_grid_spacings = math.ceil(k ** (1 / dim_x))

    # Set roll_dims to the actual index if they are specified as (-x, )
    num_dims = len(grid_shape)
    if roll_dims is not None:
        roll_dims = tuple(roll_dim % num_dims for roll_dim in roll_dims)

    # Quick calculation of nearest grid neighbour.
    x_grid_min = x_grid_flat.amin(dim=1)
    x_grid_max = x_grid_flat.amax(dim=1)
    x_grid_spacing = (x_grid_max - x_grid_min) / (grid_shape - 1)

    nearest_multi_idx = (
        x - x_grid_min[:, None, :] + x_grid_spacing[:, None, :] / 2
    ) // x_grid_spacing[:, None, :]

    # Generate a base grid for combinations of grid_spacing offsets from main neighbor.
    base_grid = torch.tensor(
        list(
            itertools.product(
                torch.arange(
                    -num_grid_spacings // 2 + num_grid_spacings % 2,
                    num_grid_spacings // 2 + 1,
                ),
                repeat=dim_x,
            )
        ),
        device=x.device,
    ).float()

    # Reshape and expand the base grid
    base_grid = base_grid.view(1, 1, -1, dim_x).expand(
        *nearest_multi_idx.shape[:-1], -1, -1
    )
    # Expand the indices of nearest neighbors to account for more than 1.
    nearest_multi_idx_expanded = nearest_multi_idx.unsqueeze(2).expand(
        -1, -1, (num_grid_spacings + 1 - num_grid_spacings % 2) ** dim_x, -1
    )
    # Generate all combinations by adding the offsets to the main neighbor.
    nearest_multi_idx = nearest_multi_idx_expanded + base_grid

    # If not rolling_dims, do not allow neighbors to go off-grid.
    # Otherwise, roll the grid along the specified dimension.
    if roll_dims is None:
        nearest_multi_idx = torch.max(
            torch.min(nearest_multi_idx, grid_shape - 1), torch.zeros_like(grid_shape)
        ).squeeze(-2)
    else:
        nearest_multi_idx = torch.cat(
            [
                (
                    torch.max(
                        torch.min(nearest_multi_idx[..., i], grid_shape[i] - 1),
                        torch.tensor(0),
                    ).unsqueeze(-1)
                    if (i not in roll_dims)
                    # else (nearest_multi_idx[..., i] % grid_shape[i]).unsqueeze(-1)
                    else (
                        (nearest_multi_idx[..., i] % grid_shape[i])
                        + (nearest_multi_idx[..., i] // grid_shape[i])
                    ).unsqueeze(-1)
                )
                for i in range(num_dims)
            ],
            dim=-1,
        ).squeeze(-2)

    # Get strides.
    strides = torch.flip(
        torch.cumprod(
            torch.cat(
                (
                    torch.ones((1,), device=grid_shape.device),
                    torch.flip(grid_shape, dims=(0,)),
                ),
                dim=0,
            ),
            dim=0,
        )[:-1],
        dims=(0,),
    )

    # (batch_size, nt, num_neighbors).
    if k == 1:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        )
    else:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        ).squeeze(-1)

    if k != 1:
        # Get mask for MHCA.
        mask = torch.ones_like(nearest_idx, dtype=torch.bool)

        # Sort nearest_idx.
        sorted_nearest_idx, indices = torch.sort(nearest_idx, dim=-1, stable=True)

        # Find first occurence where consecutive elements are different.
        first_occurrence = torch.ones_like(sorted_nearest_idx, dtype=torch.bool)
        first_occurrence[..., 1:] = (
            sorted_nearest_idx[..., 1:] != sorted_nearest_idx[..., :-1]
        )

        # Back to the original shape.
        original_indices = torch.argsort(indices, dim=-1)
        mask = torch.gather(first_occurrence, dim=-1, index=original_indices)
    else:
        mask = None

    return nearest_idx, mask


def complex_log(float_input: torch.Tensor, eps=1e-6) -> torch.ComplexType:
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi

    return torch.complex(real, imag)


def associative_scan(
    values: torch.Tensor, coeffs: torch.Tensor, dim: int
) -> torch.Tensor:
    log_values = complex_log(values.float())
    log_coeffs = complex_log(coeffs.float())
    a_star = torch.cumsum(log_coeffs, dim=dim)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
    log_x = a_star + log_x0_plus_b_star

    return torch.exp(log_x).real


def avg_pool(dim: int, **kwargs) -> torch.Tensor:
    func = (
        nn.functional.avg_pool1d,
        nn.functional.avg_pool2d,
        nn.functional.avg_pool3d,
    )[dim - 1]
    return func(**kwargs)


class UpSample(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_grid: Tuple[int, ...],
        output_grid: Tuple[int, ...],
    ):
        super().__init__()

        # How many times to upsample in each dimension.
        self.upsample_factors = tuple(
            math.ceil(o / i) for i, o in zip(input_grid, output_grid)
        )
        self.upsample_factor = math.prod(self.upsample_factors)

        self.expand_input_dim = nn.Linear(
            embed_dim, embed_dim * self.upsample_factor, bias=False
        )
        self.output_mixing = nn.Linear(embed_dim, embed_dim, bias=False)
        self.input_grid = input_grid
        self.output_grid = output_grid

    @check_shapes("x: [m, ..., d]", "return: [m, ..., d]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand_input_dim(x)

        # Not convinced this reshaping is correct...
        # e.g. [i1, i2, ...,].
        input_var_names = ["i" + str(i) for i in range(len(self.input_grid))]

        # e.g. [m1, m2, ...].
        upsample_var_names = ["m" + str(m) for m in range(len(self.upsample_factors))]
        upsample_vars = dict(zip(upsample_var_names, self.upsample_factors))

        # e.g. [o1, o2, ...].
        output_var_names = [
            "(i" + str(i) + " m" + str(i) + ")" for i in range(len(self.input_grid))
        ]

        # Now define einops strings.
        input_str = (
            "m "
            + " ".join(input_var_names)
            + " ("
            + " ".join(upsample_var_names)
            + " dout)"
        )
        output_str = "m " + " ".join(output_var_names) + " dout"

        # e.g. (m, i1, i2, i3, m1*m2*m3*dout) -> (m, i1*m1, i2*m2, i3*m3, dout).
        x = einops.rearrange(x, input_str + " -> " + output_str, **upsample_vars)

        # Now pad to correct output grid shape.
        tot_padding = [
            i * c - o
            for i, c, o in zip(self.input_grid, self.upsample_factors, self.output_grid)
        ]
        slices = tuple(
            slice(tp // 2, i * c - (tp - tp // 2))
            for tp, i, c in zip(tot_padding, self.input_grid, self.upsample_factors)
        )
        x = x[(slice(None),) + slices + (slice(None),)]
        x = self.output_mixing(x)

        return x


class DownSample(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_grid: Tuple[int, ...],
        output_grid: Tuple[int, ...],
    ):
        super().__init__()

        # How many times to downsample in each dimension.
        self.downsample_factors = tuple(
            math.ceil(i / o) for i, o in zip(input_grid, output_grid)
        )
        self.downsample_factor = math.prod(self.downsample_factors)

        self.output_projection = nn.Linear(
            self.downsample_factor * embed_dim, embed_dim, bias=False
        )
        self.input_grid = input_grid
        self.output_grid = output_grid

    @check_shapes("x: [m, ..., d]", "return: [m, ..., d]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding to the input to facilitate downsampling.
        tot_padding = [
            o * c - i
            for i, c, o in zip(
                self.input_grid, self.downsample_factors, self.output_grid
            )
        ]

        # So that the last (channel) dimension is not padded.
        padding_tuples = [0, 0]
        padding_tuples = padding_tuples + [
            pad for tp in tot_padding[::-1] for pad in [tp // 2, tp - tp // 2]
        ]

        # Could be wrong...
        x = nn.functional.pad(  # pylint: disable=not-callable
            x, padding_tuples, mode="constant", value=0
        )

        # Not convinced this reshaping is correct...
        # e.g. [i1, i2, ...].
        input_var_names = [
            "(o" + str(i) + " m" + str(i) + ")" for i in range(len(self.input_grid))
        ]

        # e.g. [m1, m2, ...].
        downsample_var_names = [
            "m" + str(i) for i in range(len(self.downsample_factors))
        ]
        downsample_vars = dict(zip(downsample_var_names, self.downsample_factors))

        # e.g. [o1, o2, ...].
        output_var_names = ["o" + str(i) for i in range(len(self.output_grid))]

        input_str = "m " + " ".join(input_var_names) + " din"
        output_str = (
            "m "
            + " ".join(output_var_names)
            + " ("
            + " ".join(downsample_var_names)
            + " din)"
        )

        # e.g. (m, o1*m1, o2*m2, o3*m3, din) -> (m, o1, o2, o3, m1*m2*m3*din).
        x = einops.rearrange(x, input_str + " -> " + output_str, **downsample_vars)

        x = self.output_projection(x)

        return x


@check_shapes("nearest_idx: [m, n]", "return: [m, d]")
def compute_cumcount_idx(nearest_idx: torch.Tensor) -> torch.Tensor:

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

    return cumcount_idx


@check_shapes(
    "nearest_idx: [m, n]",
    "cumcount_idx: [m, n]",
    "x: [m, n, d]",
    "x_grid: [m, ..., d]",
    "return[0]: [mngrid, npatch, d]",
    "return[1]: [mngrid, 1, npatch]",
)
def construct_nearest_neighbour_matrix(
    nearest_idx: torch.Tensor,
    cumcount_idx: torch.Tensor,
    x: torch.Tensor,
    x_grid: Optional[torch.Tensor] = None,
    num_grid_points: Optional[int] = None,
    concatenate_x_grid: bool = False,
):
    num_batches, num_points = x.shape[:2]

    # Flatten grid.
    if x_grid is not None:
        x_grid_flat, _ = flatten_grid(x_grid)
    else:
        assert concatenate_x_grid == False

    # For constructing tensors later on.
    n_batch_idx = torch.arange(num_batches, device=x.device).unsqueeze(-1).repeat(1, num_points)
    n_range_idx = torch.arange(num_points, device=x.device).repeat(num_batches, 1)
    m_batch_idx_flat = torch.arange(num_batches, device=x.device).repeat_interleave(num_grid_points)
    m_range_idx_flat = torch.arange(num_grid_points, device=x.device).repeat(num_batches)

    # Max patch size.
    max_patch = cumcount_idx.amax() + (2 if concatenate_x_grid else 1)

    # Create tensor with for each grid-token all nearest off-grid + itself in third axis.
    joint_grid = torch.full(
        (num_batches * num_grid_points, max_patch, x.shape[-1]),
        -torch.inf,
        device=x.device,
    )

    # Add nearest off the grid points to each on_the_grid point.
    idx_shifter = torch.arange(
        0, num_batches * num_grid_points, num_grid_points, device=x.device
    ).repeat_interleave(num_points)
    joint_grid[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = x[
        n_batch_idx.flatten(), n_range_idx.flatten()
    ]

    if concatenate_x_grid:
        joint_grid[torch.arange(num_batches * num_grid_points, device=x.device), -1] = x_grid_flat[
            m_batch_idx_flat, m_range_idx_flat
        ]

    # Mask out fake values and replace with value that won't overflow.
    att_mask = torch.ones(
        num_batches * num_grid_points, 1, max_patch, device=x.device, dtype=torch.bool
    )
    att_mask[(joint_grid.sum(-1) == -float("inf")).unsqueeze(1)] = False
    joint_grid = torch.masked_fill(joint_grid, joint_grid == -float("inf"), 0.0)

    return joint_grid, att_mask
