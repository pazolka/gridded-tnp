from typing import Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.grids import avg_pool


class PatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        input_grid: Tuple[int, ...],
        output_grid: Tuple[int, ...],
        ignore_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        dim = len(output_grid)
        if dim == 1:
            conv = nn.Conv1d
        elif dim == 2:
            conv = nn.Conv2d
        elif dim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dim}.")

        self.input_grid = input_grid
        self.output_grid = output_grid
        self.ignore_dims = tuple(ignore_dims) if ignore_dims is not None else ()
        assert all(
            in_size % out_size == 0
            for in_size, out_size in zip(input_grid, output_grid)
        ), "Input grid must be divisible by output grid."

        # Compute the kernel size, stride and padding to achieve the output grid shape.
        kernel_size = tuple(
            in_size // out_size for in_size, out_size in zip(input_grid, output_grid)
        )
        stride = kernel_size

        self.conv = conv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

    @check_shapes(
        "x: [m, ..., d]",
        "return: [m, ..., d]",
    )
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        grid_shape = torch.as_tensor(x.shape[1:-1])
        check_grid_shape = grid_shape.clone()

        # Convert self.ignore_dims to positive indexing.
        ignore_dims = tuple(
            dim + len(grid_shape) if dim < 0 else dim for dim in self.ignore_dims
        )

        for dim in ignore_dims:
            check_grid_shape = torch.cat([grid_shape[:dim], grid_shape[dim + 1 :]])

        assert torch.all(
            check_grid_shape == torch.as_tensor(self.input_grid)
        ), "Grid shape does not match input grid."

        # Merge ignored dimension with batch dimension.
        if self.ignore_dims is not None:
            ignore_dims_vars = dict(
                zip(
                    [f"d{i}" for i in ignore_dims],
                    [grid_shape[i] for i in ignore_dims],
                )
            )
            original_pattern = (
                "m " + " ".join([f"d{i}" for i in range(len(x.shape[1:-1]))]) + " d"
            )
            merged_pattern = (
                "(m "
                + " ".join([f"d{i}" for i in ignore_dims])
                + ") "
                + " ".join(
                    [f"d{i}" for i in range(len(x.shape[1:-1])) if i not in ignore_dims]
                )
                + " d"
            )
            x = einops.rearrange(x, original_pattern + " -> " + merged_pattern)

        # Perform convolution
        x = x.movedim(-1, 1)
        x = self.conv(x)
        x = x.movedim(1, -1)

        # Unmerge ignored dimensions
        if self.ignore_dims is not None:
            x = einops.rearrange(
                x, merged_pattern + " -> " + original_pattern, **ignore_dims_vars
            )

        grid_shape = torch.as_tensor(x.shape[1:-1])
        check_grid_shape = grid_shape.clone()
        for dim in ignore_dims:
            check_grid_shape = torch.cat([grid_shape[:dim], grid_shape[dim + 1 :]])

        assert torch.all(
            check_grid_shape == torch.as_tensor(self.output_grid)
        ), "Grid shape does not match output grid."

        return x

    @check_shapes(
        "x: [m, ..., d]",
        "return: [m, ..., d]",
    )
    def average_input_locations(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        grid_shape = torch.as_tensor(x.shape[1:-1])
        dim = len(grid_shape)
        coarse_x = avg_pool(
            dim=dim,
            input=x.movedim(-1, 1),
            kernel_size=self.conv.kernel_size,
            stride=self.conv.kernel_size,
        )
        return coarse_x.movedim(1, -1)
