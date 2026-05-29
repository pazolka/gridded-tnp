import math

import einops
import pytest
import torch

from tnp.utils.grids import construct_grid, nearest_gridded_neighbours


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_nearest_gridded_neighbours(ndim: int):
    """
    Test the code works with various dimensions of the grid
    """
    # Set up a ndim-dimensional grid
    grid_range = tuple(((-2.0, 2.0),) * ndim)

    points_per_dim = tuple((16 for _ in range(ndim)))
    batch_size = 2
    nt = 100
    top_k_ctot = 25
    roll_dims = (-1,)

    # Construct grid.
    x_grid = construct_grid(grid_range, points_per_dim)
    grid_shape = x_grid.shape[:-1]
    grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
    x_grid = einops.repeat(
        x_grid, grid_str + " d -> b " + grid_str + " d", b=batch_size
    )

    # Get target points.
    xt = torch.randn(batch_size, nt, len(grid_shape))

    nearest_idx, mask = nearest_gridded_neighbours(
        xt,
        x_grid,
        k=top_k_ctot,
        roll_dims=roll_dims,
    )

    # Get the number of nearest neighbours
    dim_x = xt.shape[-1]
    num_grid_spacings = math.ceil(top_k_ctot ** (1 / dim_x))
    actual_num_neigh = (
        len(
            torch.arange(
                -num_grid_spacings // 2 + num_grid_spacings % 2,
                num_grid_spacings // 2 + 1,
            )
        )
        ** dim_x
    )
    assert nearest_idx.shape == (batch_size, nt, actual_num_neigh)
    assert mask.shape == (batch_size, nt, actual_num_neigh)


@pytest.mark.parametrize("roll_dims", [None, (-1,)])
def test_regular_grid_2D(roll_dims):
    """
    Test code gives correct output for regular grid in 2D (same points_per_dim).
    There is a different grid for each batch.
    """
    k = 9

    # Grid 1
    grid_range1 = [[-4.0, 4.0], [-4.0, 4.0]]
    points_per_dim = [9, 9]
    x_grid = construct_grid(grid_range1, points_per_dim)

    # Grid 2
    grid_range2 = [[-8.0, 8.0], [-8.0, 8.0]]
    x_grid2 = construct_grid(grid_range2, points_per_dim)

    # Concat two grids along batch dimension
    x_grid = torch.concat([x_grid[None, ...], x_grid2[None, ...]])

    # Get target locations.
    x = torch.Tensor(
        [
            [[-3.9, -3.97], [1.13, 3.22], [-0.41, 2.7]],
            [[6.7, -7.6], [-3.6, -6.8], [0.6, 0.00004]],
        ]
    )

    # Do nearest gridded neighbours.
    nearest_idx, mask = nearest_gridded_neighbours(
        x,
        x_grid,
        k=k,
        roll_dims=roll_dims,
    )

    if roll_dims is None:
        true_nearest_idx = torch.Tensor(
            [
                [
                    [0, 0, 1, 0, 0, 1, 9, 9, 10],
                    [42, 43, 44, 51, 52, 53, 60, 61, 62],
                    [33, 34, 35, 42, 43, 44, 51, 52, 53],
                ],
                [
                    [54, 54, 55, 63, 63, 64, 72, 72, 73],
                    [9, 10, 11, 18, 19, 20, 27, 28, 29],
                    [30, 31, 32, 39, 40, 41, 48, 49, 50],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [True, False, True, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
                [
                    [True, False, True, True, False, True, True, False, True],
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
            ]
        )
    elif roll_dims == (-1,):
        true_nearest_idx = torch.Tensor(
            [
                [
                    [7, 0, 1, 7, 0, 1, 16, 9, 10],
                    [42, 43, 44, 51, 52, 53, 60, 61, 62],
                    [33, 34, 35, 42, 43, 44, 51, 52, 53],
                ],
                [
                    [61, 54, 55, 70, 63, 64, 79, 72, 73],
                    [9, 10, 11, 18, 19, 20, 27, 28, 29],
                    [30, 31, 32, 39, 40, 41, 48, 49, 50],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [True, True, True, False, False, False, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
                [
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
            ]
        )
    else:
        raise ValueError(f"We did not consider roll_dims {roll_dims}")

    assert torch.equal(nearest_idx, true_nearest_idx)
    assert torch.equal(mask, true_mask)


@pytest.mark.parametrize("roll_dims", [None, (-1,)])
def test_irregular_grid_2D(roll_dims):
    """
    Test code gives correct output for irregular grid in 2D (different points_per_dim).
    There is a different grid for each batch.
    """
    k = 9

    # Grid 1
    grid_range1 = [[-2.0, 2.0], [-4.0, 4.0]]
    points_per_dim = [5, 9]
    x_grid = construct_grid(grid_range1, points_per_dim)

    # Grid 2
    grid_range2 = [[-2.0, 2.0], [-8.0, 8.0]]
    x_grid2 = construct_grid(grid_range2, points_per_dim)

    # Concat two grids along batch dimension
    x_grid = torch.concat([x_grid[None, ...], x_grid2[None, ...]])

    # Get target locations.
    x = torch.Tensor(
        [
            [[-1.2, -3.3], [1.9, 3.8], [-0.41, 2.7]],
            [[0.03, -7.6], [-1.6, -6.8], [0.6, 5.6]],
        ]
    )

    # Do nearest gridded neighbours.
    nearest_idx, mask = nearest_gridded_neighbours(
        x,
        x_grid,
        k=k,
        roll_dims=roll_dims,
    )

    if roll_dims is None:
        true_nearest_idx = torch.Tensor(
            [
                [
                    [0, 1, 2, 9, 10, 11, 18, 19, 20],
                    [34, 35, 35, 43, 44, 44, 43, 44, 44],
                    [15, 16, 17, 24, 25, 26, 33, 34, 35],
                ],
                [
                    [9, 9, 10, 18, 18, 19, 27, 27, 28],
                    [0, 1, 2, 0, 1, 2, 9, 10, 11],
                    [24, 25, 26, 33, 34, 35, 42, 43, 44],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, False, True, True, False, False, False, False],
                    [True, True, True, True, True, True, True, True, True],
                ],
                [
                    [True, False, True, True, False, True, True, False, True],
                    [True, True, True, False, False, False, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
            ]
        )
    elif roll_dims == (-1,):
        true_nearest_idx = torch.Tensor(
            [
                [
                    [0, 1, 2, 9, 10, 11, 18, 19, 20],
                    [34, 35, 28, 43, 44, 37, 43, 44, 37],
                    [15, 16, 17, 24, 25, 26, 33, 34, 35],
                ],
                [
                    [16, 9, 10, 25, 18, 19, 34, 27, 28],
                    [0, 1, 2, 0, 1, 2, 9, 10, 11],
                    [24, 25, 26, 33, 34, 35, 42, 43, 44],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, True, True, True, True],
                ],
                [
                    [True, True, True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, True, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ],
            ]
        )
    else:
        raise ValueError(f"We did not consider roll_dims {roll_dims}")

    assert torch.equal(nearest_idx, true_nearest_idx)
    assert torch.equal(mask, true_mask)


@pytest.mark.parametrize("roll_dims", [None, (-2,)])
def test_irregular_grid_3D(roll_dims):
    """
    Test code gives correct output for irregular grid in 3D (different points_per_dim).
    There is a different grid for each batch.
    """
    k = 27

    # Grid 1
    grid_range1 = [[-2.0, 2.0], [-4.0, 4.0], [-1.0, 1.0]]
    points_per_dim = [5, 9, 3]
    x_grid = construct_grid(grid_range1, points_per_dim)

    # Grid 2
    grid_range2 = [[-4.0, 4.0], [-4.0, 4.0], [-2.0, 2.0]]
    x_grid2 = construct_grid(grid_range2, points_per_dim)

    # Concat two grids along batch dimension
    x_grid = torch.concat([x_grid[None, ...], x_grid2[None, ...]])

    # Get target locations.
    x = torch.Tensor(
        [[[1.3, 2.1, -0.03], [-1.9, -3.7, 0.3]], [[-3.8, 3.9, 1.6], [2.3, 0.5, -1.4]]]
    )

    # Do nearest gridded neighbours.
    nearest_idx, mask = nearest_gridded_neighbours(
        x,
        x_grid,
        k=k,
        roll_dims=roll_dims,
    )

    if roll_dims is None:
        true_nearest_idx = torch.Tensor(
            [
                [
                    [
                        69,
                        70,
                        71,
                        72,
                        73,
                        74,
                        75,
                        76,
                        77,
                        96,
                        97,
                        98,
                        99,
                        100,
                        101,
                        102,
                        103,
                        104,
                        123,
                        124,
                        125,
                        126,
                        127,
                        128,
                        129,
                        130,
                        131,
                    ],
                    [
                        0,
                        1,
                        2,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        0,
                        1,
                        2,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        27,
                        28,
                        29,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                    ],
                ],
                [
                    [
                        22,
                        23,
                        23,
                        25,
                        26,
                        26,
                        25,
                        26,
                        26,
                        22,
                        23,
                        23,
                        25,
                        26,
                        26,
                        25,
                        26,
                        26,
                        49,
                        50,
                        50,
                        52,
                        53,
                        53,
                        52,
                        53,
                        53,
                    ],
                    [
                        66,
                        66,
                        67,
                        69,
                        69,
                        70,
                        72,
                        72,
                        73,
                        93,
                        93,
                        94,
                        96,
                        96,
                        97,
                        99,
                        99,
                        100,
                        120,
                        120,
                        121,
                        123,
                        123,
                        124,
                        126,
                        126,
                        127,
                    ],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                    ],
                ],
                [
                    [
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                    ],
                ],
            ]
        )
    elif roll_dims == (-2,):
        true_nearest_idx = torch.Tensor(
            [
                [
                    [
                        69,
                        70,
                        71,
                        72,
                        73,
                        74,
                        75,
                        76,
                        77,
                        96,
                        97,
                        98,
                        99,
                        100,
                        101,
                        102,
                        103,
                        104,
                        123,
                        124,
                        125,
                        126,
                        127,
                        128,
                        129,
                        130,
                        131,
                    ],
                    [
                        21,
                        22,
                        23,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        21,
                        22,
                        23,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        48,
                        49,
                        50,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                    ],
                ],
                [
                    [
                        22,
                        23,
                        23,
                        25,
                        26,
                        26,
                        4,
                        5,
                        5,
                        22,
                        23,
                        23,
                        25,
                        26,
                        26,
                        4,
                        5,
                        5,
                        49,
                        50,
                        50,
                        52,
                        53,
                        53,
                        31,
                        32,
                        32,
                    ],
                    [
                        66,
                        66,
                        67,
                        69,
                        69,
                        70,
                        72,
                        72,
                        73,
                        93,
                        93,
                        94,
                        96,
                        96,
                        97,
                        99,
                        99,
                        100,
                        120,
                        120,
                        121,
                        123,
                        123,
                        124,
                        126,
                        126,
                        127,
                    ],
                ],
            ]
        )
        true_mask = torch.Tensor(
            [
                [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                ],
                [
                    [
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                    ],
                    [
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                        True,
                        False,
                        True,
                    ],
                ],
            ]
        )
    else:
        raise ValueError(f"We did not consider roll_dims {roll_dims}")

    assert torch.equal(nearest_idx, true_nearest_idx)
    assert torch.equal(mask, true_mask)
