import time
from typing import Union

import einops
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tnp.networks.attention import MultiHeadCrossAttention
from tnp.networks.attention_layers import MultiHeadCrossAttentionLayer
from tnp.networks.grid_encoders.pt_grid_encoders import mhca_to_grid
from tnp.utils.grids import flatten_grid, nearest_gridded_neighbours


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


def old_mhca_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
    z0_grid: torch.Tensor,
    mhca: Union[MultiHeadCrossAttention, MultiHeadCrossAttentionLayer],
) -> torch.Tensor:
    num_batches, num_points = x.shape[:2]

    # Flatten grid.
    x_grid_flat, _ = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, n).
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    n_batch_idx = torch.arange(num_batches).unsqueeze(-1).repeat(1, num_points)
    n_range_idx = torch.arange(num_points).repeat(num_batches, 1)

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

    max_patch = cumcount_idx.amax() + 2

    # Create tensor with for each grid-token all nearest off-grid + itself in third axis.
    joint_grid = torch.full(
        (num_batches * num_grid_points, max_patch, z.shape[-1]),
        -torch.inf,
        device=z_grid.device,
    )

    # Add nearest off the grid points to each on_the_grid point.
    idx_shifter = torch.arange(
        0, num_batches * num_grid_points, num_grid_points, device=z.device
    ).repeat_interleave(num_points)
    joint_grid[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = z[
        n_batch_idx.flatten(), n_range_idx.flatten()
    ]

    # Add z_grid on at the end.
    z_grid_flat, _ = flatten_grid(z_grid)
    m_batch_idx_flat = torch.arange(num_batches).repeat_interleave(num_grid_points)
    m_range_idx_flat = torch.arange(num_grid_points).repeat(num_batches)
    joint_grid[torch.arange(num_batches * num_grid_points), -1] = z_grid_flat[
        m_batch_idx_flat, m_range_idx_flat
    ]

    # Mask out fake tokens and replace with value that won't overflow.
    att_mask = torch.ones(
        num_batches * num_grid_points, 1, max_patch, device=z.device, dtype=torch.bool
    )
    att_mask[(joint_grid.sum(-1) == -float("inf")).unsqueeze(1)] = False
    joint_grid = torch.masked_fill(joint_grid, joint_grid == -float("inf"), 0.0)

    # Rearrange latents.
    z0_grid_flat, flat_to_grid_fn = flatten_grid(z0_grid)
    z0_grid_flat = einops.rearrange(z0_grid_flat, "b m e -> (b m) 1 e")

    # Finally! Do the MHCA operation.
    z_grid = mhca(z0_grid_flat, joint_grid, mask=att_mask)

    # Reshape output.
    z_grid = einops.rearrange(z_grid, "(b s) 1 e -> b s e", b=num_batches)

    # Unflatten and return.
    z_grid = flat_to_grid_fn(z_grid)

    return z_grid


def generate_fake_data(grid_shape=(8,), emb_dim=128):
    x_off_grid = torch.linspace(-1, 1, 10)[None, :, None].repeat(1, 1, len(grid_shape))
    z_off_grid = torch.randn(x_off_grid.shape[:-1] + (emb_dim,))
    x_on_grid = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(-1, 1, steps=dim, dtype=torch.float)
                for dim in grid_shape
            ],
            indexing="ij",
        ),
        dim=-1,
    )[None, ...]
    z_on_grid = torch.randn(x_on_grid.shape[:-1] + (emb_dim,))
    assert x_on_grid.dim() == len(grid_shape) + 2
    return x_off_grid, z_off_grid, x_on_grid, z_on_grid


def generate_trace(grid_shape=(8), emb_dim=128):
    x_off_grid, z_off_grid, x_on_grid, z_on_grid = generate_fake_data(
        grid_shape, emb_dim
    )
    layer = MultiHeadCrossAttentionLayer(
        embed_dim=emb_dim, num_heads=8, head_dim=16, feedforward_dim=emb_dim
    )
    latents = torch.randn(x_on_grid.shape[0], *grid_shape, emb_dim)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            mhca_to_grid(x_off_grid, z_off_grid, x_on_grid, z_on_grid, latents, layer)

    prof.export_chrome_trace("trace.json")


def test_validity(grid_shape=(8), emb_dim=128):
    x_off_grid, z_off_grid, x_on_grid, z_on_grid = generate_fake_data(
        grid_shape, emb_dim
    )

    with torch.no_grad():
        layer = MultiHeadCrossAttentionLayer(
            embed_dim=emb_dim, num_heads=8, head_dim=16, feedforward_dim=emb_dim
        )

        latents = torch.randn(x_on_grid.shape[0], *grid_shape, emb_dim)
        out_current = mhca_to_grid(
            x_off_grid, z_off_grid, x_on_grid, z_on_grid, latents, layer
        )
        out_old = old_mhca_to_grid(
            x_off_grid, z_off_grid, x_on_grid, z_on_grid, latents, layer
        )
    assert torch.equal(out_current, out_old)


def speed_test(method, grid_shape=(8), emb_dim=128, avg=100):
    x_off_grid, z_off_grid, x_on_grid, z_on_grid = generate_fake_data(
        grid_shape, emb_dim
    )
    layer = MultiHeadCrossAttentionLayer(
        embed_dim=emb_dim, num_heads=8, head_dim=16, feedforward_dim=emb_dim
    )
    latents = torch.randn(x_on_grid.shape[0], *grid_shape, emb_dim)

    start = time.time()
    with torch.no_grad():
        for _ in range(avg):
            method(x_off_grid, z_off_grid, x_on_grid, z_on_grid, latents, layer)
    avg_time = (time.time() - start) / avg * 1000
    print(f"{method.__name__} took {avg_time} ms on average over {avg} runs.")


if __name__ == "__main__":
    test_validity(grid_shape=(120,))
    test_validity(grid_shape=(120, 240))
    speed_test(
        old_mhca_to_grid,
        grid_shape=(
            120,
            240,
        ),
    )

    speed_test(
        mhca_to_grid,
        grid_shape=(
            120,
            240,
        ),
    )
