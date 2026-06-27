import einops
import torch

from tnp.networks.attention_layers import MultiHeadCrossAttentionLayer
from tnp.networks.grid_encoders.pt_grid_encoders import PseudoTokenGridEncoder
from tnp.utils.grids import flatten_grid, nearest_gridded_neighbours


def test_grid_encoder(dim: int = 1, emb_dim=128):
    x = torch.randn(8, 100, dim)
    z = torch.randn(x.shape[:-1] + (emb_dim,))

    with torch.no_grad():
        layer = MultiHeadCrossAttentionLayer(
            embed_dim=emb_dim, num_heads=8, head_dim=16, feedforward_dim=emb_dim
        )

        encoder = PseudoTokenGridEncoder(
            embed_dim=emb_dim,
            mhca_layer=layer,
            grid_range=tuple(((-1.0, 1.0),) * dim),
            points_per_dim=tuple([16]) * dim,
        )

        x_grid, z_grid = encoder(
            x,
            z,
        )

        x_grid_old, z_grid_old = old_grid_encoder(
            x,
            z,
            encoder.mhca_layer,
            encoder.grid,
            encoder.latents,
        )

    assert torch.equal(x_grid, x_grid_old)
    assert torch.equal(z_grid, z_grid_old)


def old_grid_encoder(
    x: torch.Tensor,
    z: torch.Tensor,
    mhca: MultiHeadCrossAttentionLayer,
    grid: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    # ---------------------- calculate output manually using old implementation -----------------------------------

    grid_shape = grid.shape[:-1]
    grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
    x_grid = einops.repeat(grid, grid_str + " d -> b " + grid_str + " d", b=x.shape[0])
    z_grid = einops.repeat(
        latents, grid_str + " e -> b " + grid_str + " e", b=z.shape[0]
    )

    num_batches, num_points = x.shape[:2]

    # Flatten grid.
    x_grid_flat, _ = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, n).
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    n_batch_idx = torch.arange(num_batches).unsqueeze(-1).repeat(1, num_points)
    n_range_idx = torch.arange(num_points).repeat(num_batches, 1)
    m_batch_idx = torch.arange(num_batches).unsqueeze(-1).repeat(1, num_grid_points)
    m_range_idx = torch.arange(num_grid_points).repeat(num_batches, 1)

    # Links each point to its closest grid point
    # TODO: requires SO much memory.
    nearest_mask = torch.zeros(
        num_batches, num_points, num_grid_points, device=x.device, dtype=torch.bool
    )
    nearest_mask[n_batch_idx, n_range_idx, nearest_idx] = True

    cumcount_idx = (nearest_mask.cumsum(dim=1) - 1)[
        n_batch_idx, n_range_idx, nearest_idx
    ]

    # Maximum nearest neigbours.
    max_patch = nearest_mask.sum(dim=1).amax() + 1

    # Create tensor with for, each pseudo-token, all nearest tokens.
    joint_grid = torch.full(
        (num_batches, num_grid_points, max_patch, z.shape[-1]),
        -float("inf"),
        device=z.device,
    )

    # Add nearest tokens for each pseudo-token.
    joint_grid[n_batch_idx, nearest_idx, cumcount_idx] = z

    # Flatten z_grid.
    z_grid_flat, _ = flatten_grid(z_grid)

    # Add z_grid on at the end.
    joint_grid[m_batch_idx, m_range_idx, -1] = z_grid_flat

    joint_grid = einops.rearrange(joint_grid, "b m p e -> (b m) p e")

    # Mask out fake tokens and replace with value that won't overflow.
    att_mask = torch.ones(
        num_batches * num_grid_points, 1, max_patch, device=z.device, dtype=torch.bool
    )
    att_mask[(joint_grid.sum(-1) == -float("inf")).unsqueeze(1)] = False
    joint_grid = torch.masked_fill(joint_grid, joint_grid == -float("inf"), 0.0)

    # Rearrange latents.
    z0_grid_flat, flat_to_grid_fn = flatten_grid(z_grid)
    z0_grid_flat = einops.rearrange(z0_grid_flat, "b m e -> (b m) 1 e")

    # Finally! Do the MHCA operation.
    z_grid = mhca(z0_grid_flat, joint_grid, mask=att_mask)

    # Reshape output.
    z_grid = einops.rearrange(z_grid, "(b s) 1 e -> b s e", b=num_batches)

    # Unflatten and return.
    z_grid = flat_to_grid_fn(z_grid)

    return x_grid, z_grid


if __name__ == "__main__":
    test_grid_encoder(dim=1)
    test_grid_encoder(dim=2)
    test_grid_encoder(dim=3)
