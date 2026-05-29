from abc import ABC
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from tnp.networks.attention_layers import MultiHeadCrossAttentionLayer
from tnp.utils.distances import sq_dist
from tnp.utils.grids import flatten_grid, nearest_gridded_neighbours


class GridDecoder(nn.Module, ABC):

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "return: [m, nt, d]",
    )
    def forward(
        self, xc: torch.Tensor, zc: torch.Tensor, xt: torch.Tensor, zt: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class MHCAGridDecoder(GridDecoder):
    def __init__(
        self,
        mhca_layer: MultiHeadCrossAttentionLayer,
        top_k_ctot: Optional[int] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.mhca_layer = mhca_layer
        self.top_k_ctot = top_k_ctot
        self.roll_dims = roll_dims

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "return: [m, nt, d]",
    )
    def forward(
        self, xc: torch.Tensor, zc: torch.Tensor, xt: torch.Tensor, zt: torch.Tensor
    ) -> torch.Tensor:

        zc_flat, _ = flatten_grid(zc)
        if self.top_k_ctot is not None:
            num_batches, nt = zt.shape[:2]

            # (batch_size, n, k).
            nearest_idx, mask = nearest_gridded_neighbours(
                xt,
                xc,
                k=self.top_k_ctot,
                roll_dims=self.roll_dims,
            )

            batch_idx = (
                torch.arange(num_batches, device=zt.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, nt, nearest_idx.shape[-1])
            )

            nearest_zc = zc_flat[
                batch_idx,
                nearest_idx,
            ]

            # Rearrange tokens.
            zt = einops.rearrange(zt, "b n e -> (b n) 1 e")
            nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")
            mask = einops.rearrange(mask, "b n e -> (b n) 1 e")

            # Do the MHCA operation, reshape and return.
            zt = self.mhca_layer(zt, nearest_zc, mask=mask)

            zt = einops.rearrange(zt, "(b n) 1 e -> b n e", b=num_batches)
        else:
            zt = self.mhca_layer(zt, zc_flat)

        return zt


class SetConvGridDecoder(GridDecoder):
    def __init__(
        self,
        dim: int,
        init_lengthscale: float = 0.1,
        train_lengthscale: bool = True,
        dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
        top_k_ctot: Optional[int] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.lengthscale_param = nn.Parameter(
            (torch.tensor([init_lengthscale] * dim).exp() - 1).log(),
            requires_grad=train_lengthscale,
        )
        self.dist_fn = dist_fn
        self.top_k_ctot = top_k_ctot
        self.roll_dims = roll_dims

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        zc: torch.Tensor,
        xt: torch.Tensor,
        zt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Flatten grids
        xc_flat, _ = flatten_grid(xc)  # shape (batch_size, num_grid_points, Dx)
        zc_flat, _ = flatten_grid(zc)  # shape (batch_size, num_grid_points, Dz)

        if self.top_k_ctot is not None:
            num_batches, nt = xt.shape[:2]

            nearest_idx, mask = nearest_gridded_neighbours(
                xt,
                xc,
                k=self.top_k_ctot,
                roll_dims=self.roll_dims,
            )
            batch_idx = (
                torch.arange(num_batches, device=xt.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, nt, nearest_idx.shape[-1])
            )

            nearest_zc = zc_flat[
                batch_idx,
                nearest_idx,
            ]
            nearest_xc = xc_flat[
                batch_idx,
                nearest_idx,
            ]

            # Rearrange tokens.
            nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")
            nearest_xc = einops.rearrange(nearest_xc, "b n k e -> (b n) k e")
            mask = einops.rearrange(mask, "b n e -> (b n) 1 e")

            # Compute kernel weights.
            xt_flat = einops.rearrange(xt, "b n e -> (b n) 1 e")
            weights = compute_weights(
                x1=xt_flat,
                x2=nearest_xc,
                lengthscales=self.lengthscale,
                dist_fn=self.dist_fn,
            )

            # Apply mask to weights.
            weights = weights * mask
            zt_update_flat = weights @ nearest_zc

            # Reshape output to (batch_size, num_trg, e).
            zt_update = einops.rearrange(
                zt_update_flat, "(b n) 1 e -> b n e", b=num_batches
            )

            if zt is not None:
                zt = zt + zt_update
            else:
                zt = zt_update

        else:
            # Compute kernel weights.
            weights = compute_weights(
                x1=xt,
                x2=xc_flat,
                lengthscales=self.lengthscale,
                dist_fn=self.dist_fn,
            )

            # Shape (batch_size, num_trg, num_grid_points).
            zt_update = weights @ zc_flat

            if zt is not None:
                zt = zt + zt_update
            else:
                zt = zt_update

        return zt


def compute_weights(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
) -> torch.Tensor:
    """Compute the weights for the kernel weighted sum."""

    # Expand dimensions for broadcasting.
    dists2 = dist_fn(x1, x2)

    pre_exp = torch.sum(dists2 / lengthscales.pow(2), dim=-1)
    return torch.exp(-0.5 * pre_exp)
