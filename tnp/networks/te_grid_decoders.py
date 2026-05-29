from typing import Optional, Tuple

import einops
import torch
from check_shapes import check_shapes

from tnp.networks.teattention_layers import MultiHeadCrossTEAttentionLayer
from tnp.utils.grids import flatten_grid, nearest_gridded_neighbours

from .grid_decoders import GridDecoder


class TEMHCAGridDecoder(GridDecoder):
    def __init__(
        self,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
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

        xc_flat, _ = flatten_grid(xc)
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
            nearest_xc = xc_flat[batch_idx, nearest_idx]

            # Rearrange tokens.
            zt = einops.rearrange(zt, "b n e -> (b n) 1 e")
            xt = einops.rearrange(xt, "b n e -> (b n) 1 e")
            nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")
            nearest_xc = einops.rearrange(nearest_xc, "b n k e -> (b n) k e")
            mask = einops.rearrange(mask, "b n e -> (b n) 1 e")

            # Do the MHCA operation, reshape and return.
            zt, xt = self.mhca_layer(zt, nearest_zc, xt, nearest_xc, mask=mask)

            zt = einops.rearrange(zt, "(b n) 1 e -> b n e", b=num_batches)
            xt = einops.rearrange(xt, "(b n) 1 e -> b n e", b=num_batches)
        else:
            zt, xt = self.mhca_layer(zt, zc_flat, xt, xc_flat)

        return zt
