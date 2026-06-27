import copy
import warnings
from abc import ABC
from typing import Optional, Tuple, Union

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.grids import avg_pool, coarsen_grid, flatten_grid
from .attention_layers import MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer
from .grid_decoders import GridDecoder
from .patch_encoders import PatchEncoder
from .swin_attention import HierarchicalSWINAttentionLayer, SWINAttentionLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for mhsa_layer in self.mhsa_layers:
            x = mhsa_layer(x, mask)

        return x


class TNPTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: Optional[MultiHeadSelfAttentionLayer] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc)
            elif isinstance(mhsa_layer, MultiHeadCrossAttentionLayer):
                xc = mhsa_layer(xc, xc)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc)

        return xt


class BasePerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the Perceiver encoder.

        Args:
            num_latents (int): Number of latents.
            mhsa_layer (MultiHeadSelfAttentionLayer): MHSA layer between latents.
            mhca_ctoq_layer (MultiHeadCrossAttentionLayer): MHCA layer from context to latents.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        # Initialise latents.
        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class PerceiverEncoder(BasePerceiverEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


class BaseISTEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the IST encoder.

        Args:
            num_latents (int): Number of latents.
            mhca_ctoq_layer (MultiHeadSelfAttentionLayer): MHCA layer from context to latents.
            mhca_qtoc_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to context.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers - 1)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class ISTEncoder(BaseISTEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for i, (mhca_ctoq_layer, mhca_qtot_layer) in enumerate(
            zip(self.mhca_ctoq_layers, self.mhca_qtot_layers)
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xt = mhca_qtot_layer(xt, xq)

            if i < len(self.mhca_qtoc_layers):
                xc = self.mhca_qtoc_layers[i](xc, xq)

        return xt


class GriddedTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        grid_decoder: GridDecoder,
        mhsa_layer: Union[SWINAttentionLayer, MultiHeadSelfAttentionLayer],
        patch_encoder: Optional[PatchEncoder] = None,
        top_k_ctot: Optional[int] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.grid_decoders = _get_clones(grid_decoder, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.patch_encoder = patch_encoder
        self.top_k_ctot = top_k_ctot
        self.roll_dims = roll_dims

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "mask: [m, nt, nc]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        zc: torch.Tensor,
        xt: torch.Tensor,
        zt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        if self.patch_encoder is not None:
            # TODO: what happens when grid wraps around.
            zc = self.patch_encoder(zc)
            xc = coarsen_grid(
                xc, zc.shape[1:-1], ignore_dims=self.patch_encoder.ignore_dims
            )

        for mhsa_layer, grid_decoder in zip(self.mhsa_layers, self.grid_decoders):
            if isinstance(
                mhsa_layer, (SWINAttentionLayer, HierarchicalSWINAttentionLayer)
            ):
                zc = mhsa_layer(zc)
            else:
                zc, flat_to_grid_fn = flatten_grid(zc)
                zc = mhsa_layer(zc)
                zc = flat_to_grid_fn(zc)

            zt = grid_decoder(xc, zc, xt, zt)

        return zt


class GriddedCNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        grid_decoder: GridDecoder,
        cnn: nn.Module,
        patch_encoder: Optional[PatchEncoder] = None,
        top_k_ctot: Optional[int] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.grid_decoder = grid_decoder
        self.cnn = cnn
        self.patch_encoder = patch_encoder
        self.top_k_ctot = top_k_ctot
        self.roll_dims = roll_dims

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "mask: [m, nt, nc]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        zc: torch.Tensor,
        xt: torch.Tensor,
        zt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        if self.patch_encoder is not None:
            # TODO: what happens when grid wraps around.
            zc = self.patch_encoder(zc)
            xc = avg_pool(
                dim=xc.shape[-1],
                input=xc.movedim(-1, 1),
                kernel_size=self.patch_encoder.conv.kernel_size,
                stride=self.patch_encoder.conv.stride,
            )
            xc = xc.movedim(1, -1)

        # Pass through CNN backbone.
        zc = self.cnn(zc)

        # Pass through grid decoder.
        zt = self.grid_decoder(xc, zc, xt, zt)

        return zt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
