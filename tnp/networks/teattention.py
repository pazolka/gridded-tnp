import math
from abc import ABC
from typing import Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn


class BaseMultiHeadTEAttention(nn.Module, ABC):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        kernel: nn.Module,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.kernel = kernel
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nkv, dz]",
        "zv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def propagate(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes multi-head translation equivariant attention.

        Args:
            zq (torch.Tensor): Query token.
            zk (torch.Tensor): Key token.
            zv (torch.Tensor): Value token.
            xq (torch.Tensor): Query input locations.
            xk (torch.Tensor): Key input locations.
            mask (Optional[torch.Tensor], optional): Query-key mask. Defaults to None.

        Returns:
            torch.Tensor: Output of attention mechanism.
        """
        # Compute differences.
        # (m, nq, nkv, dx).
        diff = xq[..., None, :] - xk[..., None, :, :]

        # Compute token attention.
        q = self.to_q(zq)
        k = self.to_k(zk)
        v = self.to_v(zv)

        # Each of shape (m, {num_heads, qk_dim}, n, head_dim).
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v),
        )

        # Compute attention bias.
        # (m, nq, nkv, h).
        attn_bias = self.kernel(diff)
        attn_bias = einops.rearrange(attn_bias, "m nq nkv h -> m h nq nkv")

        if mask is not None:
            mask = einops.repeat(mask, "m n p -> m h n p", h=self.num_heads)
            attn_bias = torch.masked_fill(attn_bias, ~mask, -float("inf"))

        out = (
            nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, v, attn_mask=attn_bias, scale=self.scale
            )
        )
        out = einops.rearrange(out, "m h n d -> m n (h d)")
        out = self.to_out(out)

        return out, xq


class MultiHeadTEAttention(BaseMultiHeadTEAttention):
    @check_shapes(
        "zq: [m, nq, dz]",
        "zk: [m, nkv, dz]",
        "zv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def forward(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(zq, zk, zv, xq, xk, mask)


class MultiHeadSelfTEAttention(BaseMultiHeadTEAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "z: [m, n, dz]",
        "x: [m, n, dx]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dz]",
        "return[1]: [m, n, dx]",
    )
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(z, z, z, x, x, mask)


class MultiHeadCrossTEAttention(BaseMultiHeadTEAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "zq: [m, nq, dz]",
        "zkv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def forward(
        self,
        zq: torch.Tensor,
        zkv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(zq, zkv, zkv, xq, xkv, mask)


class GriddedMultiHeadSelfTEAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        grid_shape: Tuple[int, ...],
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embed_dim)

        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        # This module expects tokens that are on a grid of this shape.
        self.grid_shape = tuple(grid_shape)
        num_grid_points = math.prod(grid_shape)
        self.attn_bias = nn.Parameter(
            torch.randn(num_heads, num_grid_points, num_grid_points),
            requires_grad=True,
        )

    @check_shapes(
        "x: [m, n, d]",
        "mask: [m, n, n]",
        "return: [m, n, d]",
    )
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure that the grid points are in the correct shape.
        assert x.shape[-2] == math.prod(self.grid_shape)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(
            lambda x: einops.rearrange(x, "m n (h d) -> m h n d", h=self.num_heads),
            (q, k, v),
        )

        attn_bias = einops.repeat(self.attn_bias, "h n1 n2 -> m h n1 n2", m=x.shape[0])
        if mask is not None:
            mask = einops.repeat(mask, "m n1 n2 -> m h n1 n2", h=self.num_heads)
            if mask.dtype == torch.bool:
                attn_bias = torch.masked_fill(attn_bias, ~mask, -float("inf"))
            else:
                attn_bias = attn_bias + mask

        out = (
            nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, v, attn_mask=attn_bias, scale=self.scale
            )
        )

        out = einops.rearrange(out, "m h n d -> m n (h d)")
        out = self.to_out(out)
        return out
