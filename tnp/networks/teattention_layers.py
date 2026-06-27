from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import BaseMultiHeadAttentionLayer
from .teattention import (
    BaseMultiHeadTEAttention,
    GriddedMultiHeadSelfTEAttention,
    MultiHeadCrossTEAttention,
    MultiHeadSelfTEAttention,
    MultiHeadTEAttention,
)


class BaseMultiHeadTEAttentionLayer(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        attention: Union[BaseMultiHeadTEAttention, partial[BaseMultiHeadTEAttention]],
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
        ff_block: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(**kwargs)

        # Feedforward model.
        if ff_block is None:
            self.ff_block = nn.Sequential(
                nn.Linear(embed_dim, feedforward_dim),
                activation,
                nn.Dropout(p_dropout),
                nn.Linear(feedforward_dim, embed_dim),
                nn.Dropout(p_dropout),
            )
        else:
            self.ff_block = ff_block

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(p_dropout)


class MultiHeadTEAttentionLayer(BaseMultiHeadTEAttentionLayer):
    def __init__(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        **kwargs,
    ):
        attention = partial(MultiHeadTEAttention, qk_dim=qk_dim, v_dim=v_dim)
        super().__init__(embed_dim=v_dim, attention=attention, **kwargs)

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
    def attn_block(
        self,
        zq: torch.Tensor,
        zk: torch.Tensor,
        zv: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zq, xq = self.attn(zq, zk, zv, xq, xk, mask=mask)
        return self.attn_dropout(zq), xq

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
        # An MHA block is just the MHA operation.
        zq, xq = self.attn_block(zq, zk, zv, xq, xk, mask)

        return zq, xq


class MultiHeadSelfTEAttentionLayer(BaseMultiHeadTEAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadSelfTEAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "z: [m, n, dz]", "x: [m, n, dx]", "mask: [m, n, n]", "return: [m, n, dz]"
    )
    def attn_block(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z, x = self.attn(z, x, mask=mask)
        return self.attn_dropout(z), x

    @check_shapes(
        "z: [m, n, dz]",
        "x: [m, n, dx]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dz]",
        "return[1]: [m, n, dx]",
    )
    def forward(
        self, z: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm_first:
            z_attn, x = self.attn_block(self.norm1(z), x, mask)
            z = z + z_attn
            z = z + self.ff_block(self.norm2(z))
        else:
            z_attn, x = self.attn_block(z, x, mask)
            z = self.norm1(z + z_attn)
            z = z + self.ff_block(self.norm2(z))

        return z, x


class MultiHeadCrossTEAttentionLayer(BaseMultiHeadTEAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadCrossTEAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "zq: [m, nq, dz]",
        "zkv: [m, nkv, dz]",
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nq, dx]",
    )
    def attn_block(
        self,
        zq: torch.Tensor,
        zkv: torch.Tensor,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zq, xq = self.attn(zq, zkv, xq, xkv, mask=mask)
        return self.attn_dropout(zq), xq

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
        if self.norm_first:
            zq_attn, xq = self.attn_block(self.norm1(zq), self.norm1(zkv), xq, xkv, mask)
            zq = zq + zq_attn
            zq = zq + self.ff_block(self.norm2(zq))
        else:
            zq_attn, xq = self.attn_block(zq, zkv, xq, xkv, mask)
            zq = self.norm1(zq + zq_attn)
            zq = zq + self.ff_block(self.norm2(zq))

        return zq, xq


class GriddedMultiHeadSelfTEAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(GriddedMultiHeadSelfTEAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def attn_block(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask)
        return self.attn_dropout(x)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn_block(self.norm1(x), mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.attn_block(x, mask))
            x = self.norm2(x + self.ff_block(x))

        return x
