from typing import Optional, Tuple, Union

import torch
from check_shapes import check_shapes


@check_shapes(
    "x: [m, ..., d]",
)
def dropout_all(
    x: torch.Tensor,
    p_dropout: Optional[float] = None,
    training: bool = True,
    dropout: Optional[torch.Tensor] = None,
    return_dropout: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if dropout is None:
        if p_dropout is None:
            return x
        if not training:
            dropout = torch.ones(x.shape[:1]).to(x)
        else:
            dropout = torch.bernoulli(torch.ones(x.shape[:1]) * (1 - p_dropout)).to(x)
            if p_dropout < 1.0:
                dropout = dropout / (1 - p_dropout)

    out = (x.transpose(0, -1) * dropout).transpose(0, -1)
    if return_dropout:
        return out, dropout

    return out
