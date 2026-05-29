import copy
from typing import Dict, List, Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.deepset import DeepSet
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder


class CNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
    ):
        super().__init__()
        self.deepset = deepset

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        zc = self.deepset(xc, yc)

        # Use same context representation for every target point.
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])

        # Concatenate xt to zc.
        zc = torch.cat((zc, xt), dim=-1)

        return zc


class MultiModalCNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
        mode_names: List[str],
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.deepset = nn.ModuleDict(
            {mode: copy.deepcopy(deepset) for mode in mode_names}
        )
        self.mode_names = mode_names

        # Shared x_encoder.
        self.x_encoder = x_encoder

        self.y_encoder = nn.ModuleDict(
            {mode: copy.deepcopy(y_encoder) for mode in mode_names}
        )

    def forward(
        self,
        xc: Dict[str, torch.Tensor],
        yc: Dict[str, torch.Tensor],
        xt: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = time_grid

        # Use separate encoders for each modality.
        xc_list = list(xc.values())
        yc_list = list(yc.values())
        nc_list = [xcm.shape[-2] for xcm in xc_list]
        x = torch.cat(xc_list, dim=1)
        x = torch.cat((x, xt), dim=1)
        x_encoded = self.x_encoder(x)

        # Split x_encoded into xc_list_encoded and xt_encoded.
        xc_list_encoded = x_encoded[:, : sum(nc_list)].split(nc_list, dim=1)
        xt_encoded = x_encoded[:, sum(nc_list) :]

        # Encode y for each modality.
        yc_encoded_list = []
        if self.y_encoder is not None:
            for mod, yc_ in zip(yc.keys(), yc_list):
                yc_encoded_list.append(self.y_encoder[mod](yc_))
        else:
            yc_encoded_list = yc_list

        # Get deepset encodings for each modality.
        zc_dict: Dict[str, torch.Tensor] = {}
        for mod, xc_encoded, yc_encoded in zip(
            yc.keys(), xc_list_encoded, yc_encoded_list
        ):
            zc_dict[mod] = self.deepset[mod](xc_encoded, yc_encoded)

        # Aggregat zc.
        zc = sum(zc_dict.values())

        # Use same context representation for every target point.
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])

        # Concatenate xt to zc.
        zt = torch.cat((zc, xt_encoded), dim=-1)

        return zt


class OOTGCNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
        y_encoder: nn.Module,
        y_grid_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.deepset = deepset
        self.y_encoder = y_encoder
        self.y_grid_encoder = y_grid_encoder
        self.x_encoder = x_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xc_grid: [m, ..., dx]",
        "yc_grid: [m, ..., dy_grid]",
        "xt: [m, nt, dx]",
        "return: [m, n, dz]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xc_grid: torch.Tensor,
        yc_grid: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        xc_grid_encoded = self.x_encoder(xc_grid)
        yc_grid_encoded = self.y_grid_encoder(yc_grid)

        # Flatten grid.
        yc_grid_encoded = yc_grid_encoded.flatten(start_dim=1, end_dim=-2)
        xc_grid_encoded = xc_grid_encoded.flatten(start_dim=1, end_dim=-2)

        # Concatenate.
        xc_encoded = torch.cat((xc_encoded, xc_grid_encoded), dim=1)
        yc_encoded = torch.cat((yc_encoded, yc_grid_encoded), dim=1)

        zc = self.deepset(xc_encoded, yc_encoded)

        # Use same context representation for every target point.
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])

        # Concatenate xt to zc.
        zt = torch.cat((zc, xt_encoded), dim=-1)
        return zt


class CNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: CNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalCNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGCNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
