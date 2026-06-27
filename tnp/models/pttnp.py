import copy
from typing import Dict, List, Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import ISTEncoder, PerceiverEncoder
from ..utils.helpers import preprocess_observations
from .base import (
    ConditionalNeuralProcess,
    MultiModalConditionalNeuralProcess,
    OOTGConditionalNeuralProcess,
)
from .tnp import TNPDecoder


class PTTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[PerceiverEncoder, ISTEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, nq, dz]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.transformer_encoder(zc, zt)
        return zt


class MultiModalPTTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[PerceiverEncoder, ISTEncoder],
        xy_encoder: nn.Module,
        zt_encoder: nn.Module,
        mode_names: List[str],
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.mode_names = mode_names

        # Shared x_encoder.
        self.x_encoder = x_encoder

        # Shared zt_encoder.
        self.zt_encoder = zt_encoder

        # Copy encoder for each mode.
        self.xy_encoder = nn.ModuleDict(
            {mode: copy.deepcopy(xy_encoder) for mode in mode_names}
        )
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

        # Encode xt.
        zt = self.zt_encoder(xt_encoded)

        # Encode y for each modality.
        yc_encoded_list = []

        if self.y_encoder is not None:
            for mod, yc_ in zip(yc.keys(), yc_list):
                yc_encoded_list.append(self.y_encoder[mod](yc_))
        else:
            yc_encoded_list = yc_list

        # Combine x and y encodings for each modality
        zc_dict: Dict[str, torch.Tensor] = {}
        for mod, xc_encoded, yc_encoded in zip(
            yc.keys(), xc_list_encoded, yc_encoded_list
        ):
            zc_dict[mod] = torch.cat((xc_encoded, yc_encoded), dim=-1)
            zc_dict[mod] = self.xy_encoder[mod](zc_dict[mod])

        # Flatten zc.
        zc = torch.cat(list(zc_dict.values()), dim=1)

        # Apply transformer encoder.
        zt = self.transformer_encoder(zc, zt)
        return zt


class OOTGPTTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[PerceiverEncoder, ISTEncoder],
        xy_encoder: nn.Module,
        xy_grid_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        y_grid_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.xy_grid_encoder = xy_grid_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.y_grid_encoder = y_grid_encoder

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
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        xc_grid_encoded = self.x_encoder(xc_grid)
        yc_grid_encoded = self.y_grid_encoder(yc_grid)

        # Tokenise.
        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zt = self.xy_encoder(zt)

        zc_grid = torch.cat((xc_grid_encoded, yc_grid_encoded), dim=-1)
        zc_grid = self.xy_grid_encoder(zc_grid)

        # Flatten grid.
        zc_grid = zc_grid.flatten(start_dim=1, end_dim=-2)

        # Concatenate.
        zc = torch.cat((zc, zc_grid), dim=1)

        # Apply transformer encoder.
        zt = self.transformer_encoder(zc, zt)
        return zt


class PTTNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: PTTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class MultiModalPTTNP(MultiModalConditionalNeuralProcess):
    def __init__(
        self,
        encoder: MultiModalPTTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class OOTGPTTNP(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGPTTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
