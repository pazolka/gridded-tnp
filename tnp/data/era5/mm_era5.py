from dataclasses import dataclass
from typing import Dict, Optional

import torch

from tnp.data.base import MultiModalBatch
from tnp.data.era5.era5 import (
    BaseERA5Batch,
    BaseERA5DataGeneratorWithReset,
    Batch,
    ERA5DataGenerator,
)
from tnp.data.era5.normalisation import locations as var_means
from tnp.data.era5.normalisation import scales as var_stds


@dataclass
class ERA5MultiModalBatch(MultiModalBatch, BaseERA5Batch, Batch):
    time_grid: Optional[torch.Tensor] = None


class MultiModalERA5DataGenerator(ERA5DataGenerator):

    def sample_batch(
        self,
        pc: float,
        pt: float,
        x_grid: torch.Tensor,
        data_vars: Dict[str, torch.Tensor],
    ) -> ERA5MultiModalBatch:

        if not self.use_time:
            time_grid = None
            data_vars = {k: data_vars[k].squeeze(1) for k in self.data_vars}
        else:
            # Get time grid.
            time_grid = x_grid[..., 0, 0, 0]

        xc: Dict[str, torch.Tensor] = {}
        yc: Dict[str, torch.Tensor] = {}
        for var, y_grid in data_vars.items():

            # Sample context mask.
            mc_grid_idx = self.sample_masks(
                prop=pc, grid_shape=y_grid.shape[1:], batch_shape=y_grid.shape[0]
            )

            batch_idx = torch.arange(x_grid.shape[0]).unsqueeze(-1)
            xc[var] = x_grid[(batch_idx,) + mc_grid_idx]
            yc[var] = y_grid[(batch_idx,) + mc_grid_idx][..., None]

        # Stack data_vars into a single tensor.
        y_grid = torch.stack([data_vars[k] for k in self.data_vars], dim=-1)

        # Sample target mask.
        mt_grid_idx = self.sample_masks(
            prop=pt, grid_shape=y_grid.shape[1:-1], batch_shape=y_grid.shape[0]
        )
        xt = x_grid[(batch_idx,) + mt_grid_idx]
        yt = y_grid[(batch_idx,) + mt_grid_idx]

        # Flatten x and y.
        x = x_grid.flatten(start_dim=1, end_dim=-2)
        y = y_grid.flatten(start_dim=1, end_dim=-2)

        return ERA5MultiModalBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            var_names=self.data_vars,
            var_means=var_means,
            var_stds=var_stds,
            time_grid=time_grid,
        )


class MultiModalERA5DataGeneratorWithReset(
    MultiModalERA5DataGenerator, BaseERA5DataGeneratorWithReset
):
    pass
