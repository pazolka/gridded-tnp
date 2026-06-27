from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tnp.data.base import Batch, OOTGBatch
from tnp.data.era5.era5 import (
    BaseERA5DataGeneratorWithReset,
    ERA5DataGenerator,
    ERA5StationDataGenerator,
)
from tnp.data.era5.normalisation import locations as var_means
from tnp.data.era5.normalisation import scales as var_stds


@dataclass
class ERA5OOTGBatch(OOTGBatch):
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]

    gridded_var_names: List[str]
    non_gridded_var_names: List[str]

    var_means: Dict[str, float]
    var_stds: Dict[str, float]


class ERA5OOTGDataGenerator(ERA5DataGenerator):
    def __init__(
        self,
        gridded_var_names: Tuple[str, ...],
        non_gridded_var_names: Tuple[str, ...],
        coarsen_factors: Tuple[int, ...] = (1, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert all(
            g in self.data_vars for g in gridded_var_names
        ), "gridded_var_names must be in data_vars."
        assert all(
            ng in self.data_vars for ng in non_gridded_var_names
        ), "non_gridded_var_names must be in data_vars."

        self.gridded_var_names = gridded_var_names
        self.non_gridded_var_names = non_gridded_var_names

        # TODO: if coarsening grid, don't load all indices.
        self.coarsen_factors = tuple(coarsen_factors)

    def coarsen_idx(
        self, idxs: List[Tuple[List, List, List]]
    ) -> List[Tuple[List, List, List]]:

        # Compute coarsened indices.
        gridded_idxs = [
            (
                idx[0],
                idx[1][:: self.coarsen_factors[0]],
                idx[2][:: self.coarsen_factors[1]],
            )
            for idx in idxs
        ]

        return gridded_idxs

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        assert self.data is not None, "Data has not been loaded. Cannot generate batch."
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Get coarsened idxs.
        gridded_idxs = self.coarsen_idx(idxs)

        # Sample context and target proportions for off-the-grid data.
        pc = self.pc_dist.sample()

        # Sample off-the-grid data.
        x_grid, data_vars = self.sample_batch_data(idxs, self.non_gridded_var_names)
        batch = self.sample_batch(pc=pc, pt=self.pt, x_grid=x_grid, data_vars=data_vars)

        # Sample on-the-grid data.
        xc_grid, gridded_data_vars = self.sample_batch_data(
            gridded_idxs, self.gridded_var_names
        )
        if not self.use_time:
            xc_grid = xc_grid.squeeze(1)[..., 1:]
            gridded_data_vars = {k: v.squeeze(1) for k, v in gridded_data_vars.items()}

        # Compute yc_grid.
        yc_grid = torch.stack(list(gridded_data_vars.values()), dim=-1)

        return ERA5OOTGBatch(
            x=batch.x,
            y=batch.y,
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            yt=batch.yt,
            xc_grid=xc_grid,
            yc_grid=yc_grid,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            gridded_var_names=self.gridded_var_names,
            non_gridded_var_names=self.non_gridded_var_names,
            var_means={k: var_means[k] for k in self.data_vars},
            var_stds={k: var_stds[k] for k in self.data_vars},
        )


class ERA5StationOOTGDataGenerator(ERA5OOTGDataGenerator, ERA5StationDataGenerator):
    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        assert self.data is not None, "Data has not been loaded. Cannot generate batch."
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Get coarsened idxs.
        gridded_idxs = self.coarsen_idx(idxs)

        # Sample context and target proportions for off-the-grid data.
        pc = self.pc_dist.sample()

        # Sample off-the-grid data.
        x_grid, non_gridded_data_vars = self.sample_batch_data(
            idxs, self.non_gridded_var_names
        )
        non_gridded_batch = self.sample_station_batch(
            pc, self.pt, x_grid, non_gridded_data_vars
        )

        # Sample on-the-grid data.
        xc_grid, gridded_data_vars = self.sample_batch_data(
            gridded_idxs, self.gridded_var_names
        )
        if not self.use_time:
            xc_grid = xc_grid.squeeze(1)[..., 1:]
            gridded_data_vars = {k: v.squeeze(1) for k, v in gridded_data_vars.items()}

        # Compute yc_grid.
        yc_grid = torch.stack(list(gridded_data_vars.values()), dim=-1)

        return ERA5OOTGBatch(
            x=non_gridded_batch.x,
            y=non_gridded_batch.y,
            xc=non_gridded_batch.xc,
            yc=non_gridded_batch.yc,
            xt=non_gridded_batch.xt,
            yt=non_gridded_batch.yt,
            xc_grid=xc_grid,
            yc_grid=yc_grid,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            gridded_var_names=self.gridded_var_names,
            non_gridded_var_names=self.non_gridded_var_names,
            var_means={k: var_means[k] for k in self.data_vars},
            var_stds={k: var_stds[k] for k in self.data_vars},
        )


class ERA5OOTGDataGeneratorWithReset(
    ERA5OOTGDataGenerator, BaseERA5DataGeneratorWithReset
):
    pass


class ERA5StationOOTGDataGeneratorWithReset(
    ERA5StationOOTGDataGenerator, BaseERA5DataGeneratorWithReset
):
    pass
