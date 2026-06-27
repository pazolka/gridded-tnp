import logging
import os
import random
import time
import warnings
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta  # type: ignore

from tnp.data.base import Batch, DataGenerator
from tnp.data.era5.normalisation import locations as var_means
from tnp.data.era5.normalisation import normalise_var
from tnp.data.era5.normalisation import scales as var_stds
from tnp.utils.grids import coarsen_grid


@dataclass
class BaseERA5Batch:
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]

    var_names: Tuple[str, ...]
    var_means: Dict[str, float]
    var_stds: Dict[str, float]


@dataclass
class ERA5Batch(Batch, BaseERA5Batch):
    time_grid: Optional[torch.Tensor] = None


class BaseERA5DataGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        time_range: Tuple[str, str] = ("2019-06-01", "2019-06-31"),
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        lon_range: Tuple[float, float] = (-180.0, 180.0),
        batch_time_steps: int = 1,
        batch_grid_size: Optional[Tuple[int, int]] = None,
        data_vars: Tuple[str, ...] = ("t2m",),
        t_spacing: int = 1,
        use_time: bool = True,
        lazy_loading: bool = True,
        wrap_longitude: bool = False,
        load_data: bool = True,
        data_dir: Optional[str] = "./data/era5",
        fnames: Optional[List[str]] = None,
        # url: str = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        url: str = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_range = time_range
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.data_vars = data_vars
        self.input_vars = ["time", "latitude", "longitude"]
        self.lazy_loading = lazy_loading

        # How large each sampled grid should be (in indicies).
        self.batch_time_steps = batch_time_steps
        self.batch_grid_size = batch_grid_size

        self.t_spacing = t_spacing
        self.use_time = use_time
        if not use_time:
            assert (
                batch_time_steps == 1
            ), "batch_time_steps must be 1 if not using time."

        # Whether we allow batches to wrap around longitudinally.
        self.wrap_longitude = wrap_longitude

        # Whether to pre-load the data.
        self.data = None
        if data_dir is None:
            raise ValueError("data_dir cannot be None")  # silence mypy
        self.data_dir = os.path.expanduser(data_dir)
        self.fnames = fnames
        self.url = url
        if load_data:
            self.load_data(time_range)

    def load_data(
        self,
        time_range: Tuple[str, str],
    ):
        def _preprocess_dataset(dataset: xr.Dataset):
            # Ensure longitudes and latitudes are in standard format.
            if dataset["longitude"].max() > 180:
                dataset = dataset.assign_coords(
                    longitude=(dataset["longitude"].values + 180) % 360 - 180
                )
            if dataset["latitude"].max() > 90:
                dataset = dataset.assign_coords(
                    latitude=dataset["latitude"].values - 90
                )

                # Sort latitude and longitude values.
            dataset = dataset.sortby(["latitude", "longitude"])

            # Change valid_time to time.
            if "valid_time" in dataset.coords:
                dataset = dataset.rename(valid_time="time")

            if "level" in dataset.coords:
                dataset = dataset.rename(level="pressure_level")

            dataset = dataset.sel(
                time=slice(*sorted(time_range)),
                latitude=slice(*sorted(self.lat_range)),
                longitude=slice(*sorted(self.lon_range)),
            )

            if "2m_temperature" in dataset.data_vars:
                dataset = dataset.rename(
                    {
                        "2m_temperature": "t2m",
                    }
                )

            if "temperature" in dataset.data_vars:
                dataset = dataset.rename({"temperature": "t"})

            if "u_component_of_wind" in dataset.data_vars:
                dataset = dataset.rename({"u_component_of_wind": "u"})

            if "v_component_of_wind" in dataset.data_vars:
                dataset = dataset.rename({"v_component_of_wind": "v"})

            # Convert pressure levels to new variables.
            if "pressure_level" in dataset.coords:
                for var in dataset.data_vars:
                    if "pressure_level" in dataset[var].coords:
                        for p in dataset[var].pressure_level.values:
                            dataset[f"{var}_{int(p)}"] = dataset[var].sel(
                                pressure_level=p
                            )

                        # Remove original variable.
                        dataset = dataset.drop_vars(var)

                # Drop pressure level coord.
                dataset = dataset.drop_vars("pressure_level")

            dataset = dataset[list(self.data_vars)]

            # Change time to hours since reference time.
            ref_np_datetime = np.datetime64("1970-01-01T00:00:00")
            hours = (dataset["time"][:].data - ref_np_datetime) / np.timedelta64(1, "h")
            dataset = dataset.assign_coords(time=hours)
            dataset = dataset.transpose("time", "latitude", "longitude")

            return dataset

        # Load datasets.
        if self.fnames is not None and self.data_dir is not None:

            # Check if any datasets are empty.
            fnames = []
            for fname in self.fnames:
                try:
                    dataset = xr.open_dataset(
                        os.path.join(self.data_dir, fname), engine="h5netcdf"
                    )
                    dataset = dataset.sel(valid_time=slice(*sorted(time_range)))
                    if dataset.valid_time.size == 0:
                        logging.debug("Skipping dataset %s...", fname)
                        continue

                    fnames.append(fname)
                except:
                    logging.debug("Unable to find %s in %s", fname, self.data_dir)

            dataset = xr.open_mfdataset(
                [os.path.join(self.data_dir, fname) for fname in fnames],
                chunks={
                    "valid_time": self.batch_time_steps,
                    "latitude": 721,
                    "longitude": 1440,
                },
                parallel=True,
                preprocess=_preprocess_dataset,
                engine="h5netcdf",
                cache=True,
                lock=False,
            )
        else:
            dataset = xr.open_zarr(
                self.url,
            )
            dataset = _preprocess_dataset(dataset)

        self.data = dataset
        if not self.lazy_loading:
            t0 = time.time()
            assert self.data is not None
            self.data = self.data.compute(
                num_workers=torch.utils.data.get_worker_info().num_workers
            )
            logging.debug("Data computed in %.2fs.", time.time() - t0)

    def sample_idx(self, batch_size: int) -> List[Tuple[List, List, List]]:
        """Samples indices used to sample dataframe.

        Args:
            batch_size (int): Batch_size.

        Returns:
            Tuple[List, List, List]: Indicies.
        """
        assert self.data is not None, "Data has not been loaded."

        # TODO: if using same location for each batch, let lat_idx starting index extend
        # to len(self.data["latitude"]) and truncate grid size.
        if self.batch_grid_size is not None:
            if len(self.data["latitude"]) >= self.batch_grid_size[0]:
                i = random.randint(
                    0, len(self.data["latitude"]) - self.batch_grid_size[0]
                )
                lat_idx = list(range(i, i + self.batch_grid_size[0]))
            else:
                raise ValueError("Grid size is too large!")

            # Allow longitude to wrap around.
            if (
                len(self.data["longitude"]) > self.batch_grid_size[1]
                and self.wrap_longitude
            ):
                i = random.randint(0, len(self.data["longitude"]))
                lon_idx = list(range(i, i + self.batch_grid_size[1]))
                lon_idx = [idx % len(self.data["longitude"]) for idx in lon_idx]
            elif len(self.data["longitude"]) >= self.batch_grid_size[1]:
                i = random.randint(
                    0, len(self.data["longitude"]) - self.batch_grid_size[1]
                )
                lon_idx = list(range(i, i + self.batch_grid_size[1]))
            else:
                raise ValueError("Grid size is too large!")
        else:
            lat_idx = list(range(len(self.data["latitude"])))
            lon_idx = list(range(len(self.data["longitude"])))

        time_idx: List[List] = []
        for _ in range(batch_size):
            i = random.randint(
                0,
                len(self.data["time"]) - self.t_spacing * self.batch_time_steps,
            )
            time_idx.append(
                list(
                    range(i, i + self.batch_time_steps * self.t_spacing, self.t_spacing)
                )
            )

        idx = [(time_idx[i], lat_idx, lon_idx) for i in range(len(time_idx))]
        return idx


class BaseERA5DataGeneratorWithReset(BaseERA5DataGenerator, ABC):
    def __init__(
        self,
        *,
        num_months: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_months = num_months

    def __iter__(self):
        self.reset_data()
        return super().__iter__()

    def reset_data(self):
        time_range = self.time_range
        datetime_str_format = "%Y-%m-%d"

        # Reset time range.
        start_date = datetime.strptime(time_range[0], datetime_str_format)
        end_date = datetime.strptime(time_range[1], datetime_str_format)

        # num_months = (end_date - start_date).months / self.num_months
        num_months = (end_date.year * 12 + end_date.month) - (
            start_date.year * 12 + start_date.month
        )
        month_offset = random.randint(0, num_months - self.num_months)
        month_diff = relativedelta(months=self.num_months)
        day_diff = relativedelta(days=1)
        new_start_date = start_date + relativedelta(months=month_offset)
        new_end_date = new_start_date + month_diff - day_diff

        new_time_range = (
            new_start_date.strftime(datetime_str_format),
            new_end_date.strftime(datetime_str_format),
        )

        logging.debug("Resetting data to %s.", new_time_range)

        self.load_data(new_time_range)


class ERA5DataGenerator(BaseERA5DataGenerator):
    def __init__(
        self,
        *,
        min_pc: float,
        max_pc: float,
        pt: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pc_dist = torch.distributions.Uniform(min_pc, max_pc)
        self.pt = pt

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        assert self.data is not None, "Data has not been loaded. Cannot generate batch."
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Sample batch data.
        x_grid, data_vars = self.sample_batch_data(idxs)

        # Sample context and target proportions.
        pc = self.pc_dist.sample()

        # Get batch.
        batch = self.sample_batch(pc=pc, pt=self.pt, x_grid=x_grid, data_vars=data_vars)
        return batch

    def sample_masks(
        self, prop: float, grid_shape: torch.Size, batch_shape: torch.Size
    ) -> torch.Tensor:

        num_data = torch.prod(torch.as_tensor(grid_shape))
        num_mask = int(num_data * prop)
        m_idx = torch.argsort(
            torch.rand(
                size=(
                    batch_shape,
                    num_data,
                )
            ),
            dim=-1,
        )[..., :num_mask]
        m_idx = torch.unravel_index(m_idx, grid_shape)

        return m_idx

    def construct_x_grid(self, idxs: List[Tuple[List, List, List]]):
        assert (
            self.data is not None
        ), "Data has not been loaded. Cannot construct x_grid."
        x_grid = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        *[
                            torch.as_tensor(
                                self.data[k][idx[i]].data, dtype=torch.float
                            )
                            for i, k in enumerate(self.input_vars)
                        ]
                    ),
                    dim=-1,
                )
                for idx in idxs
            ],
            dim=0,
        )
        return x_grid

    def construct_data_var_list(
        self,
        idxs: List[Tuple[List, List, List]],
        var_names: Tuple[str, ...],
    ):
        assert (
            self.data is not None
        ), "Data has not been loaded. Cannot construct data var list."
        data_vars_list = {
            k: [self.data[k][idx].data for idx in idxs] for k in var_names
        }

        return data_vars_list

    def sample_batch_data(
        self,
        idxs: List[Tuple[List, List, List]],
        var_names: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        assert self.data is not None, "Data has not been loaded. Cannot sample batch."
        x_grid = self.construct_x_grid(idxs)

        if var_names is None:
            var_names = self.data_vars

        # Use dask.delayed to lazily compute x_grid and y_grid
        data_vars_list = self.construct_data_var_list(idxs, var_names)

        if self.lazy_loading:
            t0 = time.time()
            data_vars_list = dask.compute(data_vars_list)[0]
            logging.debug("Data computed in %.2fs.", time.time() - t0)

        data_vars = {
            k: torch.stack(
                [torch.as_tensor(d, dtype=torch.float) for d in data_vars_list[k]]
            )
            for k in var_names
        }

        # Normalise outputs.
        data_vars = {k: normalise_var(data_vars[k], name=k) for k in var_names}

        return x_grid, data_vars

    def sample_batch(
        self,
        pc: float,
        pt: float,
        x_grid: torch.Tensor,
        data_vars: Dict[str, torch.Tensor],
    ) -> Batch:

        if not self.use_time:
            time_grid = None
            x_grid = x_grid.squeeze(1)[..., 1:]
            data_vars = {k: v.squeeze(1) for k, v in data_vars.items()}
        else:
            # Get time grid.
            time_grid = x_grid[..., 0, 0, 0]

        # Assumes same masking pattern for each grid.
        mc_grid_idx = self.sample_masks(
            prop=pc, grid_shape=x_grid.shape[1:-1], batch_shape=x_grid.shape[0]
        )
        mt_grid_idx = self.sample_masks(
            prop=pt, grid_shape=x_grid.shape[1:-1], batch_shape=x_grid.shape[0]
        )
        batch_idx = torch.arange(x_grid.shape[0]).unsqueeze(-1)

        # Stack data_vars into a single tensor.
        y_grid = torch.stack(list(data_vars.values()), dim=-1)

        # Get flattened versions.
        x = x_grid.flatten(start_dim=1, end_dim=-2)
        y = y_grid.flatten(start_dim=1, end_dim=-2)
        xc = x_grid[(batch_idx,) + mc_grid_idx]
        yc = y_grid[(batch_idx,) + mc_grid_idx]
        xt = x_grid[(batch_idx,) + mt_grid_idx]
        yt = y_grid[(batch_idx,) + mt_grid_idx]

        return ERA5Batch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            var_names=self.data_vars,
            var_means={k: var_means[k] for k in self.data_vars},
            var_stds={k: var_stds[k] for k in self.data_vars},
            time_grid=time_grid,
        )


class ERA5StationDataGenerator(ERA5DataGenerator):
    def __init__(
        self, station_coords_fname: str, predict_stations: bool = False, **kwargs
    ):
        super().__init__(**kwargs)

        station_coords_fname = os.path.expanduser(station_coords_fname)
        self.station_coords = torch.load(station_coords_fname, weights_only=True)
        self.predict_stations = predict_stations

    def compute_station_idxs(self, x_grid: torch.Tensor) -> torch.Tensor:
        # Assumes lat/lon shared across all idxs.
        lats = x_grid[0, 0, :, 0, 1]
        lons = x_grid[0, 0, 0, :, 2]

        station_idxs = torch.zeros_like(self.station_coords, dtype=torch.long)
        station_idxs[:, 0] = torch.argmin(
            torch.abs(self.station_coords[:, 0][:, None] - lats[None, ...]), dim=-1
        )
        station_idxs[:, 1] = torch.argmin(
            torch.abs(self.station_coords[:, 1][:, None] - lons[None, ...]), dim=-1
        )

        return station_idxs

    def sample_station_batch(
        self,
        pc: float,
        pt: float,
        x_grid: torch.Tensor,
        data_vars: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert (
            self.data is not None
        ), "Data has not been loaded. Cannot sample station batch."

        # Get station_idxs within lat/lon idxs.
        station_idxs = self.compute_station_idxs(x_grid)

        if not self.use_time:
            time_grid = None
            x_grid = x_grid.squeeze(1)[..., 1:]
            data_vars = {k: v.squeeze(1) for k, v in data_vars.items()}
        else:
            time_grid = x_grid[..., 0, 0, 0]

        # Assumes same masking pattern for each grid.
        nc = int(pc * len(station_idxs))
        mc_idx = torch.randperm(len(station_idxs))[:nc]
        station_idxs_c = station_idxs[mc_idx]

        # Stack data_vars into a single tensor.
        y_grid = torch.stack(list(data_vars.values()), dim=-1)

        # Set context
        xc = x_grid[..., station_idxs_c[:, 0], station_idxs_c[:, 1], :]
        yc = y_grid[..., station_idxs_c[:, 0], station_idxs_c[:, 1], :]

        # Set targets
        if self.predict_stations:
            nt = int(pt * len(station_idxs))
            mt_idx = torch.randperm(len(station_idxs))[:nt]
            station_idxs_t = station_idxs[mt_idx]
            xt = x_grid[..., station_idxs_t[:, 0], station_idxs_t[:, 1], :]
            yt = y_grid[..., station_idxs_t[:, 0], station_idxs_t[:, 1], :]
        else:
            mt_grid_idx = self.sample_masks(
                prop=pt, grid_shape=x_grid.shape[1:-1], batch_shape=x_grid.shape[0]
            )
            batch_idx = torch.arange(x_grid.shape[0]).unsqueeze(-1)
            xt = x_grid[(batch_idx,) + mt_grid_idx]
            yt = y_grid[(batch_idx,) + mt_grid_idx]

        # Don't return entire grid as far too large.
        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)

        return ERA5Batch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            var_names=self.data_vars,
            var_means={k: var_means[k] for k in self.data_vars},
            var_stds={k: var_stds[k] for k in self.data_vars},
            time_grid=time_grid,
        )

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        assert self.data is not None, "Data has not been loaded. Cannot generate batch."
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Sample batch data.
        x_grid, data_vars = self.sample_batch_data(idxs)

        # Sample context data.
        pc = self.pc_dist.sample()

        # Get station data.
        batch = self.sample_station_batch(pc, self.pt, x_grid, data_vars)
        return batch


class ERA5DataGeneratorWithReset(ERA5DataGenerator, BaseERA5DataGeneratorWithReset):
    pass


def coarsen_grid_era5(
    grid: torch.Tensor,
    output_grid: Optional[Tuple[int, ...]] = None,
    coarsen_factors: Optional[Tuple[int, ...]] = None,
    wrap_longitude: bool = False,
    lon_dim: int = -1,
) -> torch.Tensor:

    if wrap_longitude:
        warnings.warn("Assumed that the minimum longitude occurs in the first element.")
        lon_min = grid[..., 0, 0, lon_dim]
        grid = recenter_latlon_grid(grid, lon_dim)

    coarse_grid = coarsen_grid(grid, output_grid, coarsen_factors)

    if wrap_longitude:
        # Undo operations.
        coarse_grid[..., lon_dim] = coarse_grid[..., lon_dim] + lon_min[..., None, None]
        coarse_grid[..., lon_dim] = torch.where(
            coarse_grid[..., lon_dim] >= 180,
            coarse_grid[..., lon_dim] - 360,
            coarse_grid[..., lon_dim],
        )

    return coarse_grid


def recenter_latlon_grid(grid: torch.Tensor, lon_dim: int = -1):
    # Assumes first index contains smallest longitude value.
    lon_min = grid[..., 0, lon_dim]

    recentered_grid = grid.clone()
    recentered_grid[..., lon_dim] = torch.where(
        (grid[..., lon_dim] - lon_min[..., None]) < 0,
        (grid[..., lon_dim]) + 360,
        grid[..., lon_dim],
    )

    recentered_grid[..., lon_dim] = recentered_grid[..., lon_dim] - lon_min[..., None]
    return recentered_grid
