import torch
import tempfile
import os
import hiyapyco
from hydra.utils import instantiate
import xarray as xr
import numpy as np
import shutil

from tnp.utils.experiment_utils import deep_convert_dict, extract_config


def create_mock_era5_data(
    data_dir: str,
    data_vars: list,
    time_steps: int = 4,
    lat_size: int = 24,
    lon_size: int = 60,
):
    """Create mock ERA5 data files."""
    base_time = np.datetime64("2018-01-01T00:00:00")
    times = base_time + np.arange(time_steps) * np.timedelta64(1, "h")
    lats = np.linspace(49.0, 25.0, lat_size)
    lons = np.linspace(-125.0, -66.0, lon_size)

    data_arrays = {}
    for var in data_vars:
        data = np.random.randn(time_steps, lat_size, lon_size) * 10 + 20
        data_arrays[var] = xr.DataArray(
            data,
            dims=["valid_time", "latitude", "longitude"],
            coords={"valid_time": times, "latitude": lats, "longitude": lons},
        )

    dataset = xr.Dataset(data_arrays)
    os.makedirs(data_dir, exist_ok=True)
    fname = os.path.join(data_dir, "era5_test_data.nc")
    dataset.to_netcdf(fname, engine="h5netcdf")
    return ["era5_test_data.nc"]


def test_multimodal_tnp_encoder_from_configs():
    """Test MultiModalGriddedTNPEncoder using config files."""

    # Load and merge config files
    config_files = [
        "experiments/configs/models/mm-swintnp.yml",
        "experiments/configs/models/grid_encoders/pt-ge-tt.yml",
        "experiments/configs/models/grid_decoders/mhca-gd.yml",
        "experiments/configs/generators/mm-era5.yml",
    ]

    raw_config = deep_convert_dict(
        hiyapyco.load(
            config_files, method=hiyapyco.METHOD_MERGE, usedefaultyamlloader=True
        )
    )

    # Create mock data
    temp_data_dir = tempfile.mkdtemp()
    data_vars = ["u_1000", "v_1000", "u_850", "v_850", "u_700", "v_700"]
    fnames = create_mock_era5_data(temp_data_dir, data_vars, 4, 24, 60)

    # Configure for testing
    raw_config.setdefault("data", {}).update(
        {
            "data_dir": temp_data_dir,
            "train_fnames": fnames,
            "val_fnames": fnames,
        }
    )

    raw_config.setdefault("misc", {}).update(
        {
            "seed": 42,
            "num_workers": 1,
            "num_val_workers": 1,
            "logging": False,
            "pl_logging": False,
        }
    )

    raw_config.setdefault("params", {}).update(
        {
            "data_vars": data_vars,
            "time_range": ["2018-01-01", "2018-01-01"],
            "val_time_range": ["2018-01-01", "2018-01-01"],
        }
    )

    # Extract and instantiate experiment
    experiment_config, _ = extract_config(raw_config, [])
    experiment = instantiate(experiment_config)

    print(f" Model instantiated: {type(experiment.model)}")
    print(f" Parameters: {sum(p.numel() for p in experiment.model.parameters()):,}")

    # Generate test batch and run forward pass
    test_batch = experiment.generators.train.generate_batch()

    with torch.no_grad():
        if hasattr(test_batch, "time_grid") and test_batch.time_grid is not None:
            output = experiment.model(
                test_batch.xc, test_batch.yc, test_batch.xt, test_batch.time_grid
            )
        else:
            output = experiment.model(test_batch.xc, test_batch.yc, test_batch.xt)

    print(f"✓ Forward pass: {output.mean.shape}")

    # Cleanup
    shutil.rmtree(temp_data_dir, ignore_errors=True)

    return {"model": experiment.model, "output": output, "test_batch": test_batch}


if __name__ == "__main__":
    results = test_multimodal_tnp_encoder_from_configs()
    print(f" Test successful! Output: {results['output'].mean.shape}")
    print(f" Data variables in test batch: {results['test_batch'].xc.keys()}")
    key = list(results['test_batch'].xc.keys())[0]
    print(f"    Context locations for each variable shape: {results['test_batch'].xc[key].shape}")
    print(f"    Context values for each variable shape: {results['test_batch'].yc[key].shape}")
    print(f" Target locations shape: {results['test_batch'].xt.shape}")
    print(f" Target values shape: {results['test_batch'].yt.shape}")