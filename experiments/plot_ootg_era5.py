import copy
import os
from typing import Callable, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from tnp.data.era5.ootg_era5 import ERA5OOTGBatch
from tnp.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_ootg_era5(
    model: nn.Module,
    batches: List[ERA5OOTGBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 4.0),
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    colorbar: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = plot_batch.x

        with torch.no_grad():
            yplot_pred_dist = pred_fn(model, plot_batch)
            pred_mean, pred_std = (
                yplot_pred_dist.mean.cpu(),
                yplot_pred_dist.stddev.cpu(),
            )

        # Get variable means and stds.
        y_mean = torch.as_tensor(
            [batch.var_means[k] for k in batch.non_gridded_var_names]
        )
        y_std = torch.as_tensor(
            [batch.var_stds[k] for k in batch.non_gridded_var_names]
        )
        y_gridded_mean = torch.as_tensor(
            [batch.var_means[k] for k in batch.gridded_var_names]
        )
        y_gridded_std = torch.as_tensor(
            [batch.var_stds[k] for k in batch.gridded_var_names]
        )

        # Rescale inputs and outputs.
        yc = (batch.yc[0].cpu() * y_std) + y_mean
        y = (batch.y[0].cpu() * y_std) + y_mean
        yc_grid = ((batch.yc_grid[0].cpu() * y_gridded_std) + y_gridded_mean).flatten(
            0, -2
        )
        pred_mean = (pred_mean[0] * y_std) + y_mean
        pred_std = pred_std[0] * y_std
        xc = batch.xc[0].cpu()
        x = batch.x[0].cpu()
        xc_grid = batch.xc_grid[0].cpu().flatten(0, -2)

        # Colourmap limits.
        vmin = min(y.min(), y.max())
        vmax = max(y.max(), y.max())

        scatter_kwargs = {
            "s": 5,
            "marker": "s",
            "alpha": 1.0,
            "vmin": vmin,
            "vmax": vmax,
        }

        for fig_name, x_plot, y_plot in zip(
            ("context", "gridded context", "ground_truth", "pred_mean", "pred_std"),
            (xc, xc_grid, x, x, x),
            (yc, yc_grid, y, pred_mean, pred_std),
        ):

            fig = plt.figure(figsize=figsize)

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.set_axisbelow(True)

            gl = ax.gridlines(draw_labels=True)
            gl.xlabel_style = {"size": 15}
            gl.ylabel_style = {"size": 15}

            if fig_name == "pred_std":
                std_scatter_kwargs = scatter_kwargs.copy()
                std_scatter_kwargs["vmin"] = y_plot.min()
                std_scatter_kwargs["vmax"] = y_plot.max()
                sc = ax.scatter(
                    x_plot[:, -1], x_plot[:, -2], c=y_plot, **std_scatter_kwargs
                )
            elif fig_name == "gridded context":
                gridded_scatter_kwargs = scatter_kwargs.copy()
                gridded_scatter_kwargs["vmin"] = y_plot.min()
                gridded_scatter_kwargs["vmax"] = y_plot.max()
                sc = ax.scatter(
                    x_plot[:, -1], x_plot[:, -2], c=y_plot, **gridded_scatter_kwargs
                )
            else:
                sc = ax.scatter(
                    x_plot[:, -1], x_plot[:, -2], c=y_plot, **scatter_kwargs
                )

            # Add colourbar.
            if colorbar:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.09)
                cbar.solids.set(alpha=1)

            # Set lat and lon limits.
            lat_range, lon_range = batch.lat_range, batch.lon_range
            ax.set_xlim(lon_range)
            ax.set_ylim(lat_range)

            plt.tight_layout()

            fname = f"fig/{name}/{i:03d}/{fig_name}"
            if wandb.run is not None and logging:  # type: ignore[attr-defined]
                wandb.log({fname: wandb.Image(fig)})  # type: ignore[attr-defined]
            elif savefig:
                if not os.path.isdir(f"fig/{name}/{i:03d}"):
                    os.makedirs(f"fig/{name}/{i:03d}")
                plt.savefig(fname)
            else:
                plt.show()

            plt.close()
