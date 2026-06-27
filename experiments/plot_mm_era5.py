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
from tnp.data.era5.mm_era5 import ERA5MultiModalBatch
from tnp.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_mm_era5(
    model: nn.Module,
    batches: List[ERA5MultiModalBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 4.0),
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    colorbar: bool = False,
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v[:1]

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = plot_batch.x

        with torch.no_grad():
            yplot_pred_dist = pred_fn(model, plot_batch)
            pred_mean, pred_std = (
                yplot_pred_dist.mean.cpu(),
                yplot_pred_dist.stddev.cpu(),
            )

        # Get variables means and stds.
        y_mean = torch.as_tensor([batch.var_means[k] for k in batch.var_names])
        y_std = torch.as_tensor([batch.var_stds[k] for k in batch.var_names])

        # Rescale inputs and outputs.
        yc = {
            k: (v[0].cpu() * batch.var_stds[k]) + batch.var_means[k]
            for k, v in batch.yc.items()
        }
        y = (batch.y[0].cpu() * y_std) + y_mean
        pred_mean = (pred_mean[0] * y_std) + y_mean
        pred_std = pred_std[0] * y_std
        xc = {k: v[0].cpu() for k, v in batch.xc.items()}
        x = batch.x[0].cpu()

        vmin = {k: min(y[..., i].min(), yc[k].min()) for i, k in enumerate(yc.keys())}
        vmax = {k: max(y[..., i].max(), yc[k].max()) for i, k in enumerate(yc.keys())}

        scatter_kwargs = {
            "s": 10,
            "marker": "s",
            "alpha": 1.0,
            "lw": 0,
        }

        for i, var_name in enumerate(batch.var_names):
            scatter_kwargs["vmin"] = vmin[var_name]
            scatter_kwargs["vmax"] = vmax[var_name]

            for fig_name, x_plot, y_plot in zip(
                ("context", "ground_truth", "pred_mean", "pred_std"),
                (xc[var_name], x, x, x),
                (yc[var_name], y[..., i], pred_mean[..., i], pred_std[..., i]),
            ):
                # Check if a time dimension exists.
                if x_plot.shape[-1] == 3:
                    # Get unique time steps and use final one.
                    time_steps = torch.unique(x_plot[..., 0])
                    mask = x_plot[..., 0] == time_steps[-1]
                    x_plot = x_plot[mask]
                    y_plot = y_plot[mask]

                fig = plt.figure(figsize=figsize)

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                ax.set_axisbelow(True)

                gl = ax.gridlines(draw_labels=True)
                gl.xlabel_style = {"size": 15}
                gl.ylabel_style = {"size": 15}
                # ax.tick_params(axis="both", which="major", labelsize=20)

                if fig_name == "pred_std":
                    std_scatter_kwargs = scatter_kwargs
                    std_scatter_kwargs["vmin"] = y_plot.min()
                    std_scatter_kwargs["vmax"] = y_plot.max()
                    sc = ax.scatter(
                        x_plot[:, -1], x_plot[:, -2], c=y_plot, **std_scatter_kwargs
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

                fname = f"fig/{name}/{i:03d}/{var_name}/{fig_name}"
                if wandb.run is not None and logging:  # type: ignore[attr-defined]
                    wandb.log({fname: wandb.Image(fig)})  # type: ignore[attr-defined]
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}"):
                        os.makedirs(f"fig/{name}/{i:03d}")
                    plt.savefig(fname)
                else:
                    plt.show()

                plt.close()
