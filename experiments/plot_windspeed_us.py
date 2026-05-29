import copy
import os
import warnings
from functools import partial
from typing import Callable, Dict, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import anderson, kstest, norm
from torch import nn

import wandb
from tnp.data.era5.mm_era5 import ERA5MultiModalBatch
from tnp.utils.experiment_utils import np_pred_fn

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12
plt.rcParams["text.usetex"] = False
plt.rc("axes", axisbelow=True)


def vector_to_rgb(angle: np.array, absolute: np.array, max_abs: float):
    """Get the rgb value for the given `angle` and the `absolute` value

    Parameters
    ----------
    angle : float
        The angle in radians
    absolute : float
        The absolute value of the gradient

    Returns
    -------
    array_like
        The rgb value as a tuple with values [0..1]
    """
    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    return matplotlib.colors.hsv_to_rgb(
        (angle / 2 / np.pi, absolute / max_abs, absolute / max_abs)
    )


def plot_windspeed_us(
    model: nn.Module,
    batches: List[ERA5MultiModalBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (15.0, 3.0),
    name: str = "wind-speed-us",
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
    lon_range: Tuple[float, float] = (-125.0, -66.0),
    lat_range: Tuple[float, float] = (25.0, 49.0),
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
            # Move plot_batch to the same device as the model.
            for key, value in vars(plot_batch).items():
                if isinstance(value, torch.Tensor):
                    setattr(plot_batch, key, value.to(next(model.parameters()).device))
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            value[k] = v.to(next(model.parameters()).device)

            yplot_pred_dist = pred_fn(model, plot_batch)
            pred_mean, pred_std = (
                yplot_pred_dist.mean.cpu(),
                yplot_pred_dist.stddev.cpu(),
            )

        # Get variables means and stds.
        y_mean = torch.as_tensor([batch.var_means[k] for k in batch.var_names])
        y_std = torch.as_tensor([batch.var_stds[k] for k in batch.var_names])

        # Compute the predictive log-likelihoods.
        pred_ll = (
            yplot_pred_dist.log_prob(batch.y.to(yplot_pred_dist.mean.device))
            .cpu()
            .sum()
            / batch.y[..., 0].numel()
        )

        # Rescale inputs and outputs.
        # yc = {
        #     k: (v[0].cpu() * batch.var_stds[k]) + batch.var_means[k]
        #     for k, v in batch.yc.items()
        # }
        # y = (batch.y[0].cpu() * y_std) + y_mean
        y = batch.y[0].cpu()
        y = {k: y[..., i] for i, k in enumerate(batch.var_names)}
        # pred_mean = (pred_mean[0] * y_std) + y_mean
        pred_mean = pred_mean[0]
        pred_mean = {k: pred_mean[..., i] for i, k in enumerate(batch.var_names)}
        # pred_std = pred_std[0] * y_std
        pred_std = pred_std[0]
        pred_std = {k: pred_std[..., i] for i, k in enumerate(batch.var_names)}
        # xc = {k: v[0].cpu() for k, v in batch.xc.items()}
        x = batch.x[0].cpu()
        x = {k: x for k in batch.var_names}

        # vmin = {k: min(y[..., i].min(), yc[k].min()) for i, k in enumerate(yc.keys())}
        # vmax = {k: max(y[..., i].max(), yc[k].max()) for i, k in enumerate(yc.keys())}

        # Good for plotting entire US.
        # quiver_kwargs = {
        #     "width": 0.002,
        #     "scale_units": "xy",
        #     "scale": 20.0,
        #     "headwidth": 1.5,
        #     "headaxislength": 2,
        #     "headlength": 2,
        #     "transform": ccrs.PlateCarree(),
        # }
        quiver_kwargs = {
            "width": 0.004,
            "scale_units": "xy",
            "scale": 20.0,
            "headwidth": 2.0,
            "headaxislength": 2,
            "headlength": 2,
            "transform": ccrs.PlateCarree(),
        }

        variables = {
            "1000hPa": ("u_1000", "v_1000"),
            "850hPa": ("u_850", "v_850"),
            "700hPa": ("u_700", "v_700"),
        }

        # Compute y_diff.
        y_diff = {k: y[k] - pred_mean[k] for k in y.keys()}

        # Compute normalised errors.
        y_diff_normalised = {k: y_diff[k] / pred_std[k] for k in y_diff.keys()}

        # Compute Anderson-Darling test for normality.
        errors = torch.cat([v for v in y_diff_normalised.values()]).numpy()
        log_prob = norm.logpdf(errors, loc=0, scale=1)
        res = anderson(errors)
        ks_res = kstest(errors, partial(norm.cdf, loc=0, scale=1))
        print(
            f"Dataset: {i}. AD={res.statistic:.3f}. Critical values: {res.critical_values}. Significance levels: {res.significance_level}. KS={ks_res.statistic:.3f}. p-value={ks_res.pvalue:.3f}. log_prob={log_prob.mean():.3f}"
        )

        cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        # for fig_name, x_plot, y_plot in zip(
        #     # ("context", "ground_truth", "pred_mean", "pred_std"),
        #     # (xc, x, x, x),
        #     # (yc, y, pred_mean, pred_std),
        #     ("ground_truth", "pred_mean", "pred_std", "pred_error"),
        #     (x, x, x, x),
        #     (y, pred_mean, pred_std, y_diff),
        # ):
        #     cache[fig_name] = {}
        #     fig, axes = plt.subplots(
        #         figsize=figsize,
        #         nrows=1,
        #         ncols=3,
        #         dpi=200,
        #         subplot_kw=dict(projection=ccrs.PlateCarree()),
        #     )

        #     # Add features.
        #     for ax in axes.flatten():
        #         ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        #         # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        #         ax.add_feature(cfeature.LAND, linewidth=0.5)
        #         ax.add_feature(cfeature.OCEAN, linewidth=0.5)
        #         ax.set_xlim([lon_range[0], lon_range[1]])
        #         ax.set_ylim([lat_range[0], lat_range[1]])

        #     # Assumes same x for each variable.
        #     x_ = x_plot[batch.var_names[0]]
        #     for ax, (ax_name, vs) in zip(axes.flatten(), variables.items()):
        #         # Get wind speeds corresponding to the last time step.
        #         time_steps = torch.unique(x_[:, 0])
        #         mask = x_[:, 0] == time_steps[-1]

        #         # Get variables.
        #         u = y_plot[vs[0]][mask].numpy()
        #         v = y_plot[vs[1]][mask].numpy()
        #         lons = x_[mask, -1].numpy()
        #         lats = x_[mask, -2].numpy()

        #         # Get angles and lengths.
        #         angles = np.arctan2(u, v)
        #         lengths = np.sqrt(np.square(u), np.square(v))
        #         max_abs = np.max(lengths)

        #         cache[fig_name][ax_name] = {
        #             "u": u,
        #             "v": v,
        #             "lons": lons,
        #             "lats": lats,
        #             "angles": angles,
        #             "lengths": lengths,
        #             "max_abs": max_abs,
        #         }

        #     for ax, (ax_name, vs) in zip(axes.flatten(), variables.items()):
        #         u = cache[fig_name][ax_name]["u"]
        #         v = cache[fig_name][ax_name]["v"]
        #         lons = cache[fig_name][ax_name]["lons"]
        #         lats = cache[fig_name][ax_name]["lats"]
        #         angles = cache[fig_name][ax_name]["angles"]
        #         lengths = cache[fig_name][ax_name]["lengths"]
        #         max_abs = cache[fig_name][ax_name]["max_abs"]

        #         if fig_name == "pred_error":
        #             # Normalise errors by max_abs of ground truth.
        #             max_abs = cache["ground_truth"][ax_name]["max_abs"]

        #             # Scale errors by a factor of 3.
        #             lengths_ = lengths * 3.0
        #             u *= 3.0 * (1 / np.sqrt(2))
        #             v *= 3.0 * (1 / np.sqrt(2))
        #         else:
        #             lengths_ = lengths

        #         # max_abs = max(
        #         #     cache["ground_truth"][ax_name_]["max_abs"]
        #         #     for ax_name_ in variables.keys()
        #         # )
        #         # max_abs = cache["ground_truth"][ax_name]["max_abs"]

        #         # Cap lengths at max_abs.
        #         if max_abs > lengths_.max():
        #             warnings.warn(
        #                 f"max_abs {max_abs} is less than the max length {lengths_.max()}"
        #             )
        #         lengths_ = np.clip(lengths_, 0, max_abs)

        #         # Get colors.
        #         color = np.array(
        #             list(
        #                 map(
        #                     vector_to_rgb,
        #                     angles.flatten(),
        #                     lengths_.flatten(),
        #                     np.ones_like(angles.flatten()) * max_abs,
        #                 )
        #             )
        #         )

        #         # Increase length of vectors.
        #         # scale_factor = 10.0 * (1 / np.sqrt(2))
        #         scale_factor = 5.0 * (1 / np.sqrt(2))
        #         u *= scale_factor
        #         v *= scale_factor
        #         # Normalise lengths.
        #         # u *= lengths / max_abs
        #         # v *= lengths / max_abs
        #         # max_length = max_abs * 0.5
        #         # mask = lengths > max_length
        #         # u[mask] *= max_length / lengths[mask]
        #         # v[mask] *= max_length / lengths[mask]

        #         ax.quiver(lons, lats, u, v, color=color, **quiver_kwargs)
        #         ax.set_title(ax_name)

        #     plt.tight_layout()

        #     # Compute context proportion.
        #     pc = batch.xc[batch.var_names[0]].numel() / batch.x.numel()
        #     fig_name = f"{fig_name}_pc={pc:.2f}_ll={pred_ll:.2f}_range={lat_range[0]}-{lat_range[1]}_{lon_range[0]}-{lon_range[1]}.png"

        #     fname = f"fig/{name}/{i:02d}/{fig_name}"
        #     if wandb.run is not None and logging:
        #         wandb.log({fname: wandb.Image(fig)})

        #     if savefig:
        #         if not os.path.isdir(f"fig/{name}/{i:02d}"):
        #             os.makedirs(f"fig/{name}/{i:02d}")
        #         plt.savefig(fname, bbox_inches="tight")
        #     # else:
        #     #     plt.show()

        #     plt.close()

        figsize = (6, 4)
        for pl_name, vs in variables.items():
            cache[pl_name] = {}
            for fig_name, x_plot, y_plot in zip(
                (
                    "ground_truth",
                    "pred_mean",
                    "pred_std",
                    "pred_error",
                    "pred_error_normalised",
                ),
                (x, x, x, x, x),
                (y, pred_mean, pred_std, y_diff, y_diff_normalised),
            ):
                cache[pl_name][fig_name] = {}

                # Add features.
                if fig_name != "pred_error_normalised":
                    fig, ax = plt.subplots(
                        figsize=figsize,
                        dpi=200,
                        subplot_kw=dict(projection=ccrs.PlateCarree()),
                    )
                    # ax = plt.gca()
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.LAND, linewidth=0.5)
                    ax.add_feature(cfeature.OCEAN, linewidth=0.5)
                    ax.set_xlim([lon_range[0], lon_range[1]])
                    ax.set_ylim([lat_range[0], lat_range[1]])
                else:
                    fig, ax = plt.subplots(figsize=figsize, dpi=200)

                # Assumes same x for each variable.
                x_ = x_plot[batch.var_names[0]]

                # Get wind speeds corresponding to the last time step.
                time_steps = torch.unique(x_[:, 0])
                mask = x_[:, 0] == time_steps[-1]

                # Get variables.
                u = y_plot[vs[0]][mask].numpy()
                v = y_plot[vs[1]][mask].numpy()
                lons = x_[mask, -1].numpy()
                lats = x_[mask, -2].numpy()

                # Get angles and lengths.
                angles = np.arctan2(u, v)
                lengths = np.sqrt(np.square(u), np.square(v))
                max_abs = np.max(lengths)

                cache[pl_name][fig_name] = {
                    "u": u,
                    "v": v,
                    "lons": lons,
                    "lats": lats,
                    "angles": angles,
                    "lengths": lengths,
                    "max_abs": max_abs,
                }

                if fig_name == "pred_error":
                    # Normalise errors by max_abs of ground truth.
                    max_abs = cache[pl_name]["ground_truth"]["max_abs"]

                    # Scale errors by a factor of 3.
                    lengths_ = lengths * 3.0
                    u *= 3.0 * (1 / np.sqrt(2))
                    v *= 3.0 * (1 / np.sqrt(2))
                else:
                    lengths_ = lengths

                # Cap lengths at max_abs.
                if max_abs > lengths_.max():
                    warnings.warn(
                        f"max_abs {max_abs} is less than the max length {lengths_.max()}"
                    )
                lengths_ = np.clip(lengths_, 0, max_abs)

                # Get colors.
                color = np.array(
                    list(
                        map(
                            vector_to_rgb,
                            angles.flatten(),
                            lengths_.flatten(),
                            np.ones_like(angles.flatten()) * max_abs,
                        )
                    )
                )

                # Increase length of vectors.
                scale_factor = 5.0 * (1 / np.sqrt(2))
                u *= scale_factor
                v *= scale_factor

                if fig_name == "pred_std":
                    # Scale lengths by the standard deviation of the radius.
                    # Scale lengths_ equally.
                    # lengths_ = lengths_ / cache[pl_name]["ground_truth"]["max_abs"]
                    alpha = np.clip(lengths_, 0, 1)
                    ax.scatter(
                        lons,
                        lats,
                        c=lengths_,
                        alpha=alpha,
                        cmap="Reds",
                        marker="s",
                    )
                elif fig_name == "pred_error_normalised":
                    # Plot a histogram of normalised prediction errors and compare to standard normal.
                    # xmin = min(
                    #     y_diff_normalised[vs[0]].min(), y_diff_normalised[vs[1]].min()
                    # ).numpy()
                    # xmax = max(
                    #     y_diff_normalised[vs[0]].max(), y_diff_normalised[vs[1]].max()
                    # ).numpy()
                    xmin, xmax = -5, 5
                    ax.hist(
                        [
                            y_diff_normalised[vs[0]].numpy(),
                            y_diff_normalised[vs[1]].numpy(),
                        ],
                        bins=50,
                        density=True,
                        label=["u", "v"],
                        color=["tab:blue", "tab:orange"],
                        range=(xmin, xmax),
                        rwidth=1.0,
                    )
                    # ax.hist(
                    #     y_diff_normalised[vs[1]],
                    #     bins=50,
                    #     density=True,
                    #     alpha=0.5,
                    #     label=r"$v$ normalised error",
                    #     c="tab:orange",
                    #     range=(xmin, xmax),
                    # )
                    # Plot a standard normal distribution.
                    x_hist = np.linspace(xmin, xmax, 100)
                    ax.plot(
                        x_hist,
                        np.exp(-(x_hist**2) / 2) / np.sqrt(2 * np.pi),
                        c="black",
                        label=r"$\mathcal{N}(0, 1)$",
                    )
                    ax.grid(True)
                    ax.set_xlabel("Normalised error")
                    ax.set_ylabel("Density")
                    # Perform Anderson-Darling test for normality.
                    for var_name in vs:
                        errors = y_diff_normalised[var_name].numpy()
                        res = anderson(errors)
                        fig_name += f"_{var_name}_ad={res.statistic:.3f}"
                    ax.legend()
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([0, 0.45])
                else:
                    ax.quiver(lons, lats, u, v, color=color, **quiver_kwargs)

                ax.set_title(pl_name)

                plt.tight_layout()

                # Compute context proportion.
                pc = batch.xc[batch.var_names[0]].numel() / batch.x.numel()
                fig_name = f"{pl_name}_{fig_name}_pc={pc:.2f}_ll={pred_ll:.2f}_range={lat_range[0]}-{lat_range[1]}_{lon_range[0]}-{lon_range[1]}.png"

                fname = f"fig/{name}/{i:02d}/{fig_name}"
                if wandb.run is not None and logging:  # type: ignore[attr-defined]
                    wandb.log({fname: wandb.Image(fig)})  # type: ignore[attr-defined]

                if savefig:
                    if not os.path.isdir(f"fig/{name}/{i:02d}"):
                        os.makedirs(f"fig/{name}/{i:02d}")
                    plt.savefig(fname, bbox_inches="tight")

                plt.close()
