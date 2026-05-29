import copy
import os
import warnings
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from tnp.data.base import Batch
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: nn.Module,
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 16,
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    steps = int(points_per_dim * (x_range[1] - x_range[0]))

    x_plot = torch.linspace(x_range[0], x_range[1], steps).to(batches[0].xc)
    # Mesh grid if more than 1 dimension.
    if batches[0].xc.shape[-1] > 1:
        x_plot = torch.meshgrid([x_plot] * batches[0].xc.shape[-1])
        x_plot = torch.stack(x_plot, dim=-1).flatten(start_dim=0, end_dim=-2)
        x_plot = x_plot[None, ...]
    else:
        x_plot = x_plot[None, :, None]

    for i in range(num_fig):
        batch = batches[i]
        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = x_plot

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch)
            yt_pred_dist = pred_fn(model, batch)

        model_nll = -yt_pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            with torch.no_grad():
                gt_mean, gt_std, _ = batch.gt_pred(
                    xc=batch.xc,
                    yc=batch.yc,
                    xt=x_plot,
                )
                _, _, gt_loglik = batch.gt_pred(
                    xc=batch.xc,
                    yc=batch.yc,
                    xt=batch.xt,
                    yt=batch.yt,
                )
                gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

        if batch.xc.shape[-1] == 1:

            # Make figure for plotting
            fig = plt.figure(figsize=figsize)

            # Plot context and target points
            plt.scatter(
                batch.xc[0, :, 0].cpu(),
                batch.yc[0, :, 0].cpu(),
                c="k",
                label="Context",
                s=30,
            )

            plt.scatter(
                batch.xt[0, :, 0].cpu(),
                batch.yt[0, :, 0].cpu(),
                c="r",
                label="Target",
                s=30,
            )

            # Plot model predictions
            plt.plot(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu(),
                c="tab:blue",
                lw=3,
            )

            plt.fill_between(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
                mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            title_str = f"$N = {batch.xc.shape[1]}$ NLL = {model_nll:.3f}"

            if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
                # Plot ground truth
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )

                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )

                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    label="Ground truth",
                    lw=3,
                )

                title_str += f" GT NLL = {gt_nll:.3f}"

            plt.title(title_str, fontsize=24)
            plt.grid()

            # Set axis limits
            plt.xlim(x_range)
            plt.ylim(y_lim)

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)

            plt.legend(loc="upper right", fontsize=20)
            plt.tight_layout()

        elif batch.xc.shape[-1] == 2:
            fig = plt.figure(figsize=figsize)

            ax = plt.axes(projection="3d")

            # ax.scatter(
            #     batch.xc[0, :, 0].cpu(),
            #     batch.xc[0, :, 1].cpu(),
            #     batch.yc[0, :, 0].cpu(),
            #     c="k",
            #     label="Context",
            #     s=30,
            # )

            # ax.scatter(
            #     batch.xt[0, :, 0].cpu(),
            #     batch.xt[0, :, 1].cpu(),
            #     batch.yt[0, :, 0].cpu(),
            #     c="r",
            #     label="Target",
            #     s=30,
            # )

            ax.plot_trisurf(
                x_plot[0, :, 0].cpu(),
                x_plot[0, :, 1].cpu(),
                mean[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            title_str = f"$N = {batch.xc.shape[1]}$ NLL = {model_nll:.3f}"

            if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
                # Plot ground truth
                ax.plot_trisurf(
                    x_plot[0, :, 0].cpu(),
                    x_plot[0, :, 1].cpu(),
                    gt_mean[0, :].cpu(),
                    color="tab:purple",
                    alpha=0.5,
                    label="Ground truth",
                )

                title_str += f" GT NLL = {gt_nll:.3f}"

            ax.set_xlabel("$x_1$", fontsize=24)
            ax.set_ylabel("$x_2$", fontsize=24)
            ax.set_zlabel("$y$", fontsize=24)
            ax.legend(fontsize=20)
            ax.set_title(title_str, fontsize=24)
        else:
            warnings.warn(f"Unsupported number of dimensions: {batch.xc.shape[-1]}")
            return

        fname = f"fig/{name}/{i:03d}"
        if wandb.run is not None and logging:  # type: ignore[attr-defined]
            wandb.log({fname: wandb.Image(fig)})  # type: ignore[attr-defined]
        elif savefig:
            if not os.path.isdir(f"fig/{name}"):
                os.makedirs(f"fig/{name}")
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
