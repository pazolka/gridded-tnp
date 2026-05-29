import copy
import os
from typing import Callable, List, Tuple

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import wandb
from tnp.data.image import ImageBatch
from tnp.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_image(
    model: nn.Module,
    batches: List[ImageBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (24.0, 8.0),
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = einops.rearrange(plot_batch.x, "1 n1 n2 d -> 1 (n1 n2) d")
        plot_batch.mt_grid = torch.full(batch.mt_grid.shape, True)

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch)
            yt_pred_dist = pred_fn(model, batch)

            mean, std = (
                y_plot_pred_dist.mean.cpu().numpy(),
                y_plot_pred_dist.stddev.cpu().numpy(),
            )
            model_nll = (
                -yt_pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
            )

        # Reorganise into grid.
        if batch.y.shape[-1] == 1:
            # Single channel.
            yc_ = np.ma.masked_where(
                ~batch.mc_grid[..., None].cpu().numpy(),
                batch.y.cpu().numpy(),
            )
        else:
            # Three channels.
            # Masking does not work for RGB images.
            yc_ = torch.cat((batch.y, batch.mc_grid[..., None]), dim=-1).cpu().numpy()

        # Assumes same height and width.
        h, w = batch.mc_grid.shape[1:]
        yc_ = yc_[0]
        y_ = batch.y.cpu().numpy()[0]
        mean = einops.rearrange(mean, "1 (n m) d -> n m d", n=h, m=w)
        std = einops.rearrange(std, "1 (n m) d -> n m d", n=h, m=w)

        if subplots:
            # Make figure for plotting
            fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=1)

            axes[0].imshow(yc_, cmap="gray", vmax=1, vmin=0)
            axes[1].imshow(mean, cmap="gray", vmax=1, vmin=0)
            axes[2].imshow(std, cmap="gray", vmax=std.max(), vmin=std.min())

            axes[0].set_title("Context set", fontsize=18)
            axes[1].set_title("Mean prediction", fontsize=18)
            axes[2].set_title("Std prediction", fontsize=18)

            plt.suptitle(
                f"prop_ctx = {batch.xc.shape[-2] / batch.x.shape[-2]:.2f}    "
                #
                f"NLL = {model_nll:.3f}",
                fontsize=24,
            )

            fname = f"fig/{name}/{i:03d}"
            if wandb.run is not None and logging:  # type: ignore[attr-defined]
                wandb.log({fname: wandb.Image(fig)})  # type: ignore[attr-defined]
            elif savefig:
                if not os.path.isdir(f"fig/{name}"):
                    os.makedirs(f"fig/{name}")
                plt.savefig(fname)
            else:
                plt.show()

            plt.close()

        else:
            for fig_name, y_plot in zip(
                ("context", "ground_truth", "pred_mean"), (yc_, y_, mean)
            ):
                fig = plt.figure(figsize=figsize)

                plt.imshow(y_plot, vmax=1, vmin=0)
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
