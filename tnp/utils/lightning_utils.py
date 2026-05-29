import dataclasses
import time
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn

from ..data.base import Batch
from .experiment_utils import ModelCheckpointer, np_loss_fn, np_pred_fn


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        loss_fn: Callable = np_loss_fn,
        pred_fn: Callable = np_pred_fn,
        plot_fn: Optional[Callable] = None,
        checkpointer: Optional[ModelCheckpointer] = None,
        plot_interval: int = 1,
    ):
        super().__init__()

        self.model = model
        self.optimiser = (
            optimiser if optimiser is not None else torch.optim.Adam(model.parameters())
        )
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.plot_fn = plot_fn
        self.checkpointer = checkpointer
        self.plot_interval = plot_interval
        self.val_outputs: List[Any] = []
        self.test_outputs: List[Any] = []
        self.train_losses: List[Any] = []

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        loss = self.loss_fn(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach().cpu())
        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": batch}
        pred_dist = self.pred_fn(self.model, batch)
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        result["loglik"] = loglik.cpu()

        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()
        result["rmse"] = rmse

        if hasattr(batch, "var_stds"):
            if hasattr(batch, "non_gridded_var_names"):
                y_std = torch.as_tensor(
                    [batch.var_stds[k] for k in batch.non_gridded_var_names]
                ).cpu()
            else:
                assert hasattr(
                    batch, "var_names"
                ), "Batch should have 'var_names' to extract var_stds."
                y_std = torch.as_tensor(
                    [batch.var_stds[k] for k in batch.var_names]
                ).cpu()
            result["rmse"] = result["rmse"] * y_std

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            result["gt_loglik"] = gt_loglik.cpu()

        self.val_outputs.append(result)

    def test_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": _batch_to_cpu(batch)}
        pred_dist = self.pred_fn(self.model, batch)
        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        result["loglik"] = loglik.cpu()

        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()
        result["rmse"] = rmse

        if hasattr(batch, "var_stds"):
            if hasattr(batch, "non_gridded_var_names"):
                y_std = torch.as_tensor(
                    [batch.var_stds[k] for k in batch.non_gridded_var_names]
                ).cpu()
            else:
                assert hasattr(
                    batch, "var_names"
                ), "Batch should have 'var_names' to extract var_stds."
                y_std = torch.as_tensor(
                    [batch.var_stds[k] for k in batch.var_names]
                ).cpu()
            result["rmse"] = result["rmse"] * y_std

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()
            result["gt_loglik"] = gt_loglik.cpu()

        self.test_outputs.append(result)

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            return

        train_losses = torch.stack(self.train_losses)
        self.train_losses = []

        if self.checkpointer is not None:
            # For checkpointing.
            train_result = {
                "mean_loss": train_losses.mean(),
                "std_loss": train_losses.std() / (len(train_losses) ** 0.5),
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model,
                val_result=train_result,
                prefix="train_",
                update_last=True,
            )

    def on_validation_epoch_end(self) -> None:

        results = {
            k: [result[k] for result in self.val_outputs]
            for k in self.val_outputs[0].keys()
        }
        self.val_outputs = []

        loglik = torch.stack(results["loglik"])
        mean_loglik = loglik.mean()
        std_loglik = loglik.std() / (len(loglik) ** 0.5)
        self.log("val/loglik", mean_loglik)
        self.log("val/std_loglik", std_loglik)

        if "rmse" in results:
            rmse = torch.stack(results["rmse"])
            mean_rmse = rmse.mean()
            std_rmse = rmse.std() / (len(rmse) ** 0.5)
            self.log("val/rmse", mean_rmse)
            self.log("val/std_rmse", std_rmse)

        if self.checkpointer is not None:
            # For checkpointing.
            val_result = {
                "mean_loss": -mean_loglik,
                "std_loss": std_loglik,
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model,
                val_result=val_result,
                prefix="val_",
                update_last=False,
            )

        if "gt_loglik" in results:
            gt_loglik = torch.stack(results["gt_loglik"])
            mean_gt_loglik = gt_loglik.mean()
            std_gt_loglik = gt_loglik.std() / (len(gt_loglik) ** 0.5)
            self.log("val/gt_loglik", mean_gt_loglik)
            self.log("val/std_gt_loglik", std_gt_loglik)

        if self.plot_fn is not None and self.current_epoch % self.plot_interval == 0:
            self.plot_fn(
                self.model, results["batch"], f"epoch-{self.current_epoch:04d}"
            )

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            return {
                "optimizer": self.optimiser,
                "lr_scheduler": self.lr_scheduler(self.optimiser),
            }
        return self.optimiser


class LogPerformanceCallback(pl.Callback):

    def __init__(self):
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.forward_start_time = 0.0
        self.between_step_time = 0.0

    @pl.utilities.rank_zero_only
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "performance/between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
        )
        self.forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.forward_start_time
        pl_module.log(
            "performance/forward_time",
            forward_time,
            on_step=True,
            on_epoch=False,
        )
        self.backward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_after_backward(trainer, pl_module)
        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "performance/backward_time",
            backward_time,
            on_step=True,
            on_epoch=False,
        )

    @pl.utilities.rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.update_count += 1

        # Calculate total elapsed time
        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        # Calculate updates per second
        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        # Log updates per second to wandb using pl_module.log
        pl_module.log(
            "performance/average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            "performance/last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=False,
        )
        self.between_step_time = time.time()


def _batch_to_cpu(batch: Batch):
    batch_kwargs = {
        field.name: (
            getattr(batch, field.name).cpu()
            if isinstance(getattr(batch, field.name), torch.Tensor)
            else getattr(batch, field.name)
        )
        for field in dataclasses.fields(batch)
    }
    return type(batch)(**batch_kwargs)
