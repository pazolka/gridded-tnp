import logging
import warnings

import dask
import lightning.pytorch as pl
import torch
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from plot_mm_era5 import plot_mm_era5
from plot_ootg_era5 import plot_ootg_era5

from tnp.data.era5.era5 import ERA5DataGenerator
from tnp.data.era5.mm_era5 import MultiModalERA5DataGenerator
from tnp.data.era5.ootg_era5 import ERA5OOTGDataGenerator
from tnp.data.image import ImageGenerator
from tnp.data.synthetic import SyntheticGenerator
from tnp.utils.data import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
dask.config.set(scheduler="synchronous")


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_workers > 0
            else None
        ),
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=5 if experiment.misc.num_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=False,
        prefetch_factor=2 if experiment.misc.num_val_workers > 0 else None,
        pin_memory=True,
    )

    def plot_fn(model, batches, name):
        if isinstance(gen_train, SyntheticGenerator) and gen_train.dim == 1:
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        elif isinstance(gen_train, MultiModalERA5DataGenerator):
            plot_mm_era5(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        elif isinstance(gen_train, ERA5OOTGDataGenerator):
            plot_ootg_era5(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        elif isinstance(gen_train, ERA5DataGenerator):
            plot_era5(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        elif isinstance(gen_train, ImageGenerator):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        else:
            pass

    lit_model = LitWrapper(
        model=model,
        optimiser=optimiser,
        loss_fn=experiment.misc.loss_fn,
        pred_fn=experiment.misc.pred_fn,
        plot_fn=plot_fn,
        checkpointer=checkpointer,
        plot_interval=experiment.misc.plot_interval,
    )
    logger = pl.loggers.WandbLogger() if experiment.misc.logging else False
    performance_callback = LogPerformanceCallback()
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=5,
        devices=1,
        gradient_clip_val=experiment.misc.gradient_clip_val,
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        enable_progress_bar=experiment.misc.progress_bars,
        callbacks=[performance_callback],
    )

    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
