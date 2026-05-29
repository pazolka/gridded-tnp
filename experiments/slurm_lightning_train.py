import argparse
import logging
import os
import resource
import signal
import warnings
from typing import Dict, Optional, Tuple

import dask
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from plot_mm_era5 import plot_mm_era5
from plot_ootg_era5 import plot_ootg_era5

import wandb
from tnp.data.era5.era5 import ERA5DataGenerator
from tnp.data.era5.mm_era5 import MultiModalERA5DataGenerator
from tnp.data.era5.ootg_era5 import ERA5OOTGDataGenerator
from tnp.data.synthetic import SyntheticGenerator
from tnp.utils.data import adjust_num_batches
from tnp.utils.experiment_utils import deep_convert_dict, extract_config
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
dask.config.set(scheduler="synchronous")


def initialize_experiment() -> Tuple[DictConfig, Dict, Optional[str]]:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+")
    parser.add_argument("--wandb_checkpoint", type=str, default=None)
    args, config_changes = parser.parse_known_args()

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise experiment, make path.
    config, config_dict = extract_config(raw_config, config_changes)

    # Instantiate experiment and load checkpoint.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    pl.seed_everything(experiment.misc.seed)

    return experiment, config_dict, args.wandb_checkpoint


def main():
    experiment, config_dict, wandb_checkpoint = initialize_experiment()

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
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=(
            min(5, gen_train.num_batches // experiment.misc.num_workers)
            if experiment.misc.num_workers > 0
            else None
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            adjust_num_batches if experiment.misc.num_val_workers > 0 else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        prefetch_factor=(
            min(2, gen_val.num_batches // experiment.misc.num_val_workers)
            if experiment.misc.num_val_workers > 0
            else None
        ),
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
        else:
            pass

    if wandb_checkpoint:
        # Use the API to fetch the artifact.
        api = wandb.Api()
        artifact = api.artifact(wandb_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        # TODO: we will not have to specify model, optimiser etc. once using self.save_hyperparameters().
        lit_model = LitWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_file,
            map_location="cpu",
            strict=True,
            model=model,
            optimiser=optimiser,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )

    if experiment.misc.pl_logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            id=os.environ["SLURM_JOB_ID"],
            config=config_dict,
            resume="allow",
            log_model="all",
        )
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=25, save_last=True
    )
    performance_callback = LogPerformanceCallback()

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=(gen_train.num_batches // max(1, train_loader.num_workers))
        * max(1, train_loader.num_workers),
        limit_val_batches=(gen_val.num_batches // max(1, val_loader.num_workers))
        * max(1, val_loader.num_workers),
        log_every_n_steps=10,
        devices=torch.cuda.device_count(),
        num_nodes=1,
        strategy="ddp",
        gradient_clip_val=experiment.misc.gradient_clip_val,
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        enable_progress_bar=not (
            hasattr(experiment.misc, "progress_bar")
            and not experiment.misc.progress_bar
        ),
        callbacks=[checkpoint_callback, performance_callback],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Allowing opening more files.
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
