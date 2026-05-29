import argparse
import logging
import os
import resource
import warnings

import dask
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from plot_windspeed_us import plot_windspeed_us

import wandb
from tnp.utils.experiment_utils import deep_convert_dict, extract_config
from tnp.utils.lightning_utils import LitWrapper

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
dask.config.set(scheduler="synchronous")


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_path",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument("--config", type=str, nargs="+")
    args, config_changes = parser.parse_known_args()

    raw_config = deep_convert_dict(
        hiyapyco.load(
            args.config,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )
    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, config_changes)

    # Initialise wandb.
    api = wandb.Api()  # type: ignore[attr-defined]
    run = api.run(args.run_path)
    run = wandb.init(  # type: ignore[attr-defined]
        resume="must",
        project=run.project,
        name=run.name,
        id=run.id,
    )

    # Set the model to run.config.model.
    # config.model = run.config["model"]

    # Instantiate.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)

    # Load checkpoint from run artifact.
    artifact = run.use_artifact(args.checkpoint)
    artifact_dir = artifact.download()
    ckpt_file = os.path.join(artifact_dir, "model.ckpt")

    ckpt = torch.load(ckpt_file, map_location="cpu")
    print(f"Checkpoint epochs: {ckpt['epoch']}")

    if args.model_config is not None:
        raw_model_config = deep_convert_dict(
            hiyapyco.load(
                args.model_config,
                method=hiyapyco.METHOD_MERGE,
                usedefaultyamlloader=True,
            )
        )
        model_config, _ = extract_config(raw_model_config)
        model = instantiate(model_config).model
        experiment.lit_model = LitWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_file,
            map_location="cpu",
            model=model,
            strict=True,
        )
    else:
        # Load in the checkpoint.
        experiment.lit_model = LitWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_file,
            map_location="cpu",
            strict=True,
        )

    return experiment


def main():
    experiment = initialize_evaluation()

    lit_model = experiment.lit_model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    # Get a single batch from gen_test.
    iter_gen_test = iter(gen_test)

    # Set the seed.
    pl.seed_everything(experiment.misc.seed)
    batches = [next(iter_gen_test) for _ in range(experiment.misc.num_plots)]
    plot_windspeed_us(
        lit_model.model.to("cuda"),
        batches,
        num_fig=experiment.misc.num_plots,
        figsize=(15, 8),
        name=wandb.run.name,
        logging=experiment.misc.logging,
        pred_fn=lit_model.pred_fn,
        savefig=True,
        lat_range=(
            experiment.misc.lat_range
            if (
                hasattr(experiment.misc, "lat_range")
                and experiment.misc.lat_range is not None
            )
            else experiment.params.lat_range
        ),
        lon_range=(
            experiment.misc.lon_range
            if (
                hasattr(experiment.misc, "lon_range")
                and experiment.misc.lon_range is not None
            )
            else experiment.params.lon_range
        ),
    )

    wandb.run.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Allowing opening more files.
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
