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

import wandb
from tnp.utils.data import adjust_num_batches
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
    pl.seed_everything(config.misc.seed)

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

    pl.seed_everything(experiment.misc.seed)

    test_loader = torch.utils.data.DataLoader(
        gen_test,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            adjust_num_batches if experiment.misc.num_workers > 0 else None
        ),
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        prefetch_factor=(
            min(2, gen_test.num_batches // experiment.misc.num_workers)
            if experiment.misc.num_workers > 0
            else None
        ),
        pin_memory=True,
    )

    # Store number of parameters.
    num_params = sum(p.numel() for p in lit_model.model.parameters())

    trainer = pl.Trainer(devices=torch.cuda.device_count(), accelerator="auto")
    trainer.test(model=lit_model, dataloaders=test_loader)
    test_result = {
        k: [result[k] for result in lit_model.test_outputs]
        for k in lit_model.test_outputs[0].keys()
    }
    loglik = torch.stack(test_result["loglik"])
    test_result["mean_loglik"] = loglik.mean()
    test_result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)

    rmse = torch.stack(test_result["rmse"])
    test_result["mean_rmse"] = rmse.mean()
    test_result["std_rmse"] = rmse.std() / (len(rmse) ** 0.5)

    if "gt_loglik" in test_result:
        gt_loglik = torch.stack(test_result["gt_loglik"])
        test_result["mean_gt_loglik"] = gt_loglik.mean()
        test_result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

    if experiment.misc.logging:
        wandb.run.summary["num_params"] = num_params
        wandb.run.summary[f"test/{eval_name}/loglik"] = test_result["mean_loglik"]
        wandb.run.summary[f"test/{eval_name}/std_loglik"] = test_result["std_loglik"]
        wandb.run.summary[f"test/{eval_name}/rmse"] = test_result["mean_rmse"]
        wandb.run.summary[f"test/{eval_name}/std_rmse"] = test_result["std_rmse"]
        if "mean_gt_loglik" in test_result:
            wandb.run.summary[f"test/{eval_name}/gt_loglik"] = test_result[
                "mean_gt_loglik"
            ]
            wandb.run.summary[f"test/{eval_name}/std_gt_loglik"] = test_result[
                "std_gt_loglik"
            ]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Allowing opening more files.
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    main()
