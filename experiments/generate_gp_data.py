import argparse
import multiprocessing as mp
import os
from functools import partial
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def generate_and_save_batch(
    config_i: Tuple[DictConfig, int],
    gen_name: str,
    output_dir: str,
    compute_gt: bool = False,
):
    config, i = config_i

    # Check if batch already exists.
    batch_dir = os.path.join(output_dir, f"batch_{i}")
    if os.path.exists(batch_dir):
        return

    experiment = instantiate(config)
    gp_gen = getattr(experiment.generators, gen_name)

    torch.manual_seed(i)
    batch = next(iter(gp_gen))

    store_batch = {
        "x": batch.x,
        "y": batch.y,
        "xc": batch.xc,
        "yc": batch.yc,
        "xt": batch.xt,
        "yt": batch.yt,
    }

    if compute_gt:
        gt_pred = batch.gt_pred
        gt_mean, gt_std, gt_loglik = gt_pred(batch.xc, batch.yc, batch.xt, batch.yt)
        store_batch["gt_mean"] = gt_mean
        store_batch["gt_std"] = gt_std
        store_batch["gt_loglik"] = gt_loglik

    # Save batch.
    os.makedirs(batch_dir, exist_ok=True)
    for k, v in store_batch.items():
        torch.save(v, os.path.join(batch_dir, f"{k}.pt"))


def main(
    config: str,
    gen_name: str,
    output_dir: str,
    num_processes: int,
    compute_gt: bool = False,
):
    config = OmegaConf.load(config)

    # Create experiment in main process to get num_batches
    experiment = instantiate(config)
    num_batches = getattr(experiment.generators, gen_name).num_batches

    output_dir = os.path.join(output_dir, gen_name)

    args_list = [(config, i) for i in range(num_batches)]
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            generate_and_save_batch_partial = partial(
                generate_and_save_batch,
                gen_name=gen_name,
                output_dir=output_dir,
                compute_gt=compute_gt,
            )
            list(
                tqdm(
                    pool.imap(generate_and_save_batch_partial, args_list),
                    total=num_batches,
                )
            )
    else:
        for i in range(num_batches):
            generate_and_save_batch(
                (config, i),
                gen_name=gen_name,
                output_dir=output_dir,
                compute_gt=compute_gt,
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_name", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, required=False, default="./data/big-gp/"
    )
    parser.add_argument(
        "--num_processes", type=int, required=False, default=mp.cpu_count()
    )
    parser.add_argument("--compute_gt", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    main(
        args.config, args.gen_name, args.output_dir, args.num_processes, args.compute_gt
    )
