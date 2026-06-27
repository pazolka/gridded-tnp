import os
from typing import Optional

import torch

from .base import Batch, DataGenerator
from .synthetic import SyntheticBatch


class PreGeneratedSyntheticGenerator(DataGenerator):
    def __init__(self, *, data_dir: str, random_sample: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.random_sample = random_sample

        # Check that data_dir exists, and get number of files (batches) it contains.
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        self.n_batches = len(os.listdir(data_dir))
        self.batch_idx = 0

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        if batch_shape is None:
            batch_shape = torch.Size((self.batch_size,))

        # Sample random batches from data_dir.
        if self.random_sample:
            batch_idx = torch.randint(low=0, high=self.n_batches, size=())
        else:
            batch_idx = self.batch_idx % self.n_batches
            self.batch_idx += 1

        # Load batch from data_dir.
        batch_dir = os.path.join(self.data_dir, f"batch_{batch_idx}")
        xc = torch.load(os.path.join(batch_dir, "xc.pt")).to(torch.float)
        xt = torch.load(os.path.join(batch_dir, "xt.pt")).to(torch.float)
        yc = torch.load(os.path.join(batch_dir, "yc.pt")).to(torch.float)
        yt = torch.load(os.path.join(batch_dir, "yt.pt")).to(torch.float)

        if os.path.exists(os.path.join(batch_dir, "gt_mean.pt")):
            gt_mean = torch.load(os.path.join(batch_dir, "gt_mean.pt")).to(torch.float)
            gt_std = torch.load(os.path.join(batch_dir, "gt_std.pt")).to(torch.float)
            gt_loglik = torch.load(os.path.join(batch_dir, "gt_loglik.pt")).to(
                torch.float
            )
        else:
            gt_mean = None
            gt_std = None
            gt_loglik = None

        # Check that all tensors have the correct shape.
        if xc.shape[0] >= batch_shape[0]:
            # Select random subset of data.
            idx = torch.randperm(xc.shape[0])[: batch_shape[0]]
            xc = xc[idx]
            yc = yc[idx]
            xt = xt[idx]
            yt = yt[idx]
            if gt_mean is not None:
                gt_mean = gt_mean[idx]
                gt_std = gt_std[idx]
                gt_loglik = gt_loglik[idx]
        else:
            raise ValueError(f"Invalid batch shape loaded: {xc.shape[0]}")

        # Concatenate context and target data.
        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            xt=xt,
            yc=yc,
            yt=yt,
            gt_mean=gt_mean,
            gt_std=gt_std,
            gt_loglik=gt_loglik,
        )
