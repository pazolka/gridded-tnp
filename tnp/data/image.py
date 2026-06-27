from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torchvision

from ..utils.grids import flatten_grid
from .base import Batch, DataGenerator


@dataclass
class ImageBatch(Batch):
    mc_grid: torch.Tensor
    mt_grid: torch.Tensor


class ImageGenerator(DataGenerator):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        dataset: torchvision.datasets.VisionDataset,
        min_pc: float,
        max_pc: float,
        pt: Optional[float] = None,
        nt: Optional[int] = None,
        return_as_gridded: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            len(dataset[0][0].shape) == 3
        ), "dataset[0][0].shape must be (num_chanells, height, width)."

        self.dataset = dataset
        self.min_pc = min_pc
        self.max_pc = max_pc

        self.return_as_gridded = return_as_gridded

        sampler = torch.utils.data.RandomSampler(
            self.dataset, num_samples=self.samples_per_epoch
        )
        self.batch_sampler = iter(
            torch.utils.data.BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=True
            )
        )

        # Construct x values for grid.
        x = torch.stack(
            torch.meshgrid(
                *[torch.range(0, dim - 1) for dim in self.dataset[0][0][0, ...].shape]
            ),
            dim=-1,
        )
        self.grid_shape = x.shape[:-1]

        assert pt is not None or nt is not None, "pt or nt must be provided."
        self.pt = (
            pt if pt is not None else nt / (self.grid_shape[0] * self.grid_shape[1])
        )

        x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)

    def __iter__(self):
        """Reset epoch counter and batch sampler and return self."""
        # Create batch sampler.
        sampler = torch.utils.data.RandomSampler(
            self.dataset, num_samples=self.samples_per_epoch
        )
        self.batch_sampler = iter(
            torch.utils.data.BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=True
            )
        )

        if self.deterministic and self.batches is None:
            # Set deterministic seed.
            current_seed = torch.seed()
            torch.manual_seed(self.deterministic_seed)
            self.batches = [self.generate_batch() for _ in range(self.num_batches)]
            torch.manual_seed(current_seed)

        self.batch_counter = 0
        return self

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        """Generate batch of data.

        Returns:
            Batch: Tuple of tensors containing the context and target data.
        """

        # Sample context masks.
        pc = torch.rand(size=()) * (self.max_pc - self.min_pc) + self.min_pc

        # Sample batch of data.
        batch = self.sample_batch(
            pc=pc, pt=self.pt, batch_shape=torch.Size((self.batch_size,))
        )

        return batch

    def sample_masks(self, prop: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """Sample context masks.

        Returns:
            mc: Context mask.
        """

        # Sample proportions to mask.
        num_pixels = self.grid_shape[0] * self.grid_shape[1]
        num_mask = num_pixels * prop
        rand = torch.rand(size=(*batch_shape, *self.grid_shape))
        rand, flat_to_grid_fn = flatten_grid(rand[..., None])
        randperm = rand.argsort(dim=-2)
        randperm = flat_to_grid_fn(randperm)[..., 0]
        mc = randperm < num_mask

        return mc

    def sample_batch(
        self, pc: torch.Tensor, pt: torch.Tensor, batch_shape: torch.Size
    ) -> Batch:
        """Sample batch of data.

        Args:
            pc: Proportion of context points to sample.
            pt: Proportion of target points to sample.

        Returns:
            batch: Batch of data.
        """

        # Set up grids of masks.
        mc_grid = self.sample_masks(prop=pc, batch_shape=batch_shape)
        mt_grid = self.sample_masks(prop=pt, batch_shape=batch_shape)

        # Sample batch of data.
        batch_idx = next(self.batch_sampler)

        # (batch_size, num_channels, height, width).
        y_grid = torch.stack([self.dataset[idx][0] for idx in batch_idx], dim=0)
        y_grid = einops.rearrange(y_grid, "m d n1 n2 -> m n1 n2 d")

        # Input grid.
        x_grid = torch.stack(
            torch.meshgrid(
                *[torch.range(0, dim - 1) for dim in y_grid[0, ..., 0].shape]
            ),
            dim=-1,
        )
        x_grid = (x_grid - self.x_mean) / self.x_std
        x_grid = einops.repeat(x_grid, "n1 n2 d -> m n1 n2 d", m=y_grid.shape[0])

        if self.return_as_gridded:
            raise NotImplementedError

        xc = torch.stack(
            [x_grid_[mc_grid_] for x_grid_, mc_grid_ in zip(x_grid, mc_grid)]
        )
        yc = torch.stack(
            [y_grid_[mc_grid_] for y_grid_, mc_grid_ in zip(y_grid, mc_grid)]
        )
        xt = torch.stack(
            [x_grid_[mt_grid_] for x_grid_, mt_grid_ in zip(x_grid, mt_grid)]
        )
        yt = torch.stack(
            [y_grid_[mt_grid_] for y_grid_, mt_grid_ in zip(y_grid, mt_grid)]
        )
        return ImageBatch(
            x=x_grid,
            y=y_grid,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            mc_grid=mc_grid,
            mt_grid=mt_grid,
        )
