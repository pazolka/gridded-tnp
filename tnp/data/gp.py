import random
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, Union

import einops
import gpytorch
import torch

from .base import GroundTruthPredictor
from .synthetic import SyntheticBatch, SyntheticGeneratorUniformInput


class GPGenerator(ABC):
    def __init__(
        self,
        *,
        kernel: Union[
            List[Type[gpytorch.kernels.Kernel]], Type[gpytorch.kernels.Kernel]
        ],
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        noise_std: float,
        ard_num_dims: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        self.min_log10_lengthscale = torch.as_tensor(
            min_log10_lengthscale, dtype=torch.float
        )
        self.max_log10_lengthscale = torch.as_tensor(
            max_log10_lengthscale, dtype=torch.float
        )
        self.noise_std = noise_std
        self.ard_num_dims = ard_num_dims

    def set_up_gp(self) -> GroundTruthPredictor:
        # Sample lengthscale.
        if self.ard_num_dims is None:
            log10_lengthscale = (
                torch.rand(())
                * (self.max_log10_lengthscale - self.min_log10_lengthscale)
                + self.min_log10_lengthscale
            )
        else:
            log10_lengthscale = (
                torch.randn(self.ard_num_dims)
                * (self.max_log10_lengthscale - self.min_log10_lengthscale)
                + self.min_log10_lengthscale
            )

        lengthscale = 10.0**log10_lengthscale

        if isinstance(self.kernel, list):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        if hasattr(kernel, "base_kernel"):
            kernel.base_kernel.lengthscale = lengthscale
        else:
            kernel.lengthscale = lengthscale

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std**2.0

        return GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)

    def sample_outputs(
        self,
        x: torch.Tensor,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        gt_pred = self.set_up_gp()
        y = gt_pred.sample_outputs(x)
        return y, gt_pred


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        train_inputs: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        self.kernel = kernel
        self.likelihood = likelihood

        self._result_cache: Optional[Dict[str, torch.Tensor]] = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Move devices.
        old_device = xc.device
        device = self.kernel.device
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad():

                dist_ = gp_model(xt_)
                mean_ = dist_.mean
                std_ = dist_.stddev + self.likelihood.noise**0.5
                diag_dist_ = torch.distributions.Normal(mean_, std_)
                if yt is not None:
                    gt_loglik_ = diag_dist_.log_prob(yt[i, ..., 0])
                    gt_loglik_list.append(gt_loglik_)

                mean_list.append(mean_)
                std_list.append(std_)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        # Cache for deterministic validation batches.
        # Note yt is not specified when passing x_plot.
        if yt is not None:
            self._result_cache = {
                "mean": mean,
                "std": std,
                "gt_loglik": gt_loglik,
            }

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xt = xt.to(old_device)
        if yt is not None:
            yt = yt.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:

        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=self.kernel,
        )
        gp_model.eval()
        gp_model.likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = gp_model.likelihood(f)
            y = dist.sample()
            return y[..., None]


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass


class RandomScaleGPGeneratorSameInputs(RandomScaleGPGenerator):

    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        # Sample inputs, then outputs given inputs
        x = self.sample_inputs(nc=nc, nt=nt, batch_shape=torch.Size())
        gt_pred = self.set_up_gp()
        y = gt_pred.sample_outputs(x, sample_shape=batch_shape)

        x = einops.repeat(x, "n d -> b n d", b=batch_shape[0])

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred,
        )
