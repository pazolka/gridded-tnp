"""Bernoulli-Gamma likelihood for spike-and-slab modeling.

This likelihood is designed for data with:
- Point masses at zero (modeled by Bernoulli)
- Continuous positive values (modeled by Gamma)

Common applications: precipitation, presence/absence data with intensities.
"""

import torch
import torch.nn as nn
import torch.distributions as td
from abc import ABC, abstractmethod

from tnp.likelihoods.base import Likelihood


class BernoulliGammaLikelihood(Likelihood):
    """Bernoulli-Gamma mixture likelihood.
    
    Models data as a mixture of:
    - A point mass at zero (Bernoulli component)
    - Continuous positive values (Gamma component)
    
    The decoder outputs 3 parameters per target variable:
    - logits: Bernoulli log-odds of non-zero observation
    - shape: Gamma shape parameter (alpha, concentration)
    - rate: Gamma rate parameter (beta, inverse scale)
    
    Args:
        min_shape (float): Minimum Gamma shape to ensure stability. Default: 1e-4
        min_rate (float): Minimum Gamma rate to ensure stability. Default: 1e-4
    """
    
    def __init__(self, min_shape: float = 1e-4, min_rate: float = 1e-4):
        super().__init__()
        self.min_shape = min_shape
        self.min_rate = min_rate
    
    def forward(self, x: torch.Tensor) -> "BernoulliGammaDistribution":
        """
        Args:
            x: Tensor of shape [..., 3*dim_y] containing raw decoder outputs:
                - x[..., 0::3]: Bernoulli logits
                - x[..., 1::3]: Gamma shape (pre-activation)
                - x[..., 2::3]: Gamma rate (pre-activation)
        
        Returns:
            BernoulliGammaDistribution object
        """
        # Reshape to [..., dim_y, 3]
        *batch_dims, total_dim = x.shape
        dim_y = total_dim // 3
        x_reshaped = x.view(*batch_dims, dim_y, 3)
        
        # Extract parameters
        logits = x_reshaped[..., 0]  # [..., dim_y]
        shape_raw = x_reshaped[..., 1]  # [..., dim_y]
        rate_raw = x_reshaped[..., 2]  # [..., dim_y]
        
        # Apply activation functions to ensure valid parameter ranges
        probs = torch.sigmoid(logits)  # Probability of non-zero
        shape = torch.nn.functional.softplus(shape_raw) + self.min_shape
        rate = torch.nn.functional.softplus(rate_raw) + self.min_rate
        
        return BernoulliGammaDistribution(probs, shape, rate)


class BernoulliGammaDistribution:
    """Distribution for Bernoulli-Gamma mixture.
    
    This is a simple implementation that provides:
    - log_prob(): Log probability density
    - sample(): Sampling
    - mean: Expected value
    - variance: Variance (if needed)
    
    Args:
        probs: Probability of non-zero observation [..., dim_y]
        shape: Gamma shape parameter [..., dim_y]
        rate: Gamma rate parameter [..., dim_y]
    """
    
    def __init__(self, probs: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor):
        self.probs = probs
        self.shape = shape
        self.rate = rate
        
        # Create underlying distributions
        self.bernoulli = td.Bernoulli(probs=probs)
        self.gamma = td.Gamma(concentration=shape, rate=rate)
    
    @property
    def mean(self) -> torch.Tensor:
        """Expected value: E[X] = P(non-zero) * E[Gamma]"""
        return self.probs * self.gamma.mean
    
    @property
    def variance(self) -> torch.Tensor:
        """Variance using law of total variance."""
        # Var(X) = E[Var(X|Z)] + Var(E[X|Z])
        # where Z is Bernoulli indicator
        gamma_mean = self.gamma.mean
        gamma_var = self.gamma.variance
        
        # E[Var(X|Z)] = p * Var(Gamma)
        # Var(E[X|Z]) = p * (1-p) * E[Gamma]^2
        total_var = (
            self.probs * gamma_var +
            self.probs * (1 - self.probs) * gamma_mean ** 2
        )
        return total_var
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability density.
        
        Args:
            value: Observed values [..., dim_y]
        
        Returns:
            Log probability [..., dim_y]
        """
        # Identify zero and non-zero values
        is_zero = (value == 0.0)
        
        # Initialize log prob with log P(Z=0) for all values
        log_prob_zero = torch.log(1 - self.probs + 1e-8)
        
        # For non-zero values: log P(Z=1) + log Gamma(value)
        # Use where to avoid indexing issues
        log_prob_nonzero = (
            torch.log(self.probs + 1e-8) +
            self.gamma.log_prob(torch.clamp(value, min=1e-8))  # Clamp to avoid gamma(0)
        )
        
        # Select appropriate log prob based on whether value is zero
        log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)
        
        return log_prob
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw samples from the distribution.
        
        Args:
            sample_shape: Shape of samples to draw
        
        Returns:
            Samples with shape sample_shape + batch_shape + event_shape
        """
        # Sample from Bernoulli
        indicator = self.bernoulli.sample(sample_shape)
        
        # Sample from Gamma
        gamma_samples = self.gamma.sample(sample_shape)
        
        # Combine: zero if indicator=0, gamma sample otherwise
        samples = indicator * gamma_samples
        
        return samples
