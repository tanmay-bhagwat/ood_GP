"""
Baseline energy prediction metrics
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .interfaces import PredictionResult, NormalizationState


@dataclass(frozen=True)
class BaselineMetrics:
    mean_absolute_error: float
    mean_standardized_absolute_error: float
    negative_log_predictive_density: float
    mean_standardized_log_likelihood: float


def evaluate_predictions(
    prediction: PredictionResult,
    targets: torch.Tensor,
    target_normalization: NormalizationState) -> BaselineMetrics:

    mae, standardized_mae = mae(prediction, targets)
    nlpd = negative_log_predictive_density(prediction, targets)
    msll = mean_standardized_log_likelihood(prediction, targets, target_normalization)

    return BaselineMetrics(mae, standardized_mae, nlpd, msll)


def mae(
    prediction: PredictionResult,
    targets: torch.Tensor,
    variance_tolerance: float = 1.0e-10) -> tuple[float, float]:
    """
    Mean absolute error/sigma metrics
    """
    if targets.shape != prediction.mean.shape:
        raise ValueError("prediction means and targets must have matching shapes")
    minimum_variance = float(prediction.variance.min().detach().cpu())
    if minimum_variance < -variance_tolerance:
        raise ValueError(f"predictive variance is negative: {minimum_variance}")
    latent_variance = torch.clamp_min(prediction.variance, 0.0)

    absolute_error = torch.abs(prediction.mean - targets)
    standardized = absolute_error / torch.sqrt(
        latent_variance + prediction.noise_variance)
    
    return float(absolute_error.mean().detach().cpu()), float(standardized.mean().detach().cpu())


def _observation_variance(
        prediction: PredictionResult, 
        targets: torch.Tensor) -> torch.Tensor:
    """
    For adding noise variance to the signal variance and some checks for finiteness, compatibility
    """
    
    if targets.shape != prediction.mean.shape or targets.shape != prediction.variance.shape:
        raise ValueError("means, variances, and targets must have matching shapes")
    for t in (targets, prediction.mean, prediction.variance, prediction.noise_variance):
        if not torch.isfinite(t).all(): 
            raise ValueError("prediction and targets must be finite")
    if torch.any(prediction.variance < 0):
        raise ValueError("latent variance is negative")
    if torch.any(prediction.noise_variance < 0):
        raise ValueError("noise variance is negative")
    variance = prediction.variance.clamp_min(0) + prediction.noise_variance
    
    return variance


def negative_log_predictive_density(
    prediction: PredictionResult, 
    targets: torch.Tensor) -> float:
    """
    Mean pointwise Gaussian negative log predictive density (negative log of posterior over test points)
    Each observation uses its marginal variance plus observation noise

    Taken from:
    Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams, The MIT Press, 2006. ISBN 0-262-18253-X
    """
    variance = _observation_variance(prediction, targets)
    nlpd = 0.5*(torch.log(2*torch.pi*variance) +\
                (targets - prediction.mean)**2/variance)
    return nlpd.mean().item()


def mean_standardized_log_likelihood(
    prediction: PredictionResult,
    targets: torch.Tensor,
    train_norm_state: NormalizationState) -> float:
    """
    Mean of the pointwise likelihood of test points over naive train distribution (prior), 
    standardized over learned (posterior) distribution
    Taken from:
    Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams, The MIT Press, 2006. ISBN 0-262-18253-X
    """
    train_variance = _observation_variance(prediction, targets)
    train_mean = train_norm_state.mean
    msll = -negative_log_predictive_density(prediction, targets) +\
        0.5*(torch.log(2*torch.pi*train_variance) + (targets-train_mean)**2/train_variance).mean().item()
    
    return msll
