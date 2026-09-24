"""Baseline energy prediction metrics."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .interfaces import PredictionResult


@dataclass(frozen=True)
class BaselineMetrics:
    mean_absolute_error: float
    mean_standardized_absolute_error: float


def evaluate_predictions(
    prediction: PredictionResult,
    targets: torch.Tensor,
    *,
    variance_tolerance: float = 1.0e-10) -> BaselineMetrics:
    """
    Reproduce historical MAE and mean absolute error/sigma metrics
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
    return BaselineMetrics(
        mean_absolute_error=float(absolute_error.mean().detach().cpu()),
        mean_standardized_absolute_error=float(standardized.mean().detach().cpu()))
