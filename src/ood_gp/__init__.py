"""Reusable SOAP and exact-GP baseline components."""

from .interfaces import FeatureManifest, PredictionResult, SplitIndices
from .gp import ExactGPRegressor, GPConfig, StructureKernel, TrainingConfig

__all__ = [
    "ExactGPRegressor",
    "FeatureManifest",
    "GPConfig",
    "PredictionResult",
    "SplitIndices",
    "StructureKernel",
    "TrainingConfig",
]
