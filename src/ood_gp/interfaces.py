"""
Interface classes for workflow
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch


@dataclass(frozen=True)
class SplitIndices:
    """
    Interface class for train-val-test split indices from global dataset frames
    Provides index validation, save and load methods
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int
    strategy: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, dataset_size: int) -> None:
        """
        Raise when indices are invalid, duplicated, or overlapping
        """
        classes = {
            "train": np.asarray(self.train, dtype=np.int64),
            "validation": np.asarray(self.validation, dtype=np.int64),
            "test": np.asarray(self.test, dtype=np.int64)}
        for name, indices in classes.items():
            if indices.ndim != 1:
                raise ValueError(f"{name} indices must be one-dimensional")
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"{name} indices contain duplicates")
            if np.any(indices < 0):
                raise ValueError(f"{name} indices fall outside the dataset")

        if len(np.intersect1d(classes["train"], classes["validation"]))!=0:
            raise ValueError("train and validation indices overlap")
        if len(np.intersect1d(classes["train"], classes["test"]))!=0:
            raise ValueError("Train and test indices overlap")
        if len(np.intersect1d(classes["test"], classes["validation"]))!=0:
            raise ValueError("Test and validation indices overlap")
        if sum(len(indices) for indices in classes.values()) > dataset_size:
            raise ValueError("split contains more indices than the dataset")

    def save(self, path: Path) -> None:
        """
        Save indices to NPZ and provenance to an adjacent JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train=np.asarray(self.train, dtype=np.int64),
            validation=np.asarray(self.validation, dtype=np.int64),
            test=np.asarray(self.test, dtype=np.int64))
        
        manifest_path = path.with_suffix(".json")
        manifest_path.write_text(
            json.dumps({"seed": self.seed,
                        "strategy": self.strategy,
                        "metadata": dict(self.metadata)},
                indent=2,
                sort_keys=True)
            +"\n",
            encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SplitIndices":
        """
        Load a split previously written by method `save`
        """
        
        path = Path(path)
        with np.load(path) as arrays:
            train = arrays["train"].astype(np.int64, copy=True)
            validation = arrays["validation"].astype(np.int64, copy=True)
            test = arrays["test"].astype(np.int64, copy=True)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            train=train,
            validation=validation,
            test=test,
            seed=int(metadata["seed"]),
            strategy=str(metadata["strategy"]),
            metadata=metadata.get("metadata", {}))


@dataclass(frozen=True)
class FeatureManifest:
    """
    Feature provenance needed to determine GP compatibility
    """

    extractor: str
    granularity: str
    parameters: Mapping[str, Any]
    dtype: str
    feature_dimension: int
    library_versions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationState:
    """
    Statistics fitted on training data only
    """

    mean: torch.Tensor
    std: torch.Tensor
    epsilon: float = 1.0e-8

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.std + self.epsilon) + self.mean


@dataclass(frozen=True)
class PredictionResult:
    """
    Latent GP prediction and the observation-noise variance
    """

    mean: torch.Tensor
    variance: torch.Tensor
    noise_variance: torch.Tensor
