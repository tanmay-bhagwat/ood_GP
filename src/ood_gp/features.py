"""
Feature extraction and train-only normalization for atomistic structures
"""

from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence
import json
import numpy as np
import torch
from .interfaces import FeatureManifest, NormalizationState

if TYPE_CHECKING:
    from ase import Atoms

@dataclass(frozen=True)
class FeatureBatch:
    """
    A feature tensor paired with its provenance manifest
    """

    values: torch.Tensor
    manifest: FeatureManifest


@dataclass(frozen=True)
class FeatureCache:
    """
    Label-free features keyed by global source-dataset frame IDs
    """

    values: torch.Tensor
    frame_indices: np.ndarray
    manifest: FeatureManifest
    source_dataset_sha256: str

    def validate(self, dataset_size: int) -> None:
        indices = np.asarray(self.frame_indices, dtype=np.int64)
        if self.values.shape[0] != len(indices):
            raise ValueError("each feature row must have one global frame index")
        if indices.ndim != 1 or len(indices) != len(np.unique(indices)):
            raise ValueError("feature-cache frame indices must be unique and 1D")
        if np.any(indices < 0) or np.any(indices >= dataset_size):
            raise ValueError("feature-cache frame indices fall outside the dataset")
        if self.values.shape[-1] != self.manifest.feature_dimension:
            raise ValueError("feature values do not match their manifest")

    def select(self, global_indices: Sequence[int] | np.ndarray) -> FeatureBatch:
        """
        Return features in the exact order of requested global frame IDs
        """
        row_by_frame = {int(frame): row for row, frame in enumerate(self.frame_indices)}
        try:
            rows = [row_by_frame[int(frame)] for frame in global_indices]
        except KeyError as exc:
            raise ValueError(f"frame {exc.args[0]} is absent from feature cache") from exc
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        return FeatureBatch(values=self.values[row_tensor], manifest=self.manifest)


class FeatureExtractor(Protocol):
    """
    Interface shared by SOAP and future frozen-MACE extractors
    """

    def extract(self, structures: Sequence["Atoms"], **kwargs) -> FeatureBatch:
        """
        Extract features without choosing dataset splits
        """


@dataclass(frozen=True)
class SOAPConfig:
    species: tuple[str, ...]
    r_cut: float = 6.0
    sigma: float = 0.75
    n_max: int = 12
    l_max: int = 8
    periodic: bool = False
    dtype: str = "float64"
    device: str = "cpu"


class SOAPFeatureExtractor:
    """
    Atomwise DScribe SOAP features
    """

    def __init__(self, config: SOAPConfig) -> None:
        self.config = config

    def extract(self, structures: Sequence["Atoms"]) -> FeatureBatch:
        if not structures:
            raise ValueError("at least one structure is required")
        try:
            from dscribe.descriptors import SOAP
        except ImportError as exc:
            raise ImportError("SOAP extraction requires \
                              the optional runtime dependency dscribe") from exc

        soap = SOAP(
            species=list(self.config.species),
            n_max=self.config.n_max,
            l_max=self.config.l_max,
            r_cut=self.config.r_cut,
            sigma=self.config.sigma,
            periodic=self.config.periodic)
        
        ###>>> Make the actual SOAP features
        array = soap.create(list(structures))
        values = torch.as_tensor(array, dtype=_torch_dtype(self.config.dtype)).to(self.config.device)

        manifest = FeatureManifest(
            extractor="soap",
            granularity="atomwise",
            parameters={
                "species": list(self.config.species),
                "r_cut": self.config.r_cut,
                "sigma": self.config.sigma,
                "n_max": self.config.n_max,
                "l_max": self.config.l_max,
                "periodic": self.config.periodic},
            dtype=self.config.dtype,
            feature_dimension=int(values.shape[-1]),
            library_versions={"dscribe": _package_version("dscribe")})
        
        return FeatureBatch(values=values, manifest=manifest)


def fit_feature_normalization(train_features: torch.Tensor) -> NormalizationState:
    """
    Fit per-feature statistics over training structures and atoms only
    """

    if train_features.ndim == 3:
        dimensions = (0, 1)
    elif train_features.ndim == 2:
        dimensions = (0,)
    else:
        raise ValueError("features must have shape (N, D) or (N, atoms, D)")
    return NormalizationState(
        mean=train_features.mean(dim=dimensions, keepdim=True),
        std=train_features.std(dim=dimensions, keepdim=True, unbiased=True))


def fit_target_normalization(train_targets: torch.Tensor) -> NormalizationState:
    """
    Fit scalar target statistics from training labels only
    """
    
    if train_targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    return NormalizationState(
        mean=train_targets.mean(),
        std=train_targets.std(unbiased=True))


def save_feature_cache(path: Path, cache: FeatureCache) -> None:
    """
    Save label-free features, row-to-frame ID mapping, and provenance
    """

    path = Path(path)
    indices = np.asarray(cache.frame_indices, dtype=np.int64)
    cache.validate(int(indices.max()) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=cache.values.detach().cpu().numpy(),
        frame_indices=indices)
    path.with_suffix(".json").write_text(
        json.dumps({
            "feature_manifest": cache.manifest.to_dict(),
            "source_dataset_sha256": cache.source_dataset_sha256},
            indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def load_feature_cache(path: Path) -> FeatureCache:
    """
    Load a feature cache with mandatory provenance and frame mapping
    """

    path = Path(path)
    with np.load(path) as arrays:
        values = torch.from_numpy(arrays["features"].copy())
        frame_indices = arrays["frame_indices"].astype(np.int64, copy=True)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    cache = FeatureCache(
        values=values,
        frame_indices=frame_indices,
        manifest=FeatureManifest(**metadata["feature_manifest"]),
        source_dataset_sha256=str(metadata["source_dataset_sha256"]))
    cache.validate(int(frame_indices.max()) + 1)
    return cache


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return dtype


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return f"could not resolve package version for {package}"
