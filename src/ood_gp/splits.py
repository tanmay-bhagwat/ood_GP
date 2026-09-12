"""
Deterministic, explicit dataset splitting and FPS selection
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from .interfaces import SplitIndices


def random_split(
    dataset_size: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    is_proportion : bool = False,
    *,
    seed: int = 1) -> SplitIndices:
    """
    Random split with given global dataset indices
    """
    _validate_requested_size(dataset_size, train_size, 
                             validation_size, test_size, is_proportion)
    indices = np.arange(dataset_size, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_end = train_size
    validation_end = train_end + validation_size
    split = SplitIndices(
        train=indices[:train_end].copy(),
        validation=indices[train_end:validation_end].copy(),
        test=indices[validation_end:validation_end + test_size].copy(),
        seed=seed,
        strategy="historical_random")
    split.validate(dataset_size)
    return split


def farthest_point_sampling(
    descriptors: torch.Tensor,
    sample_size: int,
    *,
    seed: int = 1) -> np.ndarray:
    """
    Return row-local FPS positions using mean-pooled descriptors. Splitting deliberately runs on CPU. 
    Callers map the returned row positions through their candidate-frame array
    before treating them as dataset indices.
    """
    if descriptors.ndim not in (2, 3):
        raise ValueError("descriptors must have shape (N, D) or (N, atoms, D)")
    count = int(descriptors.shape[0])
    if sample_size < 1 or sample_size > count:
        raise ValueError("sample_size must be between one and descriptor count")

    pooled = descriptors.detach().to(device="cpu")
    if pooled.ndim == 3:
        pooled = pooled.mean(dim=1)

    rng = np.random.RandomState(seed)
    first = int(rng.randint(0, count))
    selected = [first]
    min_distance = torch.sum((pooled - pooled[first]) ** 2, dim=1)
    min_distance[first] = -torch.inf

    for _ in range(sample_size - 1):
        next_index = int(torch.argmax(min_distance).item())
        selected.append(next_index)
        distance = torch.sum((pooled - pooled[next_index]) ** 2, dim=1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance[torch.as_tensor(selected, dtype=torch.long)] = -torch.inf

    return np.asarray(selected, dtype=np.int64)


def fps_ood_split(
    energies: Sequence[float] | np.ndarray | torch.Tensor,
    descriptors: torch.Tensor,
    descriptor_frame_indices: Sequence[int] | np.ndarray,
    train_size: int,
    validation_size: int,
    test_size: int,
    *,
    seed: int = 1,
    z_threshold: float = 2.0,
    train_fps_fraction: float = 0.5,
    test_bulk_fraction: float = 0.05,
    validation_bulk_fraction: float = 0.7) -> SplitIndices:

    """
    Build the correct FPS+Bulk/OOD split in global frame indices.
    `descriptor_frame_indices` are the indices of (sub)set of frames selected from the global dataset. 
    We use `descriptor_frame_indices` to ensure train/test/val split doesn't overlap
    `z_threshold` decides the std dev limit above which we call samples 'OOD', default=2 std devs
    """

    energy_array = _as_numpy_1d(energies, "energies").astype(np.float64)
    frame_indices = _as_numpy_1d(descriptor_frame_indices, "descriptor_frame_indices").astype(np.int64)
    descriptor_values = descriptors.detach().to(device="cpu")
    dataset_size = len(energy_array)

    if int(descriptor_values.shape[0]) != len(frame_indices):
        raise ValueError("each descriptor row must have one global frame index")
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError("descriptor_frame_indices contain duplicates")
    if np.any(frame_indices < 0) or np.any(frame_indices >= dataset_size):
        raise ValueError("descriptor frame indices fall outside the dataset")
    _validate_requested_size(
        len(frame_indices), train_size, validation_size, test_size)
    
    ###>>> Make sure fraction lies in (0,1]
    if train_fps_fraction<=0 or train_fps_fraction>1:
        raise ValueError("fps_fraction must lie between zero and one")
    if test_bulk_fraction<0 or test_bulk_fraction>1:
        raise ValueError("test_bulk_fraction must lie between zero and one")
    if validation_bulk_fraction<0 or validation_bulk_fraction>1:
        raise ValueError("validation_bulk_fraction must lie between zero and one")

    rng = np.random.RandomState(seed)
    train_fps_size = int(train_size * train_fps_fraction)
    train_rng_size = train_size - train_fps_size
    if train_fps_size < 1:
        raise ValueError("train_fps_fraction selects zero FPS frames")
    candidate_rows = np.arange(len(frame_indices), dtype=np.int64)
    train_rng_rows = rng.choice(candidate_rows, train_rng_size, replace=False).astype(np.int64)
    available_fps_rows = np.setdiff1d(candidate_rows, train_rng_rows)

    ###>>> farthest_point_sampling returns fps-rows local indices, so we must match them to the right dataset row
    fps_local = farthest_point_sampling(descriptor_values[available_fps_rows], train_fps_size, seed=seed)
    fps_rows = available_fps_rows[fps_local]
    train_rows = np.concatenate((fps_rows, train_rng_rows))
    train = frame_indices[train_rows]

    ###>>> Standardize with train dataset statistics
    energy_std = energy_array[train].std()
    energy_mean = energy_array[train].mean()
    if not np.isfinite(energy_std) or energy_std == 0.0:
        raise ValueError("FPS/OOD splitting requires non-constant, finite energies")
    normalized_energy = (energy_array - energy_mean) / energy_std

    ###>>> Distinguish between bulk and ood dataset frame indices
    candidate_z = normalized_energy[frame_indices]
    bulk_global = frame_indices[np.abs(candidate_z) <= z_threshold]
    ood_global = frame_indices[np.abs(candidate_z) > z_threshold]

    ###>>> Separating total train into bulk and ood
    train_is_bulk = np.abs(normalized_energy[train]) <= z_threshold
    bulk_train = train[train_is_bulk]
    ood_train = train[~train_is_bulk]

    ###>>> Bulk/ood left after train samples chosen
    available_bulk = np.setdiff1d(bulk_global, bulk_train, assume_unique=True)
    available_ood = np.setdiff1d(ood_global, ood_train, assume_unique=True)
    n_test_bulk = int(test_size * test_bulk_fraction)
    n_test_ood = test_size - n_test_bulk
    n_validation_bulk = int(validation_size * validation_bulk_fraction)
    n_validation_ood = validation_size - n_validation_bulk
    if len(available_bulk) < n_test_bulk + n_validation_bulk:
        raise ValueError("not enough remaining bulk frames for validation and test")
    if len(available_ood) < n_test_ood + n_validation_ood:
        raise ValueError("not enough OOD frames for validation and test")

    split_rng = np.random.RandomState(seed)
    test_bulk = split_rng.choice(available_bulk, n_test_bulk, replace=False)
    test_ood = split_rng.choice(available_ood, n_test_ood, replace=False)
    validation_bulk_pool = np.setdiff1d(available_bulk, test_bulk)
    validation_ood_pool = np.setdiff1d(available_ood, test_ood)
    validation_bulk = split_rng.choice(validation_bulk_pool, n_validation_bulk, replace=False)
    validation_ood = split_rng.choice(validation_ood_pool, n_validation_ood, replace=False)

    split = SplitIndices(
        train=train,
        validation=np.concatenate((validation_bulk, validation_ood)).astype(np.int64),
        test=np.concatenate((test_bulk, test_ood)).astype(np.int64),
        seed=seed,
        strategy="fps_ood",
        metadata={
            "z_threshold": z_threshold,
            "fps_fraction": train_fps_fraction,
            "test_bulk_fraction": test_bulk_fraction,
            "validation_bulk_fraction": validation_bulk_fraction,
            "descriptor_index_contract": "row_to_global_frame"})
    split.validate(dataset_size)
    return split


def load_descriptor_cache(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load descriptors and their mandatory global frame mapping from NPZ
    """
    with np.load(Path(path)) as cache:
        if "descriptors" not in cache or "frame_indices" not in cache:
            raise ValueError("descriptor cache must contain descriptors and frame_indices")
        descriptors = torch.from_numpy(cache["descriptors"].copy())
        frame_indices = cache["frame_indices"].astype(np.int64, copy=True)
    return descriptors, frame_indices


def save_descriptor_cache(
    path: Path,
    descriptors: torch.Tensor,
    frame_indices: Sequence[int] | np.ndarray) -> None:

    """
    Persist descriptor rows with their global dataset frame mapping
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if descriptors.shape[0] != len(indices):
        raise ValueError("each descriptor row must have one global frame index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("frame indices contain duplicates")
    np.savez_compressed(
        path,
        descriptors=descriptors.detach().cpu().numpy(),
        frame_indices=indices)


def _as_numpy_1d(
    values: Sequence[float] | np.ndarray | torch.Tensor, 
    name: str) -> np.ndarray:

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validate_requested_size(
    dataset_size: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    is_proportion: bool = False) -> None:

    sizes = (train_size, validation_size, test_size)
    if any(size < 0 for size in sizes):
        raise ValueError("Split sizes cannot be negative")
    if sum(sizes) > dataset_size:
        raise ValueError("Requested split is larger than the dataset")
    if is_proportion and sum(sizes) != 1.0:
        raise ValueError("Proportions must sum to 1")
