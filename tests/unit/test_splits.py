from pathlib import Path
import numpy as np
import torch, pytest
from ood_gp.interfaces import SplitIndices
from ood_gp.splits import (
    farthest_point_sampling,
    fps_ood_split,
    random_split,
    save_descriptor_cache,
    load_descriptor_cache)


def test_valid_splits_in_random_split() -> None:
    fake_dataset_size = 20
    frame_indices = np.arange(fake_dataset_size, dtype=np.int64)
    split_indices = random_split(frame_indices, 8, 4, 2, seed=1)

    assert len(split_indices.train) == 8 and len(np.unique(split_indices.train)) == 8
    assert len(split_indices.validation) == 4 and len(np.unique(split_indices.validation)) == 4
    assert len(split_indices.test) == 2 and len(np.unique(split_indices.test)) == 2
    assert (np.intersect1d(split_indices.train, split_indices.validation)).shape[0] == 0
    assert (np.intersect1d(split_indices.train, split_indices.test)).shape[0] == 0
    assert (np.intersect1d(split_indices.test, split_indices.validation)).shape[0] == 0

    fake_dataset_size = 2
    frame_indices = np.arange(fake_dataset_size, dtype=np.int64)
    with pytest.raises(ValueError, match="larger than the dataset"):
        random_split(frame_indices, 20, 8, 4, seed=1) ##Total dataset size too small
    with pytest.raises(ValueError, match="Split sizes cannot be negative"):
        random_split(frame_indices, -20, 8, 4, seed=1) ##Train dataset size negative


def test_random_split_preserves_nonconsecutive_global_ids() -> None:
    frame_indices = np.array(
        [17, 4, 22, 9, 31, 12, 45, 6, 28, 39], dtype=np.int64)

    first = random_split(frame_indices, 4, 3, 2, seed=7)
    second = random_split(frame_indices, 4, 3, 2, seed=7)

    supplied_ids = set(frame_indices)
    assert set(first.train).issubset(supplied_ids)
    assert set(first.validation).issubset(supplied_ids)
    assert set(first.test).issubset(supplied_ids)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)


def test_random_split_rejects_invalid_frame_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        random_split(np.array([10, 10, 12]), 1, 1, 1)
    with pytest.raises(ValueError, match="negative"):
        random_split(np.array([10, -1, 12]), 1, 1, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        random_split(np.array([[10, 11], [12, 13]]), 1, 1, 1)


def test_split_validation_rejects_too_many_combined_indices() -> None:
    split = SplitIndices(
        train=np.array([10, 11, 12]),
        validation=np.array([13, 14]),
        test=np.array([15]),
        seed=1,
        strategy="historical_random")

    with pytest.raises(ValueError, match="more indices than the dataset"):
        split.validate(dataset_size=5)


def test_fps_returns_unique_local_rows() -> None:
    descriptors = torch.randn((30,2), generator=torch.Generator().manual_seed(1))
    sample_size = 10
    fps_indices = farthest_point_sampling(descriptors, sample_size, seed=1) ## Row-local indices
    print(len(fps_indices), np.unique(fps_indices))
    assert len(fps_indices) == len(np.unique(fps_indices))
    assert len(fps_indices) == sample_size


def test_fps_split_maps_rows_to_global_ids() -> None:
    energies = torch.arange(60)
    descriptors = torch.randn((30,2), generator=torch.Generator().manual_seed(1))
    descriptor_frame_indices = np.arange(30, 60, dtype=np.int64)
    train_size = 10
    val_size = 6
    test_size = 2
    splits = fps_ood_split(
        energies, 
        descriptors, 
        descriptor_frame_indices, 
        train_size=train_size, 
        validation_size=val_size, 
        test_size=test_size,
        z_threshold=float("inf"),
        test_bulk_fraction=1.0,
        validation_bulk_fraction=1.0) 

    assert len(np.unique(splits.train)) == len(splits.train)
    assert len(splits.train) == train_size
    assert len(np.unique(splits.validation)) == len(splits.validation)
    assert len(splits.validation) == val_size
    assert len(np.unique(splits.test)) == len(splits.test)
    assert len(splits.test) == test_size

    global_ids = set(descriptor_frame_indices)
    assert set(splits.train).issubset(global_ids)
    assert set(splits.validation).issubset(global_ids)
    assert set(splits.test).issubset(global_ids)
    assert set(splits.train).isdisjoint(set(splits.validation))
    assert set(splits.train).isdisjoint(set(splits.test))
    assert set(splits.test).isdisjoint(set(splits.validation))


def test_split_round_trip(tmp_path: Path) -> None:
    split = random_split(np.arange(20, dtype=np.int64), 8, 4, 5, seed=7)
    path = tmp_path / "split.npz"

    split.save(path)
    loaded = SplitIndices.load(path)

    np.testing.assert_array_equal(loaded.train, split.train)
    np.testing.assert_array_equal(loaded.validation, split.validation)
    np.testing.assert_array_equal(loaded.test, split.test)
    assert loaded.seed == split.seed
    assert loaded.strategy == split.strategy


def test_descriptor_cache_requires_row_to_frame_mapping(tmp_path: Path) -> None:
    path = tmp_path / "descriptors.npz"
    descriptors = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3)
    frame_indices = np.array([17, 4, 22, 9])

    save_descriptor_cache(path, descriptors, frame_indices)
    loaded_descriptors, loaded_indices = load_descriptor_cache(path)

    torch.testing.assert_close(loaded_descriptors, descriptors)
    np.testing.assert_array_equal(loaded_indices, frame_indices)
