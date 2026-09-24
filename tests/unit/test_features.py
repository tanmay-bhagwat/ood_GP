from pathlib import Path
import numpy as np
import pytest
import torch
from ood_gp.features import (
    SOAPConfig,
    SOAPFeatureExtractor,
    FeatureCache,
    fit_feature_normalization,
    fit_target_normalization,
    save_feature_cache,
    load_feature_cache)
from ood_gp.interfaces import FeatureManifest


###>>> Test feature normalization from train stats, transform features
def test_feature_normalization_is_fit_on_train_only() -> None:
    train = torch.tensor(
        [[[1.0, 4.0], [3.0, 8.0]], [[5.0, 12.0], [7.0, 16.0]]],
        dtype=torch.float64,
    )
    test = torch.full((2, 2, 2), 1.0e9, dtype=torch.float64)

    state = fit_feature_normalization(train)
    _ = state.transform(test)

    torch.testing.assert_close(state.mean, train.mean(dim=(0, 1), keepdim=True))
    torch.testing.assert_close(state.std, train.std(dim=(0, 1), keepdim=True, unbiased=True))


###>>> Test feature normalization from train stats, transform labels
def test_target_normalization_is_fit_on_train_only() -> None:
    train = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    state = fit_target_normalization(train)

    torch.testing.assert_close(state.mean, train.mean())
    torch.testing.assert_close(state.std, train.std(unbiased=True))


def test_soap_extraction_is_deterministic() -> None:
    pytest.importorskip("dscribe")
    ase = pytest.importorskip("ase")
    structures = [
        ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])
    ]
    extractor = SOAPFeatureExtractor(
        SOAPConfig(
            species=("H",),
            r_cut=3.0,
            sigma=0.5,
            n_max=3,
            l_max=2))

    first = extractor.extract(structures)
    second = extractor.extract(structures)

    torch.testing.assert_close(first.values, second.values, rtol=0.0, atol=0.0)
    assert first.manifest == second.manifest
    assert torch.isfinite(first.values).all()


def test_featurecache_validate() -> None:
    values = torch.randn(4, 4, 3)
    fake_dataset_size = 23
    
    manifest = FeatureManifest(
        extractor="SOAP", 
        granularity="atomwise",
        parameters={
            "r_cut" : 6.0,
            "sigma" : 0.75,
            "n_max" : 12
        },
        feature_dimension=values.size(-1),
        dtype="np.float64")
    
    frame_indices_size_mismatch = np.array([3, 4, 5]) ## Fails, less than values.size(0)
    size_mismatch_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices_size_mismatch,
        manifest=manifest,
        source_dataset_sha256="xx")
    pytest.raises(ValueError, size_mismatch_cache.validate(fake_dataset_size))

    frame_indices_neg_indices = np.array([-1, 3, 4, 9]) ## Fails, index < 0
    neg_indices_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices_neg_indices,
        manifest=manifest,
        source_dataset_sha256="xx")
    pytest.raises(ValueError, neg_indices_cache.validate(fake_dataset_size))

    frame_indices_idx_exceeds_dataset_size = np.array([17, 30, 4, 9]) ## Fails, index > fake_dataset_size
    idx_exceeds_dataset_size_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices_idx_exceeds_dataset_size,
        manifest=manifest,
        source_dataset_sha256="xx")
    pytest.raises(ValueError, idx_exceeds_dataset_size_cache.validate(fake_dataset_size))

    frame_indices_duplicate_idxs = np.array([2, 2, 5, 6]) ## Fails, non-unique elements
    duplicate_idxs_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices_duplicate_idxs,
        manifest=manifest,
        source_dataset_sha256="xx")
    pytest.raises(ValueError, duplicate_idxs_cache.validate(fake_dataset_size))

    frame_indices = np.array([17, 4, 22, 9]) ## Max index here must be len(dataset_size)-1

    cache = FeatureCache(
        values=values,
        frame_indices=frame_indices,
        manifest=manifest,
        source_dataset_sha256="xx")
    
    cache.validate(fake_dataset_size)


def test_featurecache_select():
    values = torch.randn(4, 4, 3)
    
    manifest = FeatureManifest(
        extractor="SOAP", 
        granularity="atomwise",
        parameters={
            "r_cut" : 6.0,
            "sigma" : 0.75,
            "n_max" : 12
        },
        feature_dimension=values.size(-1),
        dtype="np.float64")
    frame_indices = np.array([17, 4, 22, 9]) ## Max index here must be len(dataset_size)-1

    cache = FeatureCache(
        values=values,
        frame_indices=frame_indices,
        manifest=manifest,
        source_dataset_sha256="xx")
    
    torch.testing.assert_close(cache.select([17]).values[0,:,:], values[0,:,:])
    torch.testing.assert_close(cache.select([22]).values[0,:,:], values[2,:,:])


def test_featurecache_save(tmp_path: Path):

    values = torch.randn(4, 4, 3)
    manifest = FeatureManifest(
        extractor="SOAP", 
        granularity="atomwise",
        parameters={
            "r_cut" : 6.0,
            "sigma" : 0.75,
            "n_max" : 12
        },
        feature_dimension=values.size(-1),
        dtype="np.float64")
    frame_indices = np.array([17, 4, 22, 9]) ## Max index here must be len(dataset_size)-1

    fake_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices,
        manifest=manifest,
        source_dataset_sha256="xx")

    test_path = tmp_path / Path("unit_tests_outputs/sample_feature_cache")
    save_feature_cache(path=test_path, cache=fake_cache)
    

def test_featurecache_load(tmp_path: Path):
    test_path = tmp_path / Path("sample_feature_cache_for_loading")

    values = torch.randn(4, 4, 3)
    manifest = FeatureManifest(
        extractor="SOAP", 
        granularity="atomwise",
        parameters={
            "r_cut" : 6.0,
            "sigma" : 0.75,
            "n_max" : 12
        },
        feature_dimension=values.size(-1),
        dtype="np.float64")
    frame_indices = np.array([17, 4, 22, 9]) ## Max index here must be len(dataset_size)-1
    fake_cache = FeatureCache(
        values=values,
        frame_indices=frame_indices,
        manifest=manifest,
        source_dataset_sha256="xx")
    
    ## Previously checked the save feature
    save_feature_cache(path=test_path, cache=fake_cache)

    loaded_cache = load_feature_cache(path=test_path.with_suffix(".npz"))
    for i in range(loaded_cache.frame_indices.size):
        assert loaded_cache.frame_indices[i] == frame_indices[i]
        torch.testing.assert_close(
            loaded_cache.select([loaded_cache.frame_indices[i]]).values[0,:,:],
            values[i,:,:])
    