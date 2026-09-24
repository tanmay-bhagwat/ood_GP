from __future__ import annotations
from pathlib import Path

import torch
from ase import Atoms
import numpy as np

from ood_gp.splits import fps_ood_split, random_split
from ood_gp.features import (
    FeatureCache,
    SOAPConfig, 
    SOAPFeatureExtractor,
    save_feature_cache, load_feature_cache,
    fit_feature_normalization,
    fit_target_normalization)
from ood_gp.gp import (
    GPConfig, 
    ExactGPRegressor, 
    TrainingConfig,
    model_train)
 

def test_run_baseline(tmp_path: Path):

    dataset_path = tmp_path / "fake_dataset.npz"
    cache_path = tmp_path / "fake_cache.npz"
    split_path = tmp_path / "splits.npz"
    output_path = tmp_path / "fake_output.npz"

    ### Create fake dataset of coordinates and energies
    arr = [[[1,0,0],[0,0,i]] for i in range(10)]
    fake_coords = np.array(arr)
    fake_energies = np.arange(10,)
    global_ids = [i+10 for i in range(10)]

    ### Save it, then load it as in run_baseline.py
    np.savez_compressed(dataset_path, coordinates_key=fake_coords, energies_key=fake_energies)

    with np.load(dataset_path) as dataset:
        coords = dataset["coordinates_key"].copy()
        energies = dataset["energies_key"].copy()

    assert np.allclose(coords, fake_coords) and np.allclose(energies, fake_energies)

    ### Extract SOAP features from the coords + species
    soap_config = SOAPConfig(
        species=("H"),
        r_cut=6,
        sigma=0.75,
        n_max=12,
        l_max=8,
        periodic=False,
        dtype="float64"
    )

    extractor = SOAPFeatureExtractor(soap_config)
    batch = extractor.extract([Atoms(symbols=("H","H"), positions=positions) for positions in coords])
    cache = FeatureCache(values=batch.values, 
                         frame_indices=global_ids, 
                         manifest=batch.manifest, 
                         source_dataset_sha256="xx")
    cache.validate(len(coords))
    save_feature_cache(cache_path, cache)

    load_cache = load_feature_cache(cache_path)
    ### Make sure that the indices are loaded in the exact same order as how they were stored
    assert np.allclose(load_cache.frame_indices, cache.frame_indices)
    torch.testing.assert_close(load_cache.values, cache.values)

    ### Now we test train-test-val split strategies
    # split = fps_ood_split(
    #     energies=energies,
    #     descriptors=load_cache.values,
    #     descriptor_frame_indices=load_cache.frame_indices,
    #     train_size=4,
    #     validation_size=2,
    #     test_size=2,
    #     seed=1
    # )

    split = random_split(
        descriptor_frame_indices=load_cache.frame_indices,
        train_size=4,
        test_size=2,
        validation_size=2,
        seed=1
    )

    split.validate(10)

    ### Test save and load functionalities of SplitIndices
    split.save(split_path)
    load_split = split.load(split_path)

    np.testing.assert_array_equal(split.train, load_split.train)
    np.testing.assert_array_equal(split.test, load_split.test)
    np.testing.assert_array_equal(split.validation, load_split.validation)

    train_batch = load_cache.select(split.train)
    validation_batch = load_cache.select(split.validation)
    test_batch = load_cache.select(split.test)

    feature_norm_state = fit_feature_normalization(train_batch.values)

    mean = torch.mean(train_batch.values, dim=(0,1), keepdim=True)
    stdev = torch.std(train_batch.values, dim=(0,1), keepdim=True)
    torch.testing.assert_close(mean, feature_norm_state.mean)
    torch.testing.assert_close(stdev, feature_norm_state.std)

    train_X = feature_norm_state.transform(train_batch.values)
    test_X = feature_norm_state.transform(test_batch.values)
    validation_X = feature_norm_state.transform(validation_batch.values)

    assert torch.isfinite(train_X).all().item() and train_X.numel()
    assert train_X.shape == train_batch.values.shape
    assert torch.isfinite(test_X).all().item() and test_X.numel()
    assert test_X.shape == test_batch.values.shape
    assert torch.isfinite(validation_X).all().item() and validation_X.numel()
    assert validation_X.shape == validation_batch.values.shape

    ### Make sure transforms did not alter NormalizationState
    torch.testing.assert_close(mean, feature_norm_state.mean)
    torch.testing.assert_close(stdev, feature_norm_state.std)

    expected_std = feature_norm_state.std / (feature_norm_state.std + feature_norm_state.epsilon)

    torch.testing.assert_close(train_X.mean(dim=(0,1), keepdim=True), torch.zeros(size=(feature_norm_state.mean.shape), dtype=torch.float64))
    torch.testing.assert_close(train_X.std(dim=(0,1), keepdim=True), expected_std)
    
    frame_row_mapping = {frame:row for row,frame in enumerate(global_ids)}

    energy_tensor = torch.as_tensor(energies, dtype=torch.float64)
    train_energies = energy_tensor[[frame_row_mapping[i] for i in split.train]]
    validation_energies = energy_tensor[[frame_row_mapping[i] for i in split.validation]]
    test_energies = energy_tensor[[frame_row_mapping[i] for i in split.test]]

    energy_norm_state = fit_target_normalization(train_energies)
    mean_energy = torch.mean(train_energies)
    stdev_energy = torch.std(train_energies)

    torch.testing.assert_close(mean_energy, energy_norm_state.mean)
    torch.testing.assert_close(stdev_energy, energy_norm_state.std)

    train_y = energy_norm_state.transform(train_energies)
    test_y = energy_norm_state.transform(test_energies)
    validation_y = energy_norm_state.transform(validation_energies)

    ### Make sure transforms did not alter NormalizationState
    torch.testing.assert_close(mean_energy, energy_norm_state.mean)
    torch.testing.assert_close(stdev_energy, energy_norm_state.std)

    gp_config = GPConfig(
        feature_dimension=train_X.shape[-1],
        log_noise=-6,
        log_signal_std=3,
        log_lengthscale=0.1,
        learnable_embedding=False
    )

    model = ExactGPRegressor(gp_config)
    train_config = TrainingConfig(epochs=5)
    history = model_train(
            model,
            train_X,
            train_y,
            validation_X,
            validation_y,
            train_config)

    model.fit(train_X, train_y)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(test_X)

    assert torch.isfinite(prediction.mean).all().item() and prediction.mean.shape[0] == test_batch.values.shape[0]
    assert torch.isfinite(prediction.variance).all().item() and prediction.variance.shape[0] == test_batch.values.shape[0]