"""
Config-driven orchestration for the corrected SOAP/GP baseline
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from ase import Atoms

from .interfaces import SplitIndices
from .evaluation import BaselineMetrics, evaluate_predictions
from .features import (FeatureCache, SOAPConfig, SOAPFeatureExtractor,
                       fit_feature_normalization, fit_target_normalization,
                       load_feature_cache, save_feature_cache)
from .gp import (ExactGPRegressor, GPConfig, 
                 TrainingConfig, model_train)
from .splits import fps_ood_split, random_split

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """
    Load and minimally validate a baseline YAML configuration
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Baseline config must be dict-like")
    for section in ("experiment", "dataset", "split", "features", "gp", "training"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"baseline config is missing the {section} section")
    units = raw["dataset"].get("units", {})
    if not all(units.get(name) for name in ("energy", "force", "length")):
        raise ValueError("dataset energy, force, and length units must be explicit")
    if not raw["experiment"].get("device") or not raw["experiment"].get("dtype"):
        raise ValueError("experiment device and dtype must be explicit")
    return raw


def run_baseline(config: Mapping[str, Any]) -> BaselineMetrics:
    """
    Run and persist one corrected SOAP/exact-GP baseline experiment
    """
    experiment = config["experiment"]
    dataset_config = config["dataset"]
    split_config = config["split"]
    feature_config = config["features"]
    gp_config_raw = config["gp"]
    training_config_raw = config["training"]

    ###>>> Set seed, paths, device and dtype
    seed = int(experiment["seed"])
    _seed_everything(seed)
    device = str(experiment["device"])
    dtype = _torch_dtype(str(experiment["dtype"]))
    output_directory = Path(experiment["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_config["path"])
    LOGGER.info(f"Loading dataset from {dataset_path}")
    with np.load(dataset_path) as dataset:
        coordinates = dataset[dataset_config.get("coordinates_key", None)].copy()
        energies = dataset[dataset_config.get("energies_key", None)].copy()
        try: 
            global_frame_indices = dataset["old_indices"].copy()
        except:
            print("Could not find global ID's field, assigning default (0,...,N) ID's")
            global_frame_indices = [i for i in range(energies.shape[0])]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (frames, atoms, 3)")
    if len(coordinates) != len(energies):
        raise ValueError("coordinate and energy frame counts differ")

    symbols = list(dataset_config["symbols"])
    if len(symbols) != coordinates.shape[1]:
        raise ValueError("number of symbols does not match coordinate atom count")

    soap_config = SOAPConfig(
        species=tuple(feature_config["species"]),
        r_cut=float(feature_config["r_cut"]),
        sigma=float(feature_config["sigma"]),
        n_max=int(feature_config["n_max"]),
        l_max=int(feature_config["l_max"]),
        periodic=bool(feature_config.get("periodic", False)),
        dtype=str(experiment["dtype"]),
        device=device)
    extractor = SOAPFeatureExtractor(soap_config)
    
    ###>>> Load or create the entire features set
    dataset_sha256 = _sha256(dataset_path)
    cache_path = Path(feature_config.get(
        "cache_path", output_directory / "soap_features.npz"))
    feature_cache = _load_or_create_feature_cache(cache_path, coordinates, symbols, global_frame_indices, extractor, dataset_sha256)

    ###>>> Store train-val-test sets
    split_path = output_directory / "split_indices.npz"
    if split_path.exists() and split_path.with_suffix(".json").exists():
        split = SplitIndices.load(split_path)
        split.validate(len(energies))
        _validate_reused_split(split, split_config, seed)
    else:
        split = _make_split(split_config, energies, seed)
        split.save(split_path)

    train_batch = feature_cache.select(split.train)
    validation_batch = feature_cache.select(split.validation)
    test_batch = feature_cache.select(split.test)
    feature_normalization = fit_feature_normalization(train_batch.values)

    ###>>> Normalize features over the full set using the train set features/labels only
    train_X = feature_normalization.transform(train_batch.values).to(device=device, dtype=dtype)
    validation_X = feature_normalization.transform(validation_batch.values).to(device=device, dtype=dtype)
    test_X = feature_normalization.transform(test_batch.values).to(device=device, dtype=dtype)

    frame_row_mapping = {frame:row for row,frame in enumerate(global_frame_indices)}

    all_targets = torch.as_tensor(energies, dtype=dtype, device=device)
    train_targets_raw = all_targets[[frame_row_mapping[i] for i in split.train]]
    validation_targets_raw = all_targets[[frame_row_mapping[i] for i in split.validation]]
    test_targets_raw = all_targets[[frame_row_mapping[i] for i in split.test]]
    
    target_normalization = fit_target_normalization(train_targets_raw)
    train_y = target_normalization.transform(train_targets_raw)
    validation_y = target_normalization.transform(validation_targets_raw)
    test_y = target_normalization.transform(test_targets_raw)

    model_config = GPConfig(
        feature_dimension=int(train_X.shape[-1]),
        log_noise=float(gp_config_raw["log_noise"]),
        log_signal_std=float(gp_config_raw["log_signal_std"]),
        log_lengthscale=float(gp_config_raw["log_lengthscale"]),
        hidden_dimension=int(gp_config_raw["hidden_dimension"]),
        learnable_embedding=bool(gp_config_raw["learnable_embedding"]),
        block_size=int(gp_config_raw.get("block_size", 50)),
        jitter=float(gp_config_raw.get("jitter", 1.0e-6)))
    model = ExactGPRegressor(model_config).to(device=device, dtype=dtype)
    training_config = TrainingConfig(**training_config_raw)
    history = model_train(
        model,
        train_X,
        train_y,
        validation_X,
        validation_y,
        training_config)

    # Explicitly condition on training data after validation/model selection.
    model.fit(train_X, train_y)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(test_X)
    metrics = evaluate_predictions(prediction, test_y)

    model.save(output_directory / "gp_bundle", train_batch.manifest)
    predictions_path = output_directory / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        frame_indices=split.test,
        target=test_y.detach().cpu().numpy(),
        mean=prediction.mean.detach().cpu().numpy(),
        variance=prediction.variance.detach().cpu().numpy(),
        noise_variance=float(prediction.noise_variance.detach().cpu()))
    
    normalization_path = output_directory / "normalization.npz"
    np.savez_compressed(
        normalization_path,
        feature_mean=feature_normalization.mean.detach().cpu().numpy(),
        feature_std=feature_normalization.std.detach().cpu().numpy(),
        target_mean=target_normalization.mean.detach().cpu().numpy(),
        target_std=target_normalization.std.detach().cpu().numpy())
    
    resolved_config_path = output_directory / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8")
    
    run_manifest = {
        "seed": seed,
        "device": device,
        "dtype": str(experiment["dtype"]),
        "units": dict(dataset_config["units"]),
        "target": "normalized_energy",
        "feature_manifest": train_batch.manifest.to_dict(),
        "artifacts": {
            "dataset_sha256": dataset_sha256,
            "feature_cache": str(cache_path),
            "feature_cache_sha256": _sha256(cache_path),
            "split_sha256": _sha256(split_path),
            "normalization": normalization_path.name,
            "normalization_sha256": _sha256(normalization_path),
            "predictions": predictions_path.name,
            "predictions_sha256": _sha256(predictions_path),
            "resolved_config": resolved_config_path.name},
        "gp_config": asdict(model_config),
        "training_config": asdict(training_config),
        "metrics": asdict(metrics),
        "history": history,
        "intentional_corrections": [
            "FPS row-local indices map to global dataset frame indices",
            "test predictions explicitly condition on training data"]}
    (output_directory / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    
    return metrics


def _make_split(
    config: Mapping[str, Any], energies: np.ndarray, feature_cache: FeatureCache, seed: int) -> SplitIndices:
    strategy = str(config["strategy"])
    sizes = {
        "train_size": int(config["train_size"]),
        "validation_size": int(config["validation_size"]),
        "test_size": int(config["test_size"]),
    }
    if strategy == "random":
        return random_split(feature_cache.frame_indices, seed=seed, **sizes)
    
    if strategy == "fps_ood":
        return fps_ood_split(
            energies,
            feature_cache.values,
            feature_cache.frame_indices,
            seed=seed,
            z_threshold=float(config.get("z_threshold", 2.0)),
            train_fps_fraction=float(config.get("train_fps_fraction", 0.5)),
            test_bulk_fraction=float(config.get("test_bulk_fraction", 0.05)),
            validation_bulk_fraction=float(config.get("validation_bulk_fraction", 0.7)),
            **sizes)
    raise ValueError(f"unsupported split strategy: {strategy}")


def _validate_reused_split(
    split: SplitIndices, config: Mapping[str, Any], seed: int) -> None:
    expected_strategy = (
        "historical_random" if config["strategy"] == "random" else config["strategy"])
    expected_sizes = (
        int(config["train_size"]),
        int(config["validation_size"]),
        int(config["test_size"]))
    
    actual_sizes = (len(split.train), len(split.validation), len(split.test))
    if split.seed != seed or split.strategy != expected_strategy:
        raise ValueError("saved split provenance does not match the experiment config")
    if actual_sizes != expected_sizes:
        raise ValueError("saved split sizes do not match the experiment config")


def _load_or_create_feature_cache(
    path: Path, coordinates: np.ndarray, symbols: list[str], global_frame_indices: list[int],
    extractor: SOAPFeatureExtractor, dataset_sha256: str) -> FeatureCache:

    """
    Load compatible features, extracting the complete candidate pool once
    """

    manifest_path = path.with_suffix(".json")
    if path.exists() != manifest_path.exists():
        raise ValueError("feature cache requires both NPZ and JSON files")
    if path.exists():
        cache = load_feature_cache(path)
        cache.validate(len(coordinates))
        if cache.source_dataset_sha256 != dataset_sha256:
            raise ValueError("feature cache belongs to a different source dataset")
        _validate_soap_cache(cache, extractor.config)
        LOGGER.info("Loaded feature cache from %s", path)
        return cache

    if not path.exists() and global_frame_indices is None:
        raise ValueError("Global frame ID's must be provided when cache is being created.")
    structures = [Atoms(symbols=symbols, positions=positions)
                  for positions in coordinates]
    batch = extractor.extract(structures)
    cache = FeatureCache(
        values=batch.values.detach().cpu(),
        frame_indices=global_frame_indices,
        manifest=batch.manifest,
        source_dataset_sha256=dataset_sha256)
    cache.validate(len(coordinates))
    save_feature_cache(path, cache)
    LOGGER.info("Created feature cache at %s", path)
    return cache


def _validate_soap_cache(cache: FeatureCache, config: SOAPConfig) -> None:
    expected = {
        "species": list(config.species), "r_cut": config.r_cut,
        "sigma": config.sigma, "n_max": config.n_max,
        "l_max": config.l_max, "periodic": config.periodic}
    if (cache.manifest.extractor != "soap"
            or cache.manifest.granularity != "atomwise"
            or dict(cache.manifest.parameters) != expected
            or cache.manifest.dtype != config.dtype):
        raise ValueError("feature cache is incompatible with SOAP configuration")
    

###>>> Do we really need this?
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return dtype
