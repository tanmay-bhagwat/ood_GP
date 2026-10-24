"""
This file uses the saved SplitIndices file from the SOAP run
to create the Atoms objects needed to make the fine-tuning data for MACE
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import json, yaml
import sys

import numpy as np
import torch
from ase import Atoms
import ase.units 
from ase.io import write

from ood_gp.interfaces import SplitIndices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> None:
    """
    Load baseline YAML config and minimally validate
    """
    config = yaml.safe_load(config_path.read_text())
    for key in ("split", "dataset", "experiment"):
        if key not in config.keys():
            raise KeyError(f"{key} not found in config file")

    splits_config = config["split"]
    train_size = splits_config["train_size"]
    test_size = splits_config["test_size"]
    validation_size = splits_config["validation_size"]

    splits_path = splits_config.get("splits_path", None)
    if splits_path is None:
        raise ValueError("No valid path found for a saved SplitIndices file")

    splits_path = Path(splits_path)

    ### I am checking against a sum of train-val-test sizes instead of the full dataset size,
    ### because this is supposed to be a fine-tuning set from the exact same dataset as SOAP run
    if splits_path.exists() and splits_path.with_suffix(".json").exists():
        split = SplitIndices.load(splits_path)
        split.validate(train_size+test_size+validation_size)

    if train_size != len(split.train):
        raise ValueError("Train size does not match with SOAP train size")
    if test_size != len(split.test):
        raise ValueError("Test size does not match with SOAP test size")
    if validation_size != len(split.validation):
        raise ValueError("Validation size does not match with SOAP validation size")

    return config


def prepare_data_splits(config: dict[str, dict]) -> None:
    """
    Write the structures for fine-tuning dataset from the saved SplitIndices of SOAP run
    """
    experiment_config = config["experiment"]
    name = experiment_config.get("dtype", "float64")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{dtype} not a supported torch dtype")

    dataset_config = config["dataset"]
    dataset_path = dataset_config["path"]
    units = dataset_config["units"]

    ### Define conversion factors for energy, length units in case not in (eV, Ang)
    if units["energy"] == "kcal/mol":
        energy_conversion_factor = ase.units.kcal / ase.units.mol
    elif units["energy"] == "eV":
        energy_conversion_factor = 1
    else:
        raise ValueError("Energy units must be eV or kcal/mol")

    if units["length"] != "angstrom":
        raise ValueError("Length units must be Ang")

    splits_path = Path(config["split"]["splits_path"])
    split = SplitIndices.load(splits_path)
    
    ### Reload full dataset to pick coords, energies and other features 
    ### at each frame ID from splits
    with np.load(dataset_path) as dataset:
        global_indices = dataset[dataset_config["frames_key"]]
        coords = torch.as_tensor(dataset[dataset_config["coordinates_key"]], 
                                 dtype=dtype, device="cpu")
        energies = torch.as_tensor(dataset[dataset_config["energies_key"]], 
                                   dtype=dtype, device="cpu") * energy_conversion_factor
        forces = torch.as_tensor(dataset[dataset_config["forces_key"]], 
                                 dtype=dtype, device="cpu") * energy_conversion_factor
        symbols = dataset_config["symbols"]

    frame_row_mapping = {frame:row for row,frame in enumerate(global_indices)}

    ### Write the train-test-val structures for the saved splits from the SOAP run
    output_directory = Path(experiment_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_directory / "train.extxyz",
        "test": output_directory / "test.extxyz",
        "validation": output_directory / "validation.extxyz",
    }

    from ase.calculators.singlepoint import SinglePointCalculator
    for split_name, split_rows in (
            ("train", split.train),
            ("test", split.test),
            ("validation", split.validation)):

        structures = []
        rows = [frame_row_mapping[i] for i in split_rows]
        for j in rows:
            a = Atoms(positions=coords[j,:,:],
                      symbols=symbols,
                      pbc=False,
                      info={
                          "global_frame_id": int(global_indices[j]),
                          "dataset_row": int(j),
                          "trajectory_id": Path(dataset_path).stem
                        }
                    )
            calc = SinglePointCalculator(a,
                                         energy=float(energies[j].item()),
                                         forces=forces[j,:,:])
            a.calc = calc
            structures.append(a)

        write(output_paths[split_name], structures, format="extxyz")

    ### Write out the metadata
    dataset_metadata={
        "schema_version": 1,
        "generator":{
            "script": "scripts/prep_data.py",
            "script_sha256": _sha256("scripts/prep_data.py"),
            "ase_version": ase.__version__,
        },
        "source_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "trajectory_id": Path(dataset_path).stem,
            "frame_count": int(coords.shape[0]),
            "atom_count": int(coords.shape[1]),
            "symbols": list(symbols),
            "keys": {
                "global_frame_ids": dataset_config["frames_key"],
                "coordinates": dataset_config["coordinates_key"],
                "REF_energy": dataset_config["energies_key"],
                "REF_forces": dataset_config["forces_key"],
            },
            "input_units": dict(units),
        },
        "unit_conversion": {
            "energy_factor": float(energy_conversion_factor),
            "force_factor": float(energy_conversion_factor),
            "length_factor": 1.0,
            "output_units": {
                "energy": "eV",
                "force": "eV/angstrom",
                "length": "angstrom",
            },
        },
        "split": {
            "path": str(splits_path),
            "sha256": _sha256(splits_path),
            "metadata_path": str(splits_path.with_suffix(".json")),
            "metadata_sha256": _sha256(splits_path.with_suffix(".json")),
            "strategy": split.strategy,
            "seed": split.seed,
            "metadata": dict(split.metadata),
            "index_contract": "global_frame_id",
            "row_mapping": (
                "split ID -> row where dataset[frames_key] equals split ID"),
            "counts": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
        },
        "mace_labels": {
            "energy": "REF_energy",
            "forces": "REF_forces",
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "frame_count": len(getattr(split, name)),
            }
            for name, path in output_paths.items()
        },
    }
    (output_directory / "data_manifest.json").write_text(
        json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _test_data_export(config:dict[str, dict], sample_size:int=3):
    dataset_config = config["dataset"]
    split_config = config["split"]

    splits = SplitIndices.load(split_config["splits_path"])
    splits.validate(split_config["train_size"] + 
                    split_config["validation_size"] +
                    split_config["test_size"])

    experiment_config = config["experiment"]
    output_directory = Path(experiment_config["output_directory"])

    from ase.io import read
    # train_sample = read(output_directory/"train.extxyz", ":")
    train_sample = read(output_directory/"validation.extxyz", ":")
    assert len(train_sample) == split_config["validation_size"] > 0
    sample = train_sample[:sample_size]

    energy_unit = dataset_config["units"]["energy"]
    if energy_unit == "kcal/mol":
        factor = ase.units.kcal / ase.units.mol
    elif energy_unit == "eV":
        factor = 1.0
    else:
        raise ValueError(f"Unsupported energy unit: {energy_unit}")

    assert dataset_config["units"]["length"] == "angstrom"
    expected_force_unit = f"{energy_unit}/angstrom"
    assert dataset_config["units"]["force"] == expected_force_unit

    with np.load(dataset_config["path"]) as dataset:
        for struct in sample:
            row = struct.info["dataset_row"]
            print(row)
            orig_positions = dataset[dataset_config["coordinates_key"]][row]
            np.testing.assert_allclose(orig_positions, struct.get_positions(),
                                       atol=0, rtol=1e-8)
            
            assert dataset_config["symbols"] == struct.get_chemical_symbols()

            orig_energy = dataset[dataset_config["energies_key"]][row]
            np.testing.assert_allclose(orig_energy*factor, struct.get_potential_energy(),
                                       rtol=1e-8, atol=0)

            orig_forces = dataset[dataset_config["forces_key"]][row]
            np.testing.assert_allclose(orig_forces*factor, struct.get_forces(),
                                       rtol=1e-8, atol=1e-8)
        


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__=="__main__":

    config_path = PROJECT_ROOT/"configs/baseline_mace.yaml"
    config = load_config(config_path)
    prepare_data_splits(config)
    _test_data_export(config)
