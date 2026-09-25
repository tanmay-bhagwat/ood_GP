"""
This file uses the saved SplitIndices file from the SOAP run
to create the Atoms objects needed to make the fine-tuning data for MACE
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import torch
from ase import Atoms
import ase.units 
from ase.io import write

from ood_gp.interfaces import SplitIndices


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


def prepare_data_splits(config: dict[str, dict]) -> None:
    """
    Write the structures for fine-tuning dataset from the saved SplitIndices of SOAP run
    """
    experiment_config = config["experiment"]
    name = experiment_config.get("dtype", "float64")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{dtype} not a supported torch dtype")
    device = experiment_config.get("device", "cpu")

    dataset_config = config["dataset"]
    dataset_path = dataset_config["path"]
    units = dataset_config["units"]

    ### Define conversion factors for energy, length units in case not in (eV, Ang)
    if units["energy"] == "kcal/mol":
        energy_conversion_factor = ase.units.eV/ase.units(units["energy"])
    elif units["energy"] == "eV":
        energy_conversion_factor = 1
    else:
        raise ValueError("Energy units must be eV or kcal/mol")

    if units["length"] != "angstrom":
        raise ValueError("Length units must be Ang")
    

    splits_path = config["split"].get("split_path", None)
    split = SplitIndices.load(splits_path)
    
    ### Reload full dataset to pick coords, energies and other features 
    ### at each frame ID from splits
    with np.load(dataset_path) as dataset:
        global_indices = dataset["old_indices"]
        coords = torch.as_tensor(dataset[dataset_config["coordinates_key"]], 
                                 dtype=dtype, device=device)
        energies = torch.as_tensor(dataset[dataset_config["energies_key"]], 
                                   dtype=dtype, device=device) * energy_conversion_factor
        forces = torch.as_tensor(dataset[dataset_config["forces_key"]], 
                                 dtype=dtype, device=device) * energy_conversion_factor 
        symbols = dataset[dataset_config["symbols"]]

    frame_row_mapping = {frame:row for row,frame in enumerate(global_indices)}

    ### Write the train-test-val structures for the saved splits from the SOAP run
    count = 0
    for split_rows in (split.train, split.test, split.validation):
    
        rows = [frame_row_mapping[i] for i in split_rows]
        structures = [Atoms(positions=coords[j,:,:], 
                                symbols=symbols,
                                pbc=False,
                                info={"REF_energies":energies[j], "REF_forces":forces[j,:,:],
                                      "frame_ID":global_indices[j], "row_ID":j}) for j in rows]

        if count == 0:
            write(Path(dataset_path).parent/"train.extxyz", structures, format="extxyz")
        elif count == 1:
            write(Path(dataset_path).parent/"test.extxyz", structures, format="extxyz")
        else:
            write(Path(dataset_path).parent/"validation.extxyz", structures, format="extxyz")

        count +=1
        