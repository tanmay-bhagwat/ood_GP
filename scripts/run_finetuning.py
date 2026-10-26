from argparse import ArgumentParser
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prep_training(config: dict) -> dict:
    """
    Validate project-relative inputs, return run metadata
    """

    config = dict(config)
    input_keys = ["foundation_model", "train_file", "valid_file"]
    if config.get("test_file"):
        input_keys.append("test_file")
    input_hashes = {}
    for key in input_keys:
        path = PROJECT_ROOT / Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"{key} does not exist: {path}")
        config[key] = str(path)
        input_hashes[key] = _sha256(path)
    for key in ("model_dir", "checkpoints_dir", "log_dir", "results_dir"):
        config[key] = str(PROJECT_ROOT / Path(config[key]))

    package_versions = {}
    for package in ("mace-torch", "torch", "ase", "numpy", "PyYAML"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None

    run_metadata = {
        "status": "prepared",
        "config": config,
        "input_sha256": input_hashes,
        "environment": {"python": sys.version, "packages": package_versions},

        "finetune_model_metadata": {
            "name": config["name"],
            "foundation_model": config["foundation_model"],
            "default_dtype": config.get("default_dtype", "float64"),
            "device": config.get("device", "cuda"),
            "seed": config.get("seed", None),
            "multiheads_finetuning": config["multiheads_finetuning"],
        },

        "train_metadata": {
            "train_file": config["train_file"],
            "valid_file": config["valid_file"],
            "test_file": config.get("test_file"),
            "energy_key": config["energy_key"],
            "forces_key": config["forces_key"],
            "energy_weight": config["energy_weight"],
            "forces_weight": config["forces_weight"],
            "E0s": config["E0s"],
            "loss": config.get("loss"),
            "ema": config.get("ema", False),
            "ema_decay": config.get("ema_decay") if config.get("ema", False) else None,
            "batch_size": config["batch_size"],
            "valid_batch_size": config["valid_batch_size"],
            "max_num_epochs": config["max_num_epochs"],
            "clip_grad": config["clip_grad"],
        },

        "optimizer_metadata": {
            key: config.get(key)
            for key in ("optimizer", "lr", "amsgrad", "swa", "scheduler",
                        "scheduler_patience", "lr_factor", "patience",
                        "weight_decay", "scaling")
        },

        "eval_metadata": {
            "eval_interval": config.get("eval_interval", 1),
            "error_table": config.get("error_table", "PerAtomRMSE"),
        }
    }
    data_manifest = Path(config["train_file"]).parent / "data_manifest.json"
    if data_manifest.is_file():
        run_metadata["data_manifest"] = json.loads(data_manifest.read_text())
        run_metadata["data_manifest_sha256"] = _sha256(data_manifest)
    return run_metadata


def launch_training(config_path: Path, project_root: Path) -> None:
   
    project_root = project_root.resolve()
    config_path = (project_root / config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    subprocess.run(
        [sys.executable, "-m", "mace.cli.run_train", "--config", str(config_path)],
        cwd=project_root,
        check=True,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Finetuning settings config file (relative to project root)")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config)
    config = yaml.safe_load(config_path.read_text())
   
    metadata = prep_training(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(metadata["config"]["results_dir"]) / f"metadata_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
   
    metadata["source_config"] = str(config_path)
    metadata["config_sha256"] = _sha256(config)
    metadata["started_at"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "running"
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")

    try:
        launch_training(config, PROJECT_ROOT)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata["returncode"] = getattr(exc, "returncode", None)
        raise
    else:
        metadata["status"] = "completed"
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
