#!/usr/bin/env python3
"""
Run the config-driven SOAP/GP baseline
"""

from __future__ import annotations
import argparse
from dataclasses import asdict
import logging
from pathlib import Path
from ood_gp.baseline import load_config, run_baseline
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to baseline YAML config")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    metrics = run_baseline(load_config(arguments.config))
    logging.getLogger(__name__).info(f"Baseline metrics: {asdict(metrics)}")


if __name__ == "__main__":
    main()
