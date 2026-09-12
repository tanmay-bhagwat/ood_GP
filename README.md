# Out of Distribution sample detection using Gaussian Processes
Can we make MLIP's that understand the difference between what they don't know (epistemic uncertainty) and when they are wrong (error)? i.e. Can they be uncertainty-aware? This Gaussian Process implementation in PyTorch is my attempt to test the hypothesis that GP-based MLIP's offer an inherent advantage in this regard and can be effectively used in active learning loops to further improve performance, while offer higher data efficiency.  

End-to-end automated dataset creation and curation pipeline (configurable algorithmic datapoint selection, random and Farthest-Point Sampling implemented currently), parallelized hyperparam tuning tests and vectorized kernel computations in PyTorch. Optimal MAE=0.16, with a z-score=1.02.   

The full report can be found [here](https://drive.google.com/file/d/1jw-AT3L1ao2Id0WNQa0b654j_spL-ScU/view?usp=drive_link).

## Packaged baseline

The corrected SOAP/exact-GP baseline now lives in `src/ood_gp`; the original code is retained in `legacy/` as historical provenance. The scientific and data
interfaces are documented in `docs/BASELINE_CONTRACT.md`.

Install and run it with:

```bash
python -m pip install -e ".[test]"
python scripts/run_baseline.py configs/baseline_soap.yaml
```

The example config expects an rMD17 NPZ dataset and an NPZ FPS descriptor cache
under `data/`. The cache must contain `descriptors` and a row-aligned
`frame_indices` array. We use the frame indices as the global identifiers during train-test-val split to ensure no overlap occurs. 

Run the cheapest tests first:

```bash
python -m pytest -q tests/unit
python -m pytest -q tests/integration -m "not gpu"
```
