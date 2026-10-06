from pathlib import Path
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotData:

    train_nll: np.ndarray
    val_nll: np.ndarray
    frame_indices: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray
    predictive_variance: np.ndarray
    noise_variance: np.ndarray
    target_label: str
    variance_label: str


def load_run_data(artifacts_path: Path, normalized: bool):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if not artifacts_path.is_absolute():
        artifacts_path = PROJECT_ROOT/artifacts_path

    with np.load(artifacts_path/"normalization.npz") as norm_state:
        target_mean, target_std = norm_state["target_mean"], norm_state["target_std"]

    with np.load(artifacts_path/"predictions.npz") as arrays:
        frame_indices = _one_dimensional_array(arrays["frame_indices"], "frame_indices")
        targets = _one_dimensional_array(arrays["target"], "target")
        predictions = _one_dimensional_array(arrays["mean"], "mean")
        predictive_variance = _one_dimensional_array(arrays["variance"], "variance")
        noise_values = np.asarray(arrays["noise_variance"], dtype=np.float64)

    minimum_variance = float(predictive_variance.min())
    if minimum_variance < -1.0e-10:
        raise ValueError(
            f"predictive variance contains a negative value: {minimum_variance}")
    predictive_variance = np.maximum(predictive_variance, 0.0)

    manifest_path = artifacts_path/"run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text())
    history = run_manifest["history"]
    train_nll = _one_dimensional_array(history["train_nll"], "train_nll")
    val_nll = _one_dimensional_array(history["validation_nll"], "val_nll")

    if not normalized:
        print("Data will be unnormalized using saved mean, stdev of train set")
        predictive_variance *= target_std**2
        noise_values *= target_std**2
        predictions = predictions*target_std + target_mean
        targets = targets*target_std + target_mean
        target_label = f"{run_manifest["units"]["energy"]}"
        variance_label = f"{run_manifest["units"]["energy"]}^2"
    else:
        target_label=""
        variance_label=""

    plot_data = PlotData(
        train_nll=train_nll,
        val_nll=val_nll,
        frame_indices=frame_indices,
        targets=targets,
        predictions=predictions,
        predictive_variance=predictive_variance,
        noise_variance=noise_values,
        target_label=target_label,
        variance_label=variance_label
    )

    return plot_data


def plot_history(data:PlotData, output_dir:Path):
    plt.plot(np.arange(len(data.train_nll)), data.train_nll)
    plt.plot(np.arange(len(data.val_nll)), data.val_nll)
    plt.xlabel("Epochs")
    plt.ylabel("Negative log likelihood")
    plt.savefig(output_dir/"history.pdf", format="pdf")
    plt.cla()


def plot_predictions(data:PlotData, output_dir:Path):
    if data.target_label=="":
        plot_points = data.predictive_variance<1.0
    else:
        plot_points = data.predictive_variance<np.std(data.targets)
    plt.scatter(data.targets[plot_points], 
                data.predictions[plot_points],
                c=data.predictive_variance[plot_points],
                cmap="viridis",
                s=28, alpha=0.8,
                edgecolors="none")
    plt.colorbar(label="Predictive variance")
    plt.xlabel(f"Reference energies ({data.target_label})")
    plt.ylabel(f"Predicted energies ({data.target_label})")

    lims = [
        min(data.targets.min(), data.predictions.min()),
        max(data.targets.max(), data.predictions.max())
    ]

    plt.plot(lims, lims, 'k--')
    plt.savefig(output_dir/f"preds_{"normalized" if data.target_label=="" else "unnormalized"}.pdf", format="pdf")
    plt.cla()
    

def plot_logerrors_logvar(data:PlotData, output_dir:Path):
    error = np.abs(data.targets - data.predictions)
    uncertainty = np.sqrt(data.predictive_variance)
    plot_points = np.all(uncertainty>0) and np.all(error>0)
    if not np.any(plot_points):
        raise ValueError("Atleast one point must have positive variance and absolute error")
    
    log_uncertainty = np.log10(uncertainty)
    log_error = np.log10(error)
    plot_points = log_uncertainty<-0.25
    log_uncertainty = log_uncertainty[plot_points]
    log_error = log_error[plot_points]
    
    plt.cla()
    plt.scatter(log_error, log_uncertainty, alpha=0.7)
    plt.xlabel(r"$\log|y-\hat{y}|$")
    plt.ylabel(r"$\log\sqrt{\sigma_i^2+\sigma_n^2}$")
    plt.savefig(output_dir/"Uncertainty_error.pdf", format="pdf")
    plt.cla()


def run_plots(run_dir:Path, output_dir:Path, normalized_flag:bool):
    
    # PROJECT_ROOT = Path(__file__).resolve().parents[1]

    plot_data = load_run_data(run_dir, normalized_flag)
    figures_dir = output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_history(plot_data, figures_dir)
    plot_predictions(plot_data, figures_dir)
    plot_logerrors_logvar(plot_data, figures_dir)



def _one_dimensional_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim!=1 or arr.size==0:
        raise ValueError("Given list must be non-empty one-dimensional")
    elif not np.isfinite(arr).all():
        raise ValueError("One or more elements are not finite in given list")
    
    return arr.copy()
