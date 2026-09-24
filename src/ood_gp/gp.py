"""
Historical summed-atomic kernel and exact Gaussian-process regression
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import copy
import torch
from .interfaces import FeatureManifest, PredictionResult


@dataclass(frozen=True)
class GPConfig:
    feature_dimension: int
    log_noise: float = -6.0
    log_signal_std: float = 3.0
    log_lengthscale: float = 0.1
    hidden_dimension: int = 16
    learnable_embedding: bool = True
    block_size: int = 50
    jitter: float = 1.0e-6


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    learning_rate_lengthscale: float = 2.0e-3
    learning_rate_signal: float = 2.0e-3
    learning_rate_noise: float = 1.0e-3
    learning_rate_embedding: float = 1.0e-3
    embedding_weight_decay: float = 1.0e-4
    embedding_burn_in_epochs: int = 50
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5


class LearnableEmbedding(torch.nn.Module):
    """
    The two-layer embedding used in 16D experiment
    """

    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension, dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(values)
    

class StructureKernel(torch.nn.Module):
    """
    RBF atomic kernel summed over all cross-structure atom pairs
    """

    def __init__(self, config: GPConfig) -> None:
        super().__init__()
        self.config = config
        self.log_signal_std = torch.nn.Parameter(torch.tensor(config.log_signal_std, dtype=torch.float64))
        
        ###>>> If using learnable embedding, need D dimensional lengthscales, else use the single given lengthscale
        self.learnable_embedding = config.learnable_embedding
        if self.learnable_embedding:
            self.embedding = LearnableEmbedding(config.feature_dimension, config.hidden_dimension)
            lengthscale_size = config.hidden_dimension
            initial_lengthscale = torch.full((lengthscale_size,), config.log_lengthscale, dtype=torch.float64)
        else:
            self.embedding = None
            initial_lengthscale = torch.tensor(config.log_lengthscale, dtype=torch.float64)
        self.log_lengthscale = torch.nn.Parameter(initial_lengthscale)

    @property
    def log_sigvar(self) -> torch.nn.Parameter:
        """
        Compatibility name used by historical checkpoints and reports
        """
        
        return self.log_signal_std

    def full_kernel(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross/self covariance matrix in bounded blocks
        """
        
        if left.ndim != 3 or right.ndim != 3:
            raise ValueError("structure kernel inputs must have shape (N, atoms, D)")
        result = torch.zeros((len(left), len(right)), device=left.device, dtype=left.dtype)

        block_size = self.config.block_size
        for left_start in range(0, len(left), block_size):
            for right_start in range(0, len(right), block_size):
                left_end = min(left_start + block_size, len(left))
                right_end = min(right_start + block_size, len(right))
                result[left_start:left_end, right_start:right_end] = self._block(
                    left[left_start:left_end], right[right_start:right_end])
        return result

    def _block(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            self.embedding.to(device=left.device, dtype=left.dtype)
            left = self.embedding(left)
            right = self.embedding(right)
        lengthscale = torch.exp(self.log_lengthscale).to(device=left.device, dtype=left.dtype)
        signal_std = torch.exp(self.log_signal_std).to(device=left.device, dtype=left.dtype)
        distances = left.unsqueeze(1).unsqueeze(3) - right.unsqueeze(0).unsqueeze(2)
        exponent = -0.5 * torch.sum(distances**2 / lengthscale**2,dim=-1)
        return signal_std**2 * torch.exp(exponent).sum(dim=(2, 3))


class ExactGPRegressor(torch.nn.Module):
    """
    Dense exact GP preserving the historical noise and jitter conventions
    """

    def __init__(self, config: GPConfig) -> None:
        super().__init__()
        self.config = config
        self.log_noise = torch.nn.Parameter(
            torch.tensor(config.log_noise, dtype=torch.float64))
        self.kernel = StructureKernel(config)
        self.train_X: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None
        self._cholesky: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> "ExactGPRegressor":
        """
        Condition the GP on data; hyperparameter optimization is separate
        """
        
        if features.ndim != 3:
            raise ValueError("features must have shape (N, atoms, D)")
        if targets.ndim != 1 or len(features) != len(targets):
            raise ValueError("targets must be one-dimensional and match features")
        self.train_X = features
        self.train_y = targets
        self._cholesky = None
        self._alpha = None
        return self

    def negative_log_likelihood(self) -> torch.Tensor:
        """
        Compute the exact negative log marginal likelihood
        """
        
        features, targets = self._require_training_data()
        noise = torch.exp(self.log_noise).to(
            device=features.device, dtype=features.dtype)
        covariance = self.kernel.full_kernel(features, features)
        identity = torch.eye(
            len(features), device=features.device, dtype=features.dtype)
        
        ###>>> covar matrix is symmetric, use Cholesky decomposition to solve for alpha
        ###>>> This can be unstable especially with large kernels, so using jitter
        covariance += (noise + self.config.jitter) * identity
        cholesky = torch.linalg.cholesky(covariance)

        targets_double = targets.double()
        cholesky_double = cholesky.double()
        alpha = torch.cholesky_solve(targets_double.unsqueeze(-1), cholesky_double)
        nll = 0.5 * targets_double @ alpha.squeeze(-1)
        nll += torch.log(torch.diag(cholesky_double)).sum()
        nll += 0.5 * len(features) * torch.log(
            torch.tensor(2.0 * torch.pi, device=features.device, dtype=torch.float64))
        self._cholesky = cholesky_double
        self._alpha = alpha
        return nll

    def nll(self) -> torch.Tensor:
        """
        Historical method alias
        """
        return self.negative_log_likelihood()

    def predict(self, test_features: torch.Tensor) -> PredictionResult:
        """
        Return latent predictive mean/variance and observation noise
        """
        train_features, _ = self._require_training_data()
        self.negative_log_likelihood()
        assert self._cholesky is not None and self._alpha is not None
        K_s = self.kernel.full_kernel(test_features, train_features)
        K_ss = self.kernel.full_kernel(test_features, test_features)
        mean = K_s.double() @ self._alpha.double()
        triangular = torch.linalg.solve_triangular(
            self._cholesky, K_s.T.double(), upper=False)
        covariance = K_ss.double() - triangular.T @ triangular
        noise = torch.exp(self.log_noise).to(
            device=test_features.device, dtype=torch.float64)
        return PredictionResult(
            mean=mean.squeeze(-1),
            variance=torch.diag(covariance),
            noise_variance=noise)

    def save(self, directory: Path, feature_manifest: FeatureManifest) -> None:
        """
        Save model tensors and a JSON compatibility manifest
        """
        train_features, train_targets = self._require_training_data()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(),
             "train_X": train_features.detach().cpu(),
             "train_y": train_targets.detach().cpu()},
            directory / "gp_state.pt")
        (directory / "manifest.json").write_text(
            json.dumps(
                {"gp_config": asdict(self.config),
                 "feature_manifest": feature_manifest.to_dict()},
                indent=2,
                sort_keys=True)
                + "\n",
            encoding="utf-8")

    @classmethod
    def load(cls,
        directory: Path,
        expected_feature_manifest: FeatureManifest,
        *,
        device: str = "cpu") -> "ExactGPRegressor":
        """
        Load a bundle only when its feature provenance is compatible
        """
        directory = Path(directory)
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["feature_manifest"] != expected_feature_manifest.to_dict():
            raise ValueError("GP bundle is incompatible with the feature manifest")
        model = cls(GPConfig(**manifest["gp_config"])).to(device)
        payload: dict[str, Any] = torch.load(
            directory / "gp_state.pt", map_location=device, weights_only=True)
        model.load_state_dict(payload["state_dict"])
        model.fit(payload["train_X"].to(device), payload["train_y"].to(device))
        return model

    def _require_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_X is None or self.train_y is None:
            raise RuntimeError("fit must be called before NLL or prediction")
        return self.train_X, self.train_y


def model_train(
    model: ExactGPRegressor,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    config: TrainingConfig) -> dict[str, list[float]]:
    """
    Optimize historical GP parameters and restore train conditioning.
    Validation marginal likelihood selects the best parameter state. Validation
    data never remains attached to the model when this function returns.
    """
    parameter_groups: list[dict[str, Any]] = [
        {"params": [model.kernel.log_lengthscale],
         "lr": config.learning_rate_lengthscale},
        {"params": [model.kernel.log_signal_std],
         "lr": config.learning_rate_signal},
        {"params": [model.log_noise], 
         "lr": config.learning_rate_noise}]
    if model.kernel.embedding is not None:
        parameter_groups.append(
            {"params": model.kernel.embedding.parameters(),
             "lr": config.learning_rate_embedding,
             "weight_decay": config.embedding_weight_decay})
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience)
    history: dict[str, list[float]] = {"train_nll": [], "validation_nll": []}
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        model.train()
        if model.kernel.embedding is not None:
            embedding_enabled = epoch > config.embedding_burn_in_epochs
            for parameter in model.kernel.embedding.parameters():
                parameter.requires_grad = embedding_enabled

        model.fit(train_features, train_targets)
        train_nll = model.negative_log_likelihood()
        optimizer.zero_grad()
        train_nll.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            model.fit(validation_features, validation_targets)
            validation_nll = model.negative_log_likelihood()
        scheduler.step(validation_nll)
        train_value = float(train_nll.detach().cpu())
        validation_value = float(validation_nll.detach().cpu())
        history["train_nll"].append(train_value)
        history["validation_nll"].append(validation_value)
        if validation_value <= best_validation:
            best_validation = validation_value
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training completed without a model state")
    model.load_state_dict(best_state)
    model.fit(train_features, train_targets)
    return history
