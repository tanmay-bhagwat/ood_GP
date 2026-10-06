from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import copy

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gp import ExactGPRegressor

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
        if epoch%10==0: 
            print(f"Epoch {epoch}\n")
        model.train()
        if model.kernel.embedding is not None:
            embedding_enabled = epoch > config.embedding_burn_in_epochs
            for parameter in model.kernel.embedding.parameters():
                parameter.requires_grad = embedding_enabled

        model.condition(train_features, train_targets)
        train_nll = model.negative_log_likelihood()
        optimizer.zero_grad()
        train_nll.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            model.condition(validation_features, validation_targets)
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
    model.condition(train_features, train_targets)
    return history
