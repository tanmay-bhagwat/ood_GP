from pathlib import Path
import pytest
import torch
from ood_gp.interfaces import FeatureManifest
from ood_gp.evaluation import evaluate_predictions
from ood_gp.gp import ExactGPRegressor, GPConfig, StructureKernel


def _features() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]],
         [[1.0, 2.0], [2.0, 1.0]],
         [[3.0, 1.0], [2.0, 4.0]],
         [[4.0, 2.0], [3.0, 5.0]]],
        dtype=torch.float64
    )

def _simple_features() -> torch.Tensor:
    return torch.tensor(
        [[[1.0]],
         [[2.0]]],
        dtype=torch.float64
    )


def test_vectorized_against_naive() -> None:
    features = _features()
    config = GPConfig(
        feature_dimension=features.size(-1),
        log_noise=-6.0,
        log_signal_std=0.0,
        log_lengthscale=0.0,
        hidden_dimension=16,
        learnable_embedding=False,
        block_size=50,
        jitter=1.0e-6)
    
    lengthscale = torch.exp(torch.tensor(config.log_lengthscale))
    sigvar = torch.exp(torch.tensor(config.log_signal_std))**2 

    ### Naive implementation
    naive_calc_kernel = torch.zeros((features.shape[0], features.shape[0]), dtype=torch.float64)
    
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            dists = features[i,:,:].unsqueeze(0) - features[j,:,:].unsqueeze(1)
            
            exponent = -0.5 * torch.sum(dists**2 / lengthscale**2, dim=-1)
            naive_calc_kernel[i,j] = torch.sum(sigvar * torch.exp(exponent))

    vectorized_kernel = StructureKernel(config).full_kernel(features, features)

    torch.testing.assert_close(vectorized_kernel, naive_calc_kernel)
    

def test_predict_preserves_training_data() -> None:
    features = _features()
    targets = torch.tensor([1,2,3,4])
    config = GPConfig(
        feature_dimension=features.size(-1),
        log_noise=-6.0,
        log_signal_std=3.0,
        log_lengthscale=0.1,
        hidden_dimension=16,
        learnable_embedding=False,
        block_size=50,
        jitter=1.0e-6)
    
    train_X, val_X = features[:2,:,:], features[2:, :, :]
    train_targets, val_targets = targets[:2], targets[2:]
    gp_reg = ExactGPRegressor(config)
    gp_reg.fit(train_X, train_targets)
    gp_reg.predict(val_X)

    torch.testing.assert_close(gp_reg.train_X, train_X)
    torch.testing.assert_close(gp_reg.train_y, train_targets)


def test_prediction_on_one_point() -> None:
    
    config = GPConfig(
        feature_dimension=features.size(-1),
        log_noise=-6.0,
        log_signal_std=0.0,
        log_lengthscale=0.0,
        hidden_dimension=16,
        learnable_embedding=False,
        block_size=50,
        jitter=1.0e-6)
    
    gp_reg = ExactGPRegressor(config)
    
    features = torch.tensor([[[0.0]]])
    targets = torch.tensor([2.0])
    gp_reg.fit(features, targets)

    noise = torch.exp(torch.tensor(config.log_noise))
    ### Should be very close, did the calculation
    torch.testing.assert_close(
        gp_reg.predict(features).mean, 
        torch.full_like(features[:,0,0], targets[0]/(1.0 + noise + config.jitter)).double(),
        rtol=5e-7, atol=5e-7)
    
    nonzero_features = torch.tensor([[[1.0]], [[2.0]]])
    nonzero_targets = torch.tensor([2.0])
    gp_reg.fit(nonzero_features, nonzero_targets)

    noise = torch.exp(torch.tensor(config.log_noise))
    ### Should be very close, did the calculation
    torch.testing.assert_close(
        gp_reg.predict(nonzero_features).mean, 
        torch.full_like(nonzero_features[:,0,0], nonzero_targets[0]/(1.0 + noise + config.jitter)).double(),
        rtol=5e-7, atol=5e-7)
