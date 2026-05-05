import torch
from torch.utils.data import DataLoader
from prod.models import GPModel
from prod.train import GPTrainer
from prod.utils import *
from prod.descriptors import AtomicDescriptor
from prod.kernels import StructKernel
from prod.data_visualize import *

TRAIN_X_MEAN, TRAIN_X_STD = 0, 1
TRAIN_Y_MEAN, TRAIN_Y_STD = 0, 1

def learning_curve(train_sizes, test_size=200, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = torch.device("cpu")
    
    maes, rmses, n_trains = [], [], []
    test_size = test_size
    for train_n in train_sizes:
        
    ### load the dataset and split into train-val-test
    
        ad = AtomicDescriptor(filepath="rmd17_benzene.npz", train_size=train_n, val_size=1, test_size=test_size,
                      descriptor_type="soap", species_ls = ["C","H"], r_cut=6.0, sigma=1.0, n_max=12, l_max=8)
        train_X_norm, _, test_X_norm = ad.get_features(normalize=False)
        train_y, _, test_y = ad.get_labels(normalize=False)

        train_X_norm = (train_X_norm - TRAIN_X_MEAN)/TRAIN_X_STD
        test_X_norm = (test_X_norm - TRAIN_X_MEAN)/TRAIN_X_STD

        train_y = (train_y - TRAIN_Y_MEAN)/TRAIN_Y_STD
        test_y = (test_y - TRAIN_Y_MEAN)/TRAIN_Y_STD

        print("Finished computing, normalizing descriptors...\n")

        model = GPModel().to(device=device, dtype=torch.float64)
        model.fit(train_X_norm, train_y)
        kernel = StructKernel(D=train_X_norm.shape[-1], learnable=True).to(device=device, dtype=torch.float64)
        model.set_kernel(kernel)

        optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 2e-3},
                {"params": model.kernel.log_sigvar, "lr": 2e-3},
                {"params": model.log_noise, "lr": 2e-3}])
        epochs = epochs

        print("Starting training...")
        model.train()
        val_losses = []
        try:
            # Clear cache to maximize VRAM for the next size
            torch.cuda.empty_cache()
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = model.nll()
                loss.backward()
                optimizer.step()
                print(f"Train loss: {loss.item():.3f}")
            
            with torch.no_grad():

                model.eval()
                mu, var = model.predict(test_X_norm)
            
            # Inverse scaling if needed (assuming unit-scaled for now)
            error = mu - test_y
            rmse = torch.sqrt(torch.mean(error**2)).item()
            mae = torch.mean(torch.abs(error)).item()

            n_trains.append(train_n)
            rmses.append(rmse)
            maes.append(mae)
            print(f"N={train_n} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
            print(len(rmses), len(maes))
        except torch._C._LinAlgError:
            print(f"FAILED N={train_n}: Matrix not Positive Definite. Skipping...")
            continue
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"FAILED N={train_n}: GPU Out of Memory. Stopping analysis.")
                torch.cuda.empty_cache()
                break # Usually best to stop here as sizes only increase
            else:
                print(f"FAILED N={train_n}: {e}")
                continue

        
    return n_trains, rmses, maes


sizes = [100, 150, 200, 300, 320]
test_size=200
ad = AtomicDescriptor(filepath="rmd17_benzene.npz", train_size=800, val_size=1, test_size=1,
                      descriptor_type="soap", species_ls = ["C","H"], r_cut=6.0, sigma=1.0, n_max=12, l_max=8)
train_X_norm, _, __ = ad.get_features(normalize=False)
train_y, _, __ = ad.get_labels(normalize=False)
TRAIN_X_MEAN, TRAIN_X_STD = train_X_norm.mean(), train_X_norm.std()
TRAIN_Y_MEAN, TRAIN_Y_STD = train_y.mean(), train_y.std()

# Run analysis
train_sizes, rmses, maes = learning_curve(sizes, epochs=100)

# Plotting the results
plt.figure(figsize=(8, 5))
plt.loglog(train_sizes, rmses, 'o-', label='RMSE')
plt.loglog(train_sizes, maes, 's-', label='MAE')
plt.xlabel('Training Set Size ($N$)')
plt.ylabel('Error (Energy Units)')
plt.title('GP Learning Curve (Log-Log Scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig('learning_curve.png')
plt.show()
