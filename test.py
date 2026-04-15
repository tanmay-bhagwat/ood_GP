import torch, ase
import numpy as np
from torch.utils.data import DataLoader
from models import BPGPModel
from train import GPTrainer
from utils import *
from kernels import BPKernel
from data_visualize import *

TRAIN_X_MEAN, TRAIN_X_STD = 0, 1
TRAIN_Y_MEAN, TRAIN_Y_STD = 0, 1

def learning_curve(train_sizes, test_size=200, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = torch.device("cpu")
    
    maes, rmses = [], []
    test_size = test_size
    for train_n in train_sizes:
        
    ### load the dataset and split into train-val-test
    
        db = np.load("rmd17_benzene.npz")
        y = torch.tensor(db['energies'], device=device, dtype=torch.float64)
        y = y.double()

        test_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][-i,:,:]) for i in range(test_size)]
        test_y = y[-test_size:]
        
        train_size = train_n
        val_size = 1
        train_pts, val_pts, test_pts = train_val_test(len(y), train_size, val_size, test_size)

        train_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in range(train_n)]
        train_y = y[train_pts]
        val_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in val_pts]
        val_y = y[val_pts]

        print("Finished train-val-test splits...\n")

        train_y = (train_y - TRAIN_Y_MEAN)/TRAIN_Y_STD
        val_y = (val_y - TRAIN_Y_MEAN)/TRAIN_Y_STD
        test_y = (test_y - TRAIN_Y_MEAN)/TRAIN_Y_STD

        print("Finished normalizing targets...\n")

        ### Make the 2+3-descriptors
        train_X_norm, val_X_norm, test_X_norm = get_descriptors([train_X, val_X, test_X], 
                                                                r_cut=6.0, sigma=1.0, desc="soap", n_max=4, l_max=2, device=f"{device}")
        
        train_X_norm = (train_X_norm-TRAIN_X_MEAN)/TRAIN_X_STD 
        test_X_norm = (test_X_norm-TRAIN_X_MEAN)/TRAIN_X_STD
            

        print("Finished computing, normalizing descriptors...\n")

        ### Make a dataset and dataloader (this is extra, was not necessary in hindsight)
        traindataset = AtomicEnvDataset(train_X_norm.double(), train_y.double())
        valdataset = AtomicEnvDataset(val_X_norm.double(), val_y.double())
        testdataset = AtomicEnvDataset(test_X_norm.double(), test_y.double())

        train_loader = DataLoader(traindataset, batch_size=len(train_X_norm), shuffle=False)
        # val_loader = DataLoader(valdataset, batch_size=len(val_X_norm), shuffle=False)
        test_loader = DataLoader(testdataset, batch_size=len(test_X_norm), shuffle=False)

        print("Finished initializing dataloaders...\n")

        model = BPGPModel().to(device=device, dtype=torch.float64)
        model.fit(train_X_norm, train_y)
        kernel = BPKernel(D=train_X_norm.shape[-1], learnable=True).to(device=device, dtype=torch.float64)
        model.set_kernel(kernel)
        print("Kernel calculate...")
        model.kernel.full_kernel_block(train_X_norm, train_X_norm)
        optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 0.005},
                {"params": model.kernel.log_sigvar, "lr": 0.005},
                {"params": model.log_noise, "lr": 0.005}])
        epochs = epochs

        print("Starting training...")
        model.train()
        val_losses = []
        try:
            # Clear cache to maximize VRAM for the next size
            torch.cuda.empty_cache()
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = model.mll(update_buffers=True)
                loss.backward()
                optimizer.step()
                print(f"Train loss: {loss.item():.3f}")

            if epoch%5==0:
                with torch.no_grad():

                    model.eval()
                    mu, var = model.predict(test_X_norm)
                
                # Inverse scaling if needed (assuming unit-scaled for now)
                error = mu - test_y
                val_losses.append(error)
                rmse = torch.sqrt(torch.mean(error**2)).item()
                mae = torch.mean(torch.abs(error)).item()
                if error >= val_losses[-2]:
                    count +=1
                if count == 5:
                    print("Early stop")
                    break
            
                rmses.append(rmse)
                maes.append(mae)
                print(f"N={train_n} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
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

        
    return train_sizes, rmses, maes



sizes = [100, 150, 200, 300, 320]
test_size=200
for train_n in [400]:
    ### load the dataset and split into train-val-test
    test_size = test_size
    db = np.load("rmd17_benzene.npz")
    y = torch.tensor(db['energies'], dtype=torch.float64)
    y = y.double()

    test_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][-i,:,:]) for i in range(test_size)]
    test_y = y[-test_size:]
    
    train_size = train_n
    val_size = 1
    train_pts, val_pts, test_pts = train_val_test(len(y), train_size, val_size, test_size)

    train_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in range(train_n)]
    train_y = y[train_pts]
    val_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in val_pts]
    val_y = y[val_pts]

    print("Finished train-val-test splits...\n")

    TRAIN_Y_MEAN = train_y.mean()
    TRAIN_Y_STD = train_y.std()

    print("Finished normalizing targets...\n")

    ### Make the 2+3-descriptors
    train_X_norm, val_X_norm, test_X_norm = get_descriptors([train_X, val_X, test_X], 
                                                            r_cut=6.0, sigma=1.0, desc="soap", n_max=4, l_max=2)
    TRAIN_X_MEAN = train_X_norm.mean()
    TRAIN_X_STD = train_X_norm.std()

    
   
# Run analysis
train_sizes, rmses, maes = learning_curve(sizes)

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
