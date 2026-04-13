import torch
import ase
import numpy as np
from descriptors import BPDescriptor, soap_descriptor
from torch.utils.data import DataLoader
from models import BPGPModel
from train import GPTrainer
from utils import *
from kernels import BPKernel
from data_visualize import plot_VarError, plot_PredActualE


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


### load the dataset and split into train-val-test
db = np.load("rmd17_benzene.npz")
y = torch.tensor(db['energies'], device=device, dtype=torch.float64)
y = y.double()

train_size = 200
val_size = 20
test_size = 100

train_pts, val_pts, test_pts = train_val_test(len(y), train_size, val_size, test_size)

train_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in train_pts]
train_y = y[train_pts]
val_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in val_pts]
val_y = y[val_pts]
test_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in test_pts]
test_y = y[test_pts]
print("Finished train-val-test splits...\n")

train_y_mean = train_y.mean()
train_y_std = train_y.std()

train_y = (train_y - train_y_mean)/train_y_std
val_y = (val_y - train_y_mean)/train_y_std
test_y = (test_y - train_y_mean)/train_y_std

print("Finished normalizing targets...\n")

# ### Make the 2+3-descriptors
train_X_norm, val_X_norm, test_X_norm = get_descriptors([train_X, val_X, test_X], 
                                                        r_cut=6.0, sigma=1.0, desc="soap", n_max=4, l_max=2, device=f"{device}")
print(train_X_norm.shape)
train_mean = train_X_norm.mean()
train_std = train_X_norm.std()
train_X_norm = (train_X_norm - train_mean)/train_std
val_X_norm = (val_X_norm - train_mean)/train_std
test_X_norm = (test_X_norm - train_mean)/train_std

print("Finished computing, normalizing descriptors...\n")

### Make a dataset and dataloader (this is extra, was not necessary in hindsight)
traindataset = AtomicEnvDataset(train_X_norm, train_y)
valdataset = AtomicEnvDataset(val_X_norm, val_y)
testdataset = AtomicEnvDataset(test_X_norm, test_y)

train_loader = DataLoader(traindataset, batch_size=len(train_X_norm), shuffle=False)
val_loader = DataLoader(valdataset, batch_size=len(val_X_norm), shuffle=False)
test_loader = DataLoader(testdataset, batch_size=len(test_X_norm), shuffle=False)

print("Finished initializing dataloaders...\n")


model = BPGPModel().to(device=device)
kernel = BPKernel(D=train_X_norm.shape[-1]).to(device=device)
model.set_kernel(kernel)
optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 0.01},
        {"params": model.kernel.log_sigvar, "lr": 0.01},
        {"params": model.log_noise, "lr": 0.01}])
epochs = 100

print("Starting training...")
trainer = GPTrainer(model, optimizer, device=f"{device}")
train_loss_ls, val_loss_ls = trainer.train(epochs=epochs, data_loader=train_loader, val_loader=val_loader)


with torch.no_grad():

   model.eval()
   mean, var = model.predict(test_X_norm)
   print("mean shape:", mean.shape)
   print("var min:", var.min())
   print("var max:", var.max())
   
   print("Predictions: ", mean)
   print("Actual labels: ", test_y)

var = var.detach().cpu().numpy()
test_y = test_y.detach().cpu().numpy()
mean = mean.detach().cpu().numpy()

error = np.abs(mean-test_y)

plot_VarError(error, var)
plot_PredActualE(mean, test_y, var)