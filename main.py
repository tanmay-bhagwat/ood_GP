import torch
import numpy as np
from descriptors import BPDescriptor
from torch.utils.data import DataLoader
from models import BPGPModel
from train import GPTrainer
from utils import *
from kernels import BPKernel
from data_visualize import plot_VarError, plot_PredActualE


### load the dataset and split into train-val-test
db = np.load("rmd17_benzene.npz")
X, y = torch.tensor(db['coords']), torch.tensor(db['energies'])
X, y = X.double(), y.double()

train_size = 200
val_size = 20
test_size = 100
(train_X, train_y), (val_X, val_y), (test_X, test_y) = train_val_test(X, y, train_size, val_size, test_size)

print("Finished train-val-test splits...\n")

train_y_mean = train_y.mean()
train_y_std = train_y.std()

train_y = (train_y - train_y_mean)/train_y_std
val_y = (val_y - train_y_mean)/train_y_std
test_y = (test_y - train_y_mean)/train_y_std

print("Finished normalizing targets...\n")

### Make the 2+3-descriptors
r_cut_ls = [6.5, 4]
sigma_ls = [1.0]
n_basis = 4
train_X_norm = BPDescriptor(train_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()
val_X_norm = BPDescriptor(val_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()
test_X_norm = BPDescriptor(test_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()

print("Finished computing, normalizing descriptors...\n")


### Make a dataset and dataloader (this is extra, was not necessary in hindsight)
traindataset = AtomicEnvDataset(train_X_norm, train_y)
valdataset = AtomicEnvDataset(val_X_norm, val_y)
testdataset = AtomicEnvDataset(test_X_norm, test_y)

train_loader = DataLoader(traindataset, batch_size=len(train_X_norm), shuffle=False)
val_loader = DataLoader(valdataset, batch_size=len(val_X_norm), shuffle=False)
test_loader = DataLoader(testdataset, batch_size=len(test_X_norm), shuffle=False)

print("Finished initializing dataloaders...\n")


model = BPGPModel()
kernel = BPKernel(D=train_X_norm.shape[-1])
model.set_kernel(kernel)
optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 0.05},
        {"params": model.kernel.log_sigvar, "lr": 0.05},
        {"params": model.log_noise, "lr": 0.005}])
epochs = 50

print("Starting training...")
trainer = GPTrainer(model, optimizer)
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