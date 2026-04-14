import torch, ase
import numpy as np
from torch.utils.data import DataLoader
from models import BPGPModel
from train import GPTrainer
from utils import *
from kernels import BPKernel
from data_visualize import *


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")

### load the dataset and split into train-val-test
db = np.load("rmd17_benzene.npz")
y = torch.tensor(db['energies'], device=device, dtype=torch.float64)
y = y.double()

train_size = 400
val_size = 80
test_size = 400

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

print("Shape of train_y:", train_y.shape)
print("Finished normalizing targets...\n")

### Make the 2+3-descriptors
train_X_norm, val_X_norm, test_X_norm = get_descriptors([train_X, val_X, test_X], 
                                                        r_cut=6.0, sigma=1.0, desc="soap", n_max=12, l_max=8, device=f"{device}")
print(train_X_norm.shape)
train_mean = train_X_norm.mean()
train_std = train_X_norm.std()
train_X_norm = (train_X_norm - train_mean)/train_std
val_X_norm = (val_X_norm - train_mean)/train_std
test_X_norm = (test_X_norm - train_mean)/train_std

# save_tnsr_dct = {"train_X_norm": train_X_norm, "train_y": train_y, "val_X_norm": val_X_norm, "val_y": val_y, "test_X_norm":test_X_norm, "test_y": test_y}
# torch.save(save_tnsr_dct, "saved_features_labels.pt")
# print("Shape of train_X_norm:", train_X_norm.shape)
# print("Finished computing, normalizing descriptors...\n")

### Make a dataset and dataloader (this is extra, was not necessary in hindsight)
traindataset = AtomicEnvDataset(train_X_norm, train_y)
valdataset = AtomicEnvDataset(val_X_norm, val_y)
testdataset = AtomicEnvDataset(test_X_norm, test_y)

train_loader = DataLoader(traindataset, batch_size=len(train_X_norm), shuffle=False)
val_loader = DataLoader(valdataset, batch_size=len(val_X_norm), shuffle=False)
test_loader = DataLoader(testdataset, batch_size=len(test_X_norm), shuffle=False)

print("Finished initializing dataloaders...\n")


model = BPGPModel().to(device=device, dtype=torch.float64)
kernel = BPKernel(D=train_X_norm.shape[-1], learnable=True).to(device=device, dtype=torch.float64)
model.set_kernel(kernel)
model.fit(train_X_norm, train_y)
optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 2e-3},
        {"params": model.kernel.log_sigvar, "lr": 2e-3},
        {"params": model.log_noise, "lr": 2e-3},{"params": model.kernel.embed.parameters(), "lr":1e-3, "weight_decay":3e-4}])
epochs = 150

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
print("MAE: ",np.mean(error))
print(np.linalg.norm(error/var))
# plot_trainvsval(train_loss_ls, val_loss_ls)
plot_VarError(error, var)
plot_PredActualE(mean, test_y, var)
print("Final log_noise:", model.state_dict()['log_noise'])
print("Final log_sigvar:", kernel.state_dict()['log_sigvar'])
print("Final log_lengthscale:", kernel.state_dict()['log_lengthscale'])