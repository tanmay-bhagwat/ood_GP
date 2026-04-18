import torch
from torch.utils.data import DataLoader
from models import GPModel
from train import GPTrainer
from utils import *
from kernels import StructKernel
from data_visualize import *
from descriptors import AtomicDescriptor

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")

### load the dataset and split into train-val-test
train_size = 400
val_size = int(train_size*0.2)
test_size = 400
make_new = False
reload_states = False
calculate_desc_for_fps = False
learnable = True

if calculate_desc_for_fps:
       subset = 6000
       block_descriptors_calc("rmd17_benzene.npz", subset=subset, species_ls=["C","H"], r_cut=6.0, sigma=0.5, n_max=12, l_max=8)

if make_new:
        print("Making descriptors from scratch...")
        ad = AtomicDescriptor(filepath="rmd17_benzene.npz", train_size=train_size, val_size=val_size, test_size=test_size,
                        descriptor_type="soap", species_ls = ["C","H"], r_cut=6.0, sigma=0.5, n_max=12, l_max=8, device=device,
                        sample_strategy="stratified")
        train_X_norm, val_X_norm, test_X_norm = ad.get_features()
        train_y, val_y, test_y = ad.get_labels()

        save_tnsr_dct = {"train_X_norm": train_X_norm, "train_y": train_y, "val_X_norm": val_X_norm, "val_y": val_y, "test_X_norm":test_X_norm, "test_y": test_y}
        torch.save(save_tnsr_dct, "saved_features_labels.pt")
        print("Shape of train_X_norm:", train_X_norm.shape)

else:
        print("Loading descriptors...")
        d = torch.load("saved_features_labels.pt", map_location=device)
        train_X_norm, train_y = d['train_X_norm'].double(), d['train_y'].double()
        val_X_norm, val_y = d['val_X_norm'].double(), d['val_y'].double()
        test_X_norm, test_y = d['test_X_norm'].double(), d['test_y'].double()

if train_X_norm.shape[0] != train_size:
       raise ValueError("Train size does not match to input given!")
if val_X_norm.shape[0] != val_size:
       raise ValueError("Val size does not match to input given!")
if test_X_norm.shape[0] != test_size:
       raise ValueError("Test size does not match to input given!")

### Make a dataset and dataloader (this is extra, was not necessary in hindsight)
traindataset = AtomicEnvDataset(train_X_norm, train_y)
valdataset = AtomicEnvDataset(val_X_norm, val_y)
testdataset = AtomicEnvDataset(test_X_norm, test_y)

train_loader = DataLoader(traindataset, batch_size=len(train_X_norm), shuffle=False)
val_loader = DataLoader(valdataset, batch_size=len(val_X_norm), shuffle=False)
test_loader = DataLoader(testdataset, batch_size=len(test_X_norm), shuffle=False)
print("Finished initializing dataloaders...\n")
test_X_norm = test_X_norm.to(device=device, dtype=torch.float64)
test_y = test_y.to(device=device, dtype=torch.float64)


model = GPModel(log_noise=-10.).to(device=device, dtype=torch.float64)
kernel = StructKernel(D=train_X_norm.shape[-1], learnable=learnable, separate_ll=True).to(device=device, dtype=torch.float64)
model.set_kernel(kernel)
model.fit(train_X_norm, train_y)

if learnable:
        optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 2e-3},
                {"params": model.kernel.log_sigvar, "lr": 2e-3},
                {"params": model.log_noise, "lr": 1e-4},{"params": model.kernel.embed.parameters(), "lr":1e-3, "weight_decay":1e-4}])
else:
       optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 1e-3},
        {"params": model.kernel.log_sigvar, "lr": 1e-3},
        {"params": model.log_noise, "lr": 1e-4}])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode="min", patience=15)
epochs = 600

if reload_states:
        print("Reloading model states from checkpoint...")
        checkpoint = torch.load("best_checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(torch.equal(model.log_noise, checkpoint["model_state_dict"]['log_noise']))
        print(torch.equal(model.kernel.log_sigvar, checkpoint["model_state_dict"]['kernel.log_sigvar']))
        print(torch.equal(model.kernel.log_lengthscale, checkpoint["model_state_dict"]['kernel.log_lengthscale']))

print(torch.equal(model.log_noise, model.state_dict()['log_noise']))
print(torch.equal(model.kernel.log_sigvar, model.state_dict()['kernel.log_sigvar']))
print(torch.equal(model.kernel.log_lengthscale, model.state_dict()['kernel.log_lengthscale']))


print("Starting training...")
trainer = GPTrainer(model, optimizer, scheduler, device=f"{device}")
train_loss_ls, val_loss_ls = trainer.train(epochs=epochs, data_loader=train_loader, val_loader=val_loader, reload_states=reload_states)

with torch.no_grad():

   model.eval()
   print("Load best model to evaluate...")
   checkpoint = torch.load("best_checkpoint.pth")
   model.load_state_dict(checkpoint["model_state_dict"])
#    model.log_noise = torch.nn.Parameter(torch.tensor(-10), requires_grad=False)
#    model.kernel.log_sigvar = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
#    model.kernel.log_lengthscale = torch.nn.Parameter(torch.tensor(2.5), requires_grad=False)
   print(test_X_norm.shape)
   mean, var = model.predict(test_X_norm)
   
   print("mean shape:", mean.shape)
   print("var min:", var.min())
   print("var max:", var.max())
   
#    print("Predictions: ", mean)
#    print("Actual labels: ", test_y)


error = torch.abs(mean.cpu()-test_y.cpu())
print(error)
print("MAE: ",torch.mean(error))
print("z: ", torch.mean(error/torch.sqrt(var.cpu() + torch.exp(model.log_noise).cpu())))


var = var.detach().cpu().numpy()
test_y = test_y.detach().cpu().numpy()
mean = mean.detach().cpu().numpy()

# plot_trainvsval(train_loss_ls, val_loss_ls)
plot_VarError(error, var, torch.exp(model.log_noise).detach().cpu().numpy())
plot_PredActualE(mean, test_y, var)
print("Final log_noise:", model.log_noise.item())
print("Final log_sigvar:", kernel.log_sigvar.item())
print("Final log_lengthscale:", kernel.log_lengthscale.tolist())