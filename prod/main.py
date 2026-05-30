import torch
import os
from models import GPModel
from train import GPTrainer
from utils import *
from kernels import StructKernel
from data_visualize import *
from descriptors import AtomicDescriptor
from data_manager import DataManager


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")

### load the dataset and split into train-val-test
train_size = 400
val_size = int(train_size*0.2)
test_size = 400
make_new_descriptors = False
calculate_desc_for_fps = False
reload_states = False
learnable = True

r_cut = 6.0
sigma = 0.75
n_max = 12
l_max = 8

featuresdir_path = os.path.expandvars("/work/10132/tanmay303/ls6/me397_gp/prod/Descriptors")
os.makedirs(featuresdir_path, exist_ok=True)

if calculate_desc_for_fps:
        subset = 6000
        block_descriptors_calc("rmd17_benzene.npz", featuresdir_path=featuresdir_path, subset=subset, species_ls=["C","H"], r_cut=r_cut, sigma=sigma, n_max=n_max, l_max=l_max)


if make_new_descriptors:
        print("Making descriptors from scratch...")
        ad = AtomicDescriptor(descriptor_type="soap", 
                              species_ls = ["C","H"], r_cut=r_cut, sigma=sigma, n_max=n_max, l_max=l_max, device=device)
        dm = DataManager(rawdata_path="rmd17_benzene.npz", descriptor_engine=ad, desc_path=f"{featuresdir_path}/SoapDesc_6000.pt",
                         train_size=train_size, val_size=val_size, test_size=test_size, device=device, sample_strategy="fps_ood")
        data = dm.process_save(featuresdir_path=featuresdir_path)
else:
        dm = DataManager()
        data = dm.load_processed("Descriptors/SavedDescriptors_soap_6.0-0.75-12-8.pt")

train_X_norm, train_y = data["train_X_norm"], data["train_y"]
val_X_norm, val_y = data["val_X_norm"], data["val_y"]
test_X_norm, test_y = data["test_X_norm"], data["test_y"]
train_loader, val_loader, test_loader = dm.get_dataloaders()

config = {"device":device, "dtype":torch.float64,
          "log_noise":-6.0,
          "dim_size":train_X_norm.shape[-1], "log_lengthscale":0.1, "log_sigvar":3.0, "hidden_dim":16, "learnable":learnable,
          "max_num_epochs": 500, "train_x":train_X_norm, "train_y":train_y, "val_x":val_X_norm, "val_y":val_y}

model = GPModel(log_noise=config["log_noise"])
kernel = StructKernel(D=config["dim_size"], log_sigvar=config["log_sigvar"], log_lengthscale=config["log_lengthscale"], hidden_dim=config["hidden_dim"], learnable=config["learnable"])
model.set_kernel(kernel)
model.fit(train_X_norm, train_y)
_, optimizer_state = torch.load("/work/10132/tanmay303/ls6/ray_tmp/gp_exp/trial_66c63_00002/checkpoint_000499/checkpoint.pt", map_location=device)
optimizer = torch.optim.Adam([{"params": model.kernel.log_lengthscale, "lr": 2e-3},
        {"params": model.kernel.log_sigvar, "lr": 2e-3},
        {"params": model.log_noise, "lr": 1e-3},{"params": model.kernel.embed.parameters(), "lr":1e-3, "weight_decay":1e-4}])
optimizer.load_state_dict(optimizer_state)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode="min", patience=15)
epochs = 500

# if reload_states:
#         print("Reloading model states from checkpoint...")
#         checkpoint = torch.load("m1\\best_checkpoint.pth")
#         model.load_state_dict(checkpoint["model_state_dict"])
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
#         print(torch.equal(model.log_noise, checkpoint["model_state_dict"]['log_noise']))
#         print(torch.equal(model.kernel.log_sigvar, checkpoint["model_state_dict"]['kernel.log_sigvar']))
#         print(torch.equal(model.kernel.log_lengthscale, checkpoint["model_state_dict"]['kernel.log_lengthscale']))

# print(torch.equal(model.log_noise, model.state_dict()['log_noise']))
# print(torch.equal(model.kernel.log_sigvar, model.state_dict()['kernel.log_sigvar']))
# print(torch.equal(model.kernel.log_lengthscale, model.state_dict()['kernel.log_lengthscale']))

checkpoint_path = os.path.expandvars("/work/10132/tanmay303/ls6/me397_gp/prod/Checkpoints")
checkpoint_dir = os.makedirs(checkpoint_path, exist_ok=True)
print("Starting training...")
trainer = GPTrainer(model, optimizer, scheduler, device=f"{device}")
train_loss_ls, val_loss_ls = trainer.train(epochs=epochs, data_loader=train_loader, val_loader=val_loader, reload_states=reload_states, checkpoint_dir=checkpoint_dir)

with torch.no_grad():

   model.eval()
   print("Load best model to evaluate...")
   checkpoint = torch.load("best_checkpoint.pth")
   model.load_state_dict(checkpoint["model_state_dict"])
   model.log_noise = torch.nn.Parameter(torch.tensor(-6.0), requires_grad=False)
   model.kernel.log_sigvar = torch.nn.Parameter(torch.tensor(1.75), requires_grad=False)
   model.kernel.log_lengthscale = torch.nn.Parameter(torch.tensor(0.15), requires_grad=False)
   print(test_X_norm.shape)
   mean, var = model.predict(test_X_norm)
   
   print("mean shape:", mean.shape)
   print("var min:", var.min())
   print("var max:", var.max())
   
#    print("Predictions: ", mean)
#    print("Actual labels: ", test_y)


error = torch.abs(mean.cpu()-test_y.cpu())
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

ls = [param for param in next(model.kernel.embed.parameters())]
print(ls[0].shape, ls[1].shape, len(ls))