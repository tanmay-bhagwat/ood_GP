import torch
import os, tempfile
import ray
from models import GPModel
from kernels import StructKernel
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from utils import *
from descriptors import AtomicDescriptor, AtomicEnvDataset


class GPTrainer:
    def __init__(self, config={}):
        self.config = config
        self.device = self.config["device"]
        self.dtype = self.config["dtype"]

        self.model = GPModel(log_noise=self.config["log_noise"]).to(device=self.device, dtype=self.dtype)
        self.kernel = StructKernel(D=self.config["dim_size"], log_sigvar=self.config["log_sigvar"], log_lengthscale=self.config["log_lengthscale"], hidden_dim=config["hidden_dim"],
                                   learnable=self.config["learnable"], separate_ll=False).to(device=self.device, dtype=self.dtype)
        self.model.set_kernel(self.kernel)
        self.optimizer = torch.optim.Adam([{"params": self.model.kernel.log_lengthscale, "lr": float(self.config["lr_ll"])},
                {"params": self.model.kernel.log_sigvar, "lr": float(self.config["lr_ls"])},
                {"params": self.model.log_noise, "lr": float(self.config["lr_ln"])},
                {"params": self.model.kernel.embed.parameters(), "lr":float(self.config["lr_embed"]), "weight_decay":float(self.config["weight_decay"])}])
        
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, mode="min", patience=15)    

    
    def train_epoch(self, data_loader, val_loader):
        model = self.model
        model.train()
        
        # One batch is the whole dataset
        # ln_prior = Normal(loc=-6.5, scale=0.5)
        # ls_prior = Normal(loc=3.5, scale=0.5)
        # ll_prior = Normal(loc=0.1, scale=0.2)
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            model.fit(data, labels)
            train_loss = model.nll()

            # reg_loss = -ln_prior.log_prob(model.log_noise) + \
            #     -ls_prior.log_prob(model.kernel.log_sigvar) + -ll_prior.log_prob(model.kernel.log_lengthscale).sum()
            # train_loss += reg_loss
            
            # print(f"Batch {batch_idx}, batch loss: {train_loss:.3f} ")

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            
        # print(f"Total loss: {train_loss:.3f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (val_data, val_labels) in enumerate(val_loader):
                val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                model.fit(val_data, val_labels)
                val_loss = model.nll()
            # self.scheduler.step(val_loss)
        
        metrics = {"loss": val_loss.item()/len(val_loader)}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((model.state_dict(), self.optimizer.state_dict()), path)
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)

            # Report metrics and checkpoint to Ray Tune
            tune.report(metrics, checkpoint=checkpoint)

        return train_loss, val_loss


    def train(self, data_loader, val_loader=None, reload_states=False):

        count_val, count_tn = 0, 0
        f = 0.1
        patience, tn_limit = 25, 20

        if val_loader is None:
            raise AttributeError("Validation set must be provided for hyperparam tuning!")
        
        model = self.model.to(device=self.device)
        train_loss_ls, val_loss_ls = [], []
        epochs = config["max_num_epochs"]
        epoch = 0
        
        if tune.get_checkpoint():
            loaded_checkpoint = tune.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
                model.load_state_dict(model_state)  # restore model weights 
                self.optimizer.load_state_dict(optimizer_state)
                
        while epoch < epochs:
            if epoch <= 150:
                for param in model.kernel.embed.parameters():
                    param.requires_grad = False
            else:
                for param in model.kernel.embed.parameters():
                    param.requires_grad = True
            train_loss, val_loss = self.train_epoch(data_loader, val_loader)
            epoch += 1
            
            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            print(f"Epoch {epoch}: Train loss = {train_loss:.3f}, val loss = {val_loss:.3f}")

            # if val loss not decreasing over some iters, stop 
            if epoch >= 10: 
                # hacky way to bypass errors when restarting training at an epoch > 150 with no val loss history
                try:
                    if torch.abs(val_loss - val_loss_ls[-2]) <= 0.001:
                        count_tn += 1
                        print("======= Validation loss increased =======")
                        if count_tn == tn_limit:
                            print(f"Validation loss increased for {tn_limit} consecutive epochs, stopping...")
                            checkpoint = {"epoch":epoch, "model_state_dict": self.model.state_dict(), 
                                "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
                                "loss":val_loss}
                            torch.save(checkpoint, "last_checkpoint.pth")
                            break
                    else:
                        count_tn = 0 
                except:
                    pass
        
        return train_loss_ls, val_loss_ls
        

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load the dataset and split into train-val-test
train_size = 400
val_size = int(train_size*0.2)
test_size = 400
make_new_descriptors = False
calculate_desc_for_fps = False
reload_states = False
learnable = True

r_cut = 6.0
sigma = 1.0
n_max = 12
l_max = 8
rawdata_filepath = "rmd17_benzene.npz"

# --- First get a large descriptor database for running fps
if calculate_desc_for_fps:
    subset = 6000
    block_descriptors_calc(rawdata_filepath, subset=subset, species_ls=["C","H"], r_cut=r_cut, sigma=sigma, n_max=n_max, l_max=l_max)

# --- Either make new descriptors or reload existing ones
# --- For consistency, the database above and the train-val-test datasets below should have the same SOAP params (r_cut, sigma, n_max, l_max)
if make_new_descriptors:
    print("Making descriptors from scratch...")
    ad = AtomicDescriptor(filepath=rawdata_filepath, train_size=train_size, val_size=val_size, test_size=test_size,
                    descriptor_type="soap", species_ls = ["C","H"], r_cut=r_cut, sigma=sigma, n_max=n_max, l_max=l_max, device=device,
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
       raise ValueError("Train size does not match given input size!")
if val_X_norm.shape[0] != val_size:
       raise ValueError("Val size does not match given input size!")
if test_X_norm.shape[0] != test_size:
       raise ValueError("Test size does not match given input size!")

# --- Make a dataset and dataloader (this is extra, was not necessary in hindsight)

print("Finished initializing dataloaders...\n")
test_X_norm = test_X_norm.to(device=device, dtype=torch.float64)
test_y = test_y.to(device=device, dtype=torch.float64)

# --- Use Ray Storage to put datasets into shared memory before workers initialize for Ray processes
ray_tmp_dir = os.path.expandvars("/work/10132/tanmay303/ls6/ray_tmp")
os.makedirs(ray_tmp_dir, exist_ok=True)

ray.init(_temp_dir=ray_tmp_dir)

train_x = ray.put(train_X_norm)
train_y = ray.put(train_y)

val_x = ray.put(val_X_norm)
val_y = ray.put(val_y)

# --- Define model params and hyperparams in a single config dict
config = {"device":"cuda", "dtype":torch.float64,
          "log_noise":-6.0,
          "dim_size":train_X_norm.shape[-1], "log_lengthscale":0.1, "log_sigvar":3.0, "hidden_dim":16, "learnable":learnable,
          "lr_ll": tune.loguniform(1e-4,1e-2), "lr_ls": tune.loguniform(1e-4,1e-2), "lr_ln": tune.loguniform(1e-4,1e-2), "lr_embed": tune.loguniform(1e-4,1e-2), "weight_decay": tune.loguniform(1e-5,1e-3),
          "max_num_epochs": 500, "train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y}

def train_fn(config):

    train_X_norm = ray.get(config["train_x"])
    train_y = ray.get(config["train_y"])
    val_X_norm = ray.get(config["val_x"])
    val_y = ray.get(config["val_y"])

    traindataset = AtomicEnvDataset(train_X_norm, train_y)
    valdataset = AtomicEnvDataset(val_X_norm, val_y)

    train_loader = DataLoader(traindataset, batch_size=len(traindataset), shuffle=False)
    val_loader = DataLoader(valdataset, batch_size=len(valdataset), shuffle=False)

    trainer = GPTrainer(config=config)
    trainer.train(data_loader=train_loader, val_loader=val_loader)

def shallow_name(trial):
    return f"trial_{trial.trial_id}"

scheduler = ASHAScheduler(max_t=config["max_num_epochs"], grace_period=20, reduction_factor=2)
tuner = tune.Tuner(tune.with_resources(train_fn, resources={"cpu": 4, "gpu": 1}),
                   param_space=config,
                   run_config=tune.RunConfig(storage_path=f"{ray_tmp_dir}", name="gp_exp"),
                   tune_config=tune.TuneConfig(mode="min", metric="loss", scheduler=scheduler, num_samples=20, 
                                               trial_dirname_creator=shallow_name))

results = tuner.fit()

best_result = results.get_best_result("loss", "min")
print(f"Best trial config: {best_result.config}")
print(f"Best trial final validation loss: {best_result.metrics['loss']}")