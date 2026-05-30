import torch
from models import GPModel
from kernels import StructKernel


learnable=True
device="cpu"

d = torch.load("saved_features_labels.pt", map_location=device)
train_X_norm, train_y = d['train_X_norm'].double(), d['train_y'].double()
val_X_norm, val_y = d['val_X_norm'].double(), d['val_y'].double()
test_X_norm, test_y = d['test_X_norm'].double(), d['test_y'].double()
config = {"device":device, "dtype":torch.float64,
          "log_noise":-6.0,
          "dim_size":train_X_norm.shape[-1], "log_lengthscale":0.1, "log_sigvar":3.0, "hidden_dim":16, "learnable":learnable,
          "max_num_epochs": 500, "train_x":train_X_norm, "train_y":train_y, "val_x":val_X_norm, "val_y":val_y}

model = GPModel(log_noise=config["log_noise"])
kernel = StructKernel(D=config["dim_size"], log_sigvar=config["log_sigvar"], log_lengthscale=config["log_lengthscale"], hidden_dim=config["hidden_dim"], learnable=config["learnable"])
model.set_kernel(kernel)
model_state, optimizer_state = torch.load("/work/10132/tanmay303/ls6/ray_tmp/gp_exp/trial_66c63_00002/checkpoint_000499/checkpoint.pt", map_location=torch.device("cpu"))
model.load_state_dict(model_state)

print(model_state)
print(next(model.parameters()), next(kernel.parameters()))