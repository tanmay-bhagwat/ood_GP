import numpy as np
from torch.utils.data import Dataset
import torch, os
from descriptors import bp_descriptor_batch


class AtomicEnvDataset(Dataset):

    def __init__(self, descriptors, energy) -> None:
        super().__init__()
        self.X = descriptors
        self.y = energy

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def train_val_test(X, y, train_size=100, val_size=20, test_size=10):
    ### Define train, val, test sample sizes
    training_size = train_size
    validation_size = val_size
    test_size = test_size

    np.random.seed(1)
    shuffled_frames = [int(n) for n in range(len(y))]
    np.random.shuffle(shuffled_frames)

    training_pts = shuffled_frames[0:training_size]
    validation_pts = shuffled_frames[training_size:training_size+validation_size]
    test_pts = shuffled_frames[training_size+validation_size : training_size+validation_size+test_size]

    train_X = X[training_pts, :]
    train_y = y[training_pts]
    val_X = X[validation_pts, :]
    val_y = y[validation_pts]

    test_X = X[test_pts, :]
    test_y = y[test_pts]

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def get_normalized_descriptors(data_ls, r_cut_ls=[5.0], sigma_ls = [1.0], n_basis=4, load=False):

    train_X, val_X, test_X = data_ls
    os.chdir("./data_descriptors")
    if load==True:
        try:
            train_X_norm = torch.tensor(np.load("train_descriptors.npy"))
        except:
            raise FileNotFoundError("No train_X descriptors file found")
            
        try:
            val_X_norm = torch.tensor(np.load("val_descriptors.npy"))
        except:
            raise FileNotFoundError("No val_X descriptors file found")
            
        try:
            test_X_norm = torch.tensor(np.load("test_descriptors.npy"))
        except:
            raise FileNotFoundError("No test_X descriptors file found")
        
    else:
        train_X = bp_descriptor_batch(train_X, r_cut_ls, sigma_ls, n_basis)
        val_X = bp_descriptor_batch(val_X, r_cut_ls, sigma_ls, n_basis)
        test_X = bp_descriptor_batch(test_X, r_cut_ls, sigma_ls, n_basis)
        train_mean = train_X.mean(dim=(0,1), keepdim=True)
        train_std  = train_X.std(dim=(0,1), keepdim=True)

        train_X_norm = (train_X - train_mean)/(train_std + 1e-8)
        val_X_norm = (val_X - train_mean)/(train_std + 1e-8)
        test_X_norm = (test_X - train_mean)/(train_std + 1e-8)

        np.save("train_descriptors", train_X_norm)
        np.save("val_descriptors", val_X_norm)
        np.save("test_descriptors", test_X_norm)
        os.chdir("..")

    # print(train_X_norm.abs().max())

    return train_X_norm, val_X_norm, test_X_norm