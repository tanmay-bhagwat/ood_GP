import numpy as np
from torch.utils.data import Dataset
from descriptors import *

class AtomicEnvDataset(Dataset):

    def __init__(self, descriptors, energy) -> None:
        super().__init__()
        self.X = descriptors
        self.y = energy

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def train_val_test(db_size, train_size=100, val_size=20, test_size=10):
    ### Define train, val, test sample sizes
    training_size = train_size
    validation_size = val_size
    test_size = test_size

    np.random.seed(1)
    shuffled_frames = [int(n) for n in range(db_size)]
    np.random.shuffle(shuffled_frames)

    training_pts = shuffled_frames[0:training_size]
    validation_pts = shuffled_frames[training_size:training_size+validation_size]
    test_pts = shuffled_frames[training_size+validation_size : training_size+validation_size+test_size]

    return training_pts, validation_pts, test_pts


def get_descriptors(train_val_test, r_cut, sigma, desc="bp_desc", n_max=None, l_max=None, device="cpu"):

    train_X, val_X, test_X = train_val_test[0], train_val_test[1], train_val_test[2]
    if desc == "bp_desc":
        r_cut_ls = r_cut
        sigma_ls = sigma
        n_basis = 4
        train_X_norm = BPDescriptor(train_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()
        val_X_norm = BPDescriptor(val_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()
        test_X_norm = BPDescriptor(test_X, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()

    if desc == "soap":
        if n_max is None or l_max is None:
            raise ValueError("n_max and l_max must be provided for SOAP")
        
        r_cut = r_cut
        sigma = sigma
        train_X_norm = soap_descriptor(train_X, species_ls=["C","H"], r_cut=6.0, sigma=1.0, n_max=n_max, l_max=l_max, device=device)
        val_X_norm = soap_descriptor(val_X, species_ls=["C","H"], r_cut=6.0, sigma=1.0, n_max=n_max, l_max=l_max, device=device)
        test_X_norm = soap_descriptor(test_X, species_ls=["C","H"], r_cut=6.0, sigma=1.0, n_max=n_max, l_max=l_max, device=device)
       
    return train_X_norm, val_X_norm, test_X_norm
