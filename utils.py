import numpy as np
from torch.utils.data import Dataset
from descriptors import *
import ase

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

    results = []

    for dataset in train_val_test:
        if desc == "bp_desc":
            r_cut_ls = r_cut
            sigma_ls = sigma
            n_basis = 4
            
            desc_norm = BPDescriptor(dataset, r_cut_ls, sigma_ls, n_basis).get_normalized_descriptors()
            results.append(desc_norm)

        elif desc == "soap":
            if n_max is None or l_max is None:
                raise ValueError("n_max and l_max must be provided for SOAP")
            
            # Using the specific parameters provided in the call
            desc_norm = soap_descriptor(dataset, species_ls=["C", "H"], r_cut=r_cut, 
                                        sigma=sigma, n_max=n_max, l_max=l_max, device=device)
            results.append(desc_norm)

    return tuple(results)


def population_stats(pop_size=400):

    ### load the dataset and split into train-val-test
    
    db = np.load("rmd17_benzene.npz")
    y = torch.tensor(db['energies'], dtype=torch.float64)
    y = y.double()
    
    train_size = pop_size
    val_size = 1
    test_size = 1
    train_pts, val_pts, test_pts = train_val_test(len(y), train_size, val_size, test_size)

    train_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in range(pop_size)]
    train_y = y[train_pts]
    val_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in val_pts]
    val_y = y[val_pts]

    print("Finished train-val-test splits...\n")

    TRAIN_Y_MEAN = train_y.mean().item()
    TRAIN_Y_STD = train_y.std().item()

    print("Finished normalizing targets...\n")

    ### Make the 2+3-descriptors
    train_X_norm = get_descriptors([train_X], r_cut=6.0, sigma=1.0, desc="soap", n_max=4, l_max=2)[0]
    TRAIN_X_MEAN = train_X_norm.mean()
    TRAIN_X_STD = train_X_norm.std()

    return (TRAIN_X_MEAN, TRAIN_X_STD, TRAIN_Y_MEAN, TRAIN_Y_STD)