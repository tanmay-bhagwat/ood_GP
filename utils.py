import numpy as np
from torch.utils.data import Dataset
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


   
    