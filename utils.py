import numpy as np
from torch.utils.data import Dataset
import torch

class AtomicEnvDataset(Dataset):

    def __init__(self, descriptors, energy) -> None:
        super().__init__()
        self.X = descriptors
        self.y = energy

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def train_val_test(y, train_size=100, val_size=20, test_size=10, **kwargs):
    ### Define train, val, test sample sizes
    training_size = train_size
    validation_size = val_size
    test_size = test_size
    strategy = kwargs.get('strategy', 'random')

    np.random.seed(1)
    shuffled_frames = [int(n) for n in range(len(y))]
    np.random.shuffle(shuffled_frames)

    training_pts = shuffled_frames[0:training_size]
    validation_pts = shuffled_frames[training_size:training_size+validation_size]
    test_pts = shuffled_frames[training_size+validation_size : training_size+validation_size+test_size]

    if strategy=="random":
        print("Random sampling...")
        return training_pts, validation_pts, test_pts
        
    elif strategy=="stratified":
        
        distorted_idxs = torch.argwhere(y>=2.5).tolist()
        distorted_idxs = list([arg[0] for arg in distorted_idxs])

        test_pts = test_pts[:test_size//10]
        test_pts += list(set(distorted_idxs) - set(test_pts))[:test_size-test_size//10]
        if len(test_pts) != test_size:
            raise ValueError(f"Size of test pts is {len(test_pts)} not equal to {test_size}")
        
        return training_pts, validation_pts, test_pts

   
    