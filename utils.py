import numpy as np
from torch.utils.data import Dataset
import torch, ase
import dscribe.descriptors
from tqdm import tqdm

class AtomicEnvDataset(Dataset):

    def __init__(self, descriptors, energy) -> None:
        super().__init__()
        self.X = descriptors
        self.y = energy

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def train_val_test(db_size, train_size=100, val_size=20, test_size=10, strategy="random", **kwargs):
    
    ### Define train, val, test sample sizes
    training_size = train_size
    validation_size = val_size
    test_size = test_size
    print(test_size)
    strategy = strategy
    bulk_idxs = kwargs.get('bulk_indices', [])
    ood_idxs = kwargs.get('ood_indices', [])

    np.random.seed(1)
    shuffled_frames = [int(n) for n in range(db_size)]
    np.random.shuffle(shuffled_frames)

    if strategy=="random":
        print("Random sampling...")
        training_pts = shuffled_frames[0:training_size]
        validation_pts = shuffled_frames[training_size:training_size+validation_size]
        test_pts = shuffled_frames[training_size+validation_size : training_size+validation_size+test_size]
        return training_pts, validation_pts, test_pts
    
        
    elif strategy=="stratified":

        if len(bulk_idxs) == 0 or len(ood_idxs) == 0:
            print(f"Bulk list len: {len(bulk_idxs)}, ood_idxs len: {len(ood_idxs)}")
            raise ValueError("Stratified sampling needs both bulk_idxs and ood_idxs")
                              
        n_test_bulk = int(test_size*0.05)
        n_test_ood = test_size - n_test_bulk

        test_bulk = np.random.choice(bulk_idxs, n_test_bulk, replace=False)
        test_ood = np.random.choice(ood_idxs, n_test_ood, replace=False)
        test_pts = np.concat([test_bulk, test_ood])
        test_pts = list([idx.item() for idx in test_pts])
        if len(test_pts) != test_size:
            raise ValueError(f"Size of test pts is {len(test_pts)} not equal to {test_size}")
        
        remaining_bulk = list(set(bulk_idxs) - set(test_bulk))
        remaining_ood = list(set(ood_idxs) - set(test_ood))

        n_val_bulk = int(validation_size*0.7)
        n_val_ood = validation_size - n_val_bulk
        val_bulk = np.random.choice(remaining_bulk, n_val_bulk, replace=False)
        val_ood = np.random.choice(remaining_ood, n_val_ood, replace=False)
        validation_pts = np.concat([val_bulk, val_ood])
        validation_pts = list([idx.item() for idx in validation_pts])
        if len(validation_pts) != validation_size:
            raise ValueError(f"Size of validation pts is {len(validation_pts)} not equal to {validation_size}")
        
        return None, validation_pts, test_pts

   
def fps(descriptors:torch.Tensor, sample_size):

    orig_device = descriptors.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster = []
    print(f"Doing furthest-point-search, building dataset of size {sample_size}...")
    descr = descriptors.to(device=device)
    desc2d = descr.mean(dim=1)

    # np.random.seed(1)
    # next_idx = np.random.randint(0, descr.shape[0])
    # min_dist = torch.full((descr.shape[0],), float('inf'), device=descr.device)
    
    # for _ in tqdm(range(sample_size)):

    #     cluster.append(next_idx)
    #     next_pt = desc2d[next_idx]
        
    #     dist_to_new = torch.sum((desc2d - next_pt)**2, dim=1)
    #     min_dist = torch.minimum(min_dist, dist_to_new) # defines the boundary of the cluster
        
    #     next_idx = torch.argmax(min_dist)

    np.random.seed(1)
    next_idx = np.random.randint(0,descr.shape[0])
    cluster.append(next_idx)
    next_pt = desc2d[next_idx]

    dist_to_new = torch.sum((desc2d-next_pt)**2, dim=1).reshape(-1,1)
    min_dist = torch.full((desc2d.shape[0],1), float('inf'), device=descr.device)

    for _ in range(sample_size-1):

        min_dist = torch.hstack([dist_to_new, min_dist])
    
        next_idx = torch.argmax(torch.amin(min_dist, dim=1))
        cluster.append(next_idx)
        next_pt = desc2d[next_idx]
        dist_to_new = torch.sum((desc2d-next_pt)**2, dim=1).reshape(-1,1)
    
    return torch.tensor(cluster).to(device=orig_device)


def block_descriptors_calc(filepath, species_ls, r_cut=6.0, sigma=0.5, n_max=12, l_max=8, subset=25000):

    db = np.load(filepath)
    y = db['energies']
    y = (y-y.mean())/y.std()
    bulk_mask = np.random.choice(np.argwhere(np.abs(y)<=2).reshape(-1), subset//2, replace=False)
    outlier_mask = np.random.choice(np.argwhere(np.abs(y)>2).reshape(-1), subset//2, replace=False)
    desc = []
    dct = {}
    
    for i in tqdm(bulk_mask):
        X = ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:])
        soap = dscribe.descriptors.SOAP(species=species_ls, n_max=n_max, l_max=l_max, r_cut=r_cut, sigma=sigma, periodic=False)
        desc.append(torch.tensor(soap.create(X, n_jobs=16)))
    
    dct["idxs"] = bulk_mask

    for i in tqdm(outlier_mask):
        X = ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:])
        soap = dscribe.descriptors.SOAP(species=species_ls, n_max=n_max, l_max=l_max, r_cut=r_cut, sigma=sigma, periodic=False)
        desc.append(torch.tensor(soap.create(X, n_jobs=16)))
    
    dct["idxs"] += outlier_mask
    desc = torch.stack(desc)
    dct["saved_desc"] = desc
       
    torch.save(dct, f"SoapDesc_{len(desc)}.pt")

# block_descriptors_calc("rmd17_benzene.npz", ["C","H"], 6.0, 0.5, 12, 8, 6000)