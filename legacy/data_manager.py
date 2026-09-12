import torch, ase
import numpy as np
import os
from utils import train_val_test, fps
from descriptors import AtomicDescriptor
from torch.utils.data import DataLoader, TensorDataset


class DataManager:

    def __init__(self, rawdata_path="", descriptor_engine=None|AtomicDescriptor, device="cpu", **kwargs) -> None:
        self.rawdata_path = rawdata_path
        self.descriptor_engine = descriptor_engine

        self.train_size = kwargs.get('train_size', 400)
        self.val_size = kwargs.get('val_size', 80)
        self.test_size = kwargs.get('test_size', 400)
        self.kwargs = kwargs
        self.sample_strategy = self.kwargs.get("sample", "fps_ood")
        self.device = device

        self.X, self.y = None, None


    def _data_prep(self):
        if self.rawdata_path != "":
            db = np.load(self.rawdata_path)
            y_raw = torch.tensor(db['energies'], device="cpu")
        else:
            y_raw = self.y
            if self.X is None or self.y is None:
                raise ValueError("No filepath given and no data was assigned!")
       
        norm_y = (y_raw - y_raw.mean())/y_raw.std()

        if self.sample_strategy == "random":
            train_pts, val_pts, test_pts = train_val_test(len(norm_y), 
                                                  self.train_size, self.val_size, self.test_size, strategy=self.sample_strategy)
        
        elif self.sample_strategy == "fps_ood":
            desc_path = self.kwargs.get("desc_path", "")
            if desc_path == "":
                raise FileNotFoundError("Descriptor file not specified")

            X = torch.load(desc_path, weights_only=False)["saved_desc"]
            
            bulk_idxs = torch.where(torch.abs(norm_y)<=2)[0]
            idxs = list([idx for idx in bulk_idxs if idx < X.shape[0]])
            nonfps_idxs = np.random.choice(idxs, int(self.train_size*0.5), replace=False)
            remaining_train = list(set(idxs)-set(nonfps_idxs))
            fps_idxs = fps(X[remaining_train], sample_size=int(self.train_size*0.5))
            train_pts = torch.concat([torch.tensor(nonfps_idxs), fps_idxs])

            ood_idxs = torch.where(torch.abs(norm_y)>2)[0]
            remaining_bulk = list(set(bulk_idxs) - set(train_pts))
            _, val_pts, test_pts = train_val_test(len(norm_y),
                                                  self.train_size, self.val_size, self.test_size, strategy=self.sample_strategy,
                                                  bulk_indices=remaining_bulk, ood_indices=ood_idxs)
            
        
        if self.descriptor_engine.descriptor_type == "soap":
            symbols = self.kwargs.get("symbols", "C"*6+"H"*6)
            train_X = [ase.Atoms(symbols=symbols, positions=db['coords'][i,:,:]) for i in train_pts]
            train_y = y_raw[train_pts]
            val_X = [ase.Atoms(symbols=symbols, positions=db['coords'][i,:,:]) for i in val_pts]
            val_y = y_raw[val_pts]
            test_X = [ase.Atoms(symbols=symbols, positions=db['coords'][i,:,:]) for i in test_pts]
            test_y = y_raw[test_pts]
        
        elif self.descriptor_engine.descriptor_type == "bp":
            X = db['coords']
            train_X = db['coords'][train_pts, : ,:]
            train_y = y_raw[train_pts]
            val_X = db['coords'][val_pts, : ,:]
            val_y = y_raw[val_pts]
            test_X = db['coords'][test_pts, : ,:]
            test_y = y_raw[test_pts]

        self.descriptor_engine.train_X = train_X
        self.descriptor_engine.train_y = train_y
        self.descriptor_engine.val_X = val_X
        self.descriptor_engine.val_y = val_y
        self.descriptor_engine.test_X = test_X
        self.descriptor_engine.test_y = test_y


    def _validate_data(self):

        if self.data["train_X_norm"].shape[0] != self.train_size:
            raise ValueError("Train size does not match given input size!")
        if self.data["val_X_norm"].shape[0] != self.val_size:
            raise ValueError("Val size does not match given input size!")
        if self.data["test_X_norm"].shape[0] != self.test_size:
            raise ValueError("Test size does not match given input size!")


    def process_save(self, featuresdir_path):

        self._data_prep()
        ad = self.descriptor_engine
        train_X_norm, val_X_norm, test_X_norm = ad.get_features()
        train_y, val_y, test_y = ad.get_labels()
        self.data = {"train_X_norm": train_X_norm, "train_y": train_y, "val_X_norm": val_X_norm, "val_y": val_y, "test_X_norm":test_X_norm, "test_y": test_y}

        if self.descriptor_engine.descriptor_type == "soap":
            r_cut = self.descriptor_engine.r_cut
            sigma = self.descriptor_engine.sigma
            n_max = self.descriptor_engine.n_max
            l_max = self.descriptor_engine.l_max
            features_file = f"SavedDescriptors_{self.descriptor_engine.__name__}_{r_cut}-{sigma}-{n_max}-{l_max}.pt"

        elif self.descriptor_engine.descriptor_type == "bp":
            r_cut1, r_cut2, *_ = self.descriptor_engine.r_cut_ls + [None]
            sigma1, sigma2, *_ = self.descriptor_engine.sigma_ls + [None]
            n_max = self.descriptor_engine.n_basis
            features_file = f"SavedDescriptors_{self.descriptor_engine.__name__}_{r_cut1}-{r_cut2 if r_cut2 is not None else r_cut1}-{sigma1}-{sigma2 if sigma2 is not None else sigma1}-{n_max}.pt"
        
        featuresfile_path = os.path.join(featuresdir_path, features_file)
        print(f"Saving descriptors at {featuresfile_path}...")
        torch.save(self.data, featuresfile_path)

        return self.data


    def load_processed(self, featuresfile_path):

        print(f"Loading descriptors from {featuresfile_path}...")
        self.data = torch.load(featuresfile_path, map_location=self.device)
        self._validate_data()

        return self.data

    
    def get_dataloaders(self):
            
        traindataset = TensorDataset(self.data["train_X_norm"], self.data["train_y"])
        valdataset = TensorDataset(self.data["val_X_norm"], self.data["val_y"])
        testdataset = TensorDataset(self.data["test_X_norm"], self.data["test_y"])

        train_loader = DataLoader(traindataset, batch_size=len(self.data["train_X_norm"]), shuffle=False)
        val_loader = DataLoader(valdataset, batch_size=len(self.data["val_X_norm"]), shuffle=False)
        test_loader = DataLoader(testdataset, batch_size=len(self.data["test_X_norm"]), shuffle=False)
        print("Finished initializing dataloaders...\n")

        return train_loader, val_loader, test_loader
