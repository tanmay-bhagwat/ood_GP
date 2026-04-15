import dscribe.descriptors
import torch, ase
import numpy as np
from utils import *

### Example on using unsqueeze() to get pairwise distances
# def pairwise_distances(R:torch.Tensor):
#     """
#     Parameters
#     ---
#     all_R: ndarray
#         Positions of all atoms in config (N,atoms,3)

#     Returns
#     ---
#     distance_mat: ndarray
#         Pairwise distances (N,c)
#     """
#     # First unsqueeze at dim=1 so that we get (N,1,atoms,3), so that we have N batch size of all positions within a molec (atoms,3)
#     # Then unsqueeze at dim=2 so that we get (N,atoms,1,3) N batch size of individual position row vectors (1,3) ->
#     # This should result in (N,atoms,atoms,3) after broadcasting
#     # Then apply norm on the last dim
#     # Select unique distances using upper triangular indices only
#     # Flatten to c columns
    
#     distance_mat = torch.linalg.norm(torch.unsqueeze(R, 1) - torch.unsqueeze(R, 2), dim=-1)
#     idxs = torch.triu_indices(row=distance_mat.shape[0], col=distance_mat.shape[1], offset=1) # Triu_indices to select the unique distances
    
#     return distance_mat[:, idxs[0], idxs[1]]


class AtomicDescriptor:

    def __init__(self, filepath="", X=None, y=None, descriptor_type='soap', **kwargs):
        self.X = X
        self.y = y
        self.descriptor_type = descriptor_type.lower()
        self.params = kwargs
        self.device = kwargs.get('device', "cpu")
        self.dtype = kwargs.get('dtype', torch.float64)

        self.train_size = kwargs.get('train_size', 400)
        self.val_size = kwargs.get('train_size', 80)
        self.test_size = kwargs.get('train_size', 400)

        db = np.load(filepath)
        y = torch.tensor(db['energies'], device=self.device, dtype=self.dtype)
        train_pts, val_pts, test_pts = train_val_test(len(y), self.train_size, self.val_size, self.test_size)

        if self.descriptor_type == 'soap':
            
            self.train_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in train_pts]
            self.train_y = y[train_pts]
            self.val_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in val_pts]
            self.val_y = y[val_pts]
            self.test_X = [ase.Atoms(symbols="C"*6+"H"*6, positions=db['coords'][i,:,:]) for i in test_pts]
            self.test_y = y[test_pts]
            self.r_cut = kwargs.get('r_cut', 6.0)
            self.sigma = kwargs.get('sigma', 1.0)
            self.species_ls = kwargs.get('species_ls')
            self.n_max = kwargs.get('n_max', 12)
            self.l_max = kwargs.get('l_max', 8)

        elif self.descriptor_type == "bp":
            X = db['coords']
            self.train_X = db['coords'][train_pts, : ,:]
            self.train_y = y[train_pts]
            self.val_X = db['coords'][val_pts, : ,:]
            self.val_y = y[val_pts]
            self.test_X = db['coords'][test_pts, : ,:]
            self.test_y = y[test_pts]

        ### Default BP descriptor hyperparams if not given
        self.r_cut_ls = kwargs.get('r_cut_ls', [6.0, 3.5])
        self.sigma_ls = kwargs.get('sigma_ls', [1.0])
        self.n_basis = kwargs.get('n_basis', 4)


    def cutoff_fn(self, r, r_cut):
        return 0.5 * (torch.cos(torch.pi*r/r_cut)+1) * (r<r_cut)


    def _bp_twobody(self, R, r_cut:float=5.0, sigma:float=1.0):
        """
        Returns radial symmetry descriptors for all atoms

        Parameters
        ---
        R: torch.Tensor
            position array (N_at, 3)
        r_cut: float
            cutoff radius
        n_basis: int
            number of exp basis functions to use in expansion
        sigma: float
            std dev of exp basis functions
        
        Returns
        ---
        desc: torch.Tensor
            descriptor array (N_at, n_basis), where row i is all the radial descriptors for atom i
        """

        # Behler-Parrinello G2 symm function
        device = R.device
        r = torch.linalg.norm(R.unsqueeze(0) - R.unsqueeze(1), dim=-1) # (N_at, N_at) tensor of interatomic distances
        fc = self.cutoff_fn(r, r_cut) # Compute cutoff values for all r
        centers = torch.linspace(0, r_cut, self.n_basis, device=device, dtype=R.dtype) # Centers of n_basis exp basis functions

        arr = []
        for c in centers:
            g = torch.exp(-(r-c)**2/sigma**2)
            arr.append(torch.sum(fc*g, dim=0)) # Each arr element is (N_at,)
        
        desc = torch.stack(arr, dim=1) # (N_at, c) tensor returned
        assert desc.shape[0] == R.shape[0] and desc.shape[1] == self.n_basis
        return desc
        

    def _bp_threebody(self, R, r_cut:float=3.0, sigma:float=1.0):
        """
        Returns three-body angular descriptors for all atoms

        Parameters
        ---
        R: torch.Tensor
            position array (N_at, 3)
        r_cut: float
            cutoff radius
        sigma: float
            std dev of exp basis functions

        Returns
        ---
        desc: torch.Tensor
            descriptor array (N_at, len(features)), where row i is all the angular descriptors for atom i
        """

        # Behler-Parrinello G5 symm function
        device = R.device
        N = R.shape[0]
        desc = []
        la_ls = [-1,1]
        zeta_ls = [1,4,8,16]
        combinations = [(la, zeta) for la in la_ls for zeta in zeta_ls]

        for i in range(N):
            Ri = R - R[i,:]
            ri = torch.linalg.norm(Ri, dim=1) # (N_at,) tensor
            fc = self.cutoff_fn(ri, r_cut)
            arr = torch.zeros(N**2, len(la_ls) * len(zeta_ls), dtype=R.dtype, device=device)
            
            for j in range(N):
                Rj = R - R[j,:]
                for k in range(j+1, N):
                    if j==i or k==i:
                        continue

                    cos_th = Ri[j].dot(Ri[k])/(ri[j]*ri[k] + 1e-8)
                    
                    Rk = R - R[k,:]
                    rjk = torch.linalg.norm(Rj[i] - Rk[i])
                    fcjk = self.cutoff_fn(rjk, r_cut)
                    
                    g = torch.exp((-(ri[j])**2-(ri[k])**2-rjk**2)/sigma**2)
                    for idx, (la, zeta) in enumerate(combinations):
                        arr[j*k, idx] = (g/2**(1-zeta)) * fc[j] * fc[k] * fcjk * (1+la*cos_th)**zeta # Final arr size is (N_at-1)*(N_at-2)
        
            desc.append(arr.sum(dim=0)) # desc final size (N_at, len(combinations)) tensor returned
        desc = torch.stack(desc, dim=0)
        assert desc.shape[0] == R.shape[0] and desc.shape[1] == len(combinations)
        return desc
        

    def _bp_descriptor(self, R):

        if len(self.r_cut_ls) == 2:
            r_cut_twobody = self.r_cut_ls[0]
            r_cut_threebody = self.r_cut_ls[1]
            if r_cut_twobody < r_cut_threebody:
                raise ValueError("Two body cutoff radius must be larger than three-body cutoff")
        else:
            r_cut_twobody = self.r_cut_ls[0]
            r_cut_threebody = self.r_cut_ls[0]

        if len(self.sigma_ls) == 2:
            sigma_twobody = self.sigma_ls[0]
            sigma_threebody = self.sigma_ls[1]
        else:
            sigma_twobody = self.sigma_ls[0]
            sigma_threebody = self.sigma_ls[0]

        twobody = self._bp_twobody(R, r_cut_twobody, sigma_twobody)
        threebody = self._bp_threebody(R, r_cut_threebody, sigma_threebody)

        return torch.cat([twobody, threebody], dim=-1) #(N_at, n_basis+len(combinations)) tensor final size


    def _get_soap_descriptors(self, X, species_ls, r_cut=6.0, sigma=1.0, n_max=12, l_max=8, device="cpu"):

            soap = dscribe.descriptors.SOAP(species=species_ls, n_max=n_max, l_max=l_max, 
                                            r_cut=r_cut, sigma=sigma, periodic=False)
            desc = torch.tensor(soap.create(X))

            return desc


    def get_features(self, normalize=True):

        if self.descriptor_type == 'soap':
            train_X = self._get_soap_descriptors(self.train_X, self.species_ls, n_max=self.n_max, l_max=self.l_max,
                                                    r_cut=self.r_cut, sigma=self.sigma)
            val_X = self._get_soap_descriptors(self.val_X, self.species_ls, n_max=self.n_max, l_max=self.l_max,
                                                    r_cut=self.r_cut, sigma=self.sigma)
            test_X = self._get_soap_descriptors(self.test_X, self.species_ls, n_max=self.n_max, l_max=self.l_max,
                                                    r_cut=self.r_cut, sigma=self.sigma)

        elif self.descriptor_type == 'bp':
            desc1 = []
            for R in self.train_X:
                d = self._bp_descriptor(R)
                desc1.append(d)
            train_X = torch.stack(desc1) # To batch descriptors, will return (N_at, N_at, 2) tensor

            desc2 = []
            for R in self.val_X:
                d = self._bp_descriptor(R)
                desc2.append(d)
            val_X = torch.stack(desc2) # To batch descriptors, will return (N_at, N_at, 2) tensor

            desc3 = []
            for R in self.test_X:
                d = self._bp_descriptor(R)
                desc3.append(d)
            test_X = torch.stack(desc3) # To batch descriptors, will return (N_at, N_at, 2) tensor

        
        if normalize:
            train_mean = train_X.mean(dim=(0,1), keepdim=True)
            train_std  = train_X.std(dim=(0,1), keepdim=True)

            train_X = (train_X - train_mean)/(train_std + 1e-8)
            val_X = (val_X - train_mean)/(train_std + 1e-8)
            test_X = (test_X - train_mean)/(train_std + 1e-8)
            # print(train_X_norm.abs().max())

        return train_X, val_X, test_X
        

    def get_labels(self, normalize=True):

        if normalize:
            train_mean = self.train_y.mean()
            train_std = self.train_y.std()

            train_y = (self.train_y - train_mean)/train_std
            val_y = (self.val_y - train_mean)/train_std
            test_y = (self.test_y - train_mean)/train_std

        return train_y, val_y, test_y