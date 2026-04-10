import torch


def pairwise_distances(R:torch.Tensor):
    """
    Parameters
    ---
    all_R: ndarray
        Positions of all atoms in config (N,atoms,3)

    Returns
    ---
    distance_mat: ndarray
        Pairwise distances (N,c)
    """
    # First unsqueeze at dim=1 so that we get (N,1,atoms,3), so that we have N batch size of all positions within a molec (atoms,3)
    # Then unsqueeze at dim=2 so that we get (N,atoms,1,3) N batch size of individual position row vectors (1,3) ->
    # This should result in (N,atoms,atoms,3) after broadcasting
    # Then apply norm on the last dim
    # Select unique distances using upper triangular indices only
    # Flatten to c columns
    
    distance_mat = torch.linalg.norm(torch.unsqueeze(R, 1) - torch.unsqueeze(R, 2), dim=-1)
    idxs = torch.triu_indices(row=distance_mat.shape[0], col=distance_mat.shape[1], offset=1) # Triu_indices to select the unique distances
    
    return distance_mat[:, idxs[0], idxs[1]]



class BPDescriptor:

    def __init__(self, X) -> None:
        self.X = X


    def cutoff_fn(self, r, r_cut):
        return 0.5 * (torch.cos(torch.pi*r/r_cut)+1) * (r<r_cut)


    def bp_twobody(self, R:torch.Tensor, r_cut:float=5.0, n_basis:int=4, sigma:float=1.0):
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
        r = torch.linalg.norm(R.unsqueeze(0) - R.unsqueeze(1), dim=-1) # (N_at, N_at) tensor of interatomic distances
        fc = self.cutoff_fn(r, r_cut) # Compute cutoff values for all r
        centers = torch.linspace(0, r_cut, n_basis) # Centers of n_basis exp basis functions

        arr = []
        for c in centers:
            g = torch.exp(-(r-c)**2/sigma**2)
            arr.append(torch.sum(fc*g, dim=0)) # Each arr element is (N_at,)
        
        desc = torch.stack(arr, dim=1) # (N_at, c) tensor returned
        assert desc.shape[0] == R.shape[0] and desc.shape[1] == n_basis
        return desc
        

    def bp_threebody(self, R:torch.Tensor, r_cut:float=3.0, sigma:float=1.0):
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
        N = R.shape[0]
        desc = []
        la_ls = [-1,1]
        zeta_ls = [1,4,8,16]
        combinations = [(la, zeta) for la in la_ls for zeta in zeta_ls]

        for i in range(N):
            Ri = R - R[i,:]
            ri = torch.linalg.norm(Ri, dim=1) # (N_at,) tensor
            fc = self.cutoff_fn(ri, r_cut)
            arr = torch.zeros(N**2, len(la_ls) * len(zeta_ls))
            
            for j in range(N):
                Rj = R - R[j,:]
                for k in range(j+1, N):
                    if j==i or k==i:
                        continue
                    cos_th = Ri[j].dot(Ri[k])/(ri[j]*ri[k] + 1e-8)
                    Rk = R - R[k,:]
                    rjk = torch.linalg.norm(Rj[i,:]-Rk[i,:])
                    fcjk = self.cutoff_fn(rjk, r_cut)
                    
                    g = torch.exp((-(ri[j])**2-(ri[k])**2-rjk**2)/sigma**2)
                    for idx, (la, zeta) in enumerate(combinations):
                        arr[j*k, idx] = (g/2**(1-zeta)) * fc[j] * fc[k] * fcjk * (1+la*cos_th)**zeta # Final arr size is (N_at-1)*(N_at-2)
        
            desc.append(arr.sum(dim=0)) # desc final size (N_at, len(combinations)) tensor returned
        desc = torch.stack(desc, dim=0)
        assert desc.shape[0] == R.shape[0] and desc.shape[1] == len(combinations)
        return desc
            

    def bp_descriptor(self, R, r_cut_ls, sigma_ls, n_basis):

        if len(r_cut_ls) == 2:
            r_cut_twobody = r_cut_ls[0]
            r_cut_threebody = r_cut_ls[1]
            if r_cut_twobody < r_cut_threebody:
                raise ValueError("Two body cutoff radius must be larger than three-body cutoff")
        else:
            r_cut_twobody = r_cut_ls[0]
            r_cut_threebody = r_cut_ls[0]

        if len(sigma_ls) == 2:
            sigma_twobody = sigma_ls[0]
            sigma_threebody = sigma_ls[1]
        else:
            sigma_twobody = sigma_ls[0]
            sigma_threebody = sigma_ls[0]

        twobody = self.bp_twobody(R, r_cut_twobody, n_basis, sigma_twobody)
        threebody = self.bp_threebody(R, r_cut_threebody, sigma_threebody)

        return torch.cat([twobody, threebody], dim=-1) #(N_at, n_basis+len(combinations)) tensor final size
        # return twobody


    def bp_descriptor_batch(self, all_R, r_cut_ls, sigma_ls, n_basis):
        desc = []
        for R in all_R:
            d = self.bp_descriptor(R, r_cut_ls, sigma_ls, n_basis)
            desc.append(d)

        return torch.stack(desc) # To batch descriptors, will return (N_at, N_at, 2) tensor


def soap_descriptor():
    pass


