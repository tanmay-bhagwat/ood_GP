import torch

n_basis = 4

class LearnableEmbedding(torch.nn.Module):

    def __init__(self, D):
        super().__init__()
        hidden_dim = 32
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=D, out_features=hidden_dim, dtype=torch.float64), 
                                         torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64))
    
    def forward(self, x):
        return self.model(x)
    

class BPKernel(torch.nn.Module):

    def __init__(self, D, learnable=False):
        super().__init__()
        self.log_sigvar = torch.nn.Parameter(torch.tensor(0.01))
        self.learnable = learnable
        if learnable:
            self.embed = LearnableEmbedding(D) # Making this persistent to the kernel object
            num_features = self.embed.model[2].out_features
            self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.0]*num_features))
        else:
            self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.0]*D))
        

    def atomic_kernel(self, d1, d2):
        """
        Parameters
        ---
        d1: torch.Tensor
            Feature tensor for structure i (N_at, D)
        d2: torch.Tensor
            Feature tensor for structure j (N_at, D)
        sigvar: float
            signal variance for the kernel
        lengthscale: float
            length scale for the kernel

        Returns a (N_at, N_at) covariance tensor of all features of structure i with structure j
        """
        
        lengthscale = torch.exp(self.log_lengthscale)
        if self.learnable:
            d1 = self.embed(d1, device=d1.device)
            d2 = self.embed(d2, device=d2.device)

        dists = d1.unsqueeze(1) - d2.unsqueeze(0)
        dists = self.log_sigvar**2 * torch.sum(dists**2/(1e-8+lengthscale**2), dim=-1)

        return torch.exp(-0.5 * dists)

    def full_kernel(self, X1, X2):
        """
        Structure kernel, just a sum of atomic kernels

        Parameters
        ---
        X1: torch.Tensor
            (N, N_at, D)
        X2: torch.Tensor
            (N, N_at, D)
        
        Returns a (N, N) covariance tensor between all structures in the dataset
        """

        K = torch.stack([torch.stack([self.atomic_kernel(s1,s2).sum() for s2 in X2]) for s1 in X1])
        K = K.to(device=X1.device, dtype=X1.dtype)

        return K
    