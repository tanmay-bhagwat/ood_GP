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
        self.log_sigvar = torch.nn.Parameter(torch.tensor(-1.5))
        self.learnable = learnable
        if learnable:
            self.embed = LearnableEmbedding(D) # Making this persistent to the kernel object
            num_features = self.embed.model[2].out_features
            self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.95))
        else:
            self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.5]*D))
        

    def atomic_kernel(self, d1, d2):
        """
        Covariance of structure d1 with structure d2
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
        sigvar = torch.exp(self.log_sigvar)
        if self.learnable:
            self.embed = self.embed.to(device=d1.device)
            d1 = self.embed(d1)
            d2 = self.embed(d2)

        dists = d1.unsqueeze(-1) - d2.unsqueeze(-2) # Vectorized op
        dists = torch.sum(dists**2/(1e-8+lengthscale**2), dim=-1)

        return sigvar**2 * torch.exp(-0.5 * dists)
    

    def full_kernel(self, X1, X2):

        N1 = X1.shape[0]
        N2 = X2.shape[0]

        K = torch.zeros((N1, N2), device=X1.device, dtype=X1.dtype)
        for i in range(N1):
            for j in range(N2):
                K[i,j] = self.atomic_kernel(X1[i,:], X2[j,:]).sum()
        
        return K
    

    def full_kernel_block(self, X1, X2, block_size=50):

        N1, N2 = len(X1), len(X2)
        K = torch.zeros((N1, N2), device=X1.device, dtype=X1.dtype)
        
        for i in range(0, N1, block_size):
            for j in range(0, N2, block_size):
                end_i = min(i + block_size, N1)
                end_j = min(j + block_size, N2)
                
                # Compute K block-wise, lower simultaneous computations
                # Get (block_size, block_size) matrices out
                K[i:end_i, j:end_j] = self.sub_kernel_fullvectorized(X1[i:end_i, :, :], X2[j:end_j, :, :])
                
        return K
    
    def sub_kernel_fullvectorized(self, d1, d2):

        # Embed to lower dim = 32 if training set >= 300
        if self.learnable:
            self.embed = self.embed.to(device=d1.device)
            d1 = self.embed(d1)
            d2 = self.embed(d2)

        lengthscale = torch.exp(self.log_lengthscale)
        sigvar = torch.exp(self.log_sigvar)
        dists = d1.unsqueeze(1).unsqueeze(3) - d2.unsqueeze(0).unsqueeze(2)
        return (sigvar**2) * (torch.exp(-0.5*torch.sum(dists**2/lengthscale**2, dim=-1))).sum(dim=(2,3))
    

    def sub_kernel_partialvectorized(self, d1, d2):


        if self.learnable:
            self.embed = self.embed.to(device=d1.device, dtype=d1.dtype)
            d1 = self.embed(d1)
            d2 = self.embed(d2)

        lengthscale = torch.exp(self.log_lengthscale)
        sigvar = torch.exp(self.log_sigvar)
        ls = []
        for i in range(d1.shape[0]):
            dist_i = d1[i].unsqueeze(0).unsqueeze(2) - d2.unsqueeze(1)
            ls.append(torch.sum(sigvar**2 * torch.exp(torch.sum(dist_i**2/(lengthscale**2 + 1e-8), dim=-1)), dim=(-1,-2)))
        
        return ls