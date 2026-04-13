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
            self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.07]*self.embed.model[2].out_features))
        else:
            self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.07]*D))
        

    def atomic_kernel(self, d1, d2):
        """
        d1: torch.Tensor
            Feature tensor for atom i (N_at, D)
        d2: torch.Tensor
            Feature tensor for atom j (N_at, D)
        sigvar: float
            signal variance for the kernel
        lengthscale: float
            length scale for the kernel
        """
        sigvar = torch.exp(self.log_sigvar)
        lengthscale = torch.exp(self.log_lengthscale)

        if self.learnable:
            d1 = self.embed(d1)
            d2 = self.embed(d2)

        dists = d1.unsqueeze(1) - d2.unsqueeze(0)
        dists = sigvar**2 * torch.sum(dists**2/lengthscale**2, dim=-1)

        return torch.exp(-0.5 * dists)

    def full_kernel(self, X1, X2):
        """
        Structure kernel, just a sum of atomic kernels
        """
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        K = torch.zeros(N1, N2)

        for i in range(N1):
            for j in range(N2):
                k_atoms = self.atomic_kernel(X1[i,:], X2[j,:])
                K[i, j] = k_atoms.sum()

        return K
    