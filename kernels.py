import torch

n_basis = 4

class BPKernel(torch.nn.Module):

    def __init__(self, D):
        super().__init__()
        self.log_sigvar = torch.nn.Parameter(torch.tensor(0.01))
        self.log_lengthscale = torch.nn.Parameter(torch.tensor([0.07]*D))

    def atomic_kernel(self, d1, d2):
        """
        d1: torch.Tensor
            First feature tensor
        d2: torch.Tensor
            Second feature tensor
        sigvar: float
            signal variance for the kernel
        lengthscale: float
            length scale for the kernel
        """
        sigvar = torch.exp(self.log_sigvar)
        lengthscale = torch.exp(self.log_lengthscale)

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
                k_atoms = self.atomic_kernel(X1[i,:], X2[j, :])
                K[i, j] = k_atoms.sum()

        return K
    