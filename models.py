import torch


class BPGPModel(torch.nn.Module):

    def __init__(self, log_noise=0.75) -> None:
        super().__init__()
        self.log_noise = torch.nn.Parameter(torch.tensor(log_noise))
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def set_kernel(self, kernel):
        self.kernel = kernel

    def mll(self):
        device = self.X.device
        noise = torch.exp(self.log_noise.to(device=device, dtype=torch.float64))
        K = self.kernel.full_kernel_block(self.X, self.X)
        K += noise * torch.eye(len(self.X), device=device, dtype=self.X.dtype)

        eps = 1e-6
        K += eps * torch.eye(K.size(-1), device=device, dtype=self.X.dtype)
        # eigvals = torch.linalg.eigvalsh(K_s)
        # print("Eigvalues: ", eigvals.min(), eigvals.max())

        self.L = torch.linalg.cholesky(K)
        self.y, self.L = self.y.double(), self.L.double()
        self.alpha = torch.cholesky_solve(self.y.unsqueeze(-1), self.L)


        mll = 0.5 * self.y @ self.alpha.squeeze()
        mll += torch.log(torch.diag(self.L)).sum()
        mll += 0.5 * len(self.X)*torch.log(torch.tensor(2*torch.pi, device=device, dtype=self.X.dtype))

        return mll


    def predict(self, X_test):
        K_s = self.kernel.full_kernel_block(X_test, self.X)
        K_ss = self.kernel.full_kernel_block(X_test, X_test)

        if not self.training:
           _ = self.mll()
        mean = K_s.double() @ self.alpha.double()
        v = torch.linalg.solve_triangular(self.L, K_s.T, upper=False)

        var = K_ss - v.T @ v

        return mean.squeeze(-1), torch.diag(var)
