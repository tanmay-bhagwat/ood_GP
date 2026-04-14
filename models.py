import torch


class BPGPModel(torch.nn.Module):

    def __init__(self, log_noise=1.0) -> None:
        super().__init__()
        self.log_noise = torch.nn.Parameter(torch.tensor(log_noise))
        self.register_buffer('alpha', torch.empty(0))
        self.register_buffer('L', torch.empty(0))
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.training:
            self.alpha = torch.empty((X.shape[0],1), device=X.device, dtype=X.dtype)
            self.L = torch.empty((X.shape[0], X.shape[0]), device=X.device, dtype=X.dtype)

    def set_kernel(self, kernel):
        self.kernel = kernel

    def mll(self, update_buffers=False):
        device = self.X.device
        noise = torch.exp(self.log_noise.to(device=device, dtype=torch.float64))
        K = self.kernel.full_kernel_block(self.X, self.X)
        K += noise * torch.eye(len(self.X), device=device, dtype=self.X.dtype)

        eps = 1e-6
        K += eps * torch.eye(K.size(-1), device=device, dtype=self.X.dtype)
        # eigvals = torch.linalg.eigvalsh(K_s)
        # print("Eigvalues: ", eigvals.min(), eigvals.max())

        Lt = torch.linalg.cholesky(K)
        self.y, Lt = self.y.double(), Lt.double()
        alphat = torch.cholesky_solve(self.y.unsqueeze(-1), Lt)

        # if self.training and update_buffers:
        self.L = Lt
        self.alpha = alphat

        mll = 0.5 * self.y @ alphat.squeeze()
        mll += torch.log(torch.diag(Lt)).sum()
        mll += 0.5 * len(self.X)*torch.log(torch.tensor(2*torch.pi, device=device, dtype=self.X.dtype))

        return mll


    def predict(self, X_test):
        K_s = self.kernel.full_kernel_block(X_test, self.X)
        K_ss = self.kernel.full_kernel_block(X_test, X_test)

        mean = K_s.double() @ self.alpha.double()
        v = torch.linalg.solve_triangular(self.L, K_s.T, upper=False)

        var = K_ss - v.T @ v

        return mean.squeeze(-1), torch.diag(var)
