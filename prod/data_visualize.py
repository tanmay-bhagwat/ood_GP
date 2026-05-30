import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *

def plot_VarError(error, var, noise):
    plt.figure()
    plt.scatter(np.sqrt(var+noise), error, alpha=0.6)
    lims = [
        min(var.min(), error.min()),
        max(var.max(), error.max())
    ]

    plt.plot(lims, lims, 'k--')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"Uncertainty $\sqrt{\sigma_i^2 + \sigma_n^2}$")
    plt.ylabel("|Energy error|")
    plt.grid()
    plt.title("GP uncertainty vs error (log-log)")
    plt.savefig(f"imgs/GP_varvserror.pdf", format="pdf")
    plt.show()


def plot_PredActualE(mean, test_y, var):
    plt.figure()
    plt.scatter(test_y, mean, c=np.linspace(0.02, var.max(), len(test_y)), cmap="viridis", alpha=0.7)
    plt.colorbar(label="Predictive variance")

    lims = [
        min(test_y.min(), mean.min()),
        max(test_y.max(), mean.max())
    ]

    plt.plot(lims, lims, 'k--')
    plt.xlabel("True energy")
    plt.ylabel("Predicted energy")
    plt.title("True vs predicted energy (scaled)")
    plt.savefig(f"imgs/pred-true_energy.pdf", format="pdf")
    plt.show()


def plot_trainvsval(train_loss_ls, val_loss_ls):
    plt.figure()
    # plt.plot(np.arange(0 ,len(train_loss_ls)), train_loss_ls)
    plt.plot(np.arange(0 ,len(val_loss_ls)), val_loss_ls)

    plt.xlabel("iterations")
    plt.ylabel("loss (mll)")
    plt.savefig(f"imgs/train-val.pdf", format="pdf")
    plt.show()


def energy_analyse(db_path, desc_path, strategy):
    db = np.load("rmd17_benzene.npz")
    norm_y = (db['energies'] - db['energies'].mean())/db['energies'].std()
    print(norm_y.shape)
    ct = len(list([i for i in range(len(norm_y)) if np.abs(norm_y[i])<=2]))
    print(ct)

    plt.scatter(np.arange(0,norm_y.shape[0]), norm_y[np.arange(0,norm_y.shape[0])], alpha=0.7)
    plt.xlabel("Steps")
    plt.ylabel("Normalized energies")
    plt.axhline(2, c="k", linestyle="--")
    plt.axhline(-2, c="k", linestyle="--")
    plt.fill_between(x=np.arange(len(norm_y)), y1=-2, y2=2, hatch="///", alpha=0.6, facecolor="none")
    plt.annotate("96k points", xy=(len(norm_y)//2, 0), xytext=(len(norm_y)//2-2000, 0), fontweight="bold", fontsize=12,
                bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="black", lw=1))
    # plt.savefig("full_y.pdf", format="pdf")
    plt.close()

    plt.scatter(np.arange(norm_y.shape[0])[np.nonzero(norm_y>2)], norm_y[np.nonzero(norm_y>2)])
    plt.xlabel("Steps")
    plt.ylabel("Normalized energies")
    # plt.savefig("y_above2.pdf", format="pdf")
    plt.close()

    from prod.utils import train_val_test
    from scipy.stats import norm
    def get_gaussian_pdf(data):
        mu, std = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        p = norm.pdf(x, mu, std)
        return x,p,mu,std
    
    print("In energy analyse")

    if strategy == "random":
        train_pts, _, test_pts = train_val_test(len(norm_y), train_size=200, val_size=80, test_size=400)
    
    if strategy == "stratified":
        norm_y = torch.tensor(norm_y)
        X = torch.load(desc_path, weights_only=False)["saved_desc"]
        saved_idxs = torch.load(desc_path, weights_only=False)["idxs"]
        bulk_idxs = torch.where(torch.abs(norm_y)<=2)[0]
        idxs = list([idx for idx in bulk_idxs if idx < X.shape[0]])
        train_pts = fps(X[idxs], sample_size=200)

        ood_idxs =  torch.where(torch.abs(norm_y)>2)[0]
        remaining_bulk = list(set(bulk_idxs) - set(train_pts))
        remaining_bulk = [idx.item() for idx in remaining_bulk]
        ood_idxs = [idx.item() for idx in ood_idxs]

        _, val_pts, test_pts = train_val_test(len(norm_y), train_size=len(train_pts), val_size=80, test_size=400, strategy=strategy,
                                            bulk_indices=remaining_bulk, ood_indices=ood_idxs)
    train_color = '#1f77b4' # Muted blue
    test_color = '#ff7f0e'  # Muted orange
    plt.xlabel("Normalized energy bins")
    plt.ylabel("Count")
    plt.hist(norm_y[train_pts], bins=20, density=True, color=train_color)
    # plt.hist(norm_y[test_pts], bins=20, alpha=0.5, density=True, color=test_color)

    x_train, p_train, mu_train, std_train = get_gaussian_pdf(norm_y[train_pts])
    x_test, p_test, mu_test, std_test = get_gaussian_pdf(norm_y[test_pts])

    plt.plot(x_train, p_train, c=train_color, label=rf'Train Fit ($\mu={mu_train:.1f}, \sigma={std_train:.1f}$)')
    # plt.plot(x_test, p_test, c=test_color, label=rf'Test Fit ($\mu={mu_test:.1f}, \sigma={std_test:.1f}$)')
    plt.legend()
    plt.savefig("histplot_train.pdf", format="pdf")
    plt.show()

    features_norm = torch.linalg.norm(X[train_pts,:,:], dim=(1,2))
    plt.scatter(np.arange(X[train_pts].shape[0]), features_norm)
    plt.savefig("X_traindims_scatter.pdf", format="pdf")
    plt.show()


def min_testtrain_dist(db_path, desc_path, symbols, n_max, l_max, r_cut, sigma, strategy):
    desc_path = desc_path
    filepath = db_path
    symbols = symbols
    species_ls = list(set(symbols))

    X = torch.load(desc_path, weights_only=False)["saved_desc"]
    db = np.load(filepath)
    y_raw = torch.tensor(db['energies'], device="cpu", dtype=torch.float64)
    print(y_raw.mean(), y_raw.std())
    norm_y = (y_raw - y_raw.mean())/y_raw.std()

    train_size = 400
    val_size = int(train_size*0.2)
    test_size = train_size 
    bulk_idxs = torch.where(torch.abs(norm_y)<=2)[0]
    idxs = list([idx for idx in bulk_idxs if idx < X.shape[0]])
    nonfps_idxs = np.random.choice(idxs, int(train_size*0.5), replace=False)
    remaining_train = list([idx for idx in idxs if idx not in nonfps_idxs])
    fps_idxs = fps(X[remaining_train], sample_size=int(train_size*0.5))
    train_pts = torch.concat([torch.tensor(nonfps_idxs), fps_idxs])

    ood_idxs =  torch.where(torch.abs(norm_y)>=2)[0]
    remaining_bulk = list(set(bulk_idxs) - set(train_pts))
    _, __, test_pts = train_val_test(len(norm_y), train_size, val_size, test_size, strategy=strategy, 
                                        bulk_indices=remaining_bulk, ood_indices=ood_idxs)

    train_X = [ase.Atoms(symbols=symbols, positions=db['coords'][i,:,:]) for i in train_pts]
    test_X = [ase.Atoms(symbols=symbols, positions=db['coords'][i,:,:]) for i in test_pts]

    soap = dscribe.descriptors.SOAP(species=species_ls, n_max=n_max, l_max=l_max, 
                                                r_cut=r_cut, sigma=sigma, periodic=False)
    train_X = torch.tensor(soap.create(train_X))
    test_X = torch.tensor(soap.create(test_X))

    train_mean = train_X.mean(dim=(0,1), keepdim=True)
    train_std  = train_X.std(dim=(0,1), keepdim=True)

    train_X = (train_X - train_mean)/(train_std + 1e-8)
    test_X = (test_X - train_mean)/(train_std + 1e-8)

    train_X_global = train_X.mean(dim=1)
    test_X_global = test_X.mean(dim=1)
    dist_matrix = torch.cdist(test_X_global, train_X_global, p=2)
    ls = torch.min(dist_matrix, dim=1).values

    from scipy.stats import norm
    def get_gaussian_pdf(data):
        mu, std = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        p = norm.pdf(x, mu, std)
        return x,p,mu,std

    plt.hist(ls, bins=20, density=True)
    x,p,mu,std = get_gaussian_pdf(ls)
    print(mu, std)
    plt.plot(x, p, label=rf'Train Fit ($\mu={mu:.2f}, \sigma={std:.2f}$)')
    plt.xlabel("Min test-train distance")
    plt.legend()
    plt.savefig("data_autocorrelation.pdf", format="pdf")
    plt.show()


# energy_analyse(db_path="rmd17_benzene.npz", desc_path="SoapDesc_6000.pt", strategy="stratified")
# min_testtrain_dist(db_path="rmd17_benzene.npz", desc_path="SoapDesc_6000.pt", strategy="stratified",
                #    symbols="C"*6+"H"*6, n_max=12, l_max=8, r_cut=6, sigma=0.5)