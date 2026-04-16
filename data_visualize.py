import numpy as np
import matplotlib.pyplot as plt


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


def energy_analyse():
    db = np.load("rmd17_benzene.npz")
    y = (db['energies'] - db['energies'].mean())/db['energies'].std()
    print(y.shape)
    ct = len(list([i for i in range(len(y)) if np.abs(y[i])<=2]))
    print(ct)

    plt.scatter(np.arange(0,y.shape[0]), y[np.arange(0,y.shape[0])], alpha=0.7)
    plt.xlabel("Steps")
    plt.ylabel("Normalized energies")
    plt.axhline(2, c="k", linestyle="--")
    plt.axhline(-2, c="k", linestyle="--")
    plt.fill_between(x=np.arange(len(y)), y1=-2, y2=2, hatch="///", alpha=0.6, facecolor="none")
    plt.annotate("96k points", xy=(len(y)//2, 0), xytext=(len(y)//2-2000, 0), fontweight="bold", fontsize=12,
                bbox=dict(boxstyle="round, pad=0.3", fc="white", ec="black", lw=1))
    plt.savefig("full_y.pdf", format="pdf")
    plt.close()

    plt.scatter(np.arange(y.shape[0])[np.nonzero(y>2)], y[np.nonzero(y>2)])
    plt.xlabel("Steps")
    plt.ylabel("Normalized energies")
    plt.savefig("y_above2.pdf", format="pdf")
    plt.close()

    from utils import train_val_test
    from scipy.stats import norm
    def get_gaussian_pdf(data):
        mu, std = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        p = norm.pdf(x, mu, std)
        return x,p,mu,std


    train_pts, _, test_pts = train_val_test(len(y), train_size=400, val_size=80, test_size=400)
    train_color = '#1f77b4' # Muted blue
    test_color = '#ff7f0e'  # Muted orange
    plt.xlabel("Normalized energy bins")
    plt.ylabel("Count")
    plt.hist(y[train_pts], bins=20, density=True, color=train_color)
    plt.hist(y[test_pts], bins=20, alpha=0.5, density=True, color=test_color)

    x_train, p_train, mu_train, std_train = get_gaussian_pdf(y[train_pts])
    x_test, p_test, mu_test, std_test = get_gaussian_pdf(y[test_pts])

    plt.plot(x_train, p_train, c=train_color, label=rf'Train Fit ($\mu={mu_train:.1f}, \sigma={std_train:.1f}$)')
    plt.plot(x_test, p_test, c=test_color, label=rf'Test Fit ($\mu={mu_test:.1f}, \sigma={std_test:.1f}$)')
    plt.legend()
    plt.savefig("histplot.pdf", format="pdf")
    plt.show()