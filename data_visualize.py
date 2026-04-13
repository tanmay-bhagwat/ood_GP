import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_VarError(error, var):
    plt.figure()
    plt.scatter(var, error, alpha=0.6)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Predictive variance")
    plt.ylabel("|Energy error|")
    plt.grid()
    plt.title("GP uncertainty vs error (log-log)")
    plt.show()


def plot_PredActualE(mean, test_y, var):
    plt.figure()
    plt.scatter(test_y, mean, c=var, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Predictive variance")

    lims = [
        min(test_y.min(), mean.min()),
        max(test_y.max(), mean.max())
    ]

    plt.plot(lims, lims, 'k--')
    plt.xlabel("True energy")
    plt.ylabel("Predicted energy")
    plt.title("True vs predicted energy (scaled)")
    plt.show()


def plot_trainvsval(train_loss_ls, val_loss_ls):
    plt.figure()
    plt.plot(np.arange(0 ,len(train_loss_ls)), train_loss_ls)
    plt.plot(np.arange(0 ,len(val_loss_ls)), val_loss_ls)

    plt.xlabel("iterations")
    plt.ylabel("loss (mll)")
    plt.show()