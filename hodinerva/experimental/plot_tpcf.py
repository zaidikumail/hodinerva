from matplotlib import pyplot as plt
from pycorr import utils


def plot_cov(tpcf, savedir, savename, plt_show=True):
    corrcoef = utils.cov_to_corrcoef(tpcf.cov())
    fig = plt.figure(figsize=(4, 3.25))
    ax = plt.gca()
    c = ax.pcolor(tpcf.sep, tpcf.sep, corrcoef.T, cmap=plt.get_cmap("jet"))
    fig.colorbar(c, ax=ax)
    ax.set_title("Correlation matrix")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    fig.tight_layout()
    plt.savefig(savedir + "/" + savename + ".pdf")
    if plt_show:
        plt.show()
    plt.close()
