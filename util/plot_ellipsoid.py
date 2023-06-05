import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


def plot_result_3D(Mu_s, Sigma_s, Y):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if Y is not None:
        ax.plot(Y[0], Y[1], Y[2], 'g*', markersize=3)

    # set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=len(Mu_s[0]))
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # compute each and plot each ellipsoid iteratively
    for indx in np.arange(len(Mu_s[0])):
        # your ellispsoid and center in matrix form
        A = Sigma_s[indx]
        center = Mu_s[:, indx].reshape(3)

        # find the rotation matrix and radii of the axes
        U, s, rotation = linalg.svd(A)
        radii = np.sqrt(s) * 1.5 # set the scale factor yourself

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(indx), linewidth=0.1, alpha=0.5, shade=True) # alpha = 1
        ax.text(Mu_s[0][indx], Mu_s[1][indx], Mu_s[2][indx], str(indx + 1), fontsize=20)
    plt.show()


if __name__ == '__main__':
    Mu_s = np.load('Mu_3D.npy')
    Sigma_s = np.load('Sigma_3D.npy')
    plot_result_3D(Mu_s, Sigma_s, None)
