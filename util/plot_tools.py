import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from gmr import GMM, plot_error_ellipses
from . import data_tools



font = {'family' : 'Times New Roman',
         'size'   : 10,
         'serif':  'cmr10'
         }
mpl.rc('font', **font)
mpl.rc('text', usetex = True)


""" colors array is defined globally and used in every function """
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
"#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]


def plot_results(data, assignment_array):
    color_mapping = np.take(colors, assignment_array)

    param_dict = data_tools.extract_param(data, assignment_array)
    Priors = param_dict["Priors"]
    Mu     = param_dict["Mu"]
    Sigma  = param_dict["Sigma"]
    
    K = assignment_array.max()+1
    M = int(data.shape[1]/2)

    if M == 2:
        _, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c=color_mapping)
        gmm = GMM(K, Priors, Mu, Sigma)
        plot_error_ellipses(ax, gmm, alpha=0.3, colors=colors[0:K], factors=np.array([2.2 ]))

        for k in range(K):    
            plt.text(Mu[k, 0], Mu[k, 1], str(k+1), fontsize=20)

        ax.set_aspect('equal')
        ax.set_xlabel(r'$\xi_1$', fontsize=16)
        ax.set_ylabel(r'$\xi_2$', fontsize=16)
    
    
    elif M == 3:    
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color_mapping, s=5)

        for k in range(K):
            _, s, rotation = np.linalg.svd(Sigma[k, :, :])  # find the rotation matrix and radii of the axes
            radii = np.sqrt(s) * 2.2                        # set the scale factor yourself
            u = np.linspace(0.0, 2.0 * np.pi, 60)
            v = np.linspace(0.0, np.pi, 60)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))   # calculate cartesian coordinates for the ellipsoid surface
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + Mu[k, :]
            ax.plot_surface(x, y, z, rstride=3, cstride=3, color=colors[k], linewidth=0.1, alpha=0.3, shade=True) 
            ax.text(Mu[k, 0], Mu[k, 1], Mu[k, 2], str(k + 1), fontsize=20)

        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.set_zlabel(r'$\xi_3$')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.set_title(r'Directionality-aware Mixture Model Clustering Result')



def animate_results(data, assignment_logs):

    M = data.shape[1]/2

    def update(frame):          
        scatter.set_color(np.take(colors, assignment_logs[frame,:]))
        ax.set_title(f'Frame: {frame}')
        
    if M == 2:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        scatter = ax.scatter(data[:, 0], data[:, 1], c='k')
        ani = animation.FuncAnimation(fig, update, frames=assignment_logs.shape[0], interval=80, repeat=False)

    elif M == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='k', s=5)
        ani = animation.FuncAnimation(fig, update, frames=assignment_logs.shape[0], interval=80, repeat=False)
    
    plt.show()



def plot_logs(logNum, logLogLik):
    _, axes = plt.subplots(2, 1)
    axes[0].plot(np.arange(logNum.shape[0]), logNum, c='k')
    axes[0].set_title('Number of Components')
    axes[1].plot(np.arange(logLogLik.shape[0]), logLogLik, c='k')
    axes[1].set_title('Log Joint Likelihood')   








