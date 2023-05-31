import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from util.modelRegression import *  
import random, os



def plotResults(Data, logZ, logNum, logLogLik, animation=False):
    num, dim = Data.shape 
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    color_mapping = np.take(colors, assignment_array)

    if dim == 4:
        fig, ax = plt.subplots()
        ax.scatter(Data[:, 0], Data[:, 1], c=color_mapping)
        ax.set_aspect('equal')
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=color_mapping, s=5)


    assignment_array = regress(Data, assignment_array)       
    color_mapping = np.take(colors, assignment_array)

    if dim == 4:
        fig, ax = plt.subplots()
        ax.scatter(Data[:, 0], Data[:, 1], c=color_mapping)
        ax.set_aspect('equal')
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=color_mapping, s=5)
    ax.set_title('Clustering Result: Dataset %i Base %i Init %i Iteration %i' %(dataset_no, base, init_opt, iteration))
    
    _, axes = plt.subplots(2, 1)
    axes[0].plot(np.arange(logNum.shape[0]), logNum, c='k')
    axes[0].set_title('Number of Components')
    axes[1].plot(np.arange(logLogLik.shape[0]), logLogLik, c='k')
    axes[1].set_title('Log Joint Likelihood')

    filepath = os.path.dirname(os.path.realpath(__file__))
    logZ             = np.genfromtxt(filepath + '/data/logZ.csv', dtype=int, delimiter=None)


    logZ = np.array([[0,0,0,1,1],
                    [2,2,2,2,2]])
    total_frames = logZ.shape[0]


    # Generate some random data
    np.random.seed(0)
    x = np.random.rand(5)
    y = np.random.rand(5)
    assignment_array = np.array([0,0,0,1,1])

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] 
    color_mapping = np.take(colors, assignment_array)


    # Create a scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c='r')



    def update(frame):
        print(frame)
        scatter.set_color(np.take(colors, logZ[frame,:]))

    counter = itertools.count()
    ani = animation.FuncAnimation(fig, update, frames= total_frames, interval=500, repeat=False)

    # Show the plot
    plt.show()
