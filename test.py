import numpy as np
from scipy.stats import multivariate_normal
import argparse, subprocess, os, sys, csv, random
import matplotlib.pyplot as plt
# 

# Example log values
# log_values = np.array([1, 2, 3, 4, 5])

# # Apply log-sum-exp trick for normalization
# max_log = np.max(log_values)
# normalized_values = np.exp(log_values - max_log) / np.sum(np.exp(log_values - max_log))

# print(normalized_values)

# likelihood = multivariate_normal(mean=np.array([1,2]), cov=np.eye(2), allow_singular=True).logpdf(np.array([2,1]))
# print(likelihood)
# import matplotlib.pyplot as plt

# import numpy as np
# import random
# import pyLasaDataset as lasa
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics.pairwise import euclidean_distances

# sub_sample = 4
# #[CShape, GShape, JShape, JShape_2, LShape, NShape, PShape, RShape, Sshape, WShape,  Zshape]
# # [DoubleBendedLine, BendedLine, Sine, Leaf_1, Leaf_2, Snake， Trapezoid, Worm, Multi_Models_1]
# data = lasa.DataSet.Sshape
# dt = data.dt
# demos = data.demos # list of 7 Demo objects, each corresponding to a 
# demo_0 = demos[0]
# pos = demo_0.pos[:, ::sub_sample] # np.ndarray, shape: (2,2000)
# vel = demo_0.vel[:, ::sub_sample] # np.ndarray, shape: (2,2000) 
# Data = np.vstack((pos, vel))
# for i in np.arange(1, len(demos)):
#     pos = demos[i].pos[:, ::sub_sample]
#     vel = demos[i].vel[:, ::sub_sample]
#     Data = np.hstack((Data, np.vstack((pos, vel))))


# # Input data (XY coordinates)
# ab = np.load('array.npy')
# data = Data[:, ab].T

# # Compute pairwise distances between data points
# distances = euclidean_distances(data)

# # Compute the similarity matrix using Gaussian kernel
# sigma = 1.0  # Adjust the sigma value as needed
# similarity = np.exp(-distances ** 2 / (2 * sigma ** 2))

# # Perform spectral clustering
# num_clusters = 2  # Adjust the number of clusters as needed
# spectral_model = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
# labels = spectral_model.fit_predict(similarity)

# colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
# "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]



# color_mapping = np.take(colors, labels)
# _, ax1 = plt.subplots()
# ax1.scatter(data[:, 0], data[:, 1], c=color_mapping)
# ax1.set_aspect('equal')
# plt.show()

# Display the clustering results
# for i, label in enumerate(labels):
    # print(f"Point ({data[i, 0]}, {data[i, 1]}) belongs to cluster {label}")


filepath =os.path.dirname(os.path.realpath(__file__))

Data = np.genfromtxt(filepath + '/data/ab.csv', dtype=float, delimiter=',')
args = ['time ' + filepath + '/spectral']
completed_process     = subprocess.run(' '.join(args), shell=True)

assignment_array = np.genfromtxt(filepath + '/data/ab_output.csv', dtype=int, delimiter=',')
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
"#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

color_mapping = np.take(colors, assignment_array)
print(Data)

_, ax1 = plt.subplots()
ax1.scatter(Data[:, 0], Data[:, 1], c=color_mapping)
ax1.set_aspect('equal')
plt.show()