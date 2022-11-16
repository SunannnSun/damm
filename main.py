from util.load_data import *
from util.process_data import *
import numpy as np
import argparse
import subprocess
import os
import csv
import random


from scipy.special import logsumexp
a = np.array([-10, 2])
# print(np.log(np.sum(np.exp(a))))
# print(max(a) - logsumexp(a - max(a)))

parser = argparse.ArgumentParser(
                    prog = 'DPMM',
                    description = 'run dpmm cluster',
                    epilog = '2022, Sunan Sun <sunan@seas.upenn.edu>')

parser.add_argument('-i', '--input', type=int, default=3, help='Choose Data Input Option: 0 handrawn; 1 load handdrawn; 2 load matlab')
parser.add_argument('-d', '--data', type=int, default=1, help='Choose Matlab Dataset, default=1')
parser.add_argument('-t', '--iteration', type=int, default=50, help='Number of Sampler Iterations; default=50')
parser.add_argument('-a', '--alpha', type=float, default = 1, help='Concentration Factor; default=1')
parser.add_argument('--init', type=int, default = 10, help='number of initial clusters, 0 is one cluster per data; default=10')
args = parser.parse_args()

data_input_option = args.input
iteration         = args.iteration
alpha             = args.alpha
init_option       = args.init
dataset_no        = args.data


if data_input_option == 1:
    draw_data()
    l, t, x, y = load_data()
    Data = add_directional_features(l, t, x, y, if_normalize=True)
elif data_input_option == 2:
    data_name = 'human_demonstrated_trajectories_1.dat'
    l, t, x, y = load_data(data_name)
    Data = add_directional_features(l, t, x, y, if_normalize=True)
else:
    pkg_dir = './data/'
    chosen_data_set = dataset_no
    sub_sample = 1
    nb_trajectories = 7
    Data = load_matlab_data(pkg_dir, chosen_data_set, sub_sample, nb_trajectories)
    Data = normalize_velocity_vector(Data)
# Data = Data[:, 0:2]
num, dim = Data.shape

input_path = './data/human_demonstrated_trajectories.csv'
output_path = './data/output.csv'
with open(input_path, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(Data.shape[0]):
        data_writer.writerow(Data[i, :])


# lambda_0 = {
#     "nu_0": dim + 3,
#     "kappa_0": 1,
#     "mu_0": np.zeros(dim),
#     "sigma_0":  0.1 * np.eye(dim)
# }

sigma_0 = 0.1 * np.eye(int(dim/2+1))
sigma_0[-1, -1] = 0.001
lambda_0 = {
    "nu_0": (dim/2+1) + 3,
    "kappa_0": 1,
    "mu_0": np.zeros(int(dim/2+1)),
    "sigma_0":  sigma_0
}

params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]
# print(params)
# print(args)
# print(' '.join(args))

args = [os.path.abspath(os.getcwd()) + '/build/dpmm',
        '-n {}'.format(num),
        '-m {}'.format(dim),        
        '-i {}'.format(input_path),
        '-o {}'.format(output_path),
        '-t {}'.format(iteration),
        '-a {}'.format(alpha),
        '--init {}'.format(init_option), 
        '-p ' + ' '.join([str(p) for p in params])
]

completed_process = subprocess.run(' '.join(args), shell=True)


assignment_array = np.genfromtxt(output_path, dtype=int, delimiter=',')
# print(assignment_array.shape)


"""##### Plot Results ######"""
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
for i in range(Data.shape[0]):
    color = colors[assignment_array[i]]
    ax.scatter(Data[i, 0], Data[i, 1], c=color)
ax.set_aspect('equal')
plt.show()



# print(' '.join(args))