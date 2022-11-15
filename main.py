from util.load_data import *
from util.process_data import *
import numpy as np
import argparse
import subprocess
import os
import csv

parser = argparse.ArgumentParser(
                    prog = 'DPMM',
                    description = 'run dpmm cluster',
                    epilog = '2022, Sunan Sun <sunan@seas.upenn.edu>')

parser.add_argument('-i', '--input', type=int, default=1, help='Choose Data Input Option')
parser.add_argument('-t', '--iteration', type=int, default=2, help='Number of Sampler Iterations')

args = parser.parse_args()

data_input_option = args.input
iteration         = args.iteration

if data_input_option == 1:
    draw_data()
    l, t, x, y = load_data()
    Data = add_directional_features(l, t, x, y, if_normalize=True)
else:
    print('Not a valid option')

n, m = Data.shape

input_path = './data/human_demonstrated_trajectories.csv'
output_path = './data'
with open(input_path, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(Data.shape[0]):
        data_writer.writerow(Data[i, :])


lambda_0 = {
    "nu_0": m + 3,
    "kappa_0": m + 3,
    "mu_0": np.zeros(m),
    "sigma_0": (m + 3) * (1 * np.pi) / 180 * np.eye(m)
}

params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]
# print(params)
# print(args)
# print(' '.join(args))

args = [os.path.abspath(os.getcwd()) + '/build/dpmm',
        '-n {}'.format(n),
        '-m {}'.format(m),        
        '-i {}'.format(input_path),
        '-o {}'.format(output_path),
        '-t {}'.format(iteration),
        '-p ' + ' '.join([str(p) for p in params])
]

completed_process = subprocess.run(' '.join(args), shell=True)

# print(' '.join(args))