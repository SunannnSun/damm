from util.load_data import *
from util.process_data import *
from util.modelRegression import regress
import argparse, subprocess, os, csv, random


parser = argparse.ArgumentParser(
                    prog = 'Parallel Implemention of Dirichlet Process Mixture Model',
                    description = 'parallel implementation',
                    epilog = '2022, Sunan Sun <sunan@seas.upenn.edu>')


parser.add_argument('-i', '--input', type=int, default=2, help='Choose Data Input Option: 0 handrawn; 1 load handdrawn; 2 load matlab')
parser.add_argument('-d', '--data', type=int, default=1, help='Choose Matlab Dataset, default=1')
parser.add_argument('-t', '--iteration', type=int, default=40, help='Number of Sampler Iterations; default=50')
parser.add_argument('-a', '--alpha', type=float, default = 1, help='Concentration Factor; default=1')
parser.add_argument('--init', type=int, default = 1, help='number of initial clusters, 0 is one cluster per data; default=1')
parser.add_argument('--base', type=int, default = 0, help='sampling type; 0 is position; 1 is position+directional')
args = parser.parse_args()


input_opt         = args.input
dataset_no        = args.data
iteration         = args.iteration
alpha             = args.alpha
init_opt          = args.init
base              = args.base

input_path = './data/input.csv'
output_path = './data/output.csv'


if input_opt == 2:
    pkg_dir = './data/'
    chosen_data_set = dataset_no
    sub_sample = 1
    nb_trajectories = 7
    Data = load_matlab_data(pkg_dir, chosen_data_set, sub_sample, nb_trajectories)
    Data = normalize_velocity_vector(Data)
    # Data = Data[:, 0:2]
    # Data = Data[0:100, :]
num, dim = Data.shape                                  # always pass the full data and parse it later on


with open(input_path, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(Data.shape[0]):
        data_writer.writerow(Data[i, :])


if base == 0:  # If only Eucliden distance is taken into account
    mu_0 = np.zeros((int(Data.shape[1]/2), ))
    sigma_0 = 0.1 * np.eye(mu_0.shape[0])
    lambda_0 = {
        "nu_0": sigma_0.shape[0] + 3,
        "kappa_0": 1,
        "mu_0": mu_0, 
        "sigma_0":  sigma_0
    }
elif base == 1:
    mu_0 = np.zeros((Data.shape[1], ))
    mu_0[-1] = 1                                        # prior belief on direction; i.e. the last two entries [0, 1]
    sigma_0 = 0.1 * np.eye(mu_0.shape[0]-1)             # reduced dimension of covariance
    sigma_0[-1, -1] = 1                                 # scalar directional variance with no correlation with position
    lambda_0 = {
        "nu_0": sigma_0.shape[0] + 3,
        "kappa_0": 1,
        "mu_0": mu_0,
        "sigma_0":  sigma_0
    }


params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]


args = ['time ' + os.path.abspath(os.getcwd()) + '/main',
        '-n {}'.format(num),
        '-m {}'.format(dim),        
        '-i {}'.format(input_path),
        '-o {}'.format(output_path),
        '-t {}'.format(iteration),
        '-a {}'.format(alpha),
        '--init {}'.format(init_opt), 
        '--base {}'.format(base),
        '-p ' + ' '.join([str(p) for p in params])
]


completed_process = subprocess.run(' '.join(args), shell=True)


assignment_array = np.genfromtxt(output_path, dtype=int, delimiter=',')


"""##### Plot Results ######"""
# """
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
for i in range(Data.shape[0]):
    color = colors[assignment_array[i]]
    ax.scatter(Data[i, 0], Data[i, 1], c=color)
ax.set_aspect('equal')
# """

# assignment_array = regress(Data, assignment_array)
# fig, ax = plt.subplots()
# colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
#     "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
# for i in range(Data.shape[0]):
#     color = colors[assignment_array[i]]
#     ax.scatter(Data[i, 0], Data[i, 1], c=color)
# ax.set_aspect('equal')
plt.show()



# print(np.mean(Data, axis=0))
# print(np.cov(Data.T)*(num-1))