from util.load_data import *
from util.process_data import *
from util.modelRegression import regress
import argparse, subprocess, os, csv, random


a= [86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 403, 404, 405, 406, 407, 409, 410, 411, 412, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550]


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
    Data = Data[a, :]
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


a = np.where(assignment_array==1)[0]
print(a.tolist())

# print(np.mean(Data, axis=0))
# print(np.cov(Data.T)*(num-1))

