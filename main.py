from util.load_data import *
from util.process_data import *
from util.modelRegression import regress
import argparse, subprocess, os, csv, random


# a= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787]
a = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 304, 320, 752, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 829, 831]
# b = [47,48,49,51,52,54,55,57,59,61,63,65,67,69,71,72,73,75,78,79,80,83,84,85,88,91,93,94,95,97,98,101,102,171,172,174,175,178,179,180,181,186,188,191,192,193,194,196,197,200,201,202,204,205,209,211,213,214,216,217,218,219,221,222,275,278,282,284,287,290,291,293,296,297,298,300,301,307,309,313,314,315,316,318,320,321,322,323,324]


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
    # Data = Data[a, :]
    # Data = Data[b, :]

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

assignment_array = regress(Data, assignment_array)
fig, ax = plt.subplots()
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
for i in range(Data.shape[0]):
    color = colors[assignment_array[i]]
    ax.scatter(Data[i, 0], Data[i, 1], c=color)
ax.set_aspect('equal')
plt.show()


a = np.where(assignment_array==3)[0]
# print(len(a.tolist()))
print(a.tolist())


# print(np.mean(Data, axis=0))
# print(np.cov(Data.T)*(num-1))

