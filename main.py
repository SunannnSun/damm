from util.load_data import *
from util.process_data import *
from util.modelRegression import regress
from util.load_plot_haihui import *
import argparse, subprocess, os, csv, random


def dpmm():
    parser = argparse.ArgumentParser(
                        prog = 'Parallel Implemention of Dirichlet Process Mixture Model',
                        description = 'parallel implementation',
                        epilog = '2022, Sunan Sun <sunan@seas.upenn.edu>')


    parser.add_argument('-i', '--input', type=int, default=3, help='Choose Data Input Option: 0 handrawn; 1 load handdrawn; 2 load matlab')
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

    if input_opt == 1:
        completed_process = subprocess.run('matlab -nodesktop -sd "~/Developers/dpmm/util/drawData" -batch demo_drawData', shell=True)
        Data = np.genfromtxt('./data/human_demonstrated_trajectories_matlab.csv', dtype=float, delimiter=',')
    elif input_opt == 2:
        Data = np.genfromtxt('./data/human_demonstrated_trajectories_matlab.csv', dtype=float, delimiter=',')
    elif input_opt == 3:
        pkg_dir = './data/'
        chosen_data_set = dataset_no
        sub_sample = 1
        nb_trajectories = 7
        Data = load_matlab_data(pkg_dir, chosen_data_set, sub_sample, nb_trajectories)
        Data = normalize_velocity_vector(Data)
        # Data = Data[a, :]
        # Data = Data[b, :]
    elif input_opt == 4:          #Using Haihui's loading/plotting code
        pkg_dir = './data/'
        chosen_dataset = dataset_no  # 6 # 4 (when conducting 2D test)
        sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
        nb_trajectories = 4  # Only for real 3D data
        Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
        vel_samples = 10
        vel_size = 20
        plot_reference_trajectories_DS(Data, att, vel_samples, vel_size)
        Data = normalize_velocity_vector(Data)
        # print(np.linalg.norm(Data[:, 3:-1]))
        Data = Data[np.logical_not(np.isnan(Data[:, -1]))]  # get rid of nan points
            
        # Data = Data.T
        # fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        # ax1.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c='r', label='original demonstration', s=5)
        # plt.show()

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
        sigma_0 = 0.1 * np.eye(int(mu_0.shape[0]/2) + 1)    # reduced dimension of covariance
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
    logNum = np.genfromtxt('./data/logNum.csv', dtype=int, delimiter=',')
    logLogLik = np.genfromtxt('./data/logLogLik.csv', dtype=float, delimiter=',')

    # print(assignment_array.max())


    # print(np.amax(assignment_array))

    """##### Plot Results ######"""
    # """
    if dim == 4:
        fig, ax = plt.subplots()
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        for i in range(Data.shape[0]):
            color = colors[assignment_array[i]]
            ax.scatter(Data[i, 0], Data[i, 1], c=color)
        ax.set_aspect('equal')
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        for k in range(assignment_array.max()+1):
            color = colors[k]
            index_k = np.where(assignment_array==k)[0]
            Data_k = Data[index_k, :]
            ax.scatter(Data_k[:, 0], Data_k[:, 1], Data_k[:, 2], c=color, s=5)
    # """
    # plt.show()

    assignment_array = regress(Data, assignment_array)       #fix the scenario where small clusters vanish after regression

    # values, counts = np.unique(assignment_array, return_counts=True)
    # print(values)# print(counts)

    if dim == 4:
        fig, ax = plt.subplots()
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        for i in range(Data.shape[0]):
            color = colors[assignment_array[i]]
            ax.scatter(Data[i, 0], Data[i, 1], c=color)
        ax.set_aspect('equal')
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
            "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        for k in range(assignment_array.max()+1):
            color = colors[k]
            index_k = np.where(assignment_array==k)[0]
            Data_k = Data[index_k, :]
            ax.scatter(Data_k[:, 0], Data_k[:, 1], Data_k[:, 2], c=color, s=5)
    ax.set_title('Clustering Result: Dataset %i Base %i Init %i Iteration %i' %(dataset_no, base, init_opt, iteration))

    
    _, axes = plt.subplots(2, 1)
    axes[0].plot(np.arange(logNum.shape[0]), logNum, c='k')
    axes[0].set_title('Number of Components')
    axes[1].plot(np.arange(logLogLik.shape[0]), logLogLik, c='k')
    axes[1].set_title('Log Joint Likelihood')
    
    
    plt.show()
    # values, counts = np.unique(assignment_array, return_counts=True)
    # print(values)
    # print(counts)

    num_comp = assignment_array.max()+1
    Priors = np.zeros((num_comp, ))
    Mu = np.zeros((num_comp, int(dim/2)))
    Sigma = np.zeros((num_comp, int(dim/2), int(dim/2) ))

    for k in range(num_comp):
        data_k = Data[assignment_array==k, 0:int(dim/2)]
        Mu[k, :] = np.mean(data_k, axis=0)
        Sigma[k, :, :] = np.cov(data_k.T)
        Priors[k] = data_k.shape[0]
    Mu = Mu.T
    print(Priors)
    ds_opt_dir = './../ds-opt-py/'

    np.save(ds_opt_dir + 'distribution_difference_finding/Priors.npy', Priors)
    np.save(ds_opt_dir + 'distribution_difference_finding/Mu.npy', Mu)
    np.save(ds_opt_dir + 'distribution_difference_finding/Sigma.npy', Sigma)

    print(Mu.shape)
    print(Sigma.shape)



    # a = np.where(assignment_array==1)[0]
    # print(len(a.tolist()))
    # print(a.tolist())
    return Priors, Mu, Sigma

if __name__ == "__main__":
    dpmm()
