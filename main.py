from util.load_data import *
from util.process_data import *
from util.animate import *
from util.modelRegression import *  
from util.load_plot_haihui import *
import argparse, subprocess, os, sys, csv, random

class dpmm:
    def __init__(self, *args_):

        #     ###############################################################
        #     ################## command-line arguments #####################
        #     ############################################################### 
        parser = argparse.ArgumentParser(
                        prog = 'Parallel Implemention of Dirichlet Process Mixture Model',
                        description = 'parallel implementation',
                        epilog = '2023, Sunan Sun <sunan@seas.upenn.edu>')


        parser.add_argument('--input', type=int, default=4, help='Choose Data Input Option: 4')
        parser.add_argument('-d', '--data', type=int, default=10, help='Choose Dataset, default=10')
        parser.add_argument('-t', '--iteration', type=int, default=100, help='Number of Sampler Iterations; default=50')
        parser.add_argument('-a', '--alpha', type=float, default = 1, help='Concentration Factor; default=1')
        parser.add_argument('--init', type=int, default = 15, help='number of initial clusters, 0 is one cluster per data; default=1')
        parser.add_argument('--base', type=int, default = 1, help='clustering option; 0: position; 1: position+directional')

        args = parser.parse_args()
        self.dataset_no        = args.data
        self.iteration         = args.iteration
        self.alpha             = args.alpha
        self.init_opt          = args.init
        self.base              = args.base


        ###############################################################
        ######################### load data ###########################
        ###############################################################  
        self.filepath = os.path.dirname(os.path.realpath(__file__))
        self.input_path = self.filepath + '/data/input.csv'
        self.output_path = self.filepath + '/data/'
        
        if len(args_) == 1:
            Data = args_[0]
        else:                               # Using Haihui's loading/plotting code(default)
            pkg_dir = self.filepath + '/data/'
            chosen_dataset = self.dataset_no   
            sub_sample = 1   
            if chosen_dataset == 10:
                nb_trajectories = 4
            else:
                nb_trajectories = 6
            Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
            vel_samples = 10
            vel_size = 20

            # pkg_dir = filepath + '/data/'
            # chosen_data_set = dataset_no
            # sub_sample = 1
            # nb_trajectories = 7
            # Data = load_matlab_data(pkg_dir, chosen_data_set, sub_sample, nb_trajectories)

        Data = normalize_velocity_vector(Data)                  
        Data = Data[np.logical_not(np.isnan(Data[:, -1]))]         # get rid of nan points
        self.num, self.dim = Data.shape                                   

        with open(self.input_path, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(self.num):
                data_writer.writerow(Data[i, :])

        self.Data = Data


        ###############################################################
        ####################### hyperparameters #######################
        ###############################################################  
        mu_0 = np.zeros((self.dim, ))
        mu_0[-1] = 1                                        
        sigma_0 = 0.1 * np.eye(int(mu_0.shape[0]/2) + 1)    
        sigma_0[-1, -1] = 1                                
        lambda_0 = {
            "nu_0": sigma_0.shape[0] + 3,
            "kappa_0": 1,
            "mu_0": mu_0,
            "sigma_0":  sigma_0
        }
        self.params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]



    def begin(self):
        ###############################################################
        ####################### perform dpmm ##########################
        ###############################################################  
        filepath =self.filepath
        args = ['time ' + filepath + '/main',
                '-n {}'.format(self.num),
                '-m {}'.format(self.dim), 
                '-i {}'.format(self.input_path),
                '-o {}'.format(self.output_path),       
                '-t {}'.format(self.iteration),
                '-a {}'.format(self.alpha),
                '--init {}'.format(self.init_opt), 
                '--base {}'.format(self.base),
                '-p ' + ' '.join([str(p) for p in self.params])
        ]

        completed_process     = subprocess.run(' '.join(args), shell=True)
        self.assignment_array = np.genfromtxt(filepath + '/data/output.csv', dtype=int, delimiter=',')
        self.reg_assignment_array = regress(self.Data, self.assignment_array)       
        self.logZ             = np.genfromtxt(filepath + '/data/logZ.csv', dtype=int, delimiter=None)
        self.logNum           = np.genfromtxt(filepath + '/data/logNum.csv', dtype=int, delimiter=',')
        self.logLogLik        = np.genfromtxt(filepath + '/data/logLogLik.csv', dtype=float, delimiter=',')

        unique_elements, counts = np.unique(self.assignment_array, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Number of", element, ":", count)


    def plot(self, aniFlag=True):
        ###############################################################
        ####################### plot results ##########################
        ###############################################################
        Data = self.Data
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]


        color_mapping = np.take(colors, self.assignment_array)
        reg_color_mapping = np.take(colors, self.reg_assignment_array)


        if self.dim == 4:
            _, ax1 = plt.subplots()
            ax1.scatter(Data[:, 0], Data[:, 1], c=color_mapping)
            ax1.set_aspect('equal')

            _, ax2 = plt.subplots()
            ax2.scatter(Data[:, 0], Data[:, 1], c=reg_color_mapping)
            ax2.set_aspect('equal')


        else:
            plt.figure()
            ax1 = plt.axes(projection='3d')
            ax1.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=color_mapping, s=5)

            plt.figure()
            ax2 = plt.axes(projection='3d')
            ax2.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=reg_color_mapping, s=5)

        ax2.set_title('Clustering Result: Dataset %i Base %i Init %i Iteration %i' %(self.dataset_no, self.base, self.init_opt, self.iteration))
        

        _, axes = plt.subplots(2, 1)
        axes[0].plot(np.arange(self.logNum.shape[0]), self.logNum, c='k')
        axes[0].set_title('Number of Components')
        axes[1].plot(np.arange(self.logLogLik.shape[0]), self.logLogLik, c='k')
        axes[1].set_title('Log Joint Likelihood')



        if aniFlag:
            logZ = self.logZ
            fig_ani = plt.figure()
            ax = plt.axes(projection='3d')
            scatter = ax.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c='k', s=5)

            def update(frame):
                scatter.set_color(np.take(colors, logZ[frame,:]))

            ani = animation.FuncAnimation(fig_ani, update, frames= logZ.shape[0], interval=800, repeat=True)

        plt.show()

    def returnPara(self):
        ###############################################################
        ################# return parameters ##########################
        ###############################################################  
        dim = self.dim
        assignment_array = self.reg_assignment_array
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


        return Priors, Mu, Sigma



if __name__ == "__main__":
    """
    If no arguments are given, we run the dpmm using default settings
    """
    if(len(sys.argv) == 1):
        filepath = os.path.dirname(os.path.realpath(__file__))
        pkg_dir = filepath + '/data/'
        chosen_dataset = 10
        sub_sample = 1   
        nb_trajectories = 4   
        Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
        DPMM = dpmm(Data)
        DPMM.begin()
        DPMM.plot()

    else:
        DPMM = dpmm()
        DPMM.begin()
        DPMM.plot()
        # print(DPMM.base)

    # data_ = loadmat(r"{}".format("data/pnp_done"))
    # data = np.array(data_["Data"])
    # dpmm(data)