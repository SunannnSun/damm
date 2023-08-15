import numpy as np
import matplotlib.pyplot as plt
import argparse, subprocess, os, sys, csv, json
from damm.util import plot_tools, data_tools


def write_data(data, path):
    N = data.shape[0]
    with open(path, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for n in range(N):
            data_writer.writerow(data[n, :])


def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_json(path):
    with open(path) as json_file:
        dict = json.load(json_file)
    return dict


class damm:
    def __init__(self, *args_):
        self.file_path           = os.path.dirname(os.path.realpath(__file__))
        self.log_path           = os.path.join(self.file_path, "log", "")

        ###############################################################
        ################## command-line arguments #####################
        ############################################################### 
        parser = argparse.ArgumentParser(
                            prog = 'Directionality-aware Mixture Model',
                            description = 'Python interface with C++ source code')

        parser.add_argument('-b', '--base' , type=int, default=1  , help='Clustering option; 0: position; 1: position+directional')
        parser.add_argument('-d', '--data' , type=int, default=10 , help='Dataset number, default=10')
        parser.add_argument('-i', '--init' , type=int, default=15 , help='Number of initial clusters, 0 is one cluster per data; default=15')
        parser.add_argument('-t', '--iter' , type=int, default=200, help='Number of iterations; default=200')
        parser.add_argument('-a', '--alpha', type=float, default=1, help='Concentration Factor; default=1')

        args = parser.parse_args()
        self.base               = args.base
        self.data               = args.data
        self.init               = args.init
        self.iter               = args.iter
        self.alpha              = args.alpha

        ###############################################################
        ######################### load data ###########################
        ###############################################################  
        if len(args_) != 1:
            raise Exception("Please provide input data to initialize a damm_class")
        
        Data = args_[0]
        self.Data = data_tools.normalize_vel(Data)              
        write_data(self.Data, os.path.join(self.log_path, "input.csv"))         


        ###############################################################
        ####################### hyperparameters #######################
        ###############################################################  
        mu_0            = np.zeros((self.Data.shape[1], )) 
        mu_0[-1]        = 1                                        
        # sigma_0         = 0.1 * np.eye(int(mu_0.shape[0]/2) + 1)    
        sigma_0         = 0.1 * np.eye(int(mu_0.shape[0]))  
        sigma_0[-1, -1] = 0.1                               
        lambda_0 = {
            "nu_0"      : sigma_0.shape[0] + 3,
            "kappa_0"   : 0.1,
            "mu_0"      : mu_0,
            "sigma_0"   : sigma_0
        }
        self.params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]


    def begin(self):
        ###############################################################
        ####################### perform damm ##########################
        ###############################################################  
        args = ['time ' + os.path.join(self.file_path, "main"),
                '-n {}'.format(self.Data.shape[0]),
                '-m {}'.format(self.Data.shape[1]), 
                '-t {}'.format(self.iter),
                '-a {}'.format(self.alpha),
                '--init {}'.format(self.init), 
                '--base {}'.format(self.base),
                '--log {}'.format(self.log_path),
                '-p ' + ' '.join([str(p) for p in self.params])
        ]

        completed_process = subprocess.run(' '.join(args), shell=True)
    
        return completed_process.returncode

        
    def result(self, if_plot=True):
        Data = self.Data

        logZ             = np.genfromtxt(os.path.join(self.log_path, 'logZ.csv'     ), dtype=int,   delimiter=None )
        logNum           = np.genfromtxt(os.path.join(self.log_path, 'logNum.csv'   ), dtype=int,   delimiter=','  )
        logLogLik        = np.genfromtxt(os.path.join(self.log_path, 'logLogLik.csv'), dtype=float, delimiter=','  )
        assignment_array = np.genfromtxt(os.path.join(self.log_path, "output.csv"   ), dtype=int,   delimiter=','  )

        _, _, param_dict        = data_tools.post_process(Data, assignment_array )
        reg_assignment_array    = data_tools.regress(Data, param_dict)  
        reg_param_dict          = data_tools.extract_param(Data, reg_assignment_array)


        Priors = reg_param_dict["Priors"]
        Mu     = reg_param_dict["Mu"]
        Sigma  = reg_param_dict["Sigma"]

        if if_plot:
            plot_tools.plot_results(Data, assignment_array    )
            # plot_tools.plot_results(Data, reg_assignment_array)
            data_tools.computeBIC(Data, reg_param_dict)
            # plot_tools.animate_results(Data, logZ             )

        np.save(os.path.join(self.log_path, "Priors.npy"), Priors )
        np.save(os.path.join(self.log_path, "Mu.npy"    ), Mu     )
        np.save(os.path.join(self.log_path, "Sigma.npy" ), Sigma.T)

        json_output = {
            "name": "DAMM result",
            "K": Priors.shape[0],
            "M": Mu.shape[1],
            "Priors": Priors.tolist(),
            "Mu": Mu.ravel().tolist(),
            "Sigma": Sigma.ravel().tolist(),
        }
        write_json(json_output, os.path.join(os.path.dirname(self.file_path), 'output.json'))

        plt.show()

