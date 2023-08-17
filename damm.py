import numpy as np
import matplotlib.pyplot as plt
import argparse, subprocess, os, sys, json
from damm.util import plot_tools, data_tools


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
        self.dir_path            = os.path.dirname(self.file_path)
        self.log_path            = os.path.join(self.file_path, "log", "")

        ###############################################################
        ################## command-line arguments #####################
        ############################################################### 
        parser = argparse.ArgumentParser(
                            prog = 'Directionality-aware Mixture Model',
                            description = 'C++ Implementaion with Python Interface')

        parser.add_argument('-b', '--base' , type=int, default=1  , help='0 damm; 1 position; 2 position+directional')
        parser.add_argument('-i', '--init' , type=int, default=15 , help='Number of initial clusters, 0 is one cluster per data; default=15')
        parser.add_argument('-t', '--iter' , type=int, default=2, help='Number of iterations; default=200')
        parser.add_argument('-a', '--alpha', type=float, default=1, help='Concentration Factor; default=1')

        args = parser.parse_args()
        self.base               = args.base
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
        self.num, self.dim = self.Data.shape             


        ###############################################################
        ####################### hyperparameters #######################
        ###############################################################  
        mu_0            = np.zeros((self.dim, )) 
        sigma_0         = 0.1 * np.eye(self.dim)
        nu_0            = self.dim + 3
        kappa_0         = 1
        sigma_dir_0     = 1

        self.param = ' '.join(map(str, np.r_[sigma_dir_0, nu_0, kappa_0, mu_0.ravel(), sigma_0.ravel()]))


    def begin(self):
        ###############################################################
        ####################### perform damm ##########################
        ###############################################################  
        command_line_args = ['time ' + os.path.join(self.file_path, "main"),
                            '--base {}'.format(self.base),
                            '--init {}'.format(self.init),
                            '--iter {}'.format(self.iter),
                            '--alpha {}'.format(self.alpha),
                            '--log {}'.format(self.log_path)
        ]
        input_data  = f"{self.num}\n{self.dim}\n{' '.join(map(str, self.Data.flatten()))}\n{self.param}"

        completed_process = subprocess.run(' '.join(command_line_args), input=input_data, text=True, shell=True)
    
        return completed_process.returncode

        
    def result(self, if_plot=True):
        # Data = self.Data

        # logZ             = np.genfromtxt(os.path.join(self.log_path, 'logZ.csv'     ), dtype=int,   delimiter=None )
        # logNum           = np.genfromtxt(os.path.join(self.log_path, 'logNum.csv'   ), dtype=int,   delimiter=','  )
        # logLogLik        = np.genfromtxt(os.path.join(self.log_path, 'logLogLik.csv'), dtype=float, delimiter=','  )
        try:
            with open(os.path.join(self.log_path, "assignment.bin"), "rb") as file:
                assignment_bin = file.read()
                assignment_arr = np.frombuffer(assignment_bin, dtype=np.int32).copy().reshape(self.num, )
                if assignment_arr.min() != 0:
                    raise ValueError("Invalid assignment array")
                if not all(isinstance(value, np.int32) for value in assignment_arr):
                    raise ValueError("Invalid assignment array")
                self.assignment_arr = assignment_arr
        except FileNotFoundError:
            print("Error: assignment.bin not found.")
            sys.exit()
        except Exception as e:
            print("Error:", e)
            sys.exit()

            
                
        # _, _, param_dict        = data_tools.post_process(Data, assignment_arr )
        # reg_assignment_array    = data_tools.regress(Data, param_dict)  
        # reg_param_dict          = data_tools.extract_param(Data, reg_assignment_array)

        reg_param_dict          = data_tools.extract_param(self.Data, self.assignment_arr)
        Priors = reg_param_dict["Priors"]
        Mu     = reg_param_dict["Mu"]
        Sigma  = reg_param_dict["Sigma"]

        if if_plot:
            plot_tools.plot_results(self.Data, self.assignment_arr)
            # plot_tools.plot_results(Data, reg_assignment_array)
            # data_tools.computeBIC(Data, reg_param_dict)
            # plot_tools.animate_results(Data, logZ             )

        # np.save(os.path.join(self.log_path, "Priors.npy"), Priors )
        # np.save(os.path.join(self.log_path, "Mu.npy"    ), Mu     )
        # np.save(os.path.join(self.log_path, "Sigma.npy" ), Sigma.T)

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

