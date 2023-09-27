import numpy as np
import matplotlib.pyplot as plt
import argparse, subprocess, os, sys, json
from damm.util import plot_tools, data_tools



def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



class damm:
    def __init__(self, hyper_param_):

        self.file_path           = os.path.dirname(os.path.realpath(__file__))
        self.dir_path            = os.path.dirname(self.file_path)
        self.log_path            = os.path.join(self.file_path, "log", "")
        
        # command-line arguments 
        parser = argparse.ArgumentParser(
                            prog = 'Directionality-aware Mixture Model',
                            description = 'C++ Implementaion with Python Interface')

        parser.add_argument('-b', '--base' , type=int, default=0  , help='0 damm; 1 position; 2 position+directional')
        parser.add_argument('-i', '--init' , type=int, default=15 , help='Number of initial clusters, 0 is one cluster per data; default=15')
        parser.add_argument('-t', '--iter' , type=int, default=200, help='Number of iterations')
        parser.add_argument('-a', '--alpha', type=float, default=1, help='Concentration Factor')

        args = parser.parse_args()
        self.base               = args.base
        self.init               = args.init
        self.iter               = args.iter
        self.alpha              = args.alpha

        # load hyperparameters
        mu_0, sigma_0, nu_0, kappa_0, sigma_dir_0 = hyper_param_.values()
        self.param = ' '.join(map(str, np.r_[sigma_dir_0, nu_0, kappa_0, mu_0.ravel(), sigma_0.ravel()]))


    def begin(self, data_, *args_):

        # load and process data
        self.Data = data_tools.normalize_vel(data_)
        self.num, self.dim = self.Data.shape  

        # perform damm 
        command_line_args = ['time ' + os.path.join(self.file_path, "main"),
                            '--base {}'.format(self.base),
                            '--init {}'.format(self.init),
                            '--iter {}'.format(self.iter),
                            '--alpha {}'.format(self.alpha),
                            '--log {}'.format(self.log_path)
        ]
        if len(args_) == 0:
            input_data  = f"{self.num}\n{self.dim}\n{' '.join(map(str, self.Data.flatten()))}\n{self.param}"
        else:
            input_data  = f"{self.num}\n{self.dim}\n{' '.join(map(str, self.Data.flatten()))}\n{self.param}\n{' '.join(map(str, args_[0]))}"

        completed_process = subprocess.run(' '.join(command_line_args), input=input_data, text=True, shell=True)
        
        #store old data for incremental learning
        self.prev_data = data_

        return completed_process.returncode
    

    def begin_next(self, next_data):

        K = np.max(self.assignment_arr)+1
        prev_assignment_arr = self.assignment_arr
        next_assignment_arr = -1 * np.ones((next_data.shape[1]), dtype=np.int32)
        comb_assignment_arr = np.concatenate((prev_assignment_arr, next_assignment_arr))


        comb_data = np.hstack((self.prev_data, next_data))

        self.begin(comb_data, comb_assignment_arr)
        self.evaluate()
        self.plot()

        # next_assignment_arr = np.zeros((next_data.shape[0]))
        # for i in range(next_assignment_arr.shape[0]):
        #     next_assignment_arr[i] = 1

        # self.assignment_arr = np.vstack((self.assignment_arr, next_assignment_arr))

        # self.begin()


        
    def evaluate(self):
        print(self.Data.shape)

        # read output in binary (avoid csv reading)
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


        print(self.Data.shape)
        print(assignment_arr.shape)

        Data = self.Data
        _, _, param_dict        = data_tools.post_process(Data, assignment_arr )
        reg_assignment_array    = data_tools.regress(Data, param_dict)  
        reg_param_dict          = data_tools.extract_param(Data, reg_assignment_array)

        Priors = reg_param_dict["Priors"]
        Mu     = reg_param_dict["Mu"]
        Sigma  = reg_param_dict["Sigma"]

        json_output = {
            "name": "DAMM result",
            "K": Priors.shape[0],
            "M": Mu.shape[1],
            "Priors": Priors.tolist(),
            "Mu": Mu.ravel().tolist(),
            "Sigma": Sigma.ravel().tolist(),
        }
        write_json(json_output, os.path.join(os.path.dirname(self.file_path), 'output.json'))



    def plot(self):

        plot_tools.plot_results(self.Data, self.assignment_arr, self.base)


        # logZ             = np.genfromtxt(os.path.join(self.log_path, 'logZ.csv'     ), dtype=int,   delimiter=None )
        # logNum           = np.genfromtxt(os.path.join(self.log_path, 'logNum.csv'   ), dtype=int,   delimiter=','  )
        # logLogLik        = np.genfromtxt(os.path.join(self.log_path, 'logLogLik.csv'), dtype=float, delimiter=','  )

        # plot_tools.plot_results(Data, reg_assignment_array, self.base)
        # plot_tools.plot_logs(logNum, logLogLik)
        # data_tools.computeBIC(Data, reg_param_dict)
        # plot_tools.animate_results(Data, logZ)
            