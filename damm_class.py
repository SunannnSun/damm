import numpy as np
import matplotlib.pyplot as plt
import argparse, subprocess, os, sys, json
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from collections import OrderedDict



def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



def adjust_cov(cov, tot_scale_fact=2, rel_scale_fact=0.15):
    # print(cov)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    cov_ratio = eigenvalues_sorted[1]/eigenvalues_sorted[2]
    if cov_ratio < rel_scale_fact:
        lambda_3 = eigenvalues_sorted[2]
        lambda_2 = eigenvalues_sorted[1] + lambda_3 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_3 * (rel_scale_fact - cov_ratio)

        lambdas = np.array([lambda_1, lambda_2, lambda_3])

        L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
    else:
        L = np.diag(eigenvalues) * tot_scale_fact

    Sigma = eigenvectors @ L @ eigenvectors.T

    return Sigma




class damm_class:
    def __init__(self, x, x_dot, param_dict):
        """
        Parameters:
        -----------

        x:     (M, N) NumPy array of position input, assuming no shift (not ending at origin)

        x_dot: (M, N) NumPy array of position output (velocity)

        param_dict: a dictionary containing the hyperparameters of Gaussian conjugate prior
                    {
                        <mu_0>:       
                        <sigma_0>:
                        <nu_0>: 
                        <kappa_0>:
                        <sigma_dir>: 
                        <min_thold>:
                    }
        """


        # Define parameters and path
        self.x      = x
        self.x_dot  = x_dot
        mu_0, sigma_0, nu_0, kappa_0, sigma_dir_0, self.min_thold = param_dict.values()
        self.param  = ' '.join(map(str, np.r_[sigma_dir_0, nu_0, kappa_0, mu_0.ravel(), sigma_0.ravel()]))
        self.dir_path   = os.path.dirname(os.path.realpath(__file__))
        

        # Pre-process input data
        self.x_concat  = self._pre_process(self.x, self.x_dot) 
        self.M, self.N = self.x_concat.shape      # N is 4 or 6


        # Store command-line arguments 
        parser = argparse.ArgumentParser(
                            prog = 'Directionality-Aware Mixture Model',
                            description = 'C++ Implementaion with Python Interface')

        parser.add_argument('-b', '--base' , type=int, default=0,   help='0: Damm, 1: GMM-P, 2: GMM-PV')
        parser.add_argument('-i', '--init' , type=int, default=10,  help='Number of initial clusters')
        parser.add_argument('-t', '--iter' , type=int, default=30,  help='Number of iterations')
        parser.add_argument('-a', '--alpha', type=float, default=1, help='Concentration Factor')

        args, unknown = parser.parse_known_args()
        # args = parser.parse_args()
        self.base      = args.base
        self.init      = args.init
        self.iter      = args.iter
        self.alpha     = args.alpha



    def begin(self, *args_):
        # Pack input and arguments
        if len(args_) == 0:
            input = f"{self.M}\n{self.N}\n{' '.join(map(str, self.x_concat.flatten()))}\n{self.param}"
        else:
            input = f"{self.M}\n{self.N}\n{' '.join(map(str, self.x_concat.flatten()))}\n{self.param}\n{' '.join(map(str, args_[0]))}" # incremental learning

        args  = ['time ' + os.path.join(self.dir_path, "main"),
                            '--base {}'.format(self.base),
                            '--init {}'.format(self.init),
                            '--iter {}'.format(self.iter),
                            '--alpha {}'.format(self.alpha),
                            '--log {}'.format(self.dir_path)]


        # Run Damm 
        subprocess.run(' '.join(args), input=input, text=True, shell=True)
        
        try:
            with open(os.path.join(self.dir_path, "assignment.bin"), "rb") as file:
                assignment_bin = file.read()
                assignment_arr = np.frombuffer(assignment_bin, dtype=np.int32).copy().reshape(self.M, )
                if assignment_arr.min() != 0:
                    raise ValueError("Invalid assignment array")
                if not all(isinstance(value, np.int32) for value in assignment_arr):
                    raise ValueError("Invalid assignment array")
        except FileNotFoundError:
            print("Error: assignment.bin not found.")
            sys.exit()
        except Exception as e:
            print("Error:", e)
            sys.exit()


        # Extract Gaussians
        self._post_process(assignment_arr)
        self._extract_gaussian()


        # Return Gamma value
        return self.logProb(self.x)

    

    def _pre_process(self, x, x_dot):
        """ Extract x_dir, remove Nan values and concatenate states """

        x_dot_norm = np.linalg.norm(x_dot, axis=1)

        x     = x[x_dot_norm!=0]
        x_dot = x_dot[x_dot_norm!=0]

        x_dir = x_dot / x_dot_norm[x_dot_norm!=0].reshape(-1, 1)
        
        return np.hstack((x, x_dir)) 



    def _post_process(self, assignment_arr):
        # Delete small components
        unique_elements, counts = np.unique(assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            if  count < self.min_thold:
                indices_to_remove = np.where(assignment_arr==element)[0]
                assignment_arr  = np.delete(assignment_arr, indices_to_remove)
                self.x_concat   = np.delete(self.x_concat, indices_to_remove, axis=0)

        # Rearrange assignment label
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)
        
        unique_elements, counts = np.unique(assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Current element", element)
            print("has number", count)

        self.K = assignment_arr.max()+1
        self.assignment_arr = assignment_arr

    
            
    def _extract_gaussian(self):
        assignment_arr = self.assignment_arr
        K = self.K
        M = assignment_arr.shape[0]
        N = int(self.N/2)

        Prior  = [0] * K
        Mu      = np.zeros((K, N)) 
        Sigma   = np.zeros((K, N, N), dtype=np.float32)
        gaussian_list = []

        rearrange_K_idx  = list(OrderedDict.fromkeys(assignment_arr))
        for k in rearrange_K_idx:
            x_k                = self.x_concat[assignment_arr==k, :N]
            print(x_k.shape)
            Prior[k]           = x_k.shape[0] / M
            Mu[k, :]           = np.mean(x_k, axis=0)
            Sigma_k            = np.cov(x_k.T)            
            Sigma[k, :, :]     = adjust_cov(Sigma_k)
            # Sigma[k, :, :]     = Sigma_k

            gaussian_list.append({   
                "prior" : Prior[k],
                "mu"    : Mu[k],
                "sigma" : Sigma[k],
                "rv"    : multivariate_normal(Mu[k], Sigma[k], allow_singular=True)
            })

        self.gaussian_list = gaussian_list

        self.Prior  = Prior
        self.Mu     = Mu
        self.Sigma  = Sigma



    def elasticUpdate(self, new_traj, new_gmm_struct):

        self.x     = new_traj[:int(self.N/2), :].T
        self.x_dot = new_traj[int(self.N/2): ,:].T
        self.K = new_gmm_struct["Sigma"].shape[0] # shouldn't change
        self.M = self.x.shape[0]
        self.Prior  = new_gmm_struct["Prior"].tolist()
        self.Mu     = new_gmm_struct["Mu"].T
        self.Sigma  = new_gmm_struct["Sigma"]
        gaussian_list = []
        for k in range(self.K):
            gaussian_list.append({   
                "prior" : self.Prior[k],
                "mu"    : self.Mu[k],
                "sigma" : self.Sigma[k],
                # "sigma" : adjust_cov(self.Sigma[k]),
                "rv"    : multivariate_normal(self.Mu[k], self.Sigma[k], allow_singular=True)
            })
        self.gaussian_list = gaussian_list

        gamma = self.logProb(self.x)
        self.assignment_arr = np.argmax(gamma, axis = 0) # reverse order that we are assigning given the new gmm parameters; hence there's chance some component being empty
        unique_elements, counts = np.unique(self.assignment_arr, return_counts=True)
        for element, count in zip(unique_elements, counts):
            print("Current element", element)
            print("has number", count)
            if count == 0:
                input("Elastic update gamma gives zero count")
        return self.x, self.x_dot, self.assignment_arr, gamma



    def logProb(self, x):
        """ Compute log probability"""

        logProb = np.zeros((self.K, x.shape[0]))

        for k in range(self.K):
            prior_k, _, _, normal_k = tuple(self.gaussian_list[k].values())

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(x)

        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

        return postProb
    


    def totalProb(self, x):
        logProb = np.zeros((self.K, x.shape[0]))

        for k in range(self.K):
            prior_k, _, _, normal_k = tuple(self.gaussian_list[k].values())

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(x)

        return logsumexp(logProb, axis=0)



    def _logOut(self, *args):

        json_output = {
            "name": "Damm result",
            "K": self.Mu.shape[0],
            "M": self.Mu.shape[1],
            "Prior": self.Prior,
            "Mu": self.Mu.ravel().tolist(),
            "Sigma": self.Sigma.ravel().tolist(),
        }
        if len(js_path) == 0:
            js_path =  os.path.join(os.path.dirname(self.dir_path), 'output.json')
        write_json(json_output, js_path)


    
    # def _regress(self):
    #     """Vectorize later"""
    #     reg_assignment_array = np.zeros((self.M, ), dtype=int)
    #     for i in range(self.M):
    #         prob_array = np.zeros((self.K, ))
    #         for k in range(self.K):
    #             prob_array[k] = self.gaussian_list[k]['rv'].pdf(self.x[i, :])
    #         reg_assignment_array[i] = np.argmax(prob_array)

    #     return reg_assignment_array