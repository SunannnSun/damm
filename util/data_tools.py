import numpy as np
from scipy.stats import multivariate_normal

"""
    Module Description:  
        data_tools module contains all function used in both pre-processing and post-procesiing of data
"""


def extract_param(data, assignment_array):
    """
    Extract the parameters (i.e. mean, covariance and number of points) for each component
    
    Parameters:
        data            : (N, 2M) data array contains both position and velocity vectors
        assignment_array: (N,  ) assignment array contains the label of each data points
        
    Returns:
        {
            Priors          : (K,  ) array contains the normalized number of points in each component, where K is the estimated number of components
            Mu              : (K, M) array contains the POSITIONAL average of each component
            Signma          : (K, M, M) contains the POSITIONAL covariance of each component
        }
    """
    N = data.shape[0]
    M = int(data.shape[1] / 2)
    K = assignment_array.max()+1

    Priors  = np.zeros((K, ))
    Mu      = np.zeros((K, M))
    Sigma   = np.zeros((K, M, M))

    for k in range(K):
        data_k          = data[assignment_array==k, 0:M]
        Priors[k]       = data_k.shape[0]/N
        Mu[k, :]        = np.mean(data_k, axis=0)
        Sigma[k, :, :]  = np.cov(data_k.T)
    
    param_dict ={
        "Priors": Priors,
        "Mu" : Mu,
        "Sigma": Sigma
    }

    return param_dict



def computeBIC(data, param_dict):
    """
    Compute the BIC metric given the clustering results
    
    Parameters:
        data            : (N, 2M) data array contains both position and velocity vectors
        
    Returns:
        BIC             : scalar value measuring the model complexity
        log_liks        : the log likelihood of data given the model
    """

    Priors = param_dict["Priors"]
    Mu     = param_dict["Mu"]
    Sigma  = param_dict["Sigma"]

    K = Priors.shape[0]
    N = data.shape[0]
    M = int(data.shape[1]/2)

    num_param = K*(1+2*M+(M**2-M)/2)-1

    log_liks = 0
    for n in range(N):
        likelihood = 0
        for k in range(K):
            likelihood += Priors[k] * multivariate_normal(mean=Mu[k, :], cov=Sigma[k, :, :], allow_singular=True).pdf(data[n, 0:M])
        log_liks += np.log(likelihood)
    
    BIC = num_param*np.log(data.shape[0]) - 2*log_liks

    print("The BIC of the model is ", BIC)
    print("The log likelihood of the model is ", log_liks)

    return BIC, log_liks



def normalize_vel(data):
    """
    Normalize the velocity vector and remove nan values

    Parameters:
        data            : (2M, N) data array directly from the load_data function
        
    Returns:
        norm_data       : (N, 2M) data array contains both position and normalized velocity (i.e., directions)
    """
    
    data = data.T
    M = int(data.shape[1]/2)

    vel_data = data[:, M:]
    vel_norm = np.linalg.norm(vel_data, axis=1)

    vel_data = vel_data[vel_norm!=0]
    pos_data = data[vel_norm!=0, 0:M]
    vel_norm = vel_norm[vel_norm!=0].reshape(-1, 1)

    normalized_vel_data = vel_data / vel_norm
    norm_data = np.hstack((pos_data, normalized_vel_data))

   
    return norm_data


def post_process(data, assignment_array):
    """
    delete super tiny component
    """
    unique_elements, counts = np.unique(assignment_array, return_counts=True)
    for element, count in zip(unique_elements, counts):
        print("Number of", element+1, ":", count)
        if count < 1/20*counts.max() or  count < 60:
            indices_to_remove =  np.where(assignment_array==element)[0]
            assignment_array = np.delete(assignment_array, indices_to_remove)
            data = np.delete(data, indices_to_remove, axis=0)

    rearrange_list = []
    for idx, entry in enumerate(assignment_array):
        if not rearrange_list:
            rearrange_list.append(entry)
        if entry not in rearrange_list:
            rearrange_list.append(entry)
            assignment_array[idx] = len(rearrange_list) - 1
        else:
            assignment_array[idx] = rearrange_list.index(entry)

    unique_elements, counts = np.unique(assignment_array, return_counts=True)      
    for element, count in zip(unique_elements, counts):
            print("Number of", element+1, ":", count)

    return data, assignment_array, extract_param(data, assignment_array)



def regress(data, param_dict):
    Priors = param_dict["Priors"]
    Mu     = param_dict["Mu"]
    Sigma  = param_dict["Sigma"]

    N = data.shape[0]
    M = int(data.shape[1]/2)
    K = Priors.shape[0]
    gauss_list = []

    for k in range(K):
        gauss_list.append(multivariate_normal(mean=Mu[k, :], cov=Sigma[k, :, :], allow_singular=True))

    reg_assignment_array = np.zeros((N, ), dtype=int)
    for n in range(N):
        prob_array = np.zeros((K, ))
        for k in range(K):
            prob_array[k] = gauss_list[k].pdf(data[n, 0:M])
        reg_assignment_array[n] = np.argmax(prob_array)

    return reg_assignment_array


# def expand_cov(cov):
#     """
#     Expand the covariance matrix corresponding to the initial points
#     """

#     w, v = np.linalg.eig(cov)


    
#     pass