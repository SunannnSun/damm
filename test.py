import numpy as np
from scipy.stats import multivariate_normal


# Example log values
# log_values = np.array([1, 2, 3, 4, 5])

# # Apply log-sum-exp trick for normalization
# max_log = np.max(log_values)
# normalized_values = np.exp(log_values - max_log) / np.sum(np.exp(log_values - max_log))

# print(normalized_values)

likelihood = multivariate_normal(mean=np.array([1,2]), cov=np.eye(2), allow_singular=True).logpdf(np.array([2,1]))
print(likelihood)