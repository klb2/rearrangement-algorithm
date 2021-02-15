import numpy as np
from scipy import stats

import scratch


N = int(1e5)
qf = [stats.uniform.ppf]*5
#qf = [stats.uniform(-1, 1).ppf]*5
#qf = [stats.uniform.ppf, stats.uniform.ppf, stats.uniform.ppf]
#qf = [stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf]
#supermod_func = np.prod
#supermod_func = np.sum
supermod_func = np.min

results_low_low, results_low_up = scratch.bounds_expectation_supermod(
        qf, num_steps=N, supermod_func=supermod_func,
        method="lower", max_ra=0)
results_up_low, results_up_up = scratch.bounds_expectation_supermod(
        qf, num_steps=N, supermod_func=supermod_func,
        method="upper")
print("Results")
print(results_low_low[0], results_low_up[0])
print(results_up_low[0], results_up_up[0])
#print(results_low_low[1])
#print(supermod_func(results_low_low[1], axis=1))
