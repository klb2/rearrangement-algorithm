import numpy as np
from scipy import stats

import scratch

qf = [stats.uniform.ppf]*3
#qf = [stats.uniform.ppf, stats.uniform.ppf, stats.uniform.ppf]
#qf = [stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf]
supermod_func = np.prod
#supermod_func = np.sum

results_low, results_up = scratch.bounds_expectation_supermod(
        qf, num_steps=int(1e3), supermod_func=supermod_func,
        method="upper")
print("Results")
print(results_low[0], results_up[0])
