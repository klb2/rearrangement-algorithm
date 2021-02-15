import numpy as np
from scipy import stats

import scratch

def pareto_ppf(p, theta=2.):
    return (1.-p)**(-1./theta) - 1.

N = int(1e4)
#qf = [pareto_ppf]*3
qf = [stats.pareto(2, loc=-1).ppf]*3
func = np.max
#func = np.sum
#supermod_func = np.prod
#supermod_func = np.min

s = 2
results_low_under, results_low_over = scratch.bounds_probability(qf, s,
        num_steps=N, supermod_func=func, method="upper", max_ra=0)
#results_up_low, results_up_up = scratch.bounds_expectation_supermod(
#        qf, num_steps=N, supermod_func=supermod_func,
#        method="upper")
print("Results")
print(results_low_under[0], results_low_over[0])
