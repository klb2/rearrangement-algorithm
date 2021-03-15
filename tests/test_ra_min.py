import numpy as np
from scipy import stats

import scratch


N = int(1e3)
dist = [stats.expon]*5
#dist = [stats.uniform]*4
qf = [d.ppf for d in dist]
func = np.min

# Rearrange
## Expectation
expect_low_low, expect_low_up = scratch.bounds_expectation_supermod( qf,
        num_steps=N, supermod_func=func, method="lower", max_ra=0)
expect_up_low, expect_up_up = scratch.bounds_expectation_supermod( qf,
        num_steps=N, supermod_func=func, method="upper")

## Probability
s = .1
prob_low_under, prob_low_over = scratch.bounds_surv_probability(qf, s,
        num_steps=N, cost_func=func, method="lower", max_ra=0)
prob_up_under, prob_up_over = scratch.bounds_surv_probability(qf, s,
        num_steps=N, cost_func=func, method="upper", max_ra=0)


# Analytical
prob_up_analyt = 1.-np.max([d.cdf(s) for d in dist])
prob_low_analyt = 1.-np.minimum(sum([d.cdf(s) for d in dist]), 1)



print("Results")
print("Probability:\n{}".format(prob_up_under[1]))
#print("Expect:\n{}".format(expect_up_up[1]))
print(prob_low_under[0], prob_low_analyt, prob_low_over[0])
print(prob_up_under[0], prob_up_analyt, prob_up_over[0])
print(expect_low_low[0], expect_low_up[0])
print(expect_up_low[0], expect_up_up[0])
#print(results_low_low[0], results_low_up[0])
#print(results_up_low[0], results_up_up[0])
#print(results_low_low[1])
#print(supermod_func(results_low_low[1], axis=1))
