import base
import scratch
from scipy import stats

#qf = [stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf]
#result_low, result_up = scratch.bounds_var(0., qf, method="upper", num_steps=50)


qf = [stats.pareto(2, loc=-1).ppf]*3
alpha = .5102
results_up_under, results_up_over = scratch.bounds_var(1.-alpha, qf, method="upper", num_steps=50)
results_low_under, results_low_over = scratch.bounds_var(1.-alpha, qf, method="lower", num_steps=50)
print("Results")
print(results_low_under[0], results_low_over[0])
print(results_up_under[0], results_up_over[0])
