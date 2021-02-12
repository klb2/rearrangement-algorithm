import base
import scratch
from scipy import stats

qf = [stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf]

#base.rearrange_algorithm(.1, qf)
result_low, result_up = scratch.rearrange_algorithm(0., qf, method="upper", num_steps=50)
print("Results")
print(result_low[0])
print(result_up[0])
