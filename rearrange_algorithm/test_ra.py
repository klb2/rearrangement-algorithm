import base
import scratch
from scipy import stats

qf = [stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf, stats.expon.ppf]

#base.rearrange_algorithm(.1, qf)
scratch.rearrange_algorithm(.1, qf, method="upper", num_steps=10)
